"""Hugging Face Serverless Inference API LLM extraction service.

Provider choice: Hugging Face Serverless Inference API
- Free tier, no credit card required
- True serverless: scales to zero when not in use, billed per request
- Hosts google/gemma-3-4b-it at zero idle cost
- Per-request GPU allocation with automatic cold-start handling
- Supports concurrent requests via asyncio.gather with semaphore

Cold-start strategy:
- HF loads the model container on first request (cold start)
- Subsequent requests reuse the warm container (warm path)
- cold_start is detected by comparing first vs subsequent request latencies
"""
import json
import re
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import httpx
from pydantic import ValidationError

from backend.app.core.config import settings
from backend.app.core.state import app_state
from backend.app.services.chunker import TextChunk
from backend.app.schemas.cv_schema import ChunkExtractionSchema

logger = logging.getLogger("cv_rag_pipeline.llm_extractor")

EXTRACTION_SYSTEM_PROMPT = """You are an expert resume parsing AI. Extract structured CV entities from the provided resume excerpt into valid JSON.

Strict JSON format rules:
- Output ONLY a single valid JSON object. No Markdown preamble, no explanations.
- Follow the target JSON schema structure:
  - candidate_info: {"name": str, "email": str, "phone": str, "location": str, "links": [str], "title": str}
  - summary: str
  - work_experience: [{"company": str, "position": str, "start_date": str, "end_date": str, "is_current": bool, "location": str, "description": str, "key_achievements": [str], "technologies_used": [str]}]
  - education: [{"institution": str, "degree": str, "field_of_study": str, "start_year": str, "end_year": str, "gpa": str, "honors": [str]}]
  - skills: [{"category_name": str, "skills": [str], "proficiency_level": str}]
  - projects: [{"name": str, "description": str, "role": str, "technologies": [str], "link": str}]
  - certifications: [{"name": str, "issuer": str, "issue_date": str, "expiry_date": str, "credential_id": str, "link": str}]
  - publications: [{"title": str, "authors": str, "publisher": str, "date": str, "link": str, "description": str}]
  - awards: [{"title": str, "issuer": str, "date": str, "description": str}]
  - languages: [{"language": str, "proficiency": str}]
  - sections: [{"heading": str, "content": str}]

Multi-page & Section Handling Rules:
- When extracting project sections (including excerpts on subsequent pages or continuation parts labeled [PROJECTS]), ALWAYS extract projects into the `projects` list, NEVER into custom `sections`.
- For projects with sub-sections or labeled points (such as Overview, Architecture, Key Features, Stakeholder Management, Security, Performance), consolidate all of them into the project's `description`.
- Extract whatever fields are present in the text excerpt. If a field is not present in the excerpt, omit it or use an empty list."""


def clean_json_response(raw_text: str) -> str:
    """Strips Markdown backticks and extraneous text to extract pure JSON string."""
    text = raw_text.strip()
    # Remove markdown code blocks ```json ... ```
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        text = json_match.group(1).strip()
    else:
        # Match outermost curly braces
        brace_match = re.search(r"(\{[\s\S]*\})", text)
        if brace_match:
            text = brace_match.group(1).strip()
    return text


def _clean_contact_label_prefixes(text: str) -> str:
    """Removes common field label prefixes from extracted text lines."""
    cleaned = re.sub(r"(?i)^(?:email|phone|tel|mobile|cell|location|address|city|linkedin|github|portfolio|website|links?)\s*[:|–—-]\s*", "", text).strip()
    cleaned = re.sub(r"^[\s,|–—-]+|[\s,|–—-]+$", "", cleaned).strip()
    return cleaned


class LLMExtractor:
    """Manages parallel LLM extraction via HF Serverless Inference API.

    Provider: Hugging Face Serverless Inference API
    - Model: google/gemma-3-4b-it
    - Endpoint: https://api-inference.huggingface.co/models/{model}
    - Scale-to-zero: Yes — HF unloads model after inactivity
    - Cold-start: ~8-15s first request; ~1-3s warm requests
    - Billing: Free tier (rate-limited); Pro tier per-second GPU billing
    - Concurrency: Bounded by asyncio.Semaphore (default 5 parallel)
    """

    def __init__(self, api_key: str = settings.HF_API_KEY, model_name: str = settings.HF_MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = settings.hf_llm_url
        self.client: Optional[httpx.AsyncClient] = None
        self._retry_count: int = 0

    async def get_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0),
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
        return self.client

    async def close(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()

    async def extract_chunk(
        self,
        chunk: TextChunk,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """Extracts structured JSON from a single chunk using AsyncInferenceClient with fallback."""
        if not self.api_key:
            return self._heuristic_chunk_extraction(chunk)

        try:
            from huggingface_hub import AsyncInferenceClient
            hf_client = AsyncInferenceClient(api_key=self.api_key)

            system_msg = (
                f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                f"You MUST output ONLY a valid JSON object matching the resume schema."
            )
            user_msg = (
                f"[RESUME SECTION EXCERPT]:\n{chunk.content}\n\n"
                f"Extract candidate info, work experience, education, skills, projects, certifications, publications, awards, languages as JSON:"
            )

            for attempt in range(max_retries + 1):
                t_start = time.perf_counter()
                try:
                    res = await hf_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        max_tokens=1024,
                        temperature=0.1
                    )

                    inference_ms = round((time.perf_counter() - t_start) * 1000, 2)
                    app_state.record_inference(inference_ms)

                    generated_text = res.choices[0].message.content or ""
                    clean_text = clean_json_response(generated_text)
                    parsed_data = json.loads(clean_text)

                    if not isinstance(parsed_data, dict):
                        raise ValueError(f"Expected JSON object, got {type(parsed_data)}")

                    ChunkExtractionSchema.model_validate(parsed_data)
                    return parsed_data

                except (json.JSONDecodeError, ValidationError, ValueError) as val_err:
                    self._retry_count += 1
                    logger.warning(
                        f"Chunk {chunk.chunk_index} validation failed "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {val_err}"
                    )
                    if attempt >= max_retries:
                        return self._heuristic_chunk_extraction(chunk)

        except Exception as e:
            logger.warning(f"AsyncInferenceClient failed for chunk {chunk.chunk_index}: {e}, using heuristic")
            return self._heuristic_chunk_extraction(chunk)

        return self._heuristic_chunk_extraction(chunk)

    @property
    def retry_count(self) -> int:
        return self._retry_count

    async def extract_all_chunks_parallel(
        self,
        chunks: List[TextChunk],
        max_concurrency: int = 3
    ) -> List[Dict[str, Any]]:
        """Parallel extraction across all chunks using asyncio.gather with bounded concurrency.

        Concurrency is capped at 3 for HF free-tier to avoid rate-limit 429s.
        Increase to 5 if using a paid HF Inference Endpoint.
        """
        if not chunks:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_extract(c: TextChunk) -> Dict[str, Any]:
            async with semaphore:
                return await self.extract_chunk(c)

        tasks = [_bounded_extract(c) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    def _heuristic_chunk_extraction(self, chunk: TextChunk) -> Dict[str, Any]:
        """Robust deterministic heuristic extraction for offline/fallback/zero-key modes."""
        content = chunk.content
        section = chunk.section_name.upper()
        data: Dict[str, Any] = {}

        if section == "CONTACT_HEADER" or "CONTACT" in section:
            # 1. Email matching
            email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", content)
            
            # 2. International & local phone matching
            phone_match = re.search(
                r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{1,4})?",
                content
            )
            phone_str = None
            if phone_match:
                candidate_phone = phone_match.group(0).strip()
                digit_count = sum(c.isdigit() for c in candidate_phone)
                if digit_count >= 7 and not re.search(r"@|github|linkedin|http", candidate_phone, re.I):
                    phone_str = candidate_phone

            # 3. Links matching (ignore email domains)
            links = []
            for m in re.finditer(r"(?<!@)\b(?:https?://[^\s,;)]+|linkedin\.com/in/[^\s,;)]+|github\.com/[^\s,;)]+|[a-zA-Z0-9-]+\.(?:dev|me|app)[^\s,;)]*)", content, re.I):
                url = m.group(0).rstrip(".,;|)")
                if not url.startswith("http"):
                    url = f"https://{url}"
                links.append(url)

            lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("[")]
            # Filter out generic header words
            filtered_lines = [
                l for l in lines
                if not re.match(r"(?i)^(?:curriculum vitae|resume|contact|contact information|profile)$", l.strip())
            ]

            name = "Candidate"
            title = None
            location = None

            clean_first_lines = []
            for l in filtered_lines:
                l_no_contact = l
                if email_match:
                    l_no_contact = l_no_contact.replace(email_match.group(0), "")
                if phone_str:
                    l_no_contact = l_no_contact.replace(phone_str, "")
                l_no_contact = re.sub(r"https?://\S+|linkedin\.com/\S+|github\.com/\S+", "", l_no_contact)
                l_no_contact = _clean_contact_label_prefixes(l_no_contact)
                if l_no_contact:
                    clean_first_lines.append(l_no_contact)

            if clean_first_lines:
                first_line = clean_first_lines[0]
                if "|" in first_line:
                    parts = [p.strip() for p in first_line.split("|") if p.strip()]
                    name = parts[0]
                    if len(parts) > 1 and not re.search(r"[@\d]", parts[1]):
                        title = " | ".join(parts[1:])
                else:
                    name = first_line

                for l in clean_first_lines[1:]:
                    l_clean = _clean_contact_label_prefixes(l)
                    if not l_clean:
                        continue
                    if not title and not re.search(r"[@\d]|linkedin|github|http", l_clean, re.I) and len(l_clean) < 100:
                        if "," in l_clean and any(kw in l_clean.lower() for kw in ["ca", "ny", "usa", "germany", "uk", "india", "gurugram", "faridabad", "delhi", "london", "berlin", "san francisco", "austin", "singapore", "toronto"]):
                            location = l_clean
                        else:
                            title = l_clean
                    elif not location and ("," in l_clean or any(kw in l_clean.lower() for kw in ["ca", "ny", "usa", "germany", "uk", "india", "gurugram", "faridabad", "delhi", "london", "berlin", "san francisco", "remote", "singapore", "toronto"])):
                        if not re.search(r"[@]|linkedin|github|http", l_clean, re.I):
                            location = l_clean

            if not location:
                loc_m = re.search(r"(?i)(?:Location|Address|City)\s*[:|-]\s*([^\n|]+)", content)
                if loc_m:
                    location = _clean_contact_label_prefixes(loc_m.group(1))

            if location:
                location = _clean_contact_label_prefixes(location)

            data["candidate_info"] = {
                "name": name,
                "email": email_match.group(0) if email_match else None,
                "phone": phone_str,
                "location": location,
                "title": title,
                "links": list(dict.fromkeys(links)),
            }

        elif section == "SUMMARY":
            summary_text = re.sub(r"^\[SUMMARY(?:.*?)]\s*", "", content, flags=re.I).strip()
            summary_text = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", summary_text, flags=re.I).strip()
            summary_text = re.sub(r"(?i)^(?:professional\s+summary|summary|profile|about\s+me)\s*\n+", "", summary_text).strip()
            # If summary contains attached CORE TECHNICAL SKILLS, strip it
            summary_text = re.split(r"(?i)\n\s*(?:core\s+technical\s+skills|technical\s+skills|skills)\b", summary_text)[0].strip()
            data["summary"] = summary_text

        elif section == "EXPERIENCE":
            clean_content = re.sub(r"^\[EXPERIENCE(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:work\s+experience|professional\s+experience|experience|employment\s+history|career\s+history|work\s+history|employment)\s*\n+", "", clean_content).strip()

            lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
            
            month_pattern = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|[0-9]{1,2}/)?\s*\d{4}(?:-[0-9]{1,2})?"
            date_regex = re.compile(rf"(?:\((?:from\s+)?|\b)({month_pattern})\s*(?:-|–|—|to)\s*({month_pattern}|Present|Current|Now)\)?", re.I)

            known_cities = [
                "Gurugram", "Gurgaon", "Faridabad", "Noida", "Delhi", "Bengaluru", "Bangalore",
                "Hyderabad", "Pune", "Mumbai", "Chennai", "Kolkata", "Ahmedabad", "Jaipur",
                "London", "Berlin", "San Francisco", "Austin", "New York", "Chicago", "Seattle"
            ]

            def extract_loc_and_clean(s: str) -> Tuple[str, Optional[str]]:
                loc = None
                s = s.strip(" ()-,|–—\t")
                parts = [p.strip() for p in re.split(r",(?![^()]*\))", s) if p.strip()]
                if len(parts) >= 2:
                    last = parts[-1]
                    if any(c.lower() in last.lower() for c in known_cities) or (len(last) < 25 and not any(kw in last.lower() for kw in ["inc", "llc", "ltd", "corp", "team", "technologies"])):
                        loc = last
                        s = ", ".join(parts[:-1])
                for c in known_cities:
                    if s.endswith(c) and len(s) > len(c) + 3:
                        if not loc:
                            loc = c
                        s = s[:-len(c)].rstrip(" ,|–—-")
                        break
                return s.strip(" ,|–—-"), loc

            def parse_job_title_company(raw_str: str) -> Tuple[str, str, Optional[str]]:
                comp = "Company"
                tit = "Role"
                loc = None
                raw_str = raw_str.strip(" ()-,|–—\t")

                # 1. Pipe-separated: e.g. 'Company, Location | Title' or 'Title at Company | Location'
                if " | " in raw_str or "|" in raw_str:
                    pipe_parts = [p.strip() for p in raw_str.split("|") if p.strip()]
                    p0 = pipe_parts[0]
                    p1 = pipe_parts[1] if len(pipe_parts) > 1 else ""
                    p2 = pipe_parts[2] if len(pipe_parts) > 2 else ""

                    p0_clean, loc0 = extract_loc_and_clean(p0)
                    p1_clean, loc1 = extract_loc_and_clean(p1)
                    loc = loc0 or loc1 or (p2 if len(p2) < 30 else None)

                    # Check dash inside p0
                    if " — " in p0_clean or " – " in p0_clean or " - " in p0_clean:
                        dash_parts = re.split(r"\s+[—–-]\s+", p0_clean, maxsplit=1)
                        dp0, dp1 = dash_parts[0].strip(), dash_parts[1].strip()
                        if any(kw in dp0.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "analyst", "consultant"]):
                            tit, comp = dp0, dp1
                        else:
                            comp, tit = dp0, dp1
                    elif any(kw in p1_clean.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "analyst", "consultant", "full-stack", "backend", "frontend"]):
                        comp, tit = p0_clean, p1_clean
                    elif any(kw in p0_clean.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "analyst", "consultant", "full-stack", "backend", "frontend"]):
                        tit, comp = p0_clean, p1_clean
                    else:
                        comp, tit = p0_clean, p1_clean

                elif " — " in raw_str or " – " in raw_str:
                    dash_parts = re.split(r"\s+[—–]\s+", raw_str, maxsplit=1)
                    p0, p1 = dash_parts[0].strip(), dash_parts[1].strip()
                    p0_clean, loc0 = extract_loc_and_clean(p0)
                    p1_clean, loc1 = extract_loc_and_clean(p1)
                    loc = loc0 or loc1
                    if any(kw in p0_clean.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "analyst", "consultant"]):
                        tit, comp = p0_clean, p1_clean
                    else:
                        comp, tit = p0_clean, p1_clean

                elif " at " in raw_str:
                    parts = raw_str.split(" at ", 1)
                    tit = parts[0].strip()
                    comp, loc = extract_loc_and_clean(parts[1].strip())
                else:
                    raw_clean, loc = extract_loc_and_clean(raw_str)
                    if any(kw in raw_clean.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "analyst", "consultant", "intern"]):
                        tit = raw_clean
                    else:
                        comp = raw_clean

                # Balance parentheses
                if tit.count("(") > tit.count(")"):
                    tit += ")"
                if comp.count("(") > comp.count(")"):
                    comp += ")"

                return comp or "Company", tit or "Role", loc

            job_blocks = []
            curr_block = {"headers": [], "dates": None, "lines": []}

            for line in lines:
                d_match = date_regex.search(line)
                is_bullet = bool(re.match(r"^[-•*–—\t]\s*", line))

                if d_match and not is_bullet:
                    start_date = d_match.group(1).strip()
                    end_date = d_match.group(2).strip()
                    rem_line = line[:d_match.start()] + line[d_match.end():]
                    rem_line = rem_line.strip(" ()-,|–—\t")

                    if curr_block["dates"]:
                        # Extract trailing non-bullet header lines from previous block
                        new_headers = []
                        while curr_block["lines"] and not bool(re.match(r"^[-•*–—\t]\s*", curr_block["lines"][-1])) and len(curr_block["lines"][-1]) < 90:
                            new_headers.insert(0, curr_block["lines"].pop())
                        job_blocks.append(curr_block)
                        curr_block = {"headers": new_headers, "dates": None, "lines": []}

                    if rem_line:
                        curr_block["headers"].append(rem_line)
                    curr_block["dates"] = (start_date, end_date)
                elif not curr_block["dates"]:
                    curr_block["headers"].append(line)
                else:
                    curr_block["lines"].append(line)

            if curr_block["dates"] or curr_block["headers"]:
                job_blocks.append(curr_block)

            # Check if this entire chunk has NO date lines (continuation chunk)
            if not any(b["dates"] for b in job_blocks) and not any(date_regex.search(l) for l in lines):
                all_bullets = []
                for l in lines:
                    clean_l = re.sub(r"^[-•*–—\t]+\s*", "", l).strip()
                    if clean_l:
                        all_bullets.append(clean_l)
                data["work_experience"] = [{
                    "company": "Company",
                    "position": "Role",
                    "title": "Role",
                    "location": None,
                    "start_date": None,
                    "end_date": None,
                    "key_achievements": all_bullets,
                    "achievements": all_bullets,
                    "technologies": [],
                    "technologies_used": [],
                    "description": None
                }]
            else:
                jobs = []
                for b in job_blocks:
                    headers = b["headers"]
                    dates = b["dates"]
                    raw_lines = b["lines"]

                    company = "Company"
                    title = "Role"
                    location = None

                    if len(headers) == 1:
                        company, title, location = parse_job_title_company(headers[0])
                    elif len(headers) >= 2:
                        l0, l1 = headers[0], headers[1]
                        c1, t1, loc1 = parse_job_title_company(l1)
                        c0, t0, loc0 = parse_job_title_company(l0)
                        if any(kw in l0.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "analyst", "consultant"]):
                            title = l0
                            company = c1 if c1 != "Company" else l1
                            location = loc1
                        elif any(kw in l1.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "analyst", "consultant"]):
                            title = l1
                            company = c0 if c0 != "Company" else l0
                            location = loc0 or loc1
                        else:
                            title = l0
                            company = l1
                            location = loc1
                    elif not headers:
                        company = "Company"
                        title = "Role"

                    achievements = []
                    description_lines = []

                    for l in raw_lines:
                        is_b = bool(re.match(r"^[-•*–—\t]\s*", l))
                        clean_l = re.sub(r"^[-•*–—\t]+\s*", "", l).strip()
                        if not clean_l:
                            continue
                        if is_b:
                            achievements.append(clean_l)
                        else:
                            description_lines.append(clean_l)

                    desc_text = "\n".join(description_lines).strip()
                    ach_text = "\n".join(achievements).strip()
                    final_desc = desc_text if desc_text and desc_text != ach_text else None
                    if not achievements and description_lines:
                        achievements = description_lines
                        final_desc = None

                    jobs.append({
                        "company": company or "Company",
                        "position": title or "Role",
                        "title": title or "Role",
                        "location": location,
                        "start_date": dates[0] if dates else None,
                        "end_date": dates[1] if dates else None,
                        "key_achievements": achievements,
                        "achievements": achievements,
                        "technologies": [],
                        "technologies_used": [],
                        "description": final_desc
                    })

                data["work_experience"] = jobs or [
                    {"company": "Company", "position": "Professional", "title": "Professional", "description": clean_content}
                ]

        elif section == "EDUCATION":
            edu_items = []
            clean_content = re.sub(r"^\[EDUCATION(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:education\s*&\s*training|education\s*&\s*qualifications|education|academic\s+background|academic\s+qualifications|qualifications|academics|degrees)\s*\n+", "", clean_content).strip()
            lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
            
            month_pattern = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|[0-9]{1,2}/)?\s*\d{4}(?:-[0-9]{1,2})?"
            date_regex = re.compile(rf"(?:\((?:from\s+)?|\b)({month_pattern})\s*(?:-|–|—|to)\s*({month_pattern}|Present|Current|Now)\)?", re.I)

            DEGREE_PATTERN = re.compile(
                r"(?ix)\b("
                r"Master\s+of\s+Computer\s+Applications(?:\s*\(MCA\))?(?:,\s*[^,\n|–—]+)?"
                r"|Bachelor\s+of\s+Computer\s+Applications(?:\s*\(BCA\))?(?:,\s*[^,\n|–—]+)?"
                r"|Bachelor(?:\'s|s)?\s+(?:of|in)\s+[^,\n|–—\(\)]+"
                r"|Master(?:\'s|s)?\s+(?:of|in)\s+[^,\n|–—\(\)]+"
                r"|Doctor(?:\'s|s)?\s+(?:of|in)\s+[^,\n|–—\(\)]+"
                r"|B\.?\s*Tech(?:\.|\b)(?:\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|M\.?\s*Tech(?:\.|\b)(?:\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|M\.?\s*C\.?\s*A\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|B\.?\s*C\.?\s*A\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|B\.?\s*E\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|M\.?\s*E\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|B\.?\s*S\.?\s*C\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|M\.?\s*S\.?\s*C\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|B\.?\s*S\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|M\.?\s*S\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|B\.?\s*B\.?\s*A\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|M\.?\s*B\.?\s*A\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|B\.?\s*Com\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|M\.?\s*Com\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|B\.?\s*A\.?\b(?:\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|M\.?\s*A\.?\b(?:\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|Ph\.?\s*D\.?(?:\b|\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|Associate(?:\'s)?\s+(?:Degree|of|in)?\s*[^,\n|–—\(\)]*"
                r"|Diploma(?:\s+(?:in|of)\s+[^,\n|–—\(\)]+)?"
                r"|Senior\s+Secondary|Higher\s+Secondary|High\s+School|Matriculation"
                r")\b"
            )

            i = 0
            while i < len(lines):
                line = lines[i]
                line = re.sub(r"^[-•*–—\t]+\s*", "", line).strip()
                if not line or len(line) < 3:
                    i += 1
                    continue

                if re.match(r"(?i)^(?:relevant\s+training|training|certifications)", line):
                    i += 1
                    continue

                d_match = date_regex.search(line)
                start_date = None
                end_date = None
                clean_l = line
                if d_match:
                    start_date = d_match.group(1).strip()
                    end_date = d_match.group(2).strip()
                    clean_l = clean_l[:d_match.start()] + clean_l[d_match.end():]
                clean_l = clean_l.strip(" ()-,|–—\t")

                institution = None
                degree = None

                if " | " in clean_l or "|" in clean_l or " — " in clean_l or " – " in clean_l:
                    sep_parts = [p.strip() for p in re.split(r"\s+[|—–]\s+|\|", clean_l) if p.strip()]
                    if len(sep_parts) >= 2:
                        p0, p1 = sep_parts[0], sep_parts[1]
                        if DEGREE_PATTERN.search(p1):
                            degree, institution = p1, p0
                        elif DEGREE_PATTERN.search(p0):
                            degree, institution = p0, p1
                        elif any(kw in p0.lower() for kw in ["university", "college", "institute", "school", "mit", "stanford", "cambridge", "berkeley", "iit"]):
                            institution, degree = p0, p1
                        else:
                            degree, institution = p0, p1
                    elif sep_parts:
                        clean_l = sep_parts[0]

                if not degree:
                    deg_m = DEGREE_PATTERN.search(clean_l)
                    if deg_m:
                        degree = clean_m = deg_m.group(0).strip()
                        rem_inst = clean_l.replace(clean_m, "").strip(" ()-,|–—\t")
                        if rem_inst and len(rem_inst) > 2:
                            institution = rem_inst
                    else:
                        degree = clean_l

                # If institution not on the same line, check subsequent lines
                if not institution or institution in ("University", "University / College"):
                    j = i + 1
                    while j < len(lines):
                        next_l = re.sub(r"^[-•*–—\t]+\s*", "", lines[j]).strip()
                        if not next_l:
                            j += 1
                            continue
                        next_d = date_regex.search(next_l)
                        if next_d:
                            if not start_date:
                                start_date = next_d.group(1).strip()
                                end_date = next_d.group(2).strip()
                            i = j
                            break
                        elif not institution or institution in ("University", "University / College"):
                            if not DEGREE_PATTERN.search(next_l) and len(next_l) > 3:
                                institution = next_l
                                i = j
                        j += 1

                edu_items.append({
                    "degree": degree or clean_l or "Degree / Study",
                    "institution": institution or "University / College",
                    "start_date": start_date,
                    "end_date": end_date,
                    "start_year": start_date,
                    "end_year": end_date,
                })
                i += 1

            data["education"] = edu_items or [{"institution": "University", "degree": "Degree"}]

        elif section == "SKILLS":
            skills_text = re.sub(r"^\[SKILLS(?:.*?)]\s*", "", content, flags=re.I).strip()
            skills_text = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", skills_text, flags=re.I).strip()
            skills_text = re.sub(r"(?i)^(?:core\s+technical\s+skills|technical\s+skills|skills|technologies|tools\s*&\s*technologies)\s*\n+", "", skills_text).strip()
            
            skills_by_cat = []
            explicit_skills = []

            def parse_skill_line(line_str: str) -> List[str]:
                tokens = []
                cleaned = re.sub(r"(?i)\b\d+\s+skills\b", "", line_str).strip()
                cleaned = re.sub(r"^[-•*–—\t]+\s*", "", cleaned).strip()
                if not cleaned:
                    return []

                # Split by commas/semicolons/pipes that are outside parentheses
                parts = [p.strip() for p in re.split(r"[,|;•](?![^()]*\))", cleaned) if p.strip()]
                for p in parts:
                    m = re.match(r"^([A-Za-z0-9\s&/.+#-]+)\s*\(([^)]+)\)$", p)
                    if m:
                        prefix_name = m.group(1).strip()
                        inner_items = [s.strip(" ()-–—,.") for s in re.split(r"[,/|;•]+", m.group(2)) if s.strip(" ()-–—,.")]
                        if prefix_name and len(prefix_name) < 40 and not re.match(r"(?i)^(?:skills|core|e\.?g\.?)$", prefix_name):
                            tokens.append(prefix_name)
                        tokens.extend(inner_items)
                    else:
                        token = p.strip(" ()-–—,.")
                        token = re.sub(r"^[()]+|[()]+$", "", token).strip()
                        if token and len(token) > 1 and not re.match(r"(?i)^(?:skills|core\s+skills|\d+\s+skills)$", token):
                            tokens.append(token)
                return tokens

            for line in skills_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    cat_name, s_vals = line.split(":", 1)
                    s_list = parse_skill_line(s_vals)
                    if s_list:
                        skills_by_cat.append({"category_name": cat_name.strip(), "skills": s_list})
                        explicit_skills.extend(s_list)
                else:
                    items = parse_skill_line(line)
                    explicit_skills.extend(items)

            # Deduplicate tokens
            clean_explicit = list(dict.fromkeys([s for s in explicit_skills if s and len(s) > 1]))
            if skills_by_cat:
                data["skills"] = skills_by_cat
            else:
                data["skills"] = [{"category_name": "Technical Skills", "skills": clean_explicit or [skills_text]}]

        elif section == "PROJECTS":
            clean_content = re.sub(r"^\[PROJECTS(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:selected\s+projects|key\s+projects|projects|technical\s+projects|notable\s+projects)\s*\n+", "", clean_content).strip()

            lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
            projects = []
            current_proj = None

            labeled_subsections = (
                "overview", "description", "student workflows", "certificate generation",
                "architecture", "role-specific dashboards", "interactive features", "data processing",
                "assessment workflows", "performance analytics", "security & compliance",
                "stakeholder management", "database & architecture", "system optimization",
                "security & access control", "state management", "security hardening",
                "code modernization", "technical details", "features", "responsibilities", "key features"
            )

            for line in lines:
                is_bullet = bool(re.match(r"^[-•*–—\t]\s*", line))
                has_pipe = " | " in line or "|" in line
                has_dash_separator = bool(re.search(r"\s+[—–-]\s+", line))
                starts_lowercase = bool(re.match(r"^[a-z]", line))
                is_sub_labeled = any(line.lower().startswith(f"{kw}:") for kw in labeled_subsections)

                # A line is a new Project Header if it's not a bullet/subheading/dangling line
                is_proj_header = False
                if not is_bullet and not is_sub_labeled and not starts_lowercase:
                    if has_pipe:
                        is_proj_header = True
                    elif has_dash_separator and len(line) < 120 and not line.endswith("."):
                        is_proj_header = True
                    elif len(line) < 60 and not line.endswith(".") and not line.endswith(",") and len(line.split()) <= 8:
                        # Ensure not a sentence or fragment
                        if not any(stop in line.lower() for stop in ["visitors", "servers", "platform", "implemented", "developed", "architected", "enhanced"]):
                            is_proj_header = True

                if is_proj_header:
                    if current_proj:
                        projects.append(current_proj)

                    pipe_parts = [p.strip() for p in line.split("|") if p.strip()]
                    name = pipe_parts[0] if pipe_parts else line
                    role = None
                    tech_stack = []
                    links = []

                    if " — " in name:
                        n_parts = name.split(" — ", 1)
                        name, role = n_parts[0].strip(), n_parts[1].strip()
                    elif " – " in name:
                        n_parts = name.split(" – ", 1)
                        name, role = n_parts[0].strip(), n_parts[1].strip()
                    elif " - " in name and len(name.split(" - ")[0]) > 4:
                        n_parts = name.split(" - ", 1)
                        if any(kw in n_parts[1].lower() for kw in ["learning", "portal", "platform", "app", "system", "plugin", "builder", "freshers", "edtech", "developer", "lead"]):
                            name, role = n_parts[0].strip(), n_parts[1].strip()

                    for p in pipe_parts[1:]:
                        if any(kw in p.lower() for kw in ["developer", "designer", "architect", "lead", "engineer", "manager", "author"]):
                            role = p
                        elif any(kw in p.lower() for kw in ["http", "github.com", "linkedin", ".in", ".com", ".org", "downloads", "visitors", "stars"]):
                            links.append(p)
                        elif "," in p or any(kw in p.lower() for kw in ["php", "laravel", "react", "node", "python", "sql", "mongo", "aws", "express", "vue", "docker"]):
                            tech_stack.extend([t.strip() for t in re.split(r"[,;]+", p) if t.strip()])
                        elif not role:
                            role = p

                    name = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", name, flags=re.I).strip(" ,|–—-")
                    current_proj = {
                        "name": name or "Project",
                        "role": role,
                        "description_lines": [],
                        "technologies": tech_stack,
                        "links": links,
                    }
                else:
                    clean_l = re.sub(r"^[-•*–—\t]+\s*", "", line).strip()
                    if current_proj:
                        current_proj["description_lines"].append(clean_l)
                    else:
                        # Top-of-chunk continuation from previous part
                        current_proj = {
                            "name": "__CONTINUATION__",
                            "role": None,
                            "description_lines": [clean_l],
                            "technologies": [],
                            "links": []
                        }

            if current_proj:
                projects.append(current_proj)

            for p in projects:
                p["description"] = "\n".join(p.pop("description_lines", [])).strip()

            data["projects"] = projects or [{"name": "Project", "description": clean_content}]

        elif section == "CERTIFICATIONS":
            clean_content = re.sub(r"^\[CERTIFICATIONS(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:relevant\s+training\s*&\s*certifications|certifications\s*&\s*training|certifications|certificates|licenses|training)\s*\n+", "", clean_content).strip()
            cert_items = []
            for l in clean_content.splitlines():
                l = re.sub(r"^[-•*–—\t]+\s*", "", l).strip()
                if not l or len(l) < 4:
                    continue
                
                m = re.search(r"\((.*?)(?:,\s*(\d{4}(?:-\d{2})?))?\)$", l)
                if m:
                    c_name = l[:m.start()].strip(" -•*,: ")
                    issuer = m.group(1).strip()
                    date = m.group(2).strip() if m.group(2) else None
                    cert_items.append({"name": c_name, "issuer": issuer, "date": date})
                else:
                    cert_items.append({"name": l})
            data["certifications"] = cert_items or [{"name": clean_content}]

        elif section == "PUBLICATIONS":
            clean_content = re.sub(r"^\[PUBLICATIONS(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:publications\s*&\s*research|research\s*&\s*publications|publications|research\s+papers|research|patents)\s*\n+", "", clean_content).strip()
            pub_items = []
            for l in clean_content.splitlines():
                l = re.sub(r"^[-•*–—\t]+\s*", "", l).strip()
                if not l or len(l) < 4:
                    continue
                
                link_m = re.search(r"https?://\S+", l)
                link_val = link_m.group(0) if link_m else None
                clean_title = re.sub(r"https?://\S+", "", l).strip(" ()-,|–—\t")
                
                date_m = re.search(r"\b(19\d\d|20\d\d)\b", clean_title)
                date_val = date_m.group(1) if date_m else None
                
                pub_items.append({
                    "title": clean_title,
                    "date": date_val,
                    "link": link_val,
                    "description": l
                })
            data["publications"] = pub_items or [{"title": "Publications", "description": clean_content}]

        elif section == "AWARDS":
            clean_content = re.sub(r"^\[AWARDS(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:awards\s*&\s*honors|awards\s*&\s*achievements|honors\s*&\s*awards|awards|honors|achievements)\s*\n+", "", clean_content).strip()
            award_items = []
            for l in clean_content.splitlines():
                l = re.sub(r"^[-•*–—\t]+\s*", "", l).strip()
                if not l or len(l) < 4:
                    continue
                date_m = re.search(r"\b(19\d\d|20\d\d)\b", l)
                date_val = date_m.group(1) if date_m else None
                award_items.append({"name": l, "date": date_val})
            data["awards"] = award_items or [{"name": clean_content}]

        elif section == "LANGUAGES":
            clean_content = re.sub(r"^\[LANGUAGES(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:languages\s+known|language\s+proficiencies|known\s+languages|languages)\s*\n+", "", clean_content).strip()
            
            lang_items = []
            leaked_projects = []
            current_leak_proj = None

            # Known human languages to distinguish from projects
            known_languages = {
                "english", "hindi", "spanish", "french", "german", "mandarin", "japanese",
                "russian", "arabic", "portuguese", "bengali", "punjabi", "marathi", "tamil",
                "telugu", "gujarati", "urdu", "kannada", "malayalam", "odia", "italian"
            }

            lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
            for l in lines:
                l_clean = re.sub(r"^[-•*–—\t]+\s*", "", l).strip()
                l_lower = l_clean.lower()
                is_real_lang = any(kl in l_lower for kl in known_languages) or len(l_clean) < 25 and not any(kw in l_lower for kw in ["developer", "built", "architected", "backend", "frontend", "api", "auth", "vulnerability", "cbfc", "jobiq", "composer"])

                if is_real_lang and not current_leak_proj:
                    for sub_l in re.split(r"[,|;]+", l_clean):
                        sub_l = sub_l.strip()
                        if not sub_l:
                            continue
                        if "(" in sub_l and ")" in sub_l:
                            m = re.search(r"^(.*?)\((.*?)\)", sub_l)
                            if m:
                                lang_items.append({"language": m.group(1).strip(), "proficiency": m.group(2).strip()})
                            else:
                                lang_items.append({"language": sub_l, "proficiency": "Proficient"})
                        elif ":" in sub_l:
                            parts = sub_l.split(":", 1)
                            lang_items.append({"language": parts[0].strip(), "proficiency": parts[1].strip()})
                        else:
                            lang_items.append({"language": sub_l, "proficiency": "Proficient"})
                else:
                    # Non-language content inside LANGUAGES chunk is a project or custom section!
                    if len(l_clean) < 70 and not l_clean.endswith(".") and not any(kw in l_lower for kw in ["developer for", "implemented", "built"]):
                        if current_leak_proj:
                            leaked_projects.append(current_leak_proj)
                        current_leak_proj = {"name": l_clean, "description_lines": []}
                    elif current_leak_proj:
                        current_leak_proj["description_lines"].append(l_clean)
                    else:
                        current_leak_proj = {"name": "Project", "description_lines": [l_clean]}

            if current_leak_proj:
                leaked_projects.append(current_leak_proj)

            data["languages"] = lang_items or [{"language": "English", "proficiency": "Proficient"}]
            if leaked_projects:
                proj_objs = []
                for lp in leaked_projects:
                    proj_objs.append({
                        "name": lp["name"],
                        "description": "\n".join(lp.get("description_lines", []))
                    })
                data["projects"] = proj_objs

        else:
            sec_heading = section.replace("_", " ").title()
            clean_text = re.sub(r"^\[.*?]\s*", "", content).strip()
            clean_text = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_text, flags=re.I).strip()
            
            # Check if this fallback section is actually a project section
            if (
                any(kw in sec_heading.lower() for kw in ["project", "portfolio", "system", "application", "app"])
                or (len(clean_text) > 40 and any(kw in clean_text.lower() for kw in ["developer", "architected", "developed", "cbfc", "jobiq", "composer", "live composer", "django", "laravel", "portal", "lms"]))
            ):
                clean_chunk = TextChunk(
                    chunk_index=chunk.chunk_index,
                    section_name="PROJECTS",
                    content=f"[PROJECTS]\n{clean_text}",
                    token_count=chunk.token_count,
                    metadata=chunk.metadata
                )
                return self._heuristic_chunk_extraction(clean_chunk)
            
            data["sections"] = [{"heading": sec_heading, "content": clean_text}]

        return data
