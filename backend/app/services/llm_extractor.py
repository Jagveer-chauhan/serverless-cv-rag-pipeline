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
from typing import List, Dict, Any, Optional
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
  - languages: [{"language": str, "proficiency": str}]
  - awards: [{"title": str, "issuer": str, "date": str, "description": str}]

Extract whatever fields are present in the text excerpt. If a field is not present in the excerpt, omit it or use an empty list."""


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
                f"Extract candidate info, work experience, education, skills, projects, certifications as JSON:"
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
        section = chunk.section_name
        data: Dict[str, Any] = {}

        if section == "CONTACT_HEADER" or "CONTACT" in section:
            email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", content)
            phone_match = re.search(r"(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", content)
            
            # Find links including github.com and linkedin.com without http prefix
            links = []
            for m in re.finditer(r"https?://[^\s,;]+|(?:linkedin\.com/in/[^\s,;]+)|(?:github\.com/[^\s,;]+)", content, re.I):
                url = m.group(0)
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
                if phone_match:
                    l_no_contact = l_no_contact.replace(phone_match.group(0), "")
                l_no_contact = re.sub(r"https?://\S+|linkedin\.com/\S+|github\.com/\S+", "", l_no_contact).strip(" |,-")
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
                    if not title and not re.search(r"[@\d]|linkedin|github|http", l, re.I) and len(l) < 100:
                        if "," in l and any(kw in l.lower() for kw in ["ca", "ny", "usa", "germany", "uk", "india", "gurugram", "faridabad", "delhi", "london", "berlin", "san francisco", "austin"]):
                            location = l
                        else:
                            title = l
                    elif not location and ("," in l or any(kw in l.lower() for kw in ["ca", "ny", "usa", "germany", "uk", "india", "gurugram", "faridabad", "delhi", "london", "berlin", "san francisco", "remote"])):
                        if not re.search(r"[@]|linkedin|github|http", l, re.I):
                            location = l

            data["candidate_info"] = {
                "name": name,
                "email": email_match.group(0) if email_match else None,
                "phone": phone_match.group(0) if phone_match else None,
                "location": location,
                "title": title,
                "links": list(set(links)),
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
            jobs = []
            current_job = None

            month_pattern = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|[0-9]{1,2}/)?\s*\d{4}(?:-[0-9]{1,2})?"
            date_regex = re.compile(rf"(?:\((?:from\s+)?|\b)({month_pattern})\s*(?:-|–|—|to)\s*({month_pattern}|Present|Current|Now)\)?", re.I)

            for line in lines:
                d_match = date_regex.search(line)
                has_separators = (" | " in line or " at " in line or " @ " in line)
                is_bullet = bool(re.match(r"^[-•*–—]\s*", line)) or (":" in line and len(line.split(":")[0]) < 35 and len(line) > 80 and not d_match)
                is_new_job_header = (d_match and not is_bullet) or (has_separators and d_match)

                if is_new_job_header:
                    if current_job:
                        jobs.append(current_job)

                    header_text = line
                    start_date = None
                    end_date = None
                    if d_match:
                        start_date = d_match.group(1).strip()
                        end_date = d_match.group(2).strip()
                        header_text = header_text.replace(d_match.group(0), "").strip(" ,|–—-")

                    company = "Company"
                    position = "Software Engineer"

                    if " | " in header_text:
                        parts = [p.strip() for p in header_text.split(" | ") if p.strip()]
                        if len(parts) >= 2:
                            p0, p1 = parts[0], parts[1]
                            if any(kw in p1.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "scientist", "consultant", "analyst", "designer"]):
                                company, position = p0, p1
                            elif any(kw in p0.lower() for kw in ["developer", "engineer", "architect", "lead", "manager", "specialist", "scientist", "consultant", "analyst", "designer"]):
                                position, company = p0, p1
                            else:
                                company, position = p0, p1
                    elif " at " in header_text:
                        parts = header_text.split(" at ", 1)
                        position, company = parts[0].strip(), parts[1].strip()
                    elif " - " in header_text:
                        parts = header_text.split(" - ", 1)
                        company, position = parts[0].strip(), parts[1].strip()
                    else:
                        position = header_text

                    clean_company = company.strip(" ,|–—-")
                    clean_pos = position.strip(" ,|–—-")
                    if clean_pos.count("(") > clean_pos.count(")"):
                        clean_pos += ")"
                    if clean_company.count("(") > clean_company.count(")"):
                        clean_company += ")"

                    current_job = {
                        "company": clean_company,
                        "position": clean_pos,
                        "title": clean_pos,
                        "start_date": start_date,
                        "end_date": end_date,
                        "key_achievements": [],
                        "achievements": [],
                        "technologies": [],
                        "technologies_used": [],
                        "description_lines": []
                    }
                elif current_job:
                    clean_l = re.sub(r"^[-•*–—]\s*", "", line)
                    current_job["key_achievements"].append(clean_l)
                    current_job["achievements"].append(clean_l)
                    current_job["description_lines"].append(clean_l)

            if current_job:
                jobs.append(current_job)

            for j in jobs:
                if j.get("description_lines"):
                    j["description"] = "\n".join(j.pop("description_lines"))

            data["work_experience"] = jobs or [
                {"company": "Company", "position": "Professional", "title": "Professional", "description": clean_content}
            ]

        elif section == "EDUCATION":
            edu_items = []
            clean_content = re.sub(r"^\[EDUCATION(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:education|academic\s+background|academic\s+qualifications|qualifications|degrees)\s*\n+", "", clean_content).strip()
            lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
            
            i = 0
            while i < len(lines):
                line = lines[i]
                if re.match(r"(?i)^(?:relevant\s+training|training|certifications)", line):
                    i += 1
                    continue
                
                if re.match(r"^\b\d{4}\b$", line):
                    i += 1
                    continue

                inst = line
                deg = None
                start_year = None
                end_year = None

                # Extract date ranges like (2010-09 to 2014-06)
                date_match = re.search(r"(?:\((?:from\s+)?|\b)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2}/)?\s*\d{4}(?:-[0-9]{1,2})?)\s*(?:-|–|to)?\s*(\d{4}(?:-[0-9]{1,2})?|Present|Current|Now)?\)?", line, re.I)
                if date_match:
                    start_year = date_match.group(1).strip()
                    end_year = date_match.group(2).strip() if date_match.group(2) else start_year
                    inst = inst.replace(date_match.group(0), "").strip(" ()-,|")

                if " | " in inst:
                    parts = [p.strip() for p in inst.split(" | ") if p.strip()]
                    if len(parts) >= 2:
                        inst, deg = parts[0], parts[1]
                elif " - " in inst:
                    parts = [p.strip() for p in inst.split(" - ") if p.strip()]
                    if len(parts) >= 2:
                        inst, deg = parts[0], parts[1]

                # If degree not inline, check if next line is degree
                if not deg and i + 1 < len(lines):
                    next_l = lines[i + 1]
                    if not re.match(r"^\b\d{4}\b$", next_l) and not any(kw in next_l.lower() for kw in ["university", "college", "institute", "school"]):
                        deg = next_l
                        i += 1

                # Check if next line is year
                if i + 1 < len(lines):
                    next_l = lines[i + 1]
                    y_m = re.search(r"\b(19\d\d|20\d\d)\b", next_l)
                    if y_m:
                        end_year = y_m.group(1)
                        start_year = end_year
                        i += 1

                if inst:
                    edu_items.append({
                        "institution": inst,
                        "degree": deg or "Degree / Study",
                        "start_year": start_year,
                        "end_year": end_year,
                        "start_date": start_year,
                        "end_date": end_year,
                    })
                i += 1

            data["education"] = edu_items or [{"institution": "University", "degree": "Degree"}]

        elif section == "SKILLS":
            skills_text = re.sub(r"^\[SKILLS(?:.*?)]\s*", "", content, flags=re.I).strip()
            skills_text = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", skills_text, flags=re.I).strip()
            skills_text = re.sub(r"(?i)^(?:core\s+technical\s+skills|technical\s+skills|skills|technologies|tools\s*&\s*technologies)\s*\n+", "", skills_text).strip()
            skills_by_cat = []
            explicit_skills = []

            for line in skills_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    cat_name, s_vals = line.split(":", 1)
                    s_list = [s.strip() for s in re.split(r"[,|•;]+", s_vals) if s.strip()]
                    if s_list:
                        skills_by_cat.append({"category_name": cat_name.strip(), "skills": s_list})
                        explicit_skills.extend(s_list)
                else:
                    items = [s.strip() for s in re.split(r"[,|\n•;]+", line) if s.strip()]
                    explicit_skills.extend(items)

            if skills_by_cat:
                data["skills"] = skills_by_cat
            else:
                data["skills"] = [{"category_name": "Technical Skills", "skills": explicit_skills or [skills_text]}]

        elif section == "PROJECTS":
            clean_content = re.sub(r"^\[PROJECTS(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:key\s+projects|projects|selected\s+projects)\s*\n+", "", clean_content).strip()

            raw_blocks = re.split(r"\n\s*\n|(?<=\n)(?=[A-Z0-9][A-Za-z0-9\s,&/().–—-]+(?:\s*\|\s*|\s*–\s*|\s*-\s*)(?:Lead|Senior|Full-Stack|Core|Solo|Developer|Architect|Engineer|Designer))", clean_content)
            
            projects = []
            for b in raw_blocks:
                b = b.strip()
                if not b or len(b) < 10:
                    continue
                lines = [l.strip() for l in b.splitlines() if l.strip()]
                if not lines:
                    continue
                
                header = lines[0]
                name = header
                role = None
                if " | " in header:
                    parts = [p.strip() for p in header.split(" | ") if p.strip()]
                    name = parts[0]
                    if len(parts) > 1:
                        role = parts[1]
                
                desc_lines = lines[1:] if len(lines) > 1 else [lines[0]]
                desc = "\n".join(desc_lines)
                projects.append({
                    "name": name,
                    "role": role,
                    "description": desc
                })

            data["projects"] = projects or [{"name": "Key Projects", "description": clean_content}]

        elif section == "CERTIFICATIONS":
            clean_content = re.sub(r"^\[CERTIFICATIONS(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean_content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:relevant\s+training\s*&\s*certifications|certifications\s*&\s*training|certifications|certificates|licenses|training)\s*\n+", "", clean_content).strip()
            cert_items = []
            for l in clean_content.splitlines():
                l = l.strip(" -•*")
                if not l or len(l) < 4:
                    continue
                
                # Match name (issuer, date) or name (issuer)
                m = re.search(r"\((.*?)(?:,\s*(\d{4}(?:-\d{2})?))?\)$", l)
                if m:
                    c_name = l[:m.start()].strip(" -•*,: ")
                    issuer = m.group(1).strip()
                    date = m.group(2).strip() if m.group(2) else None
                    cert_items.append({"name": c_name, "issuer": issuer, "date": date})
                else:
                    cert_items.append({"name": l})
            data["certifications"] = cert_items or [{"name": clean_content}]

        return data
