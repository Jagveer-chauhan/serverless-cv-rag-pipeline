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
        """Extracts structured JSON from a single chunk with schema validation retry loop."""
        if not self.api_key:
            return self._heuristic_chunk_extraction(chunk)

        client = await self.get_client()
        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n[RESUME SECTION EXCERPT]:\n{chunk.content}\n\n[OUTPUT JSON]:"

        for attempt in range(max_retries + 1):
            t_start = time.perf_counter()
            try:
                response = await client.post(
                    self.api_url,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 1024,
                            "temperature": 0.1,
                            "return_full_text": False
                        },
                        "options": {"wait_for_model": True}
                    }
                )

                inference_ms = round((time.perf_counter() - t_start) * 1000, 2)
                # Record cold-start vs warm inference in app state
                app_state.record_inference(inference_ms)

                if response.status_code == 503:
                    # Model is loading — this is a cold start on the HF side
                    logger.info(f"HF model loading (cold start) for chunk {chunk.chunk_index}, waiting...")
                    await asyncio.sleep(10)
                    continue

                if response.status_code != 200:
                    logger.warning(f"HF API returned status {response.status_code}: {response.text[:200]}")
                    return self._heuristic_chunk_extraction(chunk)

                res_json = response.json()
                if isinstance(res_json, list) and len(res_json) > 0 and "generated_text" in res_json[0]:
                    generated_text = res_json[0]["generated_text"]
                elif isinstance(res_json, dict) and "generated_text" in res_json:
                    generated_text = res_json["generated_text"]
                else:
                    generated_text = str(res_json)

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
                if attempt < max_retries:
                    prompt = (
                        f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                        f"[RESUME SECTION EXCERPT]:\n{chunk.content}\n\n"
                        f"CRITICAL ERROR: Your previous response was invalid with error:\n{str(val_err)}\n"
                        f"Fix the error and output ONLY the valid JSON object:\n\n[OUTPUT JSON]:"
                    )
                else:
                    logger.error(
                        f"Exhausted validation retries on chunk {chunk.chunk_index}. "
                        f"Falling back to heuristic."
                    )
                    return self._heuristic_chunk_extraction(chunk)
            except Exception as e:
                logger.error(
                    f"Unexpected error in LLM extraction on chunk {chunk.chunk_index}: {e}",
                    exc_info=True
                )
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

            if filtered_lines:
                # First line is usually Name (or Name | Title)
                first_line = filtered_lines[0]
                if "|" in first_line:
                    parts = [p.strip() for p in first_line.split("|") if p.strip()]
                    name = parts[0]
                    if len(parts) > 1 and not re.search(r"[@\d]", parts[1]):
                        title = parts[1]
                else:
                    name = first_line

                for l in filtered_lines[1:]:
                    if not title and not re.search(r"[@\d]|linkedin|github|http", l, re.I) and len(l) < 50:
                        if "," in l and any(kw in l.lower() for kw in ["ca", "ny", "usa", "germany", "uk", "india", "london", "berlin", "san francisco", "austin"]):
                            location = l
                        else:
                            title = l
                    elif not location and ("," in l or any(kw in l.lower() for kw in ["ca", "ny", "usa", "germany", "uk", "india", "london", "berlin", "san francisco", "remote"])):
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
            summary_text = re.sub(r"(?i)^(?:professional\s+summary|summary|profile|about\s+me)\s*\n+", "", summary_text).strip()
            data["summary"] = summary_text

        elif section == "EXPERIENCE":
            exp_items = []
            clean_content = re.sub(r"^\[EXPERIENCE(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:work\s+experience|professional\s+experience|experience|employment\s+history|career\s+history|work\s+history|employment)\s*\n+", "", clean_content).strip()
            # Split jobs by double newlines or date/title header lines
            blocks = re.split(r"\n\s*\n|(?<=\n)(?=[A-Z][A-Za-z0-9\s,&/.-]+(?:\s+at\s+|\s+@\s+|\s+–\s+|\s+-\s+|\s*\|\s*|\s*\(\d{4}|\s*\d{4}))", clean_content)
            
            for b in blocks:
                b_clean = b.strip()
                if not b_clean:
                    continue
                lines = [l.strip() for l in b_clean.splitlines() if l.strip()]
                if not lines:
                    continue

                header_line = lines[0]
                company = "Company"
                position = header_line
                start_date = None
                end_date = None

                # Extract date range like (2021-03 to 2024-07), (Jan 2020 - Present), 2020 - Present
                date_match = re.search(r"(?:\((?:from\s+)?|\b)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2}/)?\s*\d{4}(?:-[0-9]{1,2})?)\s*(?:-|–|to)\s*(\d{4}(?:-[0-9]{1,2})?|Present|Current|Now)\)?", header_line, re.I)
                if date_match:
                    start_date = date_match.group(1).strip()
                    end_date = date_match.group(2).strip()
                    header_line = header_line.replace(date_match.group(0), "").strip(" ()-,|")

                if " | " in header_line:
                    parts = header_line.split(" | ", 1)
                    # Check if company | position or position | company
                    if any(kw in parts[1].lower() for kw in ["engineer", "architect", "developer", "lead", "manager", "specialist", "scientist", "consultant"]):
                        company, position = parts[0].strip(), parts[1].strip()
                    else:
                        position, company = parts[0].strip(), parts[1].strip()
                elif " at " in header_line:
                    parts = header_line.split(" at ", 1)
                    position, company = parts[0].strip(), parts[1].strip()
                elif " - " in header_line:
                    parts = header_line.split(" - ", 1)
                    position, company = parts[0].strip(), parts[1].strip()

                bullets = []
                desc_lines = []
                technologies = []

                for l in lines[1:]:
                    # Check if line contains a date range
                    l_date = re.search(r"(?:\((?:from\s+)?|\b)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2}/)?\s*\d{4}(?:-[0-9]{1,2})?)\s*(?:-|–|to)\s*(\d{4}(?:-[0-9]{1,2})?|Present|Current|Now)\)?", l, re.I)
                    if l_date and not start_date:
                        start_date = l_date.group(1).strip()
                        end_date = l_date.group(2).strip()
                        continue

                    # Check for tech list in bullet
                    tech_match = re.search(r"(?i)(?:technologies|tech stack|tools|built with):\s*(.*)", l)
                    if tech_match:
                        technologies.extend([t.strip() for t in re.split(r"[,|;]+", tech_match.group(1)) if t.strip()])
                        continue

                    if l.startswith(("-", "•", "*", "–", "—")):
                        bullets.append(l.lstrip("-•*–— ").strip())
                    else:
                        desc_lines.append(l)

                exp_items.append({
                    "company": company or "Company",
                    "position": position or "Professional",
                    "title": position or "Professional",
                    "start_date": start_date,
                    "end_date": end_date,
                    "key_achievements": bullets,
                    "achievements": bullets,
                    "technologies": technologies,
                    "technologies_used": technologies,
                    "description": "\n".join(desc_lines) if desc_lines else None
                })

            data["work_experience"] = exp_items or [
                {"company": "Company", "position": "Professional", "title": "Professional", "description": content}
            ]

        elif section == "EDUCATION":
            edu_items = []
            clean_content = re.sub(r"^\[EDUCATION(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:education|academic\s+background|academic\s+qualifications|qualifications|degrees)\s*\n+", "", clean_content).strip()
            lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
            
            i = 0
            while i < len(lines):
                line = lines[i]
                institution = line
                degree = None
                start_year = None
                end_year = None

                # Extract date ranges like (2010-09 to 2014-06)
                date_match = re.search(r"(?:\((?:from\s+)?|\b)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2}/)?\s*\d{4}(?:-[0-9]{1,2})?)\s*(?:-|–|to)\s*(\d{4}(?:-[0-9]{1,2})?|Present|Current|Now)\)?", line, re.I)
                if date_match:
                    start_year = date_match.group(1).strip()
                    end_year = date_match.group(2).strip()
                    line = line.replace(date_match.group(0), "").strip(" ()-,|")

                if " | " in line:
                    parts = [p.strip() for p in line.split(" | ") if p.strip()]
                    if len(parts) >= 2:
                        institution, degree = parts[0], parts[1]
                elif " - " in line:
                    parts = [p.strip() for p in line.split(" - ") if p.strip()]
                    if len(parts) >= 2:
                        degree, institution = parts[0], parts[1]
                else:
                    deg_match = re.search(r"(?i)\b(B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?|Bachelor[s]?|Master[s]?|Doctorate|Associate|Diploma|Degree)(?:\s+(?:of|in)\s+[A-Za-z\s]+)?", line)
                    if deg_match:
                        degree = deg_match.group(0).strip()

                edu_items.append({
                    "institution": institution or "University",
                    "degree": degree or "Degree / Study",
                    "start_year": start_year,
                    "end_year": end_year,
                    "start_date": start_year,
                    "end_date": end_year,
                })
                i += 1

            data["education"] = edu_items or [{"institution": "University", "degree": "Degree"}]

        elif section == "SKILLS":
            skills_text = re.sub(r"^\[SKILLS(?:.*?)]\s*", "", content, flags=re.I).strip()
            skills_text = re.sub(r"(?i)^(?:technical\s+skills|skills|technologies|tools\s*&\s*technologies)\s*\n+", "", skills_text).strip()
            skills_by_cat = []
            explicit_skills = []

            for line in skills_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    cat_name, s_vals = line.split(":", 1)
                    s_list = [s.strip() for s in re.split(r"[,|•]+", s_vals) if s.strip()]
                    if s_list:
                        skills_by_cat.append({"category_name": cat_name.strip(), "skills": s_list})
                        explicit_skills.extend(s_list)
                else:
                    items = [s.strip() for s in re.split(r"[,|\n•]+", line) if s.strip()]
                    explicit_skills.extend(items)

            if skills_by_cat:
                data["skills"] = skills_by_cat
            else:
                data["skills"] = [{"category_name": "Technical Skills", "skills": explicit_skills or [skills_text]}]

        elif section == "PROJECTS":
            clean_content = re.sub(r"^\[PROJECTS(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:projects|key\s+projects)\s*\n+", "", clean_content).strip()
            data["projects"] = [{"name": "Key Projects", "description": clean_content}]

        elif section == "CERTIFICATIONS":
            clean_content = re.sub(r"^\[CERTIFICATIONS(?:.*?)]\s*", "", content, flags=re.I).strip()
            clean_content = re.sub(r"(?i)^(?:certifications|certificates|licenses)\s*\n+", "", clean_content).strip()
            cert_items = []
            for l in clean_content.splitlines():
                l = l.strip(" -•*")
                if l:
                    # Match name (issuer, date) like AWS Certified Solutions Architect (Amazon, 2023-01)
                    meta_m = re.search(r"\((.*?),\s*(\d{4}(?:-\d{2})?)\)", l)
                    if meta_m:
                        c_name = l.replace(meta_m.group(0), "").strip(" -•*,")
                        cert_items.append({"name": c_name, "issuer": meta_m.group(1).strip(), "date": meta_m.group(2).strip()})
                    else:
                        cert_items.append({"name": l})
            data["certifications"] = cert_items or [{"name": clean_content}]

        return data
