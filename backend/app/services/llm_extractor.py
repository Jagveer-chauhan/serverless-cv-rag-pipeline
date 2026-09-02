"""Hugging Face Serverless LLM Extraction service with validation retry loop and asyncio.gather."""
import json
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
from pydantic import ValidationError

from backend.app.core.config import settings
from backend.app.services.chunker import TextChunk
from backend.app.schemas.cv_schema import CVExtractionSchema

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
    """Manages parallel LLM extraction and self-correcting validation retry loop."""

    def __init__(self, api_key: str = settings.HF_API_KEY, model_name: str = settings.HF_MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = settings.hf_llm_url
        self.client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(4.0, connect=1.5),
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
        # If no HF API key is configured, use fast rule-based extraction fallback for offline / test environments
        if not self.api_key:
            return self._heuristic_chunk_extraction(chunk)

        client = await self.get_client()
        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n[RESUME SECTION EXCERPT]:\n{chunk.content}\n\n[OUTPUT JSON]:"

        for attempt in range(max_retries + 1):
            try:
                response = await client.post(
                    self.api_url,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 1024,
                            "temperature": 0.1,
                            "return_full_text": False
                        }
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"HF API returned status {response.status_code}: {response.text}")
                    # Fallback to heuristic on HF error
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

                # Validate against schema (or partial schema)
                # If top-level keys exist, validate via Pydantic
                CVExtractionSchema.model_validate(parsed_data)
                return parsed_data

            except (json.JSONDecodeError, ValidationError, ValueError) as val_err:
                logger.warning(
                    f"Chunk {chunk.chunk_index} validation failed (attempt {attempt + 1}/{max_retries + 1}): {val_err}"
                )
                if attempt < max_retries:
                    # Corrective prompt loop
                    prompt = (
                        f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                        f"[RESUME SECTION EXCERPT]:\n{chunk.content}\n\n"
                        f"CRITICAL ERROR: Your previous response was invalid with error:\n{str(val_err)}\n"
                        f"Fix the error and output ONLY the valid JSON object:\n\n[OUTPUT JSON]:"
                    )
                else:
                    logger.error(f"Exhausted validation retries on chunk {chunk.chunk_index}. Falling back to heuristic.")
                    return self._heuristic_chunk_extraction(chunk)
            except Exception as e:
                logger.error(f"Unexpected error in LLM extraction on chunk {chunk.chunk_index}: {e}", exc_info=True)
                return self._heuristic_chunk_extraction(chunk)

        return self._heuristic_chunk_extraction(chunk)

    async def extract_all_chunks_parallel(
        self,
        chunks: List[TextChunk],
        max_concurrency: int = 5
    ) -> List[Dict[str, Any]]:
        """Parallel extraction across all chunks using asyncio.gather with bounded concurrency."""
        if not chunks:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_extract(c: TextChunk) -> Dict[str, Any]:
            async with semaphore:
                return await self.extract_chunk(c)

        tasks = [_bounded_extract(c) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    def _heuristic_chunk_extraction(self, chunk: TextChunk) -> Dict[str, Any]:
        """Fast, robust deterministic heuristic extraction for offline, fallback, and zero-key modes."""
        content = chunk.content
        section = chunk.section_name
        data: Dict[str, Any] = {}

        if section == "CONTACT_HEADER" or "CONTACT" in section:
            email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", content)
            phone_match = re.search(r"\(?\+?[0-9]{1,3}\)?[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}", content)
            lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("[")]
            name = lines[0] if lines else "Candidate"
            title = lines[1] if len(lines) > 1 and "@" not in lines[1] else None

            data["candidate_info"] = {
                "name": name,
                "email": email_match.group(0) if email_match else None,
                "phone": phone_match.group(0) if phone_match else None,
                "title": title,
                "links": re.findall(r"https?://[^\s]+", content),
            }

        elif section == "SUMMARY":
            summary_text = re.sub(r"^\[SUMMARY\]\s*", "", content).strip()
            data["summary"] = summary_text

        elif section == "EXPERIENCE":
            exp_items = []
            blocks = re.split(r"\n(?=[A-Z][A-Za-z0-9\s]+(?:at|@|–|-|\()|\d{4})", content)
            for b in blocks:
                b_clean = re.sub(r"^\[EXPERIENCE(?:.*?)]\s*", "", b).strip()
                if not b_clean:
                    continue
                lines = [l.strip() for l in b_clean.splitlines() if l.strip()]
                if lines:
                    title_line = lines[0]
                    # Attempt parse "Role at Company"
                    company = "Company"
                    position = title_line
                    if " at " in title_line:
                        parts = title_line.split(" at ", 1)
                        position, company = parts[0], parts[1]
                    elif " - " in title_line:
                        parts = title_line.split(" - ", 1)
                        position, company = parts[0], parts[1]
                    
                    bullets = [l.lstrip("-•* ") for l in lines[1:] if l.startswith(("-", "•", "*"))]
                    exp_items.append({
                        "company": company,
                        "position": position,
                        "key_achievements": bullets,
                        "description": " ".join(lines[1:]) if not bullets else None
                    })
            data["work_experience"] = exp_items or [{"company": "Company", "position": "Professional", "description": content}]

        elif section == "EDUCATION":
            edu_items = []
            lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("[")]
            for l in lines:
                if len(l) > 3:
                    edu_items.append({"institution": l, "degree": "Degree / Study"})
            data["education"] = edu_items or [{"institution": "University", "degree": "Degree"}]

        elif section == "SKILLS":
            skills_text = re.sub(r"^\[SKILLS(?:.*?)]\s*", "", content).strip()
            # Split by comma or newline
            raw_skills = [s.strip() for s in re.split(r"[,|\n•]+", skills_text) if s.strip() and not s.startswith("[")]
            data["skills"] = [{"category_name": "Technical Skills", "skills": raw_skills}]

        elif section == "PROJECTS":
            data["projects"] = [{"name": "Key Project", "description": content}]

        elif section == "CERTIFICATIONS":
            data["certifications"] = [{"name": content.replace("[CERTIFICATIONS]", "").strip() or "Certification"}]

        return data
