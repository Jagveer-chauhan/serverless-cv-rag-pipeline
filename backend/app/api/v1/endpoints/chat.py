"""RAG Chat endpoint with vector retrieval and Server-Sent Events (SSE) token streaming."""
import re
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from backend.app.core.config import settings
from backend.app.db.session import get_db
from backend.app.services.vector_store import search_similar_chunks

logger = logging.getLogger("cv_rag_pipeline.chat")

router = APIRouter()


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    query: str = Field(..., description="Natural language question about the candidate/CV")
    document_id: Optional[str] = Field(None, description="Optional document ID filter for targeted querying")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of vector context chunks to retrieve (spec: k=5–10)")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Prior conversation context")


RAG_SYSTEM_PROMPT = """You are an articulate, intelligent Technical Recruiter & CV Intelligence Assistant.
Your goal is to answer the user's questions about the candidate accurately, naturally, and conversationally based on the provided CV context.

Guidelines:
1. Tone: Professional, direct, engaging, and objective.
2. Focus: Answer the specific question asked directly and clearly. Do NOT repeat unrelated sections.
3. Formatting: Use clean Markdown formatting with bullet points and bold key terms for high readability.
4. Greetings: If the user says hello or greets you, reply warmly and offer help with the candidate's profile.
5. Accuracy: Rely strictly on verified facts (companies, technologies, metrics, dates, and achievements) from the CV context.
6. Missing Info: If a requested detail is not present in the CV, state clearly and politely that it is not mentioned."""


GREETING_PATTERNS = [
    r"^(?:hi|hello|hey|hola|greetings|howdy|sup|yo)\b",
    r"^(?:good\s+(?:morning|afternoon|evening|day))\b",
    r"^(?:how\s+are\s+you|who\s+are\s+you|what\s+can\s+you\s+do|help|what\s+is\s+this)\b",
    r"^(?:thanks|thank\s+you|cheers)\b",
]


def is_conversational_greeting(query: str) -> bool:
    """Detects simple conversational greetings and meta-prompts."""
    clean = query.strip().lower()
    clean = re.sub(r"[^\w\s]", "", clean)
    return any(re.search(pat, clean) for pat in GREETING_PATTERNS)


def _extract_candidate_name_and_title(chunks: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Extracts candidate name and title from chunk metadata or text content."""
    candidate_name = None
    candidate_title = None

    for c in chunks:
        meta = c.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("candidate_name"):
            candidate_name = meta["candidate_name"]
            break

    for c in chunks:
        sec = c.get("section_name", "").upper()
        content = c.get("content", "")
        if "CONTACT" in sec:
            lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("[")]
            if lines and not candidate_name:
                first = lines[0]
                if "|" in first:
                    candidate_name = first.split("|")[0].strip()
                    candidate_title = first.split("|")[1].strip()
                else:
                    candidate_name = first
            if len(lines) > 1 and not candidate_title:
                second = lines[1]
                if not re.search(r"[@\d]|http|linkedin|github", second, re.I) and len(second) < 80:
                    candidate_title = second
            break

    return candidate_name, candidate_title


def _build_greeting_response(chunks: List[Dict[str, Any]]) -> str:
    """Builds a warm, personalized greeting based on ingested candidate profile."""
    cand_name, cand_title = _extract_candidate_name_and_title(chunks)

    if cand_name and cand_name != "Candidate":
        title_str = f" ({cand_title})" if cand_title else ""
        return (
            f"Hello! I'm your AI CV Intelligence Assistant. I'm currently analyzing the profile for **{cand_name}**{title_str}.\n\n"
            f"Here are a few things you can ask me about this candidate:\n"
            f"- **Core Technical Stack**: Languages, frameworks, databases, and tools\n"
            f"- **Work Experience**: Previous companies, roles, and career accomplishments\n"
            f"- **Projects & Architecture**: Major systems, applications, and portal architectures built\n"
            f"- **Education & Certifications**: Academic qualifications and training\n\n"
            f"What would you like to explore about **{cand_name}**?"
        )
    else:
        return (
            "Hello! I'm your AI CV Intelligence Assistant. I can help you analyze candidate skills, "
            "work experience, projects, and educational background with real-time citations.\n\n"
            "How can I assist you with this candidate's resume today?"
        )


async def generate_rag_sse_stream(
    query: str,
    chunks: List[Dict[str, Any]],
    chat_history: List[ChatMessage],
    api_key: str = settings.HF_API_KEY,
    model_name: str = settings.HF_MODEL_NAME
) -> AsyncGenerator[str, None]:
    """Asynchronous generator yielding Server-Sent Events (SSE) with citations and streaming tokens."""
    if not chunks:
        yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
        welcome_msg = (
            "Welcome! No candidate CVs have been ingested yet. "
            "Please click the **'Ingest New CV'** button on the left to upload a CV (e.g. from the `samples/` directory) "
            "and start querying skills, experience, and match intelligence!"
        )
        for word in welcome_msg.split(" "):
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.015)
        yield f"event: done\ndata: {json.dumps({'status': 'complete', 'chunks_used': 0})}\n\n"
        return

    # If the user query is a simple greeting, reply naturally with conversational guidance (no citations needed)
    if is_conversational_greeting(query):
        yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
        greeting_text = _build_greeting_response(chunks)
        for word in greeting_text.split(" "):
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.015)
        yield f"event: done\ndata: {json.dumps({'status': 'complete', 'chunks_used': 0})}\n\n"
        return

    # 1. Filter and emit citations for relevant RAG chunks
    relevant_chunks = [c for c in chunks if c.get("similarity", 0) >= 0.05]
    citations_data = [
        {
            "chunk_id": c["chunk_id"],
            "section_name": c["section_name"],
            "similarity": round(c["similarity"], 3),
            "snippet": c["content"][:250] + ("..." if len(c["content"]) > 250 else ""),
        }
        for c in (relevant_chunks or chunks)[:4]
    ]
    yield f"event: citations\ndata: {json.dumps({'citations': citations_data})}\n\n"

    # 2. Formulate RAG context
    context_text = "\n\n".join(
        f"[EXCERPT {i+1} - {c['section_name']} (Relevance: {c['similarity']*100:.1f}%)]:\n{c['content']}"
        for i, c in enumerate(chunks)
    )

    # If HF API key is available, call HF AsyncInferenceClient streaming
    if api_key:
        try:
            from huggingface_hub import AsyncInferenceClient
            hf_client = AsyncInferenceClient(api_key=api_key)

            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
            ]
            for m in (chat_history or [])[-4:]:
                messages.append({"role": m.role, "content": m.content})

            user_msg = (
                f"Candidate Resume Excerpts:\n{context_text}\n\n"
                f"User Question: {query}\n\n"
                f"Please provide a natural, accurate, and structured response addressing the user's question using the context above:"
            )
            messages.append({"role": "user", "content": user_msg})

            stream = await hf_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=750,
                temperature=0.2,
                stream=True
            )

            token_emitted = False
            async for chunk_resp in stream:
                if chunk_resp.choices and len(chunk_resp.choices) > 0:
                    delta = chunk_resp.choices[0].delta
                    if delta and delta.content:
                        token_emitted = True
                        yield f"event: token\ndata: {json.dumps({'token': delta.content})}\n\n"

            if not token_emitted:
                raise ValueError("HF streaming returned empty token stream")

        except Exception as e:
            logger.warning(f"HF AsyncInferenceClient streaming failed, using fallback: {e}")
            fallback_reply = _generate_heuristic_answer(query, chunks)
            for word in fallback_reply.split(" "):
                yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                await asyncio.sleep(0.015)
    else:
        # Fast local heuristic synthesis for tests & zero-key offline mode
        fallback_reply = _generate_heuristic_answer(query, chunks)
        for word in fallback_reply.split(" "):
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.015)

    # 3. Emit completion event
    yield f"event: done\ndata: {json.dumps({'status': 'complete', 'chunks_used': len(chunks)})}\n\n"


def _generate_heuristic_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Smart heuristic natural answer synthesis for offline/fallback/zero-key mode."""
    if not chunks:
        return "I could not find any relevant information in the uploaded CV documents regarding your question."

    q_lower = query.lower()
    cand_name, cand_title = _extract_candidate_name_and_title(chunks)
    cand_prefix = f"**{cand_name}**" if cand_name else "The candidate"

    # 1. Greeting check
    if is_conversational_greeting(query):
        return _build_greeting_response(chunks)

    # 2. Skills / Stack query
    if any(k in q_lower for k in ["skill", "technolog", "stack", "tool", "language", "framework", "python", "php", "react", "node", "sql", "proficien"]):
        skill_chunks = [c for c in chunks if c.get("section_name") == "SKILLS"]
        lines = []
        if skill_chunks:
            for sc in skill_chunks:
                clean = re.sub(r"^\[.*?\]\s*", "", sc.get("content", "")).strip()
                for l in clean.splitlines():
                    l_str = l.strip(" -•*")
                    if l_str and not l_str.upper() == "SKILLS":
                        lines.append(f"- **{l_str}**" if ":" in l_str else f"- {l_str}")
        if lines:
            return (
                f"{cand_prefix} possesses the following core technical skills and proficiencies:\n\n"
                + "\n".join(lines)
            )

    # 3. Projects query
    if any(k in q_lower for k in ["project", "built", "cbfc", "erp", "lms", "portfolio", "platform", "system", "app", "developed"]):
        proj_chunks = [c for c in chunks if c.get("section_name") == "PROJECTS"]
        if proj_chunks:
            proj_blocks = []
            curr_proj = None
            for pc in proj_chunks:
                clean = re.sub(r"^\[.*?\]\s*", "", pc.get("content", "")).strip()
                clean = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean).strip()
                for line in clean.splitlines():
                    line_s = line.strip()
                    if not line_s or line_s.startswith("__CONTINUATION__"):
                        continue
                    is_header = ("|" in line_s or " – " in line_s or " — " in line_s) and not line_s.startswith(("-", "•", "*")) and not ":" in line_s.split("|")[0]
                    if is_header:
                        if curr_proj:
                            proj_blocks.append(curr_proj)
                        curr_proj = {"title": line_s.strip(" -•*"), "lines": []}
                    elif curr_proj:
                        curr_proj["lines"].append(line_s.strip(" -•*"))
                    else:
                        curr_proj = {"title": "Key Project", "lines": [line_s.strip(" -•*")]}
            if curr_proj:
                proj_blocks.append(curr_proj)

            if proj_blocks:
                formatted = []
                for p in proj_blocks:
                    b_str = "\n".join(f"- {l}" for l in p["lines"] if l)
                    formatted.append(f"### {p['title']}\n{b_str}")
                return (
                    f"Here are the notable projects developed by {cand_prefix}:\n\n"
                    + "\n\n".join(formatted)
                )

    # 4. Work Experience query
    if any(k in q_lower for k in ["experience", "work", "job", "career", "company", "companies", "role", "responsibilit", "history", "mazars", "bitstreaks", "onetick"]):
        exp_chunks = [c for c in chunks if c.get("section_name") == "EXPERIENCE"]
        if exp_chunks:
            job_blocks = []
            curr_job = None
            date_regex = re.compile(r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2}/)?\s*\d{4}\s*(?:-|–|—|to)\s*(?:\d{4}|Present|Current|Now)\b")

            for ec in exp_chunks:
                clean = re.sub(r"^\[.*?\]\s*", "", ec.get("content", "")).strip()
                clean = re.sub(r"\(Part\s+\d+/\d+\)\s*", "", clean).strip()
                for line in clean.splitlines():
                    line_s = line.strip()
                    if not line_s or line_s.startswith("__CONTINUATION__"):
                        continue
                    is_job_header = (date_regex.search(line_s) or "|" in line_s) and not line_s.startswith(("-", "•", "*"))
                    if is_job_header and (len(line_s) < 120):
                        if curr_job:
                            job_blocks.append(curr_job)
                        curr_job = {"title": line_s.strip(" -•*"), "lines": []}
                    elif curr_job:
                        curr_job["lines"].append(line_s.strip(" -•*"))
                    else:
                        curr_job = {"title": "Work Experience", "lines": [line_s.strip(" -•*")]}
            if curr_job:
                job_blocks.append(curr_job)

            if job_blocks:
                formatted = []
                for j in job_blocks:
                    b_str = "\n".join(f"- {l}" for l in j["lines"] if l)
                    formatted.append(f"### {j['title']}\n{b_str}")
                return (
                    f"Here is the professional work experience for {cand_prefix}:\n\n"
                    + "\n\n".join(formatted)
                )

    # 5. Education / Certification query
    if any(k in q_lower for k in ["education", "degree", "college", "university", "school", "certif", "course", "training", "academic"]):
        edu_chunks = [c for c in chunks if c.get("section_name") in ("EDUCATION", "CERTIFICATIONS")]
        if edu_chunks:
            items = []
            for ec in edu_chunks:
                sec_title = ec.get("section_name", "").title()
                clean = re.sub(r"^\[.*?\]\s*", "", ec.get("content", "")).strip()
                sec_items = [f"- {l.strip(' -•*')}" for l in clean.splitlines() if l.strip()]
                if sec_items:
                    items.append(f"**{sec_title}:**\n" + "\n".join(sec_items))
            if items:
                return (
                    f"Here is the educational background and certifications for {cand_prefix}:\n\n"
                    + "\n\n".join(items)
                )

    # 6. Contact & Links query
    if any(k in q_lower for k in ["contact", "email", "phone", "location", "linkedin", "github", "address", "reach", "call"]):
        contact_chunks = [c for c in chunks if "CONTACT" in c.get("section_name", "")]
        if contact_chunks:
            clean = re.sub(r"^\[.*?\]\s*", "", contact_chunks[0].get("content", "")).strip()
            return f"**Contact Information for {cand_prefix}:**\n\n{clean}"

    # 7. Summary / Overview query
    if any(k in q_lower for k in ["who is", "summary", "overview", "about", "profile", "background"]):
        summ_chunks = [c for c in chunks if c.get("section_name") == "SUMMARY"]
        if summ_chunks:
            clean = re.sub(r"^\[.*?\]\s*", "", summ_chunks[0].get("content", "")).strip()
            return f"**Professional Summary of {cand_prefix}**:\n\n{clean}"

    # 8. General Semantic Synthesis across top matching chunks
    matched_points = []
    query_tokens = set(re.findall(r"\w{3,}", q_lower))

    for c in chunks[:3]:
        sec = c.get("section_name", "General").replace("_", " ").title()
        raw_text = c.get("content", "").strip()
        clean_text = re.sub(r"^\[.*?\]\s*", "", raw_text)
        
        for line in clean_text.splitlines():
            line_str = line.strip(" -•*")
            if not line_str or line_str.upper() == sec.upper() or len(line_str) < 15:
                continue
            line_tokens = set(re.findall(r"\w{3,}", line_str.lower()))
            overlap = len(query_tokens & line_tokens)
            if overlap > 0:
                matched_points.append(f"- **[{sec}]**: {line_str}")

    if matched_points:
        # Deduplicate and return top matching points
        deduped = list(dict.fromkeys(matched_points))[:8]
        return (
            f"Based on {cand_prefix}'s CV, here are the key details regarding **\"{query}\"**:\n\n"
            + "\n".join(deduped)
        )

    # Fallback when no direct keyword overlap found
    first_chunk = chunks[0]
    sec = first_chunk.get("section_name", "General").replace("_", " ").title()
    clean = re.sub(r"^\[.*?\]\s*", "", first_chunk.get("content", "")).strip()
    preview = "\n".join([f"- {l.strip(' -•*')}" for l in clean.splitlines()[:5] if l.strip()])
    return (
        f"Regarding **\"{query}\"**, the most relevant section found in {cand_prefix}'s CV is **{sec}**:\n\n"
        f"{preview}"
    )


@router.post(
    "",
    summary="RAG Chat SSE Stream",
    description="Query uploaded CVs with Server-Sent Events (SSE) streaming and chunk citations."
)
async def chat_sse(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    try:
        # 1. Retrieve top-k relevant chunks via vector similarity
        chunks = await search_similar_chunks(
            db=db,
            query_text=request.query,
            document_id=request.document_id,
            top_k=request.top_k
        )
    except Exception as e:
        logger.warning(f"Vector search exception in chat: {e}")
        chunks = []

    # 2. Return SSE StreamingResponse
    return StreamingResponse(
        generate_rag_sse_stream(
            query=request.query,
            chunks=chunks,
            chat_history=request.chat_history or []
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )
