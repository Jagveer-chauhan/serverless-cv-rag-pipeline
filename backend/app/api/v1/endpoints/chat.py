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
    top_k: int = Field(default=10, ge=1, le=20, description="Number of vector context chunks to retrieve")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Prior conversation context")


RAG_SYSTEM_PROMPT = """You are an articulate, intelligent Technical Recruiter & CV Intelligence Assistant.
Your goal is to answer the user's questions about the candidate accurately, naturally, and conversationally based on the provided CV context.

Guidelines:
1. Tone: Professional, direct, engaging, and objective.
2. Focus: Answer the specific question asked directly and clearly. Do NOT repeat unrelated sections.
3. Brevity: For factual questions (e.g. years of experience, list of skills, number of companies), give a SHORT, direct answer — 1-3 sentences or a tight bullet list. Do NOT add lengthy explanations or padding unless the user explicitly asks for details.
4. Experience queries: If asked about years of experience (overall or in a specific skill/role), calculate from the dates in the CV and state it plainly, e.g. "5 years" or "3 years in Python (2021–2024)".
5. Skills queries: If asked what skills the candidate has, return a concise comma-separated list or short bullet list — no lengthy narrative.
6. Formatting: Use clean Markdown. Prefer inline answers over nested headers for short factual replies.
7. Greetings: If the user says hello or greets you, reply warmly and offer help with the candidate's profile.
8. Accuracy: Rely strictly on verified facts (companies, technologies, metrics, dates, and achievements) from the CV context.
9. Missing Info: If a requested detail is not present in the CV, state clearly and politely that it is not mentioned."""


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
            f"Hello! I'm your AI CV Intelligence Assistant. I'm currently analysing the profile for **{cand_name}**{title_str}.\n\n"
            f"You can ask me about:\n"
            f"- **Skills & Tech Stack** — languages, frameworks, databases, tools\n"
            f"- **Work Experience** — companies, roles, years of experience\n"
            f"- **Projects** — systems and applications built\n"
            f"- **Education & Certifications**\n\n"
            f"What would you like to know about **{cand_name}**?"
        )
    return (
        "Hello! I'm your AI CV Intelligence Assistant. I can answer questions about the "
        "candidate's skills, experience, projects, and education.\n\n"
        "How can I help you today?"
    )


# ---------------------------------------------------------------------------
# Intent classification & LLM helpers
# ---------------------------------------------------------------------------

# Map intent name → (section names to boost, keyword signals in query)
_INTENT_MAP = {
    "experience": (
        ["EXPERIENCE", "WORK", "EMPLOYMENT"],
        [
            "experience", "how many year", "years of experience", "how long", "total experience",
            "how much experience", "work history", "career", "job", "company", "companies",
            "role", "position", "employer", "employment", "worked", "work at", "work for",
        ],
    ),
    "skills": (
        ["SKILLS", "TECHNICAL", "TECHNOLOGIES", "STACK"],
        [
            "skill", "tech", "stack", "tool", "language", "framework", "library",
            "what can", "what does", "what know", "proficien", "expertise", "competenc",
            "python", "javascript", "react", "node", "php", "sql", "aws", "docker",
        ],
    ),
    "projects": (
        ["PROJECTS", "PROJECT", "PORTFOLIO"],
        ["project", "built", "developed", "application", "platform", "system", "portfolio"],
    ),
    "education": (
        ["EDUCATION", "CERTIFICATIONS", "TRAINING"],
        ["education", "degree", "university", "college", "certif", "course", "academic"],
    ),
    "contact": (
        ["CONTACT", "PERSONAL"],
        ["email", "phone", "contact", "location", "linkedin", "github", "address"],
    ),
    "summary": (
        ["SUMMARY", "PROFILE", "OBJECTIVE"],
        ["who is", "summary", "overview", "about", "profile", "background", "tell me about"],
    ),
}

# Factual queries that warrant a short, direct 1–2 sentence answer
_FACTUAL_SIGNALS = [
    "how many year", "years of experience", "how long", "total experience", "how much experience",
    "what skills", "which skills", "list of skill", "list the skill", "what languages",
    "what technologies", "tech stack", "key skills", "core skills",
    "how many companies", "number of companies", "number of year",
    "does he have", "does she have", "does the candidate",
]


def _classify_intent(query: str) -> Optional[str]:
    """Returns the dominant intent of the query, or None for general queries."""
    q = query.lower()
    for intent, (_, signals) in _INTENT_MAP.items():
        if any(sig in q for sig in signals):
            return intent
    return None


def _boost_context(
    query: str, all_chunks: List[Dict[str, Any]], top_k_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Prepends section-matched chunks to the context so the LLM always sees the right data.

    For focused queries (experience, skills, …) we guarantee the relevant section
    appears in the context even if vector similarity didn't rank it highly.
    The total list is capped at top_k + 3 to avoid bloating the prompt.
    """
    intent = _classify_intent(query)
    if not intent:
        return top_k_chunks

    target_sections, _ = _INTENT_MAP[intent]
    # Pull section-matched chunks from the full chunk list (not just top-k)
    section_chunks = [
        c for c in all_chunks
        if any(sec in (c.get("section_name") or "").upper() for sec in target_sections)
    ]
    if not section_chunks:
        return top_k_chunks

    # Merge: section chunks first, then remaining top-k, deduplicated by chunk_id
    seen_ids = set()
    merged: List[Dict[str, Any]] = []
    for c in section_chunks[:4] + top_k_chunks:
        cid = c.get("chunk_id")
        if cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(c)
    return merged[:len(top_k_chunks) + 3]


def _is_factual_query(query: str) -> bool:
    q = query.lower()
    return any(sig in q for sig in _FACTUAL_SIGNALS)


def _build_user_message(query: str, context_text: str) -> str:
    """Constructs the user turn sent to the LLM with intent-aware brevity enforcement."""
    if _is_factual_query(query):
        brevity = (
            "STRICT RULE: Your entire answer must fit in ONE short sentence or a bullet list "
            "of max 5 items. State only the direct fact — no intro, no explanation, no padding.\n\n"
        )
    else:
        brevity = ""
    return (
        f"CV Excerpts:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"{brevity}"
        f"Answer strictly from the excerpts above:"
    )


async def _call_llm_non_streaming(
    messages: List[Dict[str, str]],
    model_name: str,
    api_key: str,
    max_tokens: int = 300,
) -> str:
    """Calls the HF inference API without streaming and returns the full reply text."""
    from huggingface_hub import AsyncInferenceClient
    client = AsyncInferenceClient(api_key=api_key)
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        stream=False,
    )
    return response.choices[0].message.content or ""


def _raw_context_fallback(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Last-resort answer when no LLM key is available: surfaces the raw CV text."""
    cand_name, _ = _extract_candidate_name_and_title(chunks)
    prefix = f"**{cand_name}**" if cand_name else "The candidate"
    lines = []
    for c in chunks[:3]:
        sec = c.get("section_name", "").title()
        text = c.get("content", "").strip()
        if text:
            lines.append(f"**[{sec}]**\n{text}")
    body = "\n\n".join(lines) if lines else "No relevant CV content found."
    return (
        f"*(LLM unavailable — showing raw CV excerpts for {prefix})*\n\n"
        f"**Your question:** {query}\n\n"
        f"{body}"
    )


# ---------------------------------------------------------------------------
# Main SSE generator
# ---------------------------------------------------------------------------
async def generate_rag_sse_stream(
    query: str,
    chunks: List[Dict[str, Any]],
    chat_history: List[ChatMessage],
    api_key: str = settings.HF_API_KEY,
    model_name: str = settings.HF_MODEL_NAME,
) -> AsyncGenerator[str, None]:
    """Yields Server-Sent Events: citations → token stream → done."""

    # ── No CV ingested yet ──────────────────────────────────────────────────
    if not chunks:
        yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
        msg = (
            "Welcome! No candidate CVs have been ingested yet. "
            "Please click **'Ingest New CV'** to upload a CV and start querying."
        )
        for word in msg.split(" "):
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.015)
        yield f"event: done\ndata: {json.dumps({'status': 'complete', 'chunks_used': 0})}\n\n"
        return

    # ── Greeting — answered locally, no LLM call needed ────────────────────
    if is_conversational_greeting(query):
        yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
        text = _build_greeting_response(chunks)
        for word in text.split(" "):
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.015)
        yield f"event: done\ndata: {json.dumps({'status': 'complete', 'chunks_used': 0})}\n\n"
        return

    # ── Emit citations ───────────────────────────────────────────────────────
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

    # ── Boost context: ensure section-relevant chunks are included ───────────
    boosted_chunks = _boost_context(query, chunks, chunks)

    # ── Build context text and LLM message list ──────────────────────────────
    context_text = "\n\n".join(
        f"[EXCERPT {i+1} — {c['section_name']} (relevance {c['similarity']*100:.0f}%)]:\n{c['content']}"
        for i, c in enumerate(boosted_chunks)
    )
    user_msg = _build_user_message(query, context_text)

    # ── Intent-aware token budget ─────────────────────────────────────────────
    max_tokens = 120 if _is_factual_query(query) else 400

    base_messages: List[Dict[str, str]] = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        *[{"role": m.role, "content": m.content} for m in (chat_history or [])[-4:]],
        {"role": "user", "content": user_msg},
    ]

    # ── Try streaming LLM response ───────────────────────────────────────────
    if api_key:
        try:
            from huggingface_hub import AsyncInferenceClient
            hf_client = AsyncInferenceClient(api_key=api_key)
            stream = await hf_client.chat.completions.create(
                model=model_name,
                messages=base_messages,
                max_tokens=max_tokens,
                temperature=0.1,
                stream=True,
            )
            token_emitted = False
            async for chunk_resp in stream:
                if chunk_resp.choices:
                    delta = chunk_resp.choices[0].delta
                    if delta and delta.content:
                        token_emitted = True
                        yield f"event: token\ndata: {json.dumps({'token': delta.content})}\n\n"

            if not token_emitted:
                raise ValueError("HF streaming returned an empty token stream")

        except Exception as exc:
            logger.warning(f"HF streaming failed, attempting non-streaming fallback: {exc}")
            try:
                reply = await _call_llm_non_streaming(base_messages, model_name, api_key, max_tokens=max_tokens)
                if not reply.strip():
                    raise ValueError("Non-streaming LLM returned empty reply")
                for word in reply.split(" "):
                    yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                    await asyncio.sleep(0.01)
            except Exception as exc2:
                logger.error(f"Non-streaming LLM fallback also failed: {exc2}")
                reply = _raw_context_fallback(query, chunks)
                for word in reply.split(" "):
                    yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                    await asyncio.sleep(0.01)
    else:
        # ── No API key — surface raw context with a clear note ───────────────
        logger.warning("No HF API key configured — serving raw context fallback")
        reply = _raw_context_fallback(query, chunks)
        for word in reply.split(" "):
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.015)

    yield f"event: done\ndata: {json.dumps({'status': 'complete', 'chunks_used': len(chunks)})}\n\n"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@router.post(
    "",
    summary="RAG Chat SSE Stream",
    description="Query uploaded CVs with Server-Sent Events (SSE) streaming and chunk citations.",
)
async def chat_sse(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    try:
        chunks = await search_similar_chunks(
            db=db,
            query_text=request.query,
            document_id=request.document_id,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.warning(f"Vector search exception in chat: {e}")
        chunks = []

    return StreamingResponse(
        generate_rag_sse_stream(
            query=request.query,
            chunks=chunks,
            chat_history=request.chat_history or [],
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
