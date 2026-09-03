"""RAG Chat endpoint with vector retrieval and Server-Sent Events (SSE) token streaming."""
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator
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


RAG_SYSTEM_PROMPT = """You are an intelligent HR & Technical Recruiter Assistant. Answer the user's question accurately using ONLY the provided CV context excerpts.

Rules:
1. Always cite specific section details and achievements present in the context.
2. If the context does not contain enough information to answer, state clearly that it is not mentioned in the candidate's resume.
3. Keep your answers professional, concise, and structured with bullet points where appropriate."""


async def generate_rag_sse_stream(
    query: str,
    chunks: List[Dict[str, Any]],
    chat_history: List[ChatMessage],
    api_key: str = settings.HF_API_KEY,
    model_name: str = settings.HF_MODEL_NAME
) -> AsyncGenerator[str, None]:
    """Asynchronous generator yielding Server-Sent Events (SSE) with citations and streaming tokens."""
    # 1. Emit citations event
    citations_data = [
        {
            "chunk_id": c["chunk_id"],
            "section_name": c["section_name"],
            "similarity": c["similarity"],
            "snippet": c["content"][:250] + ("..." if len(c["content"]) > 250 else ""),
        }
        for c in chunks
    ]
    yield f"event: citations\ndata: {json.dumps({'citations': citations_data})}\n\n"

    if not chunks:
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

    # 2. Formulate RAG context
    context_text = "\n\n".join(
        f"[EXCERPT {i+1} - {c['section_name']} (Relevance: {c['similarity']*100:.1f}%)]:\n{c['content']}"
        for i, c in enumerate(chunks)
    )

    prompt = f"{RAG_SYSTEM_PROMPT}\n\n[CV CONTEXT]:\n{context_text}\n\n[USER QUESTION]:\n{query}\n\n[ASSISTANT ANSWER]:"

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
                f"[CANDIDATE CV CONTEXT]:\n{context_text}\n\n"
                f"[USER QUESTION]:\n{query}\n\n"
                f"Please provide an accurate, helpful, and professional response using ONLY the candidate CV context above:"
            )
            messages.append({"role": "user", "content": user_msg})

            stream = await hf_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=600,
                temperature=0.2,
                stream=True
            )

            async for chunk_resp in stream:
                if chunk_resp.choices and len(chunk_resp.choices) > 0:
                    delta = chunk_resp.choices[0].delta
                    if delta and delta.content:
                        yield f"event: token\ndata: {json.dumps({'token': delta.content})}\n\n"

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
    import re
    if not chunks:
        return "I could not find any relevant information in the uploaded CV documents regarding your question."

    sections: Dict[str, List[str]] = {}
    for c in chunks:
        sec = c.get("section_name", "GENERAL")
        raw_text = c.get("content", "").strip()
        clean_text = re.sub(r"^\[.*?\]\s*", "", raw_text)
        if sec not in sections:
            sections[sec] = []
        sections[sec].append(clean_text)

    ans_lines = [
        f"Here is the candidate intelligence found regarding **\"{query}\"**:\n"
    ]

    for sec, contents in sections.items():
        ans_lines.append(f"**{sec.replace('_', ' ').title()} Details:**")
        for text in contents:
            for line in text.splitlines():
                line = line.strip()
                if line and not line.upper() == sec:
                    if line.startswith(("-", "•", "*")):
                        ans_lines.append(f"- {line.lstrip('-•* ')}")
                    else:
                        ans_lines.append(f"- {line}")
        ans_lines.append("")

    return "\n".join(ans_lines)


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
