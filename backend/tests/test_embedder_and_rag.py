"""Tests for vector embeddings, similarity search, RAG verification gate, and SSE chat."""
import json
import pytest
import pytest_asyncio
try:
    import fitz
except ImportError:
    import pymupdf as fitz
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.app.main import app
from backend.app.models.base import Base
from backend.app.db.session import get_db
from backend.app.services.embedder import generate_embeddings, generate_query_embedding
from backend.app.services.vector_store import compute_cosine_similarity


@pytest_asyncio.fixture
async def override_db():
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_test_db():
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.clear()
    await test_engine.dispose()


def test_embedding_generation():
    texts = [
        "Senior Cloud Architect with AWS and Kubernetes experience.",
        "Front-end engineer building React and TypeScript applications."
    ]
    embeddings = generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    # Test query embedding
    q_emb = generate_query_embedding("Looking for AWS cloud expertise")
    assert len(q_emb) == 384

    # Cosine similarity with cloud text should be higher than frontend text
    sim_cloud = compute_cosine_similarity(q_emb, embeddings[0])
    sim_frontend = compute_cosine_similarity(q_emb, embeddings[1])

    assert sim_cloud > sim_frontend
    assert sim_cloud > 0.25


def generate_rag_test_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 72),
        "Elena Rostova\nEmail: elena@example.com\n\n"
        "SUMMARY\nExpert in distributed systems, pgvector, and serverless architectures.\n\n"
        "EXPERIENCE\nPrincipal Architect at CloudScale (2020 - Present)\n"
        "- Scaled vector search to millions of documents.\n"
        "- Designed real-time RAG pipeline achieving p95 under 5 seconds.\n\n"
        "EDUCATION\nPh.D. in Computer Science - Cambridge University\n\n"
        "SKILLS\nPostgreSQL, pgvector, Python, FastAPI, PyTorch, Docker"
    )
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.mark.asyncio
async def test_full_pipeline_upload_and_rag_ready(override_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        pdf_bytes = generate_rag_test_pdf()
        files = {"file": ("elena_rostova.pdf", pdf_bytes, "application/pdf")}

        # 1. Upload & trigger full 8-stage pipeline
        resp = await client.post("/api/v1/cvs/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "rag_ready"
        assert "within_sla" in data
        assert len(data["stages"]) >= 8
        assert "rag_verification" in data["stages"]
        doc_id = data["document_id"]

        # 2. Test SSE RAG Chat
        chat_payload = {
            "query": "What distributed systems and vector database experience does Elena have?",
            "document_id": doc_id,
            "top_k": 3
        }
        chat_resp = await client.post("/api/v1/chat", json=chat_payload)
        assert chat_resp.status_code == 200
        assert "text/event-stream" in chat_resp.headers["content-type"]

        body_text = chat_resp.text
        assert "event: citations" in body_text
        assert "event: token" in body_text
        assert "event: done" in body_text
