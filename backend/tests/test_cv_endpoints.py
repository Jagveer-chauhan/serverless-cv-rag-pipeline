"""Integration tests for CV upload, retrieval, and deletion endpoints."""
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


def generate_test_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Michael Scott\nEmail: michael@dundermifflin.com\n\nEXPERIENCE\nRegional Manager at Dunder Mifflin\n\nEDUCATION\nScranton High School\n\nSKILLS\nSales, Management, Improv")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.mark.asyncio
async def test_cv_upload_flow(override_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        pdf_bytes = generate_test_pdf()
        files = {"file": ("michael_scott.pdf", pdf_bytes, "application/pdf")}

        # 1. Upload CV
        upload_resp = await client.post("/api/v1/cvs/upload", files=files)
        assert upload_resp.status_code == 200
        data = upload_resp.json()

        assert "document_id" in data
        assert data["filename"] == "michael_scott.pdf"
        assert data["chunks_count"] > 0
        assert data["status"] == "rag_ready"
        assert "text_extraction" in data["stages"]
        assert "chunking" in data["stages"]
        doc_id = data["document_id"]

        # 2. List CVs
        list_resp = await client.get("/api/v1/cvs")
        assert list_resp.status_code == 200
        cv_list = list_resp.json()
        assert len(cv_list) >= 1
        assert any(item["id"] == doc_id for item in cv_list)

        # 3. Get CV Detail
        detail_resp = await client.get(f"/api/v1/cvs/{doc_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["id"] == doc_id
        assert len(detail["chunks"]) > 0
        assert len(detail["traces"]) >= 2

        # 4. Delete CV
        del_resp = await client.delete(f"/api/v1/cvs/{doc_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # 5. Verify 404 on deleted CV
        missing_resp = await client.get(f"/api/v1/cvs/{doc_id}")
        assert missing_resp.status_code == 404
