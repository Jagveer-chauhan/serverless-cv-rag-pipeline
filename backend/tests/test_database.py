"""Tests for Database models and async session operations."""
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from backend.app.models.base import Base
from backend.app.models.cv_document import CVDocument
from backend.app.models.cv_chunk import CVChunk
from backend.app.models.cv_processing_trace import CVProcessingTrace


@pytest_asyncio.fixture
async def async_test_session():
    """Provides an isolated in-memory SQLite async session for testing."""
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session

    await test_engine.dispose()


@pytest.mark.asyncio
async def test_cv_document_creation(async_test_session: AsyncSession):
    doc = CVDocument(
        filename="john_doe_resume.pdf",
        file_size=102400,
        content_type="application/pdf",
        status="queued",
        raw_text="Sample raw text from CV."
    )
    async_test_session.add(doc)
    await async_test_session.commit()
    await async_test_session.refresh(doc)

    assert doc.id is not None
    assert doc.filename == "john_doe_resume.pdf"
    assert doc.status == "queued"
    assert doc.created_at is not None

    doc_dict = doc.to_dict()
    assert doc_dict["filename"] == "john_doe_resume.pdf"
    assert doc_dict["file_size"] == 102400


@pytest.mark.asyncio
async def test_cv_chunk_and_trace_relationships(async_test_session: AsyncSession):
    doc = CVDocument(
        filename="jane_smith_cv.pdf",
        file_size=204800,
        status="extracting",
    )
    async_test_session.add(doc)
    await async_test_session.commit()
    await async_test_session.refresh(doc)

    # Add chunk
    chunk = CVChunk(
        document_id=doc.id,
        chunk_index=0,
        section_name="EXPERIENCE",
        content="Senior Backend Architect at Tech Corp (2020-Present)",
        token_count=12,
        metadata_json={"company": "Tech Corp"}
    )
    async_test_session.add(chunk)

    # Add trace
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    trace = CVProcessingTrace(
        document_id=doc.id,
        stage="text_extraction",
        duration_ms=45.2,
        start_time=now,
        end_time=now,
        status="success",
        metadata_json={"method": "pymupdf"}
    )
    async_test_session.add(trace)

    await async_test_session.commit()
    await async_test_session.refresh(doc)

    assert len(doc.chunks) == 1
    assert doc.chunks[0].section_name == "EXPERIENCE"
    assert len(doc.traces) == 1
    assert doc.traces[0].stage == "text_extraction"
    assert doc.traces[0].duration_ms == 45.2
