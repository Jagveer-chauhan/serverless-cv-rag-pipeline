"""Tests for PipelineTracer millisecond-level stage observability and persistence."""
import asyncio
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from backend.app.models.base import Base
from backend.app.models.cv_document import CVDocument
from backend.app.models.cv_processing_trace import CVProcessingTrace
from backend.app.observability.tracer import PipelineTracer, REQUIRED_STAGES


@pytest_asyncio.fixture
async def async_test_session():
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
    await test_engine.dispose()


@pytest.mark.asyncio
async def test_tracer_all_eight_stages():
    tracer = PipelineTracer(document_id="doc-test-123")

    # 1. text_extraction
    async with tracer.trace_stage("text_extraction", metadata={"parser": "fitz"}):
        await asyncio.sleep(0.01)

    # 2. chunking
    with tracer.trace_stage_sync("chunking", metadata={"chunks_count": 4}):
        pass

    # 3. llm_extraction
    async with tracer.trace_stage("llm_extraction", metadata={"model": "gemma-3-4b-it"}):
        await asyncio.sleep(0.02)

    # 4. validation
    with tracer.trace_stage_sync("validation", metadata={"schema": "CVExtractionSchema"}):
        pass

    # 5. merge
    with tracer.trace_stage_sync("merge", metadata={"deduped_sections": 3}):
        pass

    # 6. embedding
    async with tracer.trace_stage("embedding", metadata={"dim": 384}):
        await asyncio.sleep(0.01)

    # 7. vector_upsert
    async with tracer.trace_stage("vector_upsert", metadata={"table": "cv_chunks"}):
        await asyncio.sleep(0.01)

    # 8. rag_verification
    async with tracer.trace_stage("rag_verification", metadata={"top_1_sim": 0.94}):
        await asyncio.sleep(0.01)

    summary = tracer.get_summary()

    assert summary["document_id"] == "doc-test-123"
    assert summary["stages_count"] == 8
    assert summary["within_sla"] is True
    assert summary["sla_target_ms"] == 5000.0
    assert len(summary["missing_stages"]) == 0

    for stage_name in REQUIRED_STAGES:
        assert stage_name in summary["stages"]
        stage_info = summary["stages"][stage_name]
        assert stage_info["status"] == "success"
        assert stage_info["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_tracer_stage_failure():
    tracer = PipelineTracer(document_id="doc-fail-456")

    with pytest.raises(ValueError, match="Mock extraction error"):
        async with tracer.trace_stage("llm_extraction"):
            raise ValueError("Mock extraction error")

    summary = tracer.get_summary()
    assert "llm_extraction" in summary["stages"]
    assert summary["stages"]["llm_extraction"]["status"] == "failed"
    assert "Mock extraction error" in summary["stages"]["llm_extraction"]["error_message"]


@pytest.mark.asyncio
async def test_tracer_database_persistence(async_test_session: AsyncSession):
    # Create parent CVDocument
    doc = CVDocument(
        filename="alice_resume.pdf",
        file_size=51200,
        status="extracting"
    )
    async_test_session.add(doc)
    await async_test_session.commit()
    await async_test_session.refresh(doc)

    tracer = PipelineTracer(document_id=doc.id)
    async with tracer.trace_stage("text_extraction"):
        await asyncio.sleep(0.005)
    async with tracer.trace_stage("rag_verification"):
        await asyncio.sleep(0.005)

    persisted = await tracer.persist(async_test_session)
    assert len(persisted) == 3  # 2 stages + 1 total trace

    await async_test_session.refresh(doc)
    assert doc.total_duration_ms > 0
    assert len(doc.traces) == 3
