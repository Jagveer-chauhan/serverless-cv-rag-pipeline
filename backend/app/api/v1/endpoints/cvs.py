"""CV Ingestion, Parsing, Extraction, and Management API Endpoints."""
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from backend.app.db.session import get_db
from backend.app.models.cv_document import CVDocument
from backend.app.models.cv_chunk import CVChunk
from backend.app.models.cv_processing_trace import CVProcessingTrace
from backend.app.observability.tracer import PipelineTracer
from backend.app.services.parser import extract_text_from_pdf
from backend.app.services.chunker import chunk_cv_text, TextChunk
from backend.app.services.llm_extractor import LLMExtractor
from backend.app.services.merger import merge_extracted_chunks
from backend.app.services.pipeline import execute_cv_pipeline
from backend.app.schemas.cv_schema import CVExtractionSchema

logger = logging.getLogger("cv_rag_pipeline.api.cvs")

router = APIRouter()


# =============================================================================
# Request & Response DTO Models
# =============================================================================

class CVUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique document UUID")
    filename: str = Field(..., description="Uploaded filename")
    file_size: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Current processing status")
    chunks_count: int = Field(..., description="Number of text chunks created")
    total_duration_ms: float = Field(..., description="Total pipeline processing duration in milliseconds")
    within_sla: bool = Field(..., description="Whether total duration satisfies <= 5000ms SLA")
    sla_target_ms: float = Field(..., description="Target SLA threshold (5000.0ms)")
    stages: Dict[str, Any] = Field(default_factory=dict, description="Per-stage timing breakdown")


class ExtractRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to extract")
    chunks: Optional[List[Dict[str, Any]]] = Field(None, description="Optional custom chunk list")


class CVListItem(BaseModel):
    id: str
    filename: str
    file_size: int
    content_type: str
    status: str
    total_duration_ms: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CVDetailResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    content_type: str
    status: str
    error_message: Optional[str] = None
    total_duration_ms: Optional[float] = None
    raw_text: Optional[str] = None
    parsed_json: Optional[Any] = None
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    traces: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# =============================================================================
# Route Handlers
# =============================================================================

@router.post(
    "/upload",
    response_model=CVUploadResponse,
    summary="Upload & Process CV",
    description="Upload a CV PDF to execute the full 8-stage pipeline: extraction, chunking, LLM extraction, validation, merge, embeddings, vector upsert, and rag_ready verification."
)
async def upload_cv(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and DOCX files are supported for CV ingestion."
        )

    file_bytes = await file.read()
    file_size = len(file_bytes)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # Initialize CVDocument record in database
    doc = CVDocument(
        filename=file.filename,
        file_size=file_size,
        content_type=file.content_type or "application/pdf",
        status="extracting"
    )
    try:
        db.add(doc)
        await db.commit()
        await db.refresh(doc, attribute_names=["id", "filename", "file_size", "content_type", "status", "created_at", "updated_at"])
    except Exception as init_err:
        await db.rollback()
        logger.warning(f"Initial doc insert failed (possible missing tables), running init_db: {init_err}")
        from backend.app.db.session import init_db
        await init_db()
        db.add(doc)
        await db.commit()
        await db.refresh(doc, attribute_names=["id", "filename", "file_size", "content_type", "status", "created_at", "updated_at"])

    tracer = PipelineTracer(document_id=doc.id)

    try:
        # Execute all 8 stages sequentially under 5.0s SLA
        updated_doc, summary = await execute_cv_pipeline(
            pdf_bytes=file_bytes,
            filename=file.filename,
            doc=doc,
            db=db,
            tracer=tracer
        )

        chunk_count = summary.get("stages", {}).get("chunking", {}).get("chunks_count", 1)

        return CVUploadResponse(
            document_id=updated_doc.id,
            filename=updated_doc.filename,
            file_size=updated_doc.file_size,
            status=updated_doc.status,
            chunks_count=chunk_count,
            total_duration_ms=summary["total_duration_ms"],
            within_sla=summary["within_sla"],
            sla_target_ms=summary["sla_target_ms"],
            stages=summary["stages"]
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to process CV '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV processing pipeline failed: {str(e)}"
        )


@router.post(
    "/extract",
    summary="Extract & Merge CV JSON",
    description="Accepts pre-processed text chunks or fetches stored chunks for a CV, calls serverless Gemma with validation retry loop, and returns merged cohesive CV JSON."
)
async def extract_cv_json(
    request: ExtractRequest,
    db: AsyncSession = Depends(get_db)
):
    doc = await db.get(CVDocument, request.document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CV document '{request.document_id}' not found."
        )

    extractor = LLMExtractor()
    try:
        if request.chunks:
            text_chunks = [
                TextChunk(
                    chunk_index=i,
                    section_name=c.get("section_name", "GENERAL"),
                    content=c.get("content", ""),
                    token_count=c.get("token_count", 10)
                )
                for i, c in enumerate(request.chunks)
            ]
        else:
            text_chunks = [
                TextChunk(
                    chunk_index=c.chunk_index,
                    section_name=c.section_name,
                    content=c.content,
                    token_count=c.token_count
                )
                for c in doc.chunks
            ]

        partial_results = await extractor.extract_all_chunks_parallel(text_chunks)
        merged_schema = merge_extracted_chunks(partial_results, raw_text=doc.raw_text)
        
        doc.parsed_json = merged_schema.model_dump()
        await db.commit()

        return {
            "document_id": doc.id,
            "status": "extracted",
            "extracted_json": doc.parsed_json
        }
    finally:
        await extractor.close()


@router.get(
    "",
    response_model=List[CVListItem],
    summary="List all CV documents",
    description="Retrieve all uploaded CV documents with processing statuses and SLA timing metrics."
)
async def list_cvs(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(CVDocument).order_by(CVDocument.created_at.desc()))
        docs = result.scalars().all()
        return [
            CVListItem(
                id=d.id,
                filename=d.filename,
                file_size=d.file_size,
                content_type=d.content_type,
                status=d.status,
                total_duration_ms=d.total_duration_ms,
                created_at=d.created_at.isoformat() if d.created_at else None,
                updated_at=d.updated_at.isoformat() if d.updated_at else None,
            )
            for d in docs
        ]
    except Exception as e:
        logger.warning(f"Failed to query CV list from database: {e}")
        return []


@router.get(
    "/{cv_id}",
    response_model=CVDetailResponse,
    summary="Get CV document details",
    description="Get detailed information for a single CV, including raw text, parsed JSON, chunks, and timing traces."
)
async def get_cv_detail(cv_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(CVDocument, cv_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CV document '{cv_id}' not found."
        )

    res_chunks = await db.execute(select(CVChunk).where(CVChunk.document_id == cv_id).order_by(CVChunk.chunk_index))
    chunks_list = res_chunks.scalars().all()

    res_traces = await db.execute(select(CVProcessingTrace).where(CVProcessingTrace.document_id == cv_id).order_by(CVProcessingTrace.start_time))
    traces_list = res_traces.scalars().all()

    return CVDetailResponse(
        id=doc.id,
        filename=doc.filename,
        file_size=doc.file_size,
        content_type=doc.content_type,
        status=doc.status,
        error_message=doc.error_message,
        total_duration_ms=doc.total_duration_ms,
        raw_text=doc.raw_text,
        parsed_json=doc.parsed_json,
        chunks=[c.to_dict() for c in chunks_list],
        traces=[t.to_dict() for t in traces_list],
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
    )


@router.delete(
    "/{cv_id}",
    summary="Delete CV document",
    description="Deletes a CV document along with all its associated chunks and traces."
)
async def delete_cv(cv_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(CVDocument, cv_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CV document '{cv_id}' not found."
        )

    await db.delete(doc)
    await db.commit()
    return {"status": "deleted", "id": cv_id, "message": f"CV document {cv_id} deleted successfully."}
