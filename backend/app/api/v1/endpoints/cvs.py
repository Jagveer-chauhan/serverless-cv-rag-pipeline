"""CV Ingestion, Parsing, and Management API Endpoints."""
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from backend.app.db.session import get_db
from backend.app.models.cv_document import CVDocument
from backend.app.models.cv_chunk import CVChunk
from backend.app.observability.tracer import PipelineTracer
from backend.app.services.parser import extract_text_from_pdf
from backend.app.services.chunker import chunk_cv_text

logger = logging.getLogger("cv_rag_pipeline.api.cvs")

router = APIRouter()


class CVUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size: int
    status: str
    chunks_count: int
    total_duration_ms: float
    within_sla: bool
    sla_target_ms: float
    stages: Dict[str, Any]


class CVListItem(BaseModel):
    id: str
    filename: str
    file_size: int
    content_type: str
    status: str
    total_duration_ms: Optional[float]
    created_at: Optional[str]
    updated_at: Optional[str]


class CVDetailResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    content_type: str
    status: str
    error_message: Optional[str]
    total_duration_ms: Optional[float]
    raw_text: Optional[str]
    parsed_json: Optional[Any]
    chunks: List[Dict[str, Any]]
    traces: List[Dict[str, Any]]
    created_at: Optional[str]
    updated_at: Optional[str]


from backend.app.services.pipeline import execute_cv_pipeline

@router.post(
    "/upload",
    response_model=CVUploadResponse,
    summary="Upload & Process CV PDF",
    description="Upload a CV PDF to execute the full 8-stage pipeline: extraction, chunking, LLM extraction, validation, merge, embeddings, vector upsert, and rag_ready verification."
)
async def upload_cv(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for CV ingestion."
        )

    pdf_bytes = await file.read()
    file_size = len(pdf_bytes)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # Initialize CVDocument record
    doc = CVDocument(
        filename=file.filename,
        file_size=file_size,
        content_type=file.content_type or "application/pdf",
        status="extracting"
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    tracer = PipelineTracer(document_id=doc.id)

    try:
        # Execute all 8 stages sequentially under 5.0s SLA
        updated_doc, summary = await execute_cv_pipeline(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            doc=doc,
            db=db,
            tracer=tracer
        )

        chunk_count = len(updated_doc.chunks) if updated_doc.chunks else len(summary["stages"])

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
        logger.error(f"Failed to process CV '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV processing pipeline failed: {str(e)}"
        )

from pydantic import BaseModel, Field
from backend.app.schemas.cv_schema import CVExtractionSchema
from backend.app.services.llm_extractor import LLMExtractor
from backend.app.services.merger import merge_extracted_chunks
from backend.app.services.chunker import TextChunk

class ExtractRequest(BaseModel):
    document_id: str
    chunks: Optional[List[Dict[str, Any]]] = None

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
        # If chunks provided in body, use them, otherwise use stored chunks
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
        chunks=[c.to_dict() for c in doc.chunks],
        traces=[t.to_dict() for t in doc.traces],
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
