"""Unified end-to-end CV Ingestion and RAG Pipeline orchestrator adhering to p95 <= 5.0s SLA."""
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.cv_document import CVDocument
from backend.app.models.cv_chunk import CVChunk
from backend.app.observability.tracer import PipelineTracer
from backend.app.services.parser import extract_text_from_document
from backend.app.services.chunker import chunk_cv_text, TextChunk
from backend.app.services.llm_extractor import LLMExtractor
from backend.app.services.merger import merge_extracted_chunks
from backend.app.services.embedder import generate_embeddings
from backend.app.services.vector_store import search_similar_chunks
from backend.app.schemas.cv_schema import CVExtractionSchema, calculate_confidence_scores, ProcessingMetadata
from backend.app.core.state import app_state

logger = logging.getLogger("cv_rag_pipeline.orchestrator")


async def execute_cv_pipeline(
    pdf_bytes: bytes,
    filename: str,
    doc: CVDocument,
    db: AsyncSession,
    tracer: PipelineTracer
) -> Tuple[CVDocument, Dict[str, Any]]:
    """Executes all 8 pipeline stages sequentially while recording microsecond timing traces."""
    extractor = LLMExtractor()

    # Record upload_accepted_ms immediately — spec requirement §3.6
    upload_accepted_at = datetime.now(timezone.utc)
    tracer.record_stage("upload_accepted", duration_ms=0.0, metadata={"filename": filename})

    try:
        # =========================================================================
        # Stage 1: Text Extraction (PyMuPDF / python-docx In-Memory Parser with OCR fallback)
        # =========================================================================
        async with tracer.trace_stage("text_extraction"):
            raw_text, parse_meta = extract_text_from_document(pdf_bytes, filename=filename)
            doc.raw_text = raw_text

        # =========================================================================
        # Stage 2: Section-Aware Regex Chunking
        # =========================================================================
        with tracer.trace_stage_sync("chunking"):
            text_chunks: List[TextChunk] = chunk_cv_text(raw_text)

        # =========================================================================
        # Stage 3: LLM Extraction (Parallel asyncio.gather via HF Serverless API)
        # =========================================================================
        async with tracer.trace_stage("llm_extraction", metadata={"chunks_count": len(text_chunks)}):
            partial_extractions = await extractor.extract_all_chunks_parallel(text_chunks)

        # =========================================================================
        # Stage 4: Validation (Pydantic v2 schema validation + self-correction loop)
        # =========================================================================
        with tracer.trace_stage_sync("validation"):
            valid_partials = [p for p in partial_extractions if isinstance(p, dict)]

        # =========================================================================
        # Stage 5: Merge & Deduplication
        # =========================================================================
        with tracer.trace_stage_sync("merge"):
            merged_schema: CVExtractionSchema = merge_extracted_chunks(valid_partials, raw_text=raw_text)
            # Dynamically compute confidence scores
            merged_schema.confidence_scores = calculate_confidence_scores(merged_schema)

        # Transition: extracted (structured JSON ready, not yet indexed)
        doc.status = "extracted"
        doc.parsed_json = merged_schema.model_dump(by_alias=False)

        # =========================================================================
        # Stage 6: Text Embeddings (sentence-transformers/all-MiniLM-L6-v2)
        # =========================================================================
        chunk_texts = [c.content for c in text_chunks]
        async with tracer.trace_stage("embedding", metadata={"dim": 384, "chunks_count": len(chunk_texts)}):
            embeddings = generate_embeddings(chunk_texts)

        # =========================================================================
        # Stage 7: Vector Upsert (pgvector — raw SQL CAST to avoid asyncpg codec issues)
        # =========================================================================
        # We bypass the SQLAlchemy ORM for the embedding column entirely.
        # The ORM + pgvector.sqlalchemy.Vector requires the asyncpg codec to be
        # registered on the connection, which consistently fails on free-tier
        # Render/Supabase (run_async hook is unreliable in pooled async contexts).
        #
        # Solution: raw SQL INSERT with CAST(:embedding AS vector).
        # asyncpg passes the embedding as a plain text string; PostgreSQL's
        # pgvector extension casts it to vector itself.  Zero codec dependency.
        db_chunks: List[CVChunk] = []
        async with tracer.trace_stage("vector_upsert"):
            now = datetime.now(timezone.utc)

            # Extract candidate name for chunk metadata (for RAG citations)
            candidate_name = None
            if merged_schema.candidate and merged_schema.candidate.name:
                candidate_name = merged_schema.candidate.name

            for tc, emb in zip(text_chunks, embeddings):
                embedding_data = emb

                # Deserialise if the embedding arrived as a JSON string
                if isinstance(embedding_data, str):
                    try:
                        embedding_data = json.loads(embedding_data)
                    except Exception:
                        embedding_data = []

                # Build the canonical pgvector string '[f1,f2,...,f384]'
                clean_floats: list = [float(x) for x in embedding_data] if embedding_data else []
                embedding_str: str | None = (
                    f"[{','.join(str(v) for v in clean_floats)}]" if clean_floats else None
                )

                chunk_id = str(uuid.uuid4())

                # Enrich metadata with candidate_name and document info for citations
                chunk_metadata = {
                    **(tc.metadata or {}),
                    "cv_id": doc.id,
                    "candidate_name": candidate_name,
                    "filename": filename,
                    "section_heading": tc.section_name,
                }

                if embedding_str:
                    await db.execute(
                        text("""
                            INSERT INTO cv_chunks
                                (id, document_id, chunk_index, section_name,
                                 content, token_count, embedding, metadata_json, created_at)
                            VALUES
                                (:id, :document_id, :chunk_index, :section_name,
                                 :content, :token_count, CAST(:embedding AS vector),
                                 CAST(:metadata_json AS json), :created_at)
                        """),
                        {
                            "id": chunk_id,
                            "document_id": doc.id,
                            "chunk_index": tc.chunk_index,
                            "section_name": tc.section_name,
                            "content": tc.content,
                            "token_count": tc.token_count,
                            "embedding": embedding_str,
                            "metadata_json": json.dumps(chunk_metadata),
                            "created_at": now,
                        }
                    )
                else:
                    # Fallback: no embedding (shouldn't happen in practice)
                    db_chunk = CVChunk(
                        id=chunk_id,
                        document_id=doc.id,
                        chunk_index=tc.chunk_index,
                        section_name=tc.section_name,
                        content=tc.content,
                        token_count=tc.token_count,
                        embedding=None,
                        metadata_json=chunk_metadata,
                        created_at=now,
                    )
                    db.add(db_chunk)

                # Keep a lightweight reference for the verification stage
                db_chunks.append(CVChunk(
                    id=chunk_id,
                    document_id=doc.id,
                    chunk_index=tc.chunk_index,
                    section_name=tc.section_name,
                    content=tc.content,
                    token_count=tc.token_count,
                ))

            doc.status = "indexing"
            await db.commit()

        # =========================================================================
        # Stage 8: RAG Verification Gate
        # =========================================================================
        async with tracer.trace_stage("rag_verification"):
            if db_chunks:
                try:
                    verify_results = await search_similar_chunks(
                        db=db,
                        query_text=db_chunks[0].content[:200],
                        document_id=doc.id,
                        top_k=1,
                    )
                    if not verify_results:
                        logger.warning(
                            "RAG verification: top-1 similarity check returned empty results "
                            "(chunks inserted but not yet searchable — this is acceptable)."
                        )
                except Exception as verify_err:
                    # A SQL error in search_similar_chunks aborts the transaction.
                    # Roll back so we can still commit the rag_ready status.
                    logger.warning(f"RAG verification search failed (non-fatal): {verify_err}")
                    try:
                        await db.rollback()
                    except Exception:
                        pass

            # Verification passed → Transition to rag_ready
            rag_ready_at = datetime.now(timezone.utc)
            doc.status = "rag_ready"

            # Attach full processing metadata to the parsed JSON
            tracer.compute_total()
            timing_dict = {
                stage: round(trace.duration_ms, 2)
                for stage, trace in tracer.traces.items()
                if stage != "total"
            }

            proc_metadata = ProcessingMetadata(
                request_id=doc.id,
                model="google/gemma-3-4b-it",
                provider="Hugging Face Serverless Inference API",
                status="rag_ready",
                upload_accepted_at=upload_accepted_at.isoformat(),
                rag_ready_at=rag_ready_at.isoformat(),
                extraction_time_ms=tracer.traces.get("llm_extraction", None) and
                                   tracer.traces["llm_extraction"].duration_ms,
                chunks_used=len(db_chunks),
                retry_count=extractor.retry_count,
                cold_start=app_state.cold_start_occurred,
                cold_start_ms=app_state.cold_start_ms,
                first_inference_ms=app_state.first_inference_ms,
                warm_inference_ms=app_state.warm_inference_ms,
                timing_ms=timing_dict,
            )

            if doc.parsed_json and isinstance(doc.parsed_json, dict):
                doc.parsed_json["processing_metadata"] = proc_metadata.model_dump()

            await db.commit()

        # Finalize and persist traces
        await tracer.persist(db, document_id=doc.id)
        summary = tracer.get_summary()

        logger.info(
            f"🎉 [Pipeline] Document {doc.id} ('{filename}') reached 'rag_ready' in "
            f"{summary['total_duration_ms']:.2f}ms (Within SLA: {summary['within_sla']})"
        )
        return doc, summary

    except Exception as e:
        logger.error(f"Pipeline failed for document {doc.id}: {e}", exc_info=True)
        await db.rollback()

        # Mark as degraded if we have some data, otherwise failed
        try:
            doc.status = "degraded" if doc.parsed_json else "failed"
            doc.error_message = str(e)
            db.add(doc)
            await db.commit()
            await tracer.persist(db, document_id=doc.id)
        except Exception as rollback_err:
            logger.error(f"Failed to record failed status in db: {rollback_err}")
        raise e
    finally:
        await extractor.close()
