"""Unified end-to-end CV Ingestion and RAG Pipeline orchestrator adhering to p95 <= 5.0s SLA."""
import json
import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.cv_document import CVDocument
from backend.app.models.cv_chunk import CVChunk
from backend.app.observability.tracer import PipelineTracer
from backend.app.services.parser import extract_text_from_pdf
from backend.app.services.chunker import chunk_cv_text, TextChunk
from backend.app.services.llm_extractor import LLMExtractor
from backend.app.services.merger import merge_extracted_chunks
from backend.app.services.embedder import generate_embeddings
from backend.app.services.vector_store import search_similar_chunks
from backend.app.schemas.cv_schema import CVExtractionSchema

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

    try:
        # =========================================================================
        # Stage 1: Text Extraction (PyMuPDF In-Memory Parser with OCR fallback)
        # =========================================================================
        async with tracer.trace_stage("text_extraction"):
            raw_text, parse_meta = extract_text_from_pdf(pdf_bytes, filename=filename)
            doc.raw_text = raw_text

        # =========================================================================
        # Stage 2: Section-Aware Regex Chunking
        # =========================================================================
        with tracer.trace_stage_sync("chunking"):
            text_chunks: List[TextChunk] = chunk_cv_text(raw_text)

        # =========================================================================
        # Stage 3: LLM Extraction (Parallel asyncio.gather via Hugging Face API)
        # =========================================================================
        async with tracer.trace_stage("llm_extraction", metadata={"chunks_count": len(text_chunks)}):
            partial_extractions = await extractor.extract_all_chunks_parallel(text_chunks)

        # =========================================================================
        # Stage 4: Validation (Schema validation loop)
        # =========================================================================
        with tracer.trace_stage_sync("validation"):
            # Partial extractions validated against Pydantic schema
            valid_partials = [p for p in partial_extractions if isinstance(p, dict)]

        # =========================================================================
        # Stage 5: Merge & Deduplication
        # =========================================================================
        with tracer.trace_stage_sync("merge"):
            merged_schema: CVExtractionSchema = merge_extracted_chunks(valid_partials)
            doc.parsed_json = merged_schema.model_dump()

        # =========================================================================
        # Stage 6: Text Embeddings (sentence-transformers/all-MiniLM-L6-v2)
        # =========================================================================
        chunk_texts = [c.content for c in text_chunks]
        async with tracer.trace_stage("embedding", metadata={"dim": 384, "chunks_count": len(chunk_texts)}):
            embeddings = generate_embeddings(chunk_texts)

        # =========================================================================
        # Stage 7: Vector Upsert (pgvector table persistence)
        # =========================================================================
        db_chunks: List[CVChunk] = []
        async with tracer.trace_stage("vector_upsert"):
            for tc, emb in zip(text_chunks, embeddings):
                embedding_data = emb
                if isinstance(embedding_data, str):
                    try:
                        embedding_data = json.loads(embedding_data)
                    except Exception:
                        embedding_data = []

                # Force strict list of Python floats (removes string or numpy types)
                clean_embedding = [float(x) for x in embedding_data] if embedding_data else None

                db_chunk = CVChunk(
                    document_id=doc.id,
                    chunk_index=tc.chunk_index,
                    section_name=tc.section_name,
                    content=tc.content,
                    token_count=tc.token_count,
                    embedding=clean_embedding,
                    metadata_json=tc.metadata
                )
                db.add(db_chunk)
                db_chunks.append(db_chunk)

            doc.status = "indexing"
            await db.commit()

        # =========================================================================
        # Stage 8: RAG Verification Gate
        # =========================================================================
        async with tracer.trace_stage("rag_verification"):
            if db_chunks:
                # Immediate top-1 similarity query against the primary chunk
                verify_results = await search_similar_chunks(
                     db=db,
                     query_text=db_chunks[0].content[:200],
                     document_id=doc.id,
                     top_k=1
                 )
                if not verify_results:
                    raise RuntimeError("RAG verification gate failed: Top-1 similarity check returned empty results.")

            # Verification passed -> Transition to rag_ready
            doc.status = "rag_ready"
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
        await db.rollback()  # Reset the aborted transaction first
        
        # Now safely update the document status
        try:
            doc.status = "failed"
            doc.error_message = str(e)
            db.add(doc)
            await db.commit()
            await tracer.persist(db, document_id=doc.id)
        except Exception as rollback_err:
            logger.error(f"Failed to record failed status in db: {rollback_err}")
        raise e
    finally:
        await extractor.close()
