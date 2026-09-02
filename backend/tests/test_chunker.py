"""Tests for regex-based section-aware chunking."""
from backend.app.services.chunker import chunk_cv_text, estimate_token_count


def test_chunk_cv_text_sections():
    raw_cv = """
    Alex Johnson
    alex@example.com | San Francisco, CA

    SUMMARY
    Senior Full Stack Engineer with 7 years of experience in microservices and LLM systems.

    EXPERIENCE
    Staff Software Engineer at AI Innovations (2022 - Present)
    - Built real-time RAG applications with sub-second streaming.
    - Led a team of 6 engineers across frontend and backend.

    Senior Backend Developer at DataCore (2018 - 2022)
    - Scaled relational databases to handle 50k QPS.

    EDUCATION
    Bachelor of Science in Computer Science, UC Berkeley (2014 - 2018)

    SKILLS
    Python, TypeScript, React, PostgreSQL, FastAPI, Docker, Kubernetes

    PROJECTS
    Serverless RAG Engine - Built an open-source tool for document parsing and pgvector search.
    """
    chunks = chunk_cv_text(raw_cv)

    assert len(chunks) >= 4
    section_names = [c.section_name for c in chunks]

    assert "SUMMARY" in section_names
    assert "EXPERIENCE" in section_names
    assert "EDUCATION" in section_names
    assert "SKILLS" in section_names

    # Verify context headers are preserved in chunk content
    for chunk in chunks:
        assert chunk.content.startswith(f"[{chunk.section_name}]")
        assert chunk.token_count > 0
        assert chunk.chunk_index >= 0


def test_empty_cv_chunking():
    chunks = chunk_cv_text("")
    assert chunks == []
