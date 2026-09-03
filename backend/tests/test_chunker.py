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


def test_chunker_avoids_false_positive_section_splits_inside_projects():
    raw_cv = """
    Jane Doe
    jane@example.com

    SUMMARY
    Senior Engineer.

    SELECTED PROJECTS

    Institute Management ERP | Lead Architect
    Overview: Designed ERP for universities.
    Technologies: PHP, Laravel, PostgreSQL, Docker
    Key Highlights: Delivered on time with zero downtime.
    Activities: Coordinated sprint planning.
    Training: Conducted onboarding sessions.

    Tracked.ai - Learning Management System | Full-Stack Developer
    Overview: Australian LMS platform.
    Experience with real-time sockets and WebSockets.
    """
    chunks = chunk_cv_text(raw_cv)
    section_names = [c.section_name for c in chunks]

    # Should NOT have SKILLS, CERTIFICATIONS, ADDITIONAL, or duplicate EXPERIENCE created from inline bullets!
    assert "SKILLS" not in section_names
    assert "CERTIFICATIONS" not in section_names
    assert "ADDITIONAL" not in section_names
    assert "PROJECTS" in section_names


def test_chunker_handles_multi_page_project_continuation_headers():
    raw_cv = """
    John Developer
    john@example.com

    PROJECTS (PAGE 1)
    Project A | Developer
    Built service A.

    PROJECTS (CONTINUED)
    Project B | Developer
    Built service B.
    """
    chunks = chunk_cv_text(raw_cv)
    for c in chunks:
        if c.section_name != "CONTACT_HEADER":
            assert c.section_name == "PROJECTS"

