"""Tests for LLM extraction and chunk merge/deduplication logic."""
import pytest
from backend.app.services.chunker import TextChunk
from backend.app.services.llm_extractor import LLMExtractor
from backend.app.services.merger import merge_extracted_chunks


@pytest.mark.asyncio
async def test_llm_extractor_parallel_extraction():
    extractor = LLMExtractor(api_key="")  # Uses fast fallback heuristic
    
    chunks = [
        TextChunk(
            chunk_index=0,
            section_name="CONTACT_HEADER",
            content="Sarah Connor\nEmail: sarah@skynet-resistance.org\nPhone: +1 800-555-0199\nhttps://github.com/sarah-c",
            token_count=15
        ),
        TextChunk(
            chunk_index=1,
            section_name="SUMMARY",
            content="[SUMMARY]\nStrategic Defense Leader with 10+ years organizing autonomous robotics counter-measures.",
            token_count=14
        ),
        TextChunk(
            chunk_index=2,
            section_name="EXPERIENCE",
            content="[EXPERIENCE]\nCommander at Resistance HQ (2020 - Present)\n- Led defensive operations across 5 divisions.\n- Built real-time comms network.",
            token_count=20
        ),
        TextChunk(
            chunk_index=3,
            section_name="SKILLS",
            content="[SKILLS]\nTactics, Python, Distributed Systems, Cryptography",
            token_count=8
        )
    ]

    partial_results = await extractor.extract_all_chunks_parallel(chunks, max_concurrency=4)
    assert len(partial_results) == 4

    # Test merge logic
    merged_schema = merge_extracted_chunks(partial_results)

    assert merged_schema.candidate_info is not None
    assert merged_schema.candidate_info.name == "Sarah Connor"
    assert merged_schema.candidate_info.email == "sarah@skynet-resistance.org"
    assert "Defense Leader" in (merged_schema.summary or "")
    assert len(merged_schema.work_experience) >= 1
    assert len(merged_schema.skills) >= 1

    await extractor.close()


def test_merge_deduplication():
    partial_a = {
        "candidate_info": {"name": "John Smith", "email": "john@example.com"},
        "work_experience": [
            {"company": "Google", "position": "Senior Engineer", "key_achievements": ["Launched Feature A"]}
        ],
        "skills": [{"category_name": "Backend", "skills": ["Python", "FastAPI"]}]
    }

    partial_b = {
        "candidate_info": {"phone": "+1 555-0100"},
        "work_experience": [
            {"company": "Google", "position": "Senior Engineer", "key_achievements": ["Optimized Latency B"]}
        ],
        "skills": [{"category_name": "Backend", "skills": ["FastAPI", "SQLAlchemy", "PostgreSQL"]}]
    }

    merged = merge_extracted_chunks([partial_a, partial_b])

    assert merged.candidate_info.name == "John Smith"
    assert merged.candidate_info.phone == "+1 555-0100"
    
    # Verify work experience deduplicated into 1 entry with merged achievements
    assert len(merged.work_experience) == 1
    assert len(merged.work_experience[0].key_achievements) == 2

    # Verify skills deduplicated
    backend_skills = next(s for s in merged.skills if s.category_name == "Backend")
    assert len(backend_skills.skills) == 4  # Python, FastAPI, SQLAlchemy, PostgreSQL
