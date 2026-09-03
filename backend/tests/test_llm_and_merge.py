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

    assert merged_schema.candidate is not None
    assert merged_schema.candidate.name == "Sarah Connor"
    assert merged_schema.candidate.email == "sarah@skynet-resistance.org"
    assert "Defense Leader" in (merged_schema.summary or "")
    assert len(merged_schema.experience) >= 1
    assert merged_schema.skills is not None

    await extractor.close()


def test_merge_deduplication():
    partial_a = {
        "candidate": {"name": "John Smith", "email": "john@example.com"},
        "experience": [
            {"company": "Google", "title": "Senior Engineer", "achievements": ["Launched Feature A"]}
        ],
        "skills": {"explicit": ["Python", "FastAPI"]}
    }

    partial_b = {
        "candidate": {"phone": "+1 555-0100"},
        "experience": [
            {"company": "Google", "title": "Senior Engineer", "achievements": ["Optimized Latency B"]}
        ],
        "skills": {"explicit": ["FastAPI", "SQLAlchemy", "PostgreSQL"]}
    }

    merged = merge_extracted_chunks([partial_a, partial_b])

    assert merged.candidate.name == "John Smith"
    assert merged.candidate.phone == "+1 555-0100"
    
    # Verify work experience deduplicated into 1 entry with merged achievements
    assert len(merged.experience) == 1
    assert len(merged.experience[0].achievements) == 2

    # Verify skills deduplicated
    assert len(merged.skills.explicit) == 4  # Python, FastAPI, SQLAlchemy, PostgreSQL


def test_chunk_extraction_schema_with_dynamic_sections():
    from backend.app.schemas.cv_schema import ChunkExtractionSchema

    # Validate that ChunkExtractionSchema accepts chunk-level extractions with projects, languages, awards, position
    raw_chunk = {
        "candidate_info": {
            "name": "Alex Chen",
            "email": "alex@example.com",
            "phone": "+1 555 0199",
            "location": "San Francisco, CA"
        },
        "work_experience": [
            {
                "company": "Stripe",
                "position": "Staff Engineer",
                "start_date": "2020",
                "end_date": "2024",
                "key_achievements": ["Built high-throughput payment pipeline"],
                "technologies_used": ["Go", "Kafka", "PostgreSQL"]
            }
        ],
        "education": [
            {
                "institution": "Stanford University",
                "degree": "M.S. in Computer Science",
                "start_year": "2018",
                "end_year": "2020"
            }
        ],
        "projects": [
            {"name": "Distributed Cache", "description": "High performance Raft cluster"}
        ],
        "languages": [
            {"language": "English", "proficiency": "Native"},
            {"language": "Mandarin", "proficiency": "Fluent"}
        ],
        "awards": [
            {"title": "Hackathon Winner", "issuer": "TechCrunch"}
        ]
    }

    validated = ChunkExtractionSchema.model_validate(raw_chunk)
    assert validated.candidate.name == "Alex Chen"
    assert validated.experience[0].title == "Staff Engineer"
    assert validated.education[0].start_date == "2018"
    assert len(validated.projects) == 1
    assert len(validated.languages) == 2

    # Now verify merger properly routes these into the assignment CVExtractionSchema
    merged = merge_extracted_chunks([raw_chunk])
    assert merged.candidate.name == "Alex Chen"
    assert merged.experience[0].company == "Stripe"
    assert merged.experience[0].title == "Staff Engineer"
    assert "English" in merged.derived.languages
    assert "Mandarin" in merged.derived.languages
    assert len(merged.projects) == 1
    assert merged.projects[0].name == "Distributed Cache"
    assert len(merged.awards) == 1
    assert merged.awards[0].name == "Hackathon Winner"
    assert len(merged.languages) == 2


def test_merge_arbitrary_skill_dictionaries():
    raw_chunk = {
        "skills": {
            "languages": ["Python", "Rust", "Go"],
            "frameworks": ["FastAPI", "Next.js"],
            "databases": ["PostgreSQL", "Redis"],
            "soft_skills": ["Leadership", "Mentoring"]
        }
    }

    merged = merge_extracted_chunks([raw_chunk])
    assert "Python" in merged.skills.explicit
    assert "Rust" in merged.skills.explicit
    assert "FastAPI" in merged.skills.explicit
    assert "PostgreSQL" in merged.skills.explicit
    assert "Leadership" in merged.skills.soft_skills
