"""Tests for Pydantic v2 CVExtractionSchema and dynamic attribute behavior."""
import pytest
from pydantic import ValidationError
from backend.app.schemas.cv_schema import (
    CVExtractionSchema,
    CandidateInfo,
    WorkExperienceItem,
    EducationItem,
    SkillCategory,
)


def test_schema_valid_and_dynamic_extras():
    # Nested models allow dynamic arbitrary fields (extra='allow')
    exp_item = WorkExperienceItem(
        company="Antigravity AI",
        position="Senior Staff Architect",
        start_date="2022-01",
        is_current=True,
        security_clearance="Top Secret",  # Dynamic extra attribute
        patent_ids=["US123456", "US789012"],  # Dynamic extra attribute
        budget_managed_usd=5000000  # Dynamic extra attribute
    )

    assert exp_item.company == "Antigravity AI"
    assert getattr(exp_item, "security_clearance") == "Top Secret"
    assert getattr(exp_item, "budget_managed_usd") == 5000000

    cand_info = CandidateInfo(
        name="Dr. Evelyn Wright",
        email="evelyn@example.com",
        github_handle="evelyn-wright",  # Dynamic extra attribute
        preferred_pronouns="She/Her"  # Dynamic extra attribute
    )
    assert cand_info.name == "Dr. Evelyn Wright"
    assert getattr(cand_info, "github_handle") == "evelyn-wright"

    root_schema = CVExtractionSchema(
        candidate_info=cand_info,
        summary="Leading AI and distributed systems researcher.",
        work_experience=[exp_item],
        education=[EducationItem(institution="MIT", degree="Ph.D.", field_of_study="Artificial Intelligence")],
        skills=[SkillCategory(category_name="ML", skills=["PyTorch", "Transformers"])]
    )

    data = root_schema.model_dump()
    assert data["candidate_info"]["name"] == "Dr. Evelyn Wright"
    assert data["work_experience"][0]["company"] == "Antigravity AI"
    assert data["work_experience"][0]["security_clearance"] == "Top Secret"


def test_root_schema_forbids_unknown_top_level_fields():
    # Top-level model has ConfigDict(extra='forbid')
    with pytest.raises(ValidationError):
        CVExtractionSchema(
            summary="A short summary",
            unknown_top_level_hallucination="Should trigger ValidationError"
        )
