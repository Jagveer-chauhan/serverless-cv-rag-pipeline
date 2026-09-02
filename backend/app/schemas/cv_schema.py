"""Pydantic v2 schemas for structured CV extraction matching assignment specification."""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field


class DynamicBaseModel(BaseModel):
    """Nested model allowing dynamic extra fields for flexible candidate extraction."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class CandidateLinks(DynamicBaseModel):
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    other: List[str] = Field(default_factory=list)


class CandidateInfo(DynamicBaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: Optional[Union[CandidateLinks, Dict[str, Any], List[str]]] = None
    title: Optional[str] = None


class WorkExperienceItem(DynamicBaseModel):
    company: str
    title: str = Field(..., description="Position or role title")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    inferred_skills: List[str] = Field(default_factory=list)


class EducationItem(DynamicBaseModel):
    institution: str
    degree: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[Union[str, float]] = None
    notes: Optional[str] = None


class SkillsBlock(DynamicBaseModel):
    explicit: List[str] = Field(default_factory=list)
    inferred: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)


class CertificationItem(DynamicBaseModel):
    name: str
    issuer: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None


class CareerGapItem(DynamicBaseModel):
    start_date: Optional[str] = Field(None, alias="from")
    end_date: Optional[str] = Field(None, alias="to")
    reason: Optional[str] = None


class DerivedInsights(DynamicBaseModel):
    years_of_experience: Optional[float] = None
    seniority_level: Optional[str] = None  # Junior, Mid, Senior, Lead, Principal, Executive
    career_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    career_trajectory: Optional[str] = None
    domain: Optional[str] = None
    languages: List[str] = Field(default_factory=list)


class InferredSignals(DynamicBaseModel):
    leadership_signals: List[str] = Field(default_factory=list)
    communication_style: Optional[str] = None
    potential_red_flags: List[str] = Field(default_factory=list)
    career_objective: Optional[str] = None


class CustomSection(DynamicBaseModel):
    heading: str
    content: str


class ConfidenceScores(DynamicBaseModel):
    overall: float = 0.92
    experience_dates: float = 0.88
    inferred_skills: float = 0.85


class ProcessingMetadata(DynamicBaseModel):
    request_id: Optional[str] = None
    model: str = "google/gemma-3-4b-it"
    status: str = "rag_ready"
    upload_accepted_at: Optional[str] = None
    rag_ready_at: Optional[str] = None
    extraction_time_ms: Optional[float] = None
    chunks_used: int = 0
    retry_count: int = 0
    cold_start: bool = False
    timing_ms: Dict[str, float] = Field(default_factory=dict)


class CVExtractionSchema(BaseModel):
    """Fixed top-level schema supporting dynamic sections per assignment specification."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate: Optional[CandidateInfo] = Field(default=None, alias="candidate_info")
    summary: Optional[str] = None
    experience: List[WorkExperienceItem] = Field(default_factory=list, alias="work_experience")
    education: List[EducationItem] = Field(default_factory=list)
    skills: Optional[Union[SkillsBlock, List[Any], Dict[str, Any]]] = None
    certifications: List[CertificationItem] = Field(default_factory=list)
    derived: Optional[DerivedInsights] = None
    inferred: Optional[InferredSignals] = None
    sections: List[CustomSection] = Field(default_factory=list, description="Dynamic custom sections (Publications, Volunteering, Projects)")
    raw_text: Optional[str] = None
    confidence_scores: Optional[ConfidenceScores] = Field(default_factory=ConfidenceScores)
    processing_metadata: Optional[ProcessingMetadata] = None
