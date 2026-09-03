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
    company: str = Field(default="Company")
    title: str = Field(default="Role", alias="position", description="Position or role title")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    location: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list, alias="technologies_used")
    achievements: List[str] = Field(default_factory=list, alias="key_achievements")
    inferred_skills: List[str] = Field(default_factory=list)


class EducationItem(DynamicBaseModel):
    institution: str = Field(default="University")
    degree: Optional[str] = None
    start_date: Optional[str] = Field(default=None, alias="start_year")
    end_date: Optional[str] = Field(default=None, alias="end_year")
    location: Optional[str] = None
    gpa: Optional[Union[str, float]] = None
    notes: Optional[str] = Field(default=None, alias="honors")


class SkillsBlock(DynamicBaseModel):
    explicit: List[str] = Field(default_factory=list)
    inferred: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)


class CertificationItem(DynamicBaseModel):
    name: str = Field(default="Certification")
    issuer: Optional[str] = None
    date: Optional[str] = Field(default=None, alias="issue_date")
    url: Optional[str] = Field(default=None, alias="link")


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


class ProjectItem(DynamicBaseModel):
    name: str = Field(default="Project", alias="title")
    role: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list, alias="technologies_used")
    links: List[str] = Field(default_factory=list, alias="link")


class AwardItem(DynamicBaseModel):
    name: str = Field(default="Award", alias="title")
    issuer: Optional[str] = Field(default=None, alias="organization")
    date: Optional[str] = Field(default=None, alias="year")
    description: Optional[str] = None


class PublicationItem(DynamicBaseModel):
    title: str = Field(default="Publication", alias="name")
    authors: Optional[str] = None
    publisher: Optional[str] = Field(default=None, alias="journal")
    date: Optional[str] = Field(default=None, alias="year")
    link: Optional[str] = Field(default=None, alias="url")
    description: Optional[str] = None


class LanguageItem(DynamicBaseModel):
    language: str = Field(default="Language", alias="name")
    proficiency: Optional[str] = Field(default=None, alias="level")


class ChunkExtractionSchema(DynamicBaseModel):
    """Flexible chunk extraction schema allowing chunk-level partial extractions."""
    candidate: Optional[CandidateInfo] = Field(default=None, alias="candidate_info")
    summary: Optional[str] = None
    experience: List[WorkExperienceItem] = Field(default_factory=list, alias="work_experience")
    education: List[EducationItem] = Field(default_factory=list)
    skills: Optional[Union[SkillsBlock, List[Any], Dict[str, Any]]] = None
    certifications: List[Union[CertificationItem, Dict[str, Any]]] = Field(default_factory=list)
    projects: List[Union[ProjectItem, Dict[str, Any], str]] = Field(default_factory=list)
    awards: List[Union[AwardItem, Dict[str, Any], str]] = Field(default_factory=list)
    publications: List[Union[PublicationItem, Dict[str, Any], str]] = Field(default_factory=list)
    languages: List[Union[LanguageItem, Dict[str, Any], str]] = Field(default_factory=list)
    sections: List[Union[CustomSection, Dict[str, Any]]] = Field(default_factory=list)


class ConfidenceScores(DynamicBaseModel):
    """Dynamically calculated extraction confidence scores (not hardcoded)."""
    overall: float = Field(default=0.0, description="Overall extraction confidence 0.0–1.0")
    experience_dates: float = Field(default=0.0, description="Confidence in experience date parsing")
    inferred_skills: float = Field(default=0.0, description="Confidence in inferred skills")


class ProcessingMetadata(DynamicBaseModel):
    """Full performance trace object as required by the assignment specification."""
    request_id: Optional[str] = None
    model: str = "google/gemma-3-4b-it"
    provider: str = "Hugging Face Serverless Inference API"
    status: str = "rag_ready"
    upload_accepted_at: Optional[str] = None
    rag_ready_at: Optional[str] = None
    extraction_time_ms: Optional[float] = None
    chunks_used: int = 0
    retry_count: int = 0
    # Cold-start tracking (required by spec §3.6)
    cold_start: bool = Field(default=False, alias="is_cold_start")
    cold_start_ms: Optional[float] = Field(default=None, alias="model_loading_time_ms")
    first_inference_ms: Optional[float] = None
    warm_inference_ms: Optional[float] = None
    provider_queue_time_ms: Optional[float] = None
    timing_ms: Dict[str, float] = Field(default_factory=dict)


class CVExtractionSchema(BaseModel):
    """Fixed top-level schema supporting dynamic sections per assignment specification.

    Top level is fixed (extra='forbid') with populate_by_name=True so both
    field names and aliases are accepted during model_validate().
    Nested models use extra='allow' to capture fields not in the base schema.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # populate_by_name=True ensures both 'candidate' and 'candidate_info' work
    candidate: Optional[CandidateInfo] = Field(default=None, alias="candidate_info")
    summary: Optional[str] = None
    experience: List[WorkExperienceItem] = Field(default_factory=list, alias="work_experience")
    education: List[EducationItem] = Field(default_factory=list)
    skills: Optional[Union[SkillsBlock, List[Any], Dict[str, Any]]] = None
    certifications: List[CertificationItem] = Field(default_factory=list)
    projects: List[Union[ProjectItem, Dict[str, Any]]] = Field(default_factory=list)
    awards: List[Union[AwardItem, Dict[str, Any]]] = Field(default_factory=list)
    publications: List[Union[PublicationItem, Dict[str, Any]]] = Field(default_factory=list)
    languages: List[Union[LanguageItem, Dict[str, Any], str]] = Field(default_factory=list)
    derived: Optional[DerivedInsights] = None
    inferred: Optional[InferredSignals] = None
    sections: List[CustomSection] = Field(
        default_factory=list,
        description="Dynamic custom sections (Publications, Volunteering, Projects, etc.)"
    )
    raw_text: Optional[str] = None
    confidence_scores: Optional[ConfidenceScores] = Field(default_factory=ConfidenceScores)
    processing_metadata: Optional[ProcessingMetadata] = None


def calculate_confidence_scores(schema: CVExtractionSchema) -> ConfidenceScores:
    """Dynamically calculates extraction confidence based on fields populated vs expected."""
    # Overall: ratio of non-empty top-level fields
    filled = sum([
        1 if schema.candidate and schema.candidate.name else 0,
        1 if schema.summary else 0,
        1 if schema.experience else 0,
        1 if schema.education else 0,
        1 if schema.skills else 0,
        1 if schema.certifications or schema.projects else 0,
        1 if schema.derived else 0,
        1 if schema.inferred else 0,
    ])
    overall = round(filled / 8, 2)

    # Experience dates confidence
    exp_with_dates = [
        e for e in schema.experience
        if e.start_date or e.end_date
    ] if schema.experience else []
    exp_date_conf = (
        round(len(exp_with_dates) / len(schema.experience), 2)
        if schema.experience else 0.5
    )

    # Inferred skills confidence
    inferred_count = len(schema.inferred.leadership_signals) if schema.inferred else 0
    inferred_conf = min(1.0, round(inferred_count / 3, 2)) if inferred_count else 0.5

    return ConfidenceScores(
        overall=overall,
        experience_dates=exp_date_conf,
        inferred_skills=inferred_conf,
    )
