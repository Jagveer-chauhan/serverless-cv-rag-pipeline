"""Pydantic v2 schemas for structured CV extraction and dynamic attribute validation."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, EmailStr


class DynamicBaseModel(BaseModel):
    """Base model allowing dynamic extra fields for flexible extraction."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class CandidateInfo(DynamicBaseModel):
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Primary email address")
    phone: Optional[str] = Field(None, description="Phone / contact number")
    location: Optional[str] = Field(None, description="City, State, Country")
    links: Optional[List[str]] = Field(default_factory=list, description="LinkedIn, GitHub, Portfolio URLs")
    title: Optional[str] = Field(None, description="Professional headline or title")


class WorkExperienceItem(DynamicBaseModel):
    company: str = Field(..., description="Name of the employing company or organization")
    position: str = Field(..., description="Job title / role")
    start_date: Optional[str] = Field(None, description="Start date (e.g., '2021-06' or 'June 2021')")
    end_date: Optional[str] = Field(None, description="End date (e.g., 'Present' or '2023-12')")
    is_current: bool = Field(False, description="Whether this is the current job")
    location: Optional[str] = Field(None, description="Office location or Remote")
    description: Optional[str] = Field(None, description="Role summary and duties")
    key_achievements: List[str] = Field(default_factory=list, description="Quantified achievements and impact bullets")
    technologies_used: List[str] = Field(default_factory=list, description="Tools and technologies utilized")


class EducationItem(DynamicBaseModel):
    institution: str = Field(..., description="University, College, or School name")
    degree: Optional[str] = Field(None, description="Degree earned (e.g., 'B.S.', 'M.S.', 'Ph.D.')")
    field_of_study: Optional[str] = Field(None, description="Major, discipline, or field of study")
    start_year: Optional[str] = Field(None, description="Start year")
    end_year: Optional[str] = Field(None, description="Graduation year or expected graduation")
    gpa: Optional[str] = Field(None, description="Grade point average or honors")
    honors: List[str] = Field(default_factory=list, description="Academic honors or scholarships")


class SkillCategory(DynamicBaseModel):
    category_name: str = Field(..., description="Category (e.g., 'Languages', 'Frameworks', 'Databases', 'Cloud')")
    skills: List[str] = Field(default_factory=list, description="List of individual skill names")
    proficiency_level: Optional[str] = Field(None, description="Beginner, Intermediate, Advanced, Expert")


class ProjectItem(DynamicBaseModel):
    name: str = Field(..., description="Project title or name")
    description: Optional[str] = Field(None, description="Project summary and impact")
    role: Optional[str] = Field(None, description="Role in project")
    technologies: List[str] = Field(default_factory=list, description="Tech stack utilized")
    link: Optional[str] = Field(None, description="URL or repository link")


class CertificationItem(DynamicBaseModel):
    name: str = Field(..., description="Certification or license title")
    issuer: Optional[str] = Field(None, description="Issuing organization (e.g., AWS, GCP, Microsoft)")
    issue_date: Optional[str] = Field(None, description="Date issued")
    expiry_date: Optional[str] = Field(None, description="Expiration date")
    credential_id: Optional[str] = Field(None, description="Credential or verification ID")
    link: Optional[str] = Field(None, description="Verification URL")


class LanguageItem(DynamicBaseModel):
    language: str = Field(..., description="Language name (e.g., 'English', 'Spanish')")
    proficiency: Optional[str] = Field(None, description="Proficiency level (e.g., 'Native', 'Fluent', 'Professional')")


class AwardItem(DynamicBaseModel):
    title: str = Field(..., description="Award or honor title")
    issuer: Optional[str] = Field(None, description="Granting organization")
    date: Optional[str] = Field(None, description="Date received")
    description: Optional[str] = Field(None, description="Details or criteria")


class CVExtractionSchema(BaseModel):
    """Root Pydantic v2 schema for structured CV extraction.
    
    Fixed top-level fields ensure consistent API contracts, while nested models
    use ConfigDict(extra='allow') for dynamic candidate attributes.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate_info: Optional[CandidateInfo] = Field(default=None, description="Contact and personal information")
    summary: Optional[str] = Field(default=None, description="Professional summary or objective statement")
    work_experience: List[WorkExperienceItem] = Field(default_factory=list, description="Work and employment history")
    education: List[EducationItem] = Field(default_factory=list, description="Academic degrees and qualifications")
    skills: List[SkillCategory] = Field(default_factory=list, description="Categorized technical and soft skills")
    projects: List[ProjectItem] = Field(default_factory=list, description="Key projects and portfolios")
    certifications: List[CertificationItem] = Field(default_factory=list, description="Certificates and licenses")
    languages: List[LanguageItem] = Field(default_factory=list, description="Spoken/written languages")
    awards: List[AwardItem] = Field(default_factory=list, description="Honors, awards, and recognitions")
    additional_info: Optional[Dict[str, Any]] = Field(default=None, description="Any other pertinent resume details")
