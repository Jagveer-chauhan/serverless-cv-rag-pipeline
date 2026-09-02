"""Deduplication and merge service for combining partial chunk JSON extractions."""
import logging
from typing import List, Dict, Any, Optional
from backend.app.schemas.cv_schema import (
    CVExtractionSchema,
    CandidateInfo,
    WorkExperienceItem,
    EducationItem,
    SkillCategory,
    ProjectItem,
    CertificationItem,
    LanguageItem,
    AwardItem,
)

logger = logging.getLogger("cv_rag_pipeline.merger")


def merge_extracted_chunks(partial_extractions: List[Dict[str, Any]]) -> CVExtractionSchema:
    """Merges and deduplicates multiple partial chunk extractions into a cohesive CVExtractionSchema."""
    if not partial_extractions:
        return CVExtractionSchema()

    merged_candidate_info: Dict[str, Any] = {}
    merged_summary: Optional[str] = None
    merged_experience: List[Dict[str, Any]] = []
    merged_education: List[Dict[str, Any]] = []
    merged_skills_by_category: Dict[str, List[str]] = {}
    merged_projects: List[Dict[str, Any]] = []
    merged_certifications: List[Dict[str, Any]] = []
    merged_languages: List[Dict[str, Any]] = []
    merged_awards: List[Dict[str, Any]] = []
    merged_additional: Dict[str, Any] = {}

    for extract in partial_extractions:
        if not isinstance(extract, dict):
            continue

        # 1. Candidate Info
        cand = extract.get("candidate_info")
        if cand and isinstance(cand, dict):
            for k, v in cand.items():
                if v:
                    if k == "links" and isinstance(v, list):
                        curr_links = set(merged_candidate_info.get("links", []))
                        curr_links.update(v)
                        merged_candidate_info["links"] = list(curr_links)
                    elif not merged_candidate_info.get(k):
                        merged_candidate_info[k] = v

        # 2. Summary
        summ = extract.get("summary")
        if summ and isinstance(summ, str) and len(summ.strip()) > len(merged_summary or ""):
            merged_summary = summ.strip()

        # 3. Work Experience
        exp_list = extract.get("work_experience")
        if exp_list and isinstance(exp_list, list):
            for item in exp_list:
                if isinstance(item, dict) and item.get("company"):
                    merged_experience.append(item)

        # 4. Education
        edu_list = extract.get("education")
        if edu_list and isinstance(edu_list, list):
            for item in edu_list:
                if isinstance(item, dict) and item.get("institution"):
                    merged_education.append(item)

        # 5. Skills
        skills_data = extract.get("skills")
        if skills_data and isinstance(skills_data, list):
            for sc in skills_data:
                if isinstance(sc, dict):
                    cat = sc.get("category_name", "Technical Skills")
                    skill_items = sc.get("skills", [])
                    if cat not in merged_skills_by_category:
                        merged_skills_by_category[cat] = []
                    for s in skill_items:
                        if s and s not in merged_skills_by_category[cat]:
                            merged_skills_by_category[cat].append(s)

        # 6. Projects
        proj_list = extract.get("projects")
        if proj_list and isinstance(proj_list, list):
            for p in proj_list:
                if isinstance(p, dict) and p.get("name"):
                    merged_projects.append(p)

        # 7. Certifications
        cert_list = extract.get("certifications")
        if cert_list and isinstance(cert_list, list):
            for c in cert_list:
                if isinstance(c, dict) and c.get("name"):
                    merged_certifications.append(c)

        # 8. Languages
        lang_list = extract.get("languages")
        if lang_list and isinstance(lang_list, list):
            for l in lang_list:
                if isinstance(l, dict) and l.get("language"):
                    merged_languages.append(l)

        # 9. Awards
        award_list = extract.get("awards")
        if award_list and isinstance(award_list, list):
            for a in award_list:
                if isinstance(a, dict) and a.get("title"):
                    merged_awards.append(a)

        # 10. Additional info
        add_info = extract.get("additional_info")
        if add_info and isinstance(add_info, dict):
            merged_additional.update(add_info)

    # Deduplicate Work Experience
    deduped_exp = _deduplicate_work_experience(merged_experience)

    # Deduplicate Education
    deduped_edu = _deduplicate_education(merged_education)

    # Construct SkillCategory objects
    final_skills = [
        SkillCategory(category_name=cat, skills=skills_list)
        for cat, skills_list in merged_skills_by_category.items()
        if skills_list
    ]

    # Deduplicate Projects
    deduped_projects = _deduplicate_by_name(merged_projects, key="name", model_cls=ProjectItem)

    # Deduplicate Certifications
    deduped_certs = _deduplicate_by_name(merged_certifications, key="name", model_cls=CertificationItem)

    # Deduplicate Languages
    deduped_languages = _deduplicate_by_name(merged_languages, key="language", model_cls=LanguageItem)

    # Deduplicate Awards
    deduped_awards = _deduplicate_by_name(merged_awards, key="title", model_cls=AwardItem)

    candidate_obj = CandidateInfo.model_validate(merged_candidate_info) if merged_candidate_info else None

    final_schema = CVExtractionSchema(
        candidate_info=candidate_obj,
        summary=merged_summary,
        work_experience=deduped_exp,
        education=deduped_edu,
        skills=final_skills,
        projects=deduped_projects,
        certifications=deduped_certs,
        languages=deduped_languages,
        awards=deduped_awards,
        additional_info=merged_additional or None,
    )
    logger.info(
        f"Merged CV JSON: {len(deduped_exp)} exp, {len(deduped_edu)} edu, "
        f"{len(final_skills)} skill categories, {len(deduped_projects)} projects."
    )
    return final_schema


def _deduplicate_work_experience(items: List[Dict[str, Any]]) -> List[WorkExperienceItem]:
    seen = {}
    for it in items:
        company = str(it.get("company", "")).strip().lower()
        position = str(it.get("position", "")).strip().lower()
        key = (company, position)
        if key not in seen:
            seen[key] = it
        else:
            # Merge achievements and technologies
            existing = seen[key]
            curr_ach = existing.get("key_achievements", [])
            new_ach = it.get("key_achievements", [])
            existing["key_achievements"] = list(set(curr_ach + new_ach))

            curr_tech = existing.get("technologies_used", [])
            new_tech = it.get("technologies_used", [])
            existing["technologies_used"] = list(set(curr_tech + new_tech))

    return [WorkExperienceItem.model_validate(val) for val in seen.values()]


def _deduplicate_education(items: List[Dict[str, Any]]) -> List[EducationItem]:
    seen = {}
    for it in items:
        inst = str(it.get("institution", "")).strip().lower()
        deg = str(it.get("degree", "")).strip().lower()
        key = (inst, deg)
        if key not in seen:
            seen[key] = it
    return [EducationItem.model_validate(val) for val in seen.values()]


def _deduplicate_by_name(items: List[Dict[str, Any]], key: str, model_cls: Any) -> List[Any]:
    seen = {}
    for it in items:
        name_val = str(it.get(key, "")).strip().lower()
        if name_val and name_val not in seen:
            seen[name_val] = it
    return [model_cls.model_validate(val) for val in seen.values()]
