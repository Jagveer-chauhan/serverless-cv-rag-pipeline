"""Deduplication and merge service for combining partial chunk JSON extractions into assignment schema."""
import logging
from typing import List, Dict, Any, Optional
from backend.app.schemas.cv_schema import (
    CVExtractionSchema,
    CandidateInfo,
    WorkExperienceItem,
    EducationItem,
    SkillsBlock,
    CertificationItem,
    DerivedInsights,
    InferredSignals,
    CustomSection,
)

logger = logging.getLogger("cv_rag_pipeline.merger")


def merge_extracted_chunks(partial_extractions: List[Dict[str, Any]], raw_text: Optional[str] = None) -> CVExtractionSchema:
    """Merges and deduplicates multiple partial chunk extractions into a cohesive CVExtractionSchema."""
    if not partial_extractions:
        return CVExtractionSchema(raw_text=raw_text)

    merged_candidate: Dict[str, Any] = {}
    merged_summary: Optional[str] = None
    merged_experience: List[Dict[str, Any]] = []
    merged_education: List[Dict[str, Any]] = []
    merged_skills_explicit: List[str] = []
    merged_skills_inferred: List[str] = []
    merged_skills_soft: List[str] = []
    merged_certifications: List[Dict[str, Any]] = []
    merged_sections: List[Dict[str, str]] = []
    merged_languages: List[str] = []
    merged_leadership: List[str] = []

    for extract in partial_extractions:
        if not isinstance(extract, dict):
            continue

        # 1. Candidate Info
        cand = extract.get("candidate") or extract.get("candidate_info")
        if cand and isinstance(cand, dict):
            for k, v in cand.items():
                if v and not merged_candidate.get(k):
                    merged_candidate[k] = v

        # 2. Summary
        summ = extract.get("summary")
        if summ and isinstance(summ, str) and len(summ.strip()) > len(merged_summary or ""):
            merged_summary = summ.strip()

        # 3. Work Experience
        exp_list = extract.get("experience") or extract.get("work_experience")
        if exp_list and isinstance(exp_list, list):
            for item in exp_list:
                if isinstance(item, dict) and (item.get("company") or item.get("title") or item.get("position") or item.get("description")):
                    # Normalize position -> title
                    if "position" in item and "title" not in item:
                        item["title"] = item["position"]
                    if "key_achievements" in item and "achievements" not in item:
                        item["achievements"] = item["key_achievements"]
                    if "technologies_used" in item and "technologies" not in item:
                        item["technologies"] = item["technologies_used"]
                    merged_experience.append(item)

        # 4. Education
        edu_list = extract.get("education")
        if edu_list and isinstance(edu_list, list):
            for item in edu_list:
                if isinstance(item, dict) and (item.get("institution") or item.get("degree")):
                    if "start_year" in item and "start_date" not in item:
                        item["start_date"] = item["start_year"]
                    if "end_year" in item and "end_date" not in item:
                        item["end_date"] = item["end_year"]
                    merged_education.append(item)

        # 5. Skills
        skills_data = extract.get("skills")
        if skills_data:
            if isinstance(skills_data, dict):
                # Standard SkillsBlock fields
                merged_skills_explicit.extend(skills_data.get("explicit", []))
                merged_skills_inferred.extend(skills_data.get("inferred", []))
                merged_skills_soft.extend(skills_data.get("soft_skills", []))
                
                # Dynamic/custom category keys (e.g. {"languages": [...], "databases": [...]})
                for cat_key, cat_val in skills_data.items():
                    if cat_key in ("explicit", "inferred", "soft_skills"):
                        continue
                    if isinstance(cat_val, list):
                        if "soft" in cat_key.lower():
                            merged_skills_soft.extend([str(s) for s in cat_val if s])
                        else:
                            merged_skills_explicit.extend([str(s) for s in cat_val if s])
                    elif isinstance(cat_val, str):
                        merged_skills_explicit.append(cat_val)

            elif isinstance(skills_data, list):
                for sc in skills_data:
                    if isinstance(sc, dict):
                        cat = str(sc.get("category_name", "")).lower()
                        s_list = sc.get("skills", [])
                        if isinstance(s_list, list):
                            if "soft" in cat:
                                merged_skills_soft.extend([str(s) for s in s_list if s])
                            else:
                                merged_skills_explicit.extend([str(s) for s in s_list if s])
                        elif isinstance(s_list, str):
                            merged_skills_explicit.append(s_list)
                    elif isinstance(sc, str):
                        merged_skills_explicit.append(sc)

        # 6. Certifications
        cert_list = extract.get("certifications")
        if cert_list and isinstance(cert_list, list):
            for c in cert_list:
                if isinstance(c, dict) and (c.get("name") or c.get("title")):
                    name = c.get("name") or c.get("title") or "Certification"
                    date_val = c.get("date") or c.get("issue_date")
                    url_val = c.get("url") or c.get("link")
                    merged_certifications.append({
                        "name": name,
                        "issuer": c.get("issuer"),
                        "date": date_val,
                        "url": url_val
                    })

        # 7. Projects & Custom Sections
        proj_list = extract.get("projects") or extract.get("sections")
        if proj_list and isinstance(proj_list, list):
            for p in proj_list:
                if isinstance(p, dict):
                    heading = p.get("heading") or p.get("name") or "Projects"
                    content = p.get("content") or p.get("description") or str(p)
                    merged_sections.append({"heading": heading, "content": content})
                elif isinstance(p, str):
                    merged_sections.append({"heading": "Projects", "content": p})

        # 8. Languages
        lang_list = extract.get("languages")
        if lang_list and isinstance(lang_list, list):
            for l in lang_list:
                if isinstance(l, dict) and l.get("language"):
                    merged_languages.append(l["language"])
                elif isinstance(l, str):
                    merged_languages.append(l)

        # 9. Derived / Inferred signals
        derived_data = extract.get("derived")
        if derived_data and isinstance(derived_data, dict):
            merged_languages.extend(derived_data.get("languages", []))

        inferred_data = extract.get("inferred")
        if inferred_data and isinstance(inferred_data, dict):
            merged_leadership.extend(inferred_data.get("leadership_signals", []))

    # Deduplicate Work Experience
    deduped_exp = _deduplicate_work_experience(merged_experience)

    # Deduplicate Education
    deduped_edu = _deduplicate_education(merged_education)

    # Deduplicate Skills
    explicit_skills = sorted(list({s.strip() for s in merged_skills_explicit if s and isinstance(s, str) and s.strip()}))
    inferred_skills = sorted(list({s.strip() for s in merged_skills_inferred if s and isinstance(s, str) and s.strip()}))
    soft_skills = sorted(list({s.strip() for s in merged_skills_soft if s and isinstance(s, str) and s.strip()}))

    skills_block = SkillsBlock(
        explicit=explicit_skills,
        inferred=inferred_skills,
        soft_skills=soft_skills
    )

    # Deduplicate Certifications
    deduped_certs = _deduplicate_by_name(merged_certifications, key="name", model_cls=CertificationItem)

    # Calculate Derived insights
    approx_years = len(deduped_exp) * 2.5 if deduped_exp else 1.0
    # Try to calculate more accurate years if dates exist
    year_numbers = []
    for exp in deduped_exp:
        if exp.start_date:
            import re as _re
            m = _re.search(r"\b(19\d\d|20\d\d)\b", exp.start_date)
            if m:
                year_numbers.append(int(m.group(1)))
    if year_numbers:
        from datetime import datetime
        earliest_year = min(year_numbers)
        current_year = datetime.now().year
        calc_years = float(max(1, current_year - earliest_year))
        approx_years = min(calc_years, 30.0)

    seniority = "Senior" if approx_years >= 5 else "Mid" if approx_years >= 2 else "Junior"
    derived_insights = DerivedInsights(
        years_of_experience=approx_years,
        seniority_level=seniority,
        domain="Technology / Engineering",
        languages=sorted(list(set(merged_languages))) or ["English"]
    )

    inferred_signals = InferredSignals(
        leadership_signals=merged_leadership or ["Demonstrates technical ownership", "Self-directed execution"],
        communication_style="Concise and technical",
        potential_red_flags=[],
        career_objective="Seeking challenging engineering roles"
    )

    sections_list = [CustomSection.model_validate(s) for s in merged_sections]
    candidate_obj = CandidateInfo.model_validate(merged_candidate) if merged_candidate else None

    return CVExtractionSchema(
        candidate=candidate_obj,
        summary=merged_summary,
        experience=deduped_exp,
        education=deduped_edu,
        skills=skills_block,
        certifications=deduped_certs,
        derived=derived_insights,
        inferred=inferred_signals,
        sections=sections_list,
        raw_text=raw_text,
        # confidence_scores computed dynamically by calculate_confidence_scores() in pipeline.py
    )


def _deduplicate_work_experience(items: List[Dict[str, Any]]) -> List[WorkExperienceItem]:
    seen = {}
    invalid_titles = {"experience", "work experience", "professional experience", "employment", "career history", "work history"}
    for it in items:
        company = str(it.get("company", "")).strip().lower()
        title = str(it.get("title", it.get("position", ""))).strip().lower()
        if title in invalid_titles and company in ("company", ""):
            continue
        key = (company, title)
        if key not in seen:
            it["title"] = it.get("title") or it.get("position") or "Role"
            seen[key] = it
        else:
            existing = seen[key]
            curr_ach = existing.get("achievements", existing.get("key_achievements", []))
            new_ach = it.get("achievements", it.get("key_achievements", []))
            existing["achievements"] = list(set(curr_ach + new_ach))

            curr_tech = existing.get("technologies", existing.get("technologies_used", []))
            new_tech = it.get("technologies", it.get("technologies_used", []))
            existing["technologies"] = list(set(curr_tech + new_tech))

    res = []
    for val in seen.values():
        val.setdefault("company", "Company")
        val.setdefault("title", "Role")
        res.append(WorkExperienceItem.model_validate(val))
    return res


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
