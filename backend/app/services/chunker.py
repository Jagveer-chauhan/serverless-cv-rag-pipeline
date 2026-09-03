"""Regex-based section-aware CV chunking service."""
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field

# Regex patterns matching standard resume section headings
SECTION_HEADER_PATTERNS = [
    (r"(?i)^(?:professional\s+summary|executive\s+summary|career\s+summary|summary\s+of\s+qualifications|profile\s+summary|career\s+profile|summary|profile|about\s+me|about|career\s+objective|objective|personal\s+statement)\b", "SUMMARY"),
    (r"(?i)^(?:work\s+experience|professional\s+experience|experience|employment\s+history|career\s+history|work\s+history|employment|experience\s*&\s*projects|career\s+highlights|relevant\s+experience|work\s+background|professional\s+background|internships?|internship\s+experience)\b", "EXPERIENCE"),
    (r"(?i)^(?:relevant\s+training\s*&\s*certifications|certifications\s*&\s*training|training\s*&\s*certifications|certifications|certificates|licenses\s*&\s*certifications|licenses|licensures|courses\s*&\s*certifications|accreditations|courses|training|relevant\s+training|credentials|professional\s+certifications)\b", "CERTIFICATIONS"),
    (r"(?i)^(?:education\s*&\s*training|education\s*&\s*certifications|education\s*&\s*qualifications|education|academic\s+background|academic\s+qualifications|academic\s+history|academic\s+credentials|qualifications|academics|degrees)\b", "EDUCATION"),
    (r"(?i)^(?:core\s+technical\s+skills|technical\s+skills|technical\s+proficiencies|technical\s+toolbox|technical\s+expertise|core\s+competencies|core\s+skills|areas\s+of\s+expertise|skills\s*&\s*expertise|skills\s*&\s*competencies|skills\s*&\s*abilities|skills\s*&\s*tools|skills\s*/\s*tools|skills|key\s+skills|it\s+skills|technologies|tools\s*&\s*technologies|tech\s+stack|proficiencies|competencies)\b", "SKILLS"),
    (r"(?i)^(?:key\s+projects|projects|selected\s+projects|personal\s+projects|notable\s+projects|portfolio\s+projects|major\s+projects|project\s+experience|academic\s+projects|software\s+projects|open\s+source(?:\s+contributions)?)\b", "PROJECTS"),
    (r"(?i)^(?:publications\s*&\s*research|research\s*&\s*publications|publications|research\s+papers|research|papers|patents\s*&\s*publications|patents|journals\s*&\s*conferences)\b", "PUBLICATIONS"),
    (r"(?i)^(?:awards\s*&\s*honors|awards\s*&\s*achievements|honors\s*&\s*awards|awards|honors|achievements|accolades|accomplishments)\b", "AWARDS"),
    (r"(?i)^(?:languages\s+known|language\s+proficiencies|language\s+proficiency|known\s+languages|languages)\b", "LANGUAGES"),
    (r"(?i)^(?:volunteer\s+experience|volunteering|volunteer\s+work|volunteer|extracurricular\s+activities|extracurricular|interests|hobbies\s*&\s*interests|hobbies|activities|leadership\s+activities|leadership)\b", "ADDITIONAL"),
]

MAX_CHUNK_CHAR_SIZE = 1200  # ~250-300 tokens


@dataclass
class TextChunk:
    chunk_index: int
    section_name: str
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "section_name": self.section_name,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


def estimate_token_count(text: str) -> int:
    """Fast approximation of token count (avg ~4 chars per token)."""
    return max(1, len(text.strip().split()))


def chunk_cv_text(raw_text: str) -> List[TextChunk]:
    """Splits raw CV text into section-aware chunks with preserved context headers."""
    if not raw_text or not raw_text.strip():
        return []

    lines = [line.rstrip() for line in raw_text.splitlines()]
    sections: List[Dict[str, Any]] = []
    
    current_section = "CONTACT_HEADER"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue

        # Check if line matches a known section header
        matched_section = None
        # Headers are usually short (< 70 chars) and standalone or formatted
        if len(stripped) < 70:
            # Strip markdown (#, *), bullet markers, numbers ("1. ", "II. "), and trailing colons
            clean_line = re.sub(r"^[\s#*_\-–—]+|[\s:*_\-–—]+$", "", stripped).strip()
            clean_line = re.sub(r"^(?:[0-9]+|[A-Za-z]|[IVXLCDM]+)[\.\)\-]\s*", "", clean_line).strip()
            clean_line = re.sub(r"^[\s#*_\-–—]+|[\s:*_\-–—]+$", "", clean_line).strip()
            for pattern, sec_name in SECTION_HEADER_PATTERNS:
                if re.match(pattern, clean_line):
                    matched_section = sec_name
                    break

        if matched_section:
            # Commit previous section
            if current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    sections.append({"section": current_section, "text": sec_text})
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    # Commit last section
    if current_lines:
        sec_text = "\n".join(current_lines).strip()
        if sec_text:
            sections.append({"section": current_section, "text": sec_text})

    # Subdivide large sections while preserving section context prefix
    final_chunks: List[TextChunk] = []
    chunk_idx = 0

    for sec in sections:
        sec_name = sec["section"]
        sec_text = sec["text"]

        if len(sec_text) <= MAX_CHUNK_CHAR_SIZE:
            formatted_content = f"[{sec_name}]\n{sec_text}"
            final_chunks.append(
                TextChunk(
                    chunk_index=chunk_idx,
                    section_name=sec_name,
                    content=formatted_content,
                    token_count=estimate_token_count(formatted_content),
                    metadata={"section": sec_name, "part": 1, "total_parts": 1}
                )
            )
            chunk_idx += 1
        else:
            # Split section by paragraphs or lines
            paragraphs = sec_text.split("\n\n")
            sub_chunks = []
            curr_buf: List[str] = []
            curr_len = 0

            for p in paragraphs:
                p_len = len(p)
                if curr_len + p_len > MAX_CHUNK_CHAR_SIZE and curr_buf:
                    sub_chunks.append("\n\n".join(curr_buf))
                    curr_buf = [p]
                    curr_len = p_len
                else:
                    curr_buf.append(p)
                    curr_len += p_len + 2

            if curr_buf:
                sub_chunks.append("\n\n".join(curr_buf))

            total_parts = len(sub_chunks)
            for part_num, sub_content in enumerate(sub_chunks, start=1):
                part_prefix = f"[{sec_name}] (Part {part_num}/{total_parts})\n"
                formatted_sub = f"{part_prefix}{sub_content.strip()}"
                final_chunks.append(
                    TextChunk(
                        chunk_index=chunk_idx,
                        section_name=sec_name,
                        content=formatted_sub,
                        token_count=estimate_token_count(formatted_sub),
                        metadata={"section": sec_name, "part": part_num, "total_parts": total_parts}
                    )
                )
                chunk_idx += 1

    return final_chunks
