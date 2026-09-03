"""Regex-based section-aware CV chunking service."""
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field

# Regex patterns matching standard resume section headings (strictly anchored to prevent matching inline text)
SECTION_HEADER_PATTERNS = [
    (r"(?i)^(?:professional\s+summary|executive\s+summary|career\s+summary|summary\s+of\s+qualifications|profile\s+summary|career\s+profile|about\s+me|about|career\s+objective|objective|personal\s+statement|summary|profile)(?:\s*\(.*?\))?[:\s\-–—]*$", "SUMMARY"),
    (r"(?i)^(?:work\s+experience|professional\s+experience|employment\s+history|career\s+history|work\s+history|employment|experience\s*&\s*projects|career\s+highlights|relevant\s+experience|work\s+background|professional\s+background|professional\s+journey|internships?|internship\s+experience|industry\s+experience|experience)(?:\s*\(.*?\))?[:\s\-–—]*$", "EXPERIENCE"),
    (r"(?i)^(?:relevant\s+training\s*&\s*certifications|certifications\s*&\s*training|training\s*&\s*certifications|certifications\s*&\s*licenses|licenses\s*&\s*certifications|courses\s*&\s*certifications|accreditations|certifications|certificates|licenses|licensures|courses|relevant\s+training|credentials|professional\s+certifications|training)(?:\s*\(.*?\))?[:\s\-–—]*$", "CERTIFICATIONS"),
    (r"(?i)^(?:education\s*&\s*training|education\s*&\s*certifications|education\s*&\s*qualifications|academic\s+background|academic\s+qualifications|academic\s+history|academic\s+credentials|educational\s+background|qualifications|education|academics|degrees)(?:\s*\(.*?\))?[:\s\-–—]*$", "EDUCATION"),
    (r"(?i)^(?:core\s+technical\s+skills|technical\s+skills|technical\s+proficiencies|technical\s+toolbox|technical\s+expertise|core\s+competencies|core\s+skills|areas\s+of\s+expertise|skills\s*&\s*expertise|skills\s*&\s*competencies|skills\s*&\s*abilities|skills\s*&\s*tools|skills\s*/\s*tools|skills\s*&\s*frameworks|key\s+skills|it\s+skills|technologies|tools\s*&\s*technologies|tech\s+stack|technological\s+skills|skills|proficiencies|competencies)(?:\s*\(.*?\))?[:\s\-–—]*$", "SKILLS"),
    (r"(?i)^(?:key\s+projects(?:\s*&\s*(?:contributions|portfolio|achievements|highlights))?|projects\s*&\s*(?:contributions|portfolio|achievements|highlights|systems)|selected\s+projects|personal\s+projects|notable\s+projects|portfolio\s+projects|major\s+projects(?:\s*&\s*systems(?:\s+developed)?)?|project\s+experience|academic\s+projects|software\s+projects|technical\s+projects|client\s+projects|live\s+projects|freelance\s+projects|systems?\s+developed|products?\s+developed|key\s+systems|key\s+highlights|projects\s*&\s*highlights|projects\s+undertaken|representative\s+projects|recent\s+projects|project\s+work|projects|portfolio|open\s+source(?:\s+contributions)?)(?:\s*\(.*?\))?[:\s\-–—]*$", "PROJECTS"),
    (r"(?i)^(?:publications\s*&\s*research|research\s*&\s*publications|research\s+papers|patents\s*&\s*publications|journals\s*&\s*conferences|publications|research|papers|patents)(?:\s*\(.*?\))?[:\s\-–—]*$", "PUBLICATIONS"),
    (r"(?i)^(?:awards\s*&\s*honors|awards\s*&\s*achievements|honors\s*&\s*awards|accomplishments|accolades|fellowships?|awards|honors|achievements)(?:\s*\(.*?\))?[:\s\-–—]*$", "AWARDS"),
    (r"(?i)^(?:languages\s+known|language\s+proficiencies|language\s+proficiency|known\s+languages|spoken\s+languages|languages)(?:\s*\(.*?\))?[:\s\-–—]*$", "LANGUAGES"),
    (r"(?i)^(?:volunteer\s+experience|volunteering|volunteer\s+work|extracurricular\s+activities|extracurricular|interests|hobbies\s*&\s*interests|leadership\s+activities|activities|volunteer|hobbies|leadership)(?:\s*\(.*?\))?[:\s\-–—]*$", "ADDITIONAL"),
]

MAX_CHUNK_CHAR_SIZE = 2500  # ~500-600 tokens (optimal for multi-page CV section continuity)
PAGE_NO_REGEX = re.compile(
    r"(?i)^(?:page\s+\d+(?:\s*(?:of|/)\s*\d+)?|\d+\s*(?:of|/)\s*\d+|-\s*\d+\s*-|page\s*\|\s*\d+|\d+\s*\|\s*page)\s*$"
)


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

        # Skip standalone page numbering artifacts
        if PAGE_NO_REGEX.match(stripped):
            continue

        # Check if line matches a known section header
        matched_section = None
        # Headers are short (< 60 chars) and standalone
        is_bullet = bool(re.match(r"^[-•*–—\t]\s*", stripped))
        if not is_bullet and len(stripped) < 60:
            # Strip markdown (#, *), bullet markers, numbers ("1. ", "II. "), and trailing colons/dashes
            clean_line = re.sub(r"^[\s#*_\-–—]+|[\s:*_\-–—]+$", "", stripped).strip()
            clean_line = re.sub(r"^(?:[0-9]+|[A-Za-z]|[IVXLCDM]+)[\.\)\-]\s*", "", clean_line).strip()
            clean_line = re.sub(r"^[\s#*_\-–—]+|[\s:*_\-–—]+$", "", clean_line).strip()

            # Ensure this is not a property line with inline content (e.g. "Technologies: React, Node...")
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
            # Split section by paragraphs or entries
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
