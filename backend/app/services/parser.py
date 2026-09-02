"""In-memory PDF text extraction service using PyMuPDF with pytesseract OCR fallback."""
import io
import logging
from typing import Tuple, Dict, Any, Optional
try:
    import fitz
except ImportError:
    import pymupdf as fitz
from PIL import Image

logger = logging.getLogger("cv_rag_pipeline.parser")

MIN_TEXT_DENSITY_CHARS_PER_PAGE = 50


def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> Tuple[str, Dict[str, Any]]:
    """Extracts text from PDF bytes in-memory using PyMuPDF with OCR fallback.
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    if not pdf_bytes:
        raise ValueError("PDF content is empty.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    if page_count == 0:
        raise ValueError("PDF document has 0 pages.")

    pages_text = []
    total_chars = 0

    for page_idx in range(page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text") or ""
        pages_text.append(text)
        total_chars += len(text.strip())

    char_density = total_chars / page_count
    logger.info(
        f"PyMuPDF extracted {total_chars} chars across {page_count} pages "
        f"(density: {char_density:.1f} chars/page) for '{filename}'"
    )

    # Trigger OCR fallback if text density is below threshold
    if char_density < MIN_TEXT_DENSITY_CHARS_PER_PAGE or total_chars < 30:
        logger.warning(
            f"Low text density detected ({char_density:.1f} < {MIN_TEXT_DENSITY_CHARS_PER_PAGE}). "
            f"Triggering pytesseract OCR fallback for '{filename}'..."
        )
        ocr_text, ocr_meta = _extract_via_ocr_fallback(doc, filename)
        doc.close()
        return ocr_text, ocr_meta

    full_text = "\n\n".join(pages_text).strip()
    doc.close()

    metadata = {
        "parser": "pymupdf_direct",
        "page_count": page_count,
        "char_count": len(full_text),
        "char_density": round(char_density, 2),
        "ocr_triggered": False,
    }
    return full_text, metadata


def _extract_via_ocr_fallback(doc: fitz.Document, filename: str) -> Tuple[str, Dict[str, Any]]:
    """Fallback OCR extraction using pytesseract when direct text extraction fails."""
    try:
        import pytesseract
    except ImportError:
        logger.error("pytesseract is not installed. Returning partial direct text.")
        return "", {"parser": "fallback_unavailable", "ocr_triggered": True, "error": "pytesseract missing"}

    ocr_pages_text = []
    total_ocr_chars = 0
    page_count = len(doc)

    for page_idx in range(page_count):
        page = doc.load_page(page_idx)
        # Render page to high-res pixmap in memory (150 DPI for fast OCR)
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            page_text = pytesseract.image_to_string(img)
        except Exception as ocr_err:
            logger.warning(f"OCR failed on page {page_idx + 1} of '{filename}': {ocr_err}")
            page_text = ""
        ocr_pages_text.append(page_text)
        total_ocr_chars += len(page_text.strip())

    full_ocr_text = "\n\n".join(ocr_pages_text).strip()
    metadata = {
        "parser": "pytesseract_ocr",
        "page_count": page_count,
        "char_count": len(full_ocr_text),
        "char_density": round(total_ocr_chars / max(page_count, 1), 2),
        "ocr_triggered": True,
    }
    logger.info(f"pytesseract OCR completed for '{filename}' with {total_ocr_chars} characters.")
    return full_ocr_text, metadata
