import io
import re
import gc
import base64
import logging
from typing import Optional, List, Dict, Any
from PIL import Image
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


class DocumentExtractor:
    """
    High-Performance Indian Bank Sanction Letter Extractor.
    - SOTA Mistral OCR (`mistral-ocr-latest`) for highest accuracy table & scanned PDF parsing.
    - Multi-page native PyMuPDF vector text, table, and annotation extraction across all pages/annexures/footnotes.
    - Cloud-optimized Tesseract fallback for pure image scans without external keys.
    """

    @classmethod
    def extract_pages_with_mistral_ocr(
        cls, pdf_bytes: bytes, mistral_api_key: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Directly invokes Mistral AI's document OCR endpoint and preserves page boundaries."""
        api_key = mistral_api_key or getattr(settings, "MISTRAL_API_KEY", "")
        if not api_key or "your_mistral" in api_key.lower():
            return None

        try:
            base64_data = base64.b64encode(pdf_bytes).decode("utf-8")
            payload = {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_data}",
                },
                "include_image_base64": False,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=httpx.Timeout(45.0, connect=5.0)) as client:
                resp = client.post("https://api.mistral.ai/v1/ocr", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                pages = data.get("pages", [])
                parsed_pages: List[Dict[str, Any]] = []
                for p in pages:
                    idx = p.get("index", 0)
                    md = p.get("markdown", "").strip()
                    if md:
                        cleaned = cls._clean_extracted_text(md)
                        parsed_pages.append(
                            {
                                "page": idx + 1,
                                "text": cleaned,
                                "tables": [],
                                "engine": "mistral_ocr",
                            }
                        )
                if parsed_pages:
                    return parsed_pages
        except Exception as e:
            logger.warning(f"Mistral OCR failed: {e}. Falling back to PyMuPDF native extraction.")
        return None

    @classmethod
    def extract_with_mistral_ocr(cls, pdf_bytes: bytes, mistral_api_key: Optional[str] = None) -> Optional[str]:
        pages = cls.extract_pages_with_mistral_ocr(pdf_bytes, mistral_api_key)
        if not pages:
            return None
        return "\n\n".join(f"--- Page {p['page']} (Mistral OCR) ---\n{p['text']}" for p in pages)

    @classmethod
    def extract_pages_from_pdf_bytes(
        cls, pdf_bytes: bytes, mistral_api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts structured text and tables page-by-page across every page, annexure,
        paragraph, table, and footnote, preserving exact source page numbers.
        """
        mistral_pages = cls.extract_pages_with_mistral_ocr(pdf_bytes, mistral_api_key)
        if mistral_pages and sum(len(p["text"]) for p in mistral_pages) >= 50:
            return mistral_pages

        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is not installed. Please install it using `pip install pymupdf`."
            )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            total_pages = len(doc)
            if total_pages == 0:
                return []

            pages_to_extract = min(total_pages, 50)
            extracted_pages: List[Dict[str, Any]] = []

            # PASS 1: Native Vector Text, Tables & Annotations across ALL pages
            for page_idx in range(pages_to_extract):
                page = doc[page_idx]
                page_sections: List[str] = []
                extracted_tables: List[List[List[str]]] = []

                # 1. Extract geometrically sorted text first (captures tables, paragraphs, footnotes, annexures
                #    without PyMuPDF find_tables() span/decimal reordering like "1860000 . , .")
                sorted_text = page.get_text("text", sort=True).strip()
                if sorted_text:
                    page_sections.append(f"--- Page {page_idx + 1} Document Text ---\n{sorted_text}")

                # 2. Extract structured vector tables as metadata; only append to text stream if sorted_text is empty
                try:
                    tabs = page.find_tables()
                    if tabs.tables:
                        table_texts = []
                        for t_idx, tab in enumerate(tabs.tables):
                            rows = tab.extract()
                            if rows:
                                clean_rows: List[List[str]] = []
                                table_lines = [f"--- Page {page_idx + 1} Structured Table {t_idx + 1} ---"]
                                for row in rows:
                                    clean_row = [str(cell or "").replace("\n", " ").strip() for cell in row]
                                    if any(clean_row):
                                        clean_rows.append(clean_row)
                                        table_lines.append(" | ".join(clean_row))
                                if clean_rows:
                                    extracted_tables.append(clean_rows)
                                    table_texts.append("\n".join(table_lines))
                        if not sorted_text and table_texts:
                            page_sections.extend(table_texts)
                except Exception as e:
                    logger.debug(f"Table extraction note on page {page_idx + 1}: {e}")

                # 3. Extract any PDF annotations / sticky notes
                try:
                    annot_texts = []
                    for annot in page.annots() or []:
                        info = annot.info or {}
                        content = (info.get("content") or "").strip()
                        if content:
                            annot_texts.append(content)
                    if annot_texts:
                        page_sections.append(
                            f"--- Page {page_idx + 1} Annotations ---\n" + "\n".join(annot_texts)
                        )
                except Exception:
                    pass

                raw_page_text = "\n\n".join(page_sections).strip()
                cleaned_page_text = cls._clean_extracted_text(raw_page_text)
                extracted_pages.append(
                    {
                        "page": page_idx + 1,
                        "text": cleaned_page_text,
                        "raw_text": raw_page_text,
                        "tables": extracted_tables,
                        "engine": "pymupdf",
                    }
                )

            total_chars = sum(len(p["text"]) for p in extracted_pages)
            if total_chars >= 50:
                return [p for p in extracted_pages if p["text"]]

            # PASS 2: Tesseract OCR for pure flat image scans
            if pytesseract is not None:
                ocr_pages: List[Dict[str, Any]] = []
                pages_to_ocr = min(total_pages, 5)
                for page_idx in range(pages_to_ocr):
                    page = doc[page_idx]
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                        text = pytesseract.image_to_string(img, config="--psm 6").strip()
                        del pix
                        del img
                        gc.collect()
                        if text:
                            ocr_pages.append(
                                {
                                    "page": page_idx + 1,
                                    "text": cls._clean_extracted_text(text),
                                    "tables": [],
                                    "engine": "tesseract_ocr",
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Tesseract OCR failed on page {page_idx + 1}: {e}")

                if ocr_pages and sum(len(p["text"]) for p in ocr_pages) > 20:
                    return ocr_pages

            return [p for p in extracted_pages if p["text"]]
        finally:
            doc.close()

    @classmethod
    def extract_text_from_pdf_bytes(cls, pdf_bytes: bytes, mistral_api_key: Optional[str] = None) -> str:
        pages = cls.extract_pages_from_pdf_bytes(pdf_bytes, mistral_api_key)
        if not pages:
            return ""
        chunks = []
        for p in pages:
            engine_tag = " (Mistral OCR)" if p.get("engine") == "mistral_ocr" else ""
            if p["text"].startswith("--- Page"):
                chunks.append(p["text"])
            else:
                chunks.append(f"--- Page {p['page']}{engine_tag} ---\n{p['text']}")
        return "\n\n".join(chunks).strip()

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Standardizes currency symbols, whitespace, and OCR number spacing without corrupting prose."""
        if not text:
            return ""

        text = text.replace("\xa0", " ").replace("\u200b", "")
        # Standardize Rupee glyphs (unicode ₹, Rs., INR, and Indian font backtick ` before numbers)
        text = re.sub(r"[\u20B9₹]|`(?=\s*\d)|\bRs\.?\s*|\bINR\b\s*", "INR ", text)
        # Fix OCR spaced numbers in Indian denominations e.g. "45 ,00 ,000" or "45,00 ,000" -> "45,00,000"
        text = re.sub(r"(\d)\s+,\s*(\d)", r"\1,\2", text)
        text = re.sub(r"(\d),\s+(\d)", r"\1,\2", text)
        text = re.sub(r"(\d)\s+,\s*(\d)", r"\1,\2", text)
        # Fix spaced decimals e.g. "10 .05" -> "10.05"
        text = re.sub(r"(\d)\s*\.\s*(\d+)", r"\1.\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
