import io
import re
import gc
import base64
from typing import Optional
from PIL import Image
import httpx
from app.config import settings

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
    - Ultra-fast native vector text extraction (<0.05s) for instant digital parsing.
    - Cloud-optimized Tesseract fallback for environments without external keys.
    """

    @classmethod
    def extract_with_mistral_ocr(cls, pdf_bytes: bytes, mistral_api_key: Optional[str] = None) -> Optional[str]:
        """Directly invokes Mistral AI's state-of-the-art document OCR endpoint."""
        api_key = mistral_api_key or getattr(settings, "MISTRAL_API_KEY", "")
        if not api_key:
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
            with httpx.Client(timeout=45.0) as client:
                resp = client.post("https://api.mistral.ai/v1/ocr", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                pages = data.get("pages", [])
                markdown_pages = []
                for p in pages:
                    idx = p.get("index", 0)
                    md = p.get("markdown", "").strip()
                    if md:
                        markdown_pages.append(f"--- Page {idx + 1} (Mistral OCR) ---\n{md}")
                if markdown_pages:
                    return cls._clean_extracted_text("\n\n".join(markdown_pages))
        except Exception as e:
            # If Mistral OCR fails (e.g. invalid key or network timeout), gracefully fall back
            pass
        return None

    @classmethod
    def extract_text_from_pdf_bytes(cls, pdf_bytes: bytes, mistral_api_key: Optional[str] = None) -> str:
        # =====================================================================
        # PASS 0: SOTA Mistral OCR (`mistral-ocr-latest`) if API key is present
        # =====================================================================
        mistral_text = cls.extract_with_mistral_ocr(pdf_bytes, mistral_api_key)
        if mistral_text and len(mistral_text) >= 50:
            return mistral_text

        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is not installed. Please install it using `pip install pymupdf`."
            )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return ""

        # =====================================================================
        # PASS 1: Ultra-Fast Native Vector Text & Table Extraction (<20ms per page)
        # =====================================================================
        extracted_pages = []
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_sections = []

            # 1. Extract any structured vector tables with find_tables()
            try:
                tabs = page.find_tables()
                if tabs.tables:
                    for t_idx, tab in enumerate(tabs.tables):
                        rows = tab.extract()
                        if rows:
                            table_lines = [f"\n--- Page {page_idx + 1} Structured Table {t_idx + 1} ---"]
                            for row in rows:
                                clean_row = [str(cell or "").replace("\n", " ").strip() for cell in row]
                                if any(clean_row):
                                    table_lines.append(" | ".join(clean_row))
                            page_sections.append("\n".join(table_lines))
            except Exception:
                pass

            # 2. Extract geometrically sorted text (top-to-bottom, left-to-right)
            # sort=True prevents column interleaving and scrambled multi-column table cells
            sorted_text = page.get_text("text", sort=True).strip()
            if sorted_text:
                page_sections.append(f"\n--- Page {page_idx + 1} Document Text ---\n{sorted_text}")

            if page_sections:
                extracted_pages.append("\n".join(page_sections))

        combined_text = "\n\n".join(extracted_pages).strip()

        # If document contains digital text (>= 50 chars), return immediately!
        # Runs in 0.05 seconds, uses < 10 MB RAM, and never blocks the server.
        if len(combined_text) >= 50:
            doc.close()
            return cls._clean_extracted_text(combined_text)

        # =====================================================================
        # PASS 2: Cloud-Optimized OCR (ONLY for camera photos / pure flat scans)
        # =====================================================================
        if pytesseract is not None:
            ocr_pages = []
            # On bank sanction letters, all key loan parameters are on Pages 1-2.
            # Limiting to 2 pages avoids 60s CPU timeouts on cloud free tiers.
            pages_to_ocr = min(total_pages, 2)

            for page_idx in range(pages_to_ocr):
                page = doc[page_idx]
                try:
                    # 120 DPI grayscale is ideal: low RAM usage + high OCR accuracy
                    pix = page.get_pixmap(dpi=120)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                    text = pytesseract.image_to_string(img, config="--psm 6").strip()
                    if text:
                        ocr_pages.append(f"--- Page {page_idx + 1} (OCR) ---\n{text}")

                    # Explicitly free memory
                    del pix
                    del img
                    gc.collect()
                except Exception:
                    pass

            ocr_text = "\n\n".join(ocr_pages).strip()
            if len(ocr_text) > 20:
                doc.close()
                return cls._clean_extracted_text(ocr_text)

        doc.close()
        return cls._clean_extracted_text(combined_text)

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Standardizes currency, whitespace, and formatting."""
        if not text:
            return ""

        text = text.replace("\xa0", " ").replace("\u200b", "")
        # Standardize Rupee symbol or Indian font backtick (glyph 96 / ASCII `) to INR
        text = re.sub(r"[\u20B9₹`]|Rs\.?|INR", "INR ", text)
        # Fix OCR spaced numbers in Indian denominations e.g. "45 ,00 ,000" -> "45,00,000"
        text = re.sub(r"(\d+)\s*,\s*(\d+)", r"\1,\2", text)
        text = re.sub(r"(\d+)\s*,\s*(\d+)", r"\1,\2", text)
        # Fix spaced decimals e.g. "10 .05" -> "10.05"
        text = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
