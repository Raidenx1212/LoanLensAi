import io
import re
import gc
from typing import Optional
from PIL import Image

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
    High-Performance, Cloud-Safe Indian Bank Sanction Letter Extractor.
    Designed for low-resource environments (Render 512MB RAM / 0.1 CPU).
    - Instant vector text extraction for digital PDFs (<0.05s).
    - Fast 2-page max OCR for scanned image PDFs (<15s) with strict memory cleanup.
    """

    @classmethod
    def extract_text_from_pdf_bytes(cls, pdf_bytes: bytes) -> str:
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
        # PASS 1: Ultra-Fast Native Vector Text Extraction (<10ms per page)
        # =====================================================================
        extracted_pages = []
        for page_idx in range(total_pages):
            page = doc[page_idx]
            # get_text('text') reliably extracts all tabular cells, text blocks, and loan values
            text = page.get_text("text").strip()
            if text:
                extracted_pages.append(f"--- Page {page_idx + 1} ---\n{text}")

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
        text = re.sub(r"[\u20B9₹]|Rs\.?|INR", "INR ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
