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
    High-Performance, Memory-Safe Indian Bank Sanction Letter Extractor.
    Optimized for cloud environments (512MB RAM limit).
    Prioritizes ultra-fast native vector text/tables; uses low-memory OCR only if document is scanned.
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

        extracted_pages = []

        # =====================================================================
        # PASS 1: Native Vector Text & Table Extraction (<10 MB RAM, 50ms)
        # =====================================================================
        for page_idx in range(total_pages):
            page = doc[page_idx]

            # 1. Native table extraction
            table_markdown = []
            try:
                table_finder = page.find_tables()
                if table_finder and table_finder.tables:
                    for tab in table_finder.tables:
                        grid = tab.extract()
                        if grid and len(grid) >= 2:
                            header = " | ".join(str(c or "").strip() for c in grid[0])
                            sep = " | ".join(["---"] * len(grid[0]))
                            rows = [" | ".join(str(c or "").strip() for c in row) for row in grid[1:]]
                            table_markdown.append(f"\n| {header} |\n| {sep} |\n" + "\n".join(f"| {r} |" for r in rows) + "\n")
            except Exception:
                pass

            page_text = page.get_text("text").strip()

            if table_markdown:
                page_text += "\n\n### Formatted Tables:\n" + "\n".join(table_markdown)

            extracted_pages.append(page_text)

        full_text = "\n\n--- Page Break ---\n\n".join(extracted_pages).strip()

        # If native text extraction succeeded (>= 50 chars), return immediately!
        # This completely avoids heavy OCR and prevents Render 512MB OOM crashes.
        if len(full_text) >= 50:
            doc.close()
            return cls._clean_extracted_text(full_text)

        # =====================================================================
        # PASS 2: Memory-Constrained Low-DPI OCR (ONLY for pure scanned images)
        # =====================================================================
        if pytesseract is not None:
            ocr_pages = []
            # Limit OCR to first 4 pages to protect 512MB memory limit
            max_ocr_pages = min(total_pages, 4)
            for page_idx in range(max_ocr_pages):
                page = doc[page_idx]
                try:
                    # Render at 130 DPI (sharp enough for OCR, 1/4th the RAM of 300 DPI)
                    pix = page.get_pixmap(dpi=130)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")  # Convert to grayscale
                    ocr_text = pytesseract.image_to_string(img, config="--psm 6").strip()
                    ocr_pages.append(ocr_text)

                    # Explicitly release pixel buffers to keep RAM low
                    del pix
                    del img
                    gc.collect()
                except Exception:
                    pass

            ocr_combined = "\n\n--- Page Break ---\n\n".join(ocr_pages).strip()
            if len(ocr_combined) > 20:
                doc.close()
                return cls._clean_extracted_text(ocr_combined)

        doc.close()
        return cls._clean_extracted_text(full_text)

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Cleans currency symbols, whitespace, and formatting artifacts."""
        if not text:
            return ""

        # Normalize non-breaking spaces
        text = text.replace("\xa0", " ").replace("\u200b", "")

        # Standardize Indian Rupee representations to INR
        text = re.sub(r"[\u20B9₹]|Rs\.?|INR", "INR ", text)

        # Normalize redundant blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
