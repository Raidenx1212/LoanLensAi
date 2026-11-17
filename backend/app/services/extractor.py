import io
import re
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
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


class DocumentExtractor:
    """
    High-Fidelity Indian Bank Sanction Letter text & table extractor.
    Preserves tabular structures using PyMuPDF4LLM Markdown parser,
    with fallbacks to PyMuPDF table finder and vector text.
    """

    @classmethod
    def extract_text_from_pdf_bytes(cls, pdf_bytes: bytes) -> str:
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is not installed. Please install it using `pip install pymupdf`."
            )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            doc.close()
            return ""

        extracted_text = ""

        # 1. Primary Strategy: PyMuPDF4LLM Markdown conversion (preserves tables perfectly)
        if pymupdf4llm is not None:
            try:
                markdown_text = pymupdf4llm.to_markdown(doc)
                if markdown_text and len(markdown_text.strip()) > 50:
                    extracted_text = markdown_text
            except Exception:
                extracted_text = ""

        # 2. Secondary Strategy: Page-by-page table extraction + text extraction
        if not extracted_text.strip():
            page_chunks = []
            for page_idx in range(len(doc)):
                page = doc[page_idx]

                # Extract formatted tables if present
                tables_md = []
                try:
                    tabs = page.find_tables()
                    for tab in tabs:
                        df = tab.extract()
                        if df:
                            # Convert 2D list to markdown table
                            header = " | ".join(str(c or "").strip() for c in df[0])
                            sep = " | ".join(["---"] * len(df[0]))
                            rows = [" | ".join(str(c or "").strip() for c in row) for row in df[1:]]
                            tables_md.append(f"\n{header}\n{sep}\n" + "\n".join(rows) + "\n")
                except Exception:
                    pass

                text = page.get_text("text").strip()

                # Scanned image detection
                if len(text) < 40 and pytesseract is not None:
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(img, config="--psm 6").strip()
                        if ocr_text:
                            text = ocr_text
                    except Exception:
                        pass

                page_content = text
                if tables_md:
                    page_content += "\n\n### Extracted Tables:\n" + "\n".join(tables_md)

                page_chunks.append(f"--- Page {page_idx + 1} ---\n{page_content}")

            extracted_text = "\n\n".join(page_chunks)

        doc.close()

        # Clean up common PDF formatting artifacts
        extracted_text = cls._clean_extracted_text(extracted_text)
        return extracted_text

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Cleans Indian currency symbols, non-breaking spaces, and formatting quirks."""
        if not text:
            return ""

        # Normalize non-breaking spaces
        text = text.replace("\xa0", " ").replace("\u200b", "")

        # Standardize Indian Rupee representations to INR
        text = re.sub(r"[\u20B9₹]|Rs\.?|INR", "INR ", text)

        # Normalize multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
