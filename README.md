# LoanLens (India-Focused AI Loan Auditor)

LoanLens is an AI-powered financial auditor built specifically for Indian borrowers. It analyzes Bank Sanction Letters and Key Fact Statements (KFS) in PDF format to uncover hidden charges, verify stated vs. reducing balance EMIs, audit statutory 18% GST on banking fees, and compute the true cost of borrowing (**True APR / IRR**).

---

## 🛡️ Core Architectural Rule: Strict LLM vs. Math Isolation

1. **Zero Financial Hallucinations**: Under no circumstances does the LLM perform arithmetic, compound interest, fee summation, or APR calculations.
2. **LLM Role (OpenRouter / Qwen-2.5-72B)**:
   - Structured JSON parameter extraction from unstructured OCR/PDF text into strict Pydantic schemas.
   - Translating and explaining deterministic audit findings in plain **English, Hindi (हिन्दी), or Kannada (ಕನ್ನಡ)** with actionable borrower negotiation checklists.
3. **Python Financial Engine (`app/services/auditor.py`)**:
   - 100% deterministic execution of reducing balance amortization formulas.
   - Detection of deceptive flat rates disguised as reducing rates.
   - Statutory Indian GST (18% under SAC 9971) calculation on processing & documentation charges.
   - Exact True APR calculation using Newton-Raphson Internal Rate of Return (IRR).
   - RBI regulatory compliance auditing (e.g. RBI circular RBI/2014-15/168 on floating rate foreclosure fee prohibition).

---

## 📁 Project Structure

```text
loanlens/
├── backend/
│   ├── app/
│   │   ├── config.py                  # Pydantic Settings & environment variables
│   │   ├── main.py                    # FastAPI app (UI serving + API routes)
│   │   ├── core/
│   │   │   └── database.py            # SQLite / PostgreSQL engine & session
│   │   ├── models/
│   │   │   └── audit_record.py        # SQLAlchemy audit persistence model
│   │   ├── schemas/
│   │   │   └── loan.py                # Pydantic schemas (Extraction, Audit, Comparison)
│   │   └── services/
│   │       ├── auditor.py             # Deterministic financial math & RBI audit engine
│   │       ├── extractor.py           # PyMuPDF + Tesseract OCR fallback
│   │       └── llm_service.py         # OpenRouter API client (Qwen-2.5-72B)
│   ├── tests/
│   │   ├── test_auditor.py            # Pure deterministic unit tests
│   │   ├── test_api.py                # FastAPI endpoint tests
│   │   └── generate_sample_pdf.py     # Generates realistic Indian bank sanction letters
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html                     # Responsive single-page web dashboard
│   ├── package.json                   # React + Vite dependencies
│   └── src/                           # Modular React components
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Setup Environment
```powershell
cd loanlens/backend

# Copy .env.example to .env and insert your OpenRouter API key
Copy-Item .env.example .env
```

Edit `backend/.env`:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
DATABASE_URL=sqlite:///./loanlens.db
```
*(Note: If no API key is provided, LoanLens runs with an automatic local heuristic extraction and template explainer mode for offline development)*.

### 2. Run Deterministic Unit Tests
```powershell
python -m pytest tests/test_auditor.py -v
```

### 3. Launch LoanLens Server & Dashboard
```powershell
cd loanlens/backend
uvicorn app.main:app --reload --port 8000
```

- **Interactive Web Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Swagger Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 Key API Endpoints

### 1. `POST /api/v1/audit`
- **Input**: Multipart `file` (PDF) + `language` (`en` | `hi` | `kn`)
- **Pipeline**: PyMuPDF OCR $\rightarrow$ OpenRouter JSON extraction $\rightarrow$ Deterministic Python Audit $\rightarrow$ Multilingual Narrative Explanation $\rightarrow$ DB Storage.

### 2. `POST /api/v1/audit/manual`
- **Input**: Direct JSON matching `ExtractedLoanTerms`.
- **Purpose**: Direct calculation of True APR, GST, and Discrepancies without requiring a PDF upload.

### 3. `POST /api/v1/compare`
- **Input**: Two loan offer objects (`loan_a`, `loan_b`).
- **Output**: Strict ranking by Total Cost of Borrowing, showing the exact savings in Rupees and True APR comparison.
