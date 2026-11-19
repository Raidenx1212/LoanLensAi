# 🔍 LoanLens — AI-Powered Indian Loan Auditor

> **Uncover hidden bank charges, verify stated vs. reducing EMIs, audit statutory 18% GST, and calculate your True APR in seconds.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-619DC6?style=for-the-badge&logo=render)](https://loanlens-ai-auditor.onrender.com)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-A6C8E2?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-F4BAA3?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![RBI Fair Lending](https://img.shields.io/badge/RBI-KFS%20Compliant-F4F0E2?style=for-the-badge&logo=shield)](https://rbi.org.in)

---

## 🛑 The Problem

When taking a personal, home, or vehicle loan in India, what banks advertise is rarely what you actually pay:

1. **The "Headline Rate" Trap**: Lenders advertise attractive rates like **10.5%**, but after upfront deductions and processing fees, your **True Effective APR** often exceeds **16%**.
2. **Disguised Flat Rates**: NBFCs often quote a "7% flat rate" which sounds cheap, but equals an exorbitant **13%+ reducing rate**.
3. **Mandatory 18% GST on Fees**: Bank processing fees and documentation charges attract 18% GST (SAC 9971), reducing the actual cash disbursed to your bank account.
4. **Forced Bundled Insurance**: Banks frequently deduct ₹10,000–₹25,000 upfront for "Loan Protection" or "Credit Shield" insurance without telling you it is legally optional.
5. **Illegal Foreclosure Penalties**: Despite the Reserve Bank of India strictly prohibiting pre-payment penalties on floating rate personal loans (*RBI/2014-15/168*), banks still sneak 2–4% exit penalties into sanction letters.
6. **Complex Fine Print**: Bank Sanction Letters and Key Fact Statements (KFS) are dense, multi-page PDFs full of legal jargon that everyday borrowers cannot decipher.

---

## 💡 The Solution

**LoanLens** is an intelligent loan auditor designed specifically for Indian borrowers. It automates financial due diligence:

- 📄 **Upload Any Sanction Letter PDF**: Instantly parses multi-page PDFs from SBI, HDFC, ICICI, Axis, Bajaj Finance, and any RBI-regulated lender.
- 🧮 **Strict LLM / Math Isolation (Zero Hallucinations)**:
  - **AI (OpenRouter / Qwen-2.5-72B)** *only* extracts structured loan terms from unstructured text and drafts plain-language explanations.
  - **100% Deterministic Python Engine** calculates reducing balance EMIs, 18% GST, net in-hand cash, and **True APR via Newton-Raphson IRR**.
- 🛡️ **RBI Regulatory Compliance Audit**: Automatically detects illegal foreclosure clauses, predatory rate spreads, and bundled insurance traps.
- 🗣️ **Multilingual Borrower Guide**: Explains the audit findings in plain **English, Hindi (हिन्दी), or Kannada (ಕನ್ನಡ)** with a direct question-and-answer checklist to negotiate with your bank branch manager.
- ⚖️ **Loan Offer Comparison**: Compare two competing loan offers side-by-side to find out which one saves you real cash.

---

## ⚡ How It Works

```
┌─────────────────────────────────┐
│ 1. Upload Bank Sanction PDF/KFS │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 2. Ultra-Fast Text & Table OCR  │  <-- PyMuPDF (<0.05s) + Tesseract fallback
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 3. Structured Parameter Parsing │  <-- Qwen-2.5-72B (Strictly Zero Math)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 4. Deterministic Financial Math │  <-- Reducing EMI, 18% GST, Newton-Raphson IRR
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 5. RBI Compliance & True APR    │  <-- Flags illegal penalties, fee spreads
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 6. Multilingual Borrower Report │  <-- Actionable checklist (EN / HI / KN)
└─────────────────────────────────┘
```

---

## 🌐 Live Web App

Try the live application right now without installation:
👉 **[https://loanlens-ai-auditor.onrender.com](https://loanlens-ai-auditor.onrender.com)**

---

## 🚀 Quickstart (Run Locally)

### 1. Clone & Setup
```bash
git clone https://github.com/Raidenx1212/LoanLensAi.git
cd LoanLensAi/backend

# Create virtual environment & install requirements
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
cp .env.example .env
```
Add your OpenRouter API key for high-accuracy AI extraction:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```
*(Note: If no API key is set, LoanLens automatically runs in offline heuristic mode with full deterministic auditing).*

### 3. Run Tests & Launch Server
```bash
# Verify financial math unit tests
python -m pytest tests/test_auditor.py -v

# Start the application
uvicorn app.main:app --reload --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📑 Core API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET`, `HEAD` | Interactive Web Dashboard |
| `/health` | `GET`, `HEAD` | Cloud health check probe |
| `/api/v1/audit` | `POST` | Upload PDF $\rightarrow$ OCR $\rightarrow$ Deterministic Audit $\rightarrow$ Multilingual report |
| `/api/v1/audit/manual` | `POST` | Direct JSON parameter audit without a PDF |
| `/api/v1/compare` | `POST` | Side-by-side comparison of 2 loan offers ranked by total borrowing cost |

---

## 📜 Regulatory References
- **RBI Circular RBI/2014-15/168**: Strict prohibition on pre-payment/foreclosure penalties on floating-rate individual loans.
- **RBI Key Fact Statement (KFS) Guidelines (April 2024)**: Mandatory disclosure of Total Cost of Borrowing and All-Inclusive APR.
- **GST SAC 9971**: Statutory 18% Goods and Services Tax applicable to banking and loan processing fees in India.
