import json
import logging
import re
from typing import Dict, Any, Optional
import httpx
from app.schemas.loan import (
    ExtractedLoanTerms,
    LoanAuditReport,
    Language,
    InterestRateType,
    LoanCategory,
)
from app.config import settings

logger = logging.getLogger(__name__)


class OpenRouterLLMService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "qwen/qwen-2.5-72b-instruct"

    async def _call_openrouter(
        self, messages: list, temperature: float = 0.0, response_format: dict = None, custom_api_key: Optional[str] = None
    ) -> str:
        key = custom_api_key or self.api_key or settings.OPENROUTER_API_KEY
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://loanlens.ai",
            "X-Title": "LoanLens AI Auditor",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def extract_loan_parameters(
        self, raw_document_text: str, custom_api_key: Optional[str] = None
    ) -> ExtractedLoanTerms:
        """
        STRICT ARCHITECTURAL RULE:
        The LLM extracts numbers verbatim as found in the text.
        It must never calculate EMIs, GST, or APR.
        """
        active_key = custom_api_key or self.api_key or settings.OPENROUTER_API_KEY

        # If API key is missing, use enhanced pattern-matching extraction fallback
        if not active_key:
            logger.warning(
                "OPENROUTER_API_KEY is missing. Using heuristic regex extraction fallback."
            )
            return self._heuristic_fallback_extraction(raw_document_text)

        system_prompt = (
            "You are a Senior Indian Banking & Fintech Document Specialist.\n"
            "Your task is to extract exact loan parameters from Indian Bank Sanction Letters or "
            "RBI Key Fact Statements (KFS).\n\n"
            "STRICT ARCHITECTURAL RULE:\n"
            "1. Extract ONLY the exact parameters explicitly written in the document.\n"
            "2. DO NOT perform any mathematical calculations, interest compounding, or rounding.\n"
            "3. If a value (such as processing fee or documentation charges) is not mentioned, return 0.0.\n"
            "4. Convert tenure to months (e.g., '5 Years' -> 60, '15 Years' -> 180, '20 Years' -> 240).\n"
            "5. Convert Lakhs/Crores to numerical INR (e.g. '15 Lakhs' or '15.5 Lac' -> 1550000.0, '1 Crore' -> 10000000.0).\n"
            "6. Distinguish whether interest is stated as 'reducing' balance or 'flat' balance.\n"
            "7. Detect whether any insurance (credit shield, life/health) is added to the principal or bundled.\n"
            "8. Identify any foreclosure/pre-payment penalty clauses verbatim.\n\n"
            "Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "lender_name": "string or null",\n'
            '  "loan_category": "personal | home | vehicle | business | education | other",\n'
            '  "principal_amount": float,\n'
            '  "stated_interest_rate": float,\n'
            '  "interest_type": "reducing | flat | floating | fixed | unknown",\n'
            '  "tenure_months": int,\n'
            '  "stated_emi": float or null,\n'
            '  "processing_fee": float,\n'
            '  "documentation_charges": float,\n'
            '  "bundled_insurance": float,\n'
            '  "is_insurance_capitalized": bool,\n'
            '  "prepayment_penalty_terms": "string or null",\n'
            '  "has_prepayment_penalty": bool\n'
            "}"
        )

        user_content = f"Sanction Letter / KFS Content:\n```markdown\n{raw_document_text[:16000]}\n```"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response_text = await self._call_openrouter(
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                custom_api_key=active_key,
            )
            clean_json = json.loads(response_text)
            extracted = ExtractedLoanTerms(**clean_json)
            extracted.extraction_method = "openrouter_llm"
            return extracted
        except Exception as e:
            logger.error(f"OpenRouter extraction failed: {e}. Falling back to heuristic extractor.")
            fallback = self._heuristic_fallback_extraction(raw_document_text)
            fallback.extraction_method = f"offline_heuristic_fallback (LLM Error: {str(e)[:60]})"
            return fallback

    async def generate_multilingual_explanation(
        self, audit: LoanAuditReport, language: Language, custom_api_key: Optional[str] = None
    ) -> str:
        """
        Explains deterministic audit findings to an Indian borrower.
        Supported languages: English, Hindi, Kannada.
        """
        active_key = custom_api_key or self.api_key or settings.OPENROUTER_API_KEY
        if not active_key:
            return self._local_template_explanation(audit, language)

        audit_summary_for_llm = {
            "lender": audit.lender_name,
            "principal": audit.principal,
            "stated_rate": f"{audit.stated_annual_rate}%",
            "effective_annual_apr": f"{audit.effective_annual_apr}%",
            "rate_spread": f"{audit.rate_inflation_spread}%",
            "stated_emi": audit.stated_emi,
            "calculated_reducing_emi": audit.calculated_reducing_emi,
            "is_emi_matching": audit.is_emi_matching,
            "net_disbursed_amount": audit.net_disbursed_amount,
            "fees_and_18_pct_gst": audit.total_upfront_fees_incl_gst,
            "bundled_insurance": audit.bundled_insurance_amount,
            "total_borrowing_cost": audit.total_cost_of_borrowing,
            "rbi_compliance_score": f"{audit.rbi_compliance_score}/100",
            "discrepancies": [
                {"title": d.title, "description": d.description, "severity": d.severity}
                for d in audit.discrepancies
            ],
        }

        lang_instructions = {
            Language.ENGLISH: "Write the response in clear, empowering, consumer-friendly English.",
            Language.HINDI: "Write the complete response in natural, fluent Hindi (Devanagari script). Use easy-to-understand loan terms like ऋण (loan), किश्त (EMI), वास्तविक ब्याज दर (True APR), अतिरिक्त शुल्क (hidden charges).",
            Language.KANNADA: "Write the complete response in natural, professional Kannada script (ಕನ್ನಡ). Use clear loan concepts such as ಸಾಲ (loan), ಮಾಸಿಕ ಕಂತು (EMI), ನೈಜ ಬಡ್ಡಿದರ (True APR), ಮುಂಗಡ ಕಡಿತಗಳು (upfront deductions).",
        }

        system_prompt = (
            "You are LoanLens Explainer, a trusted Indian financial ombudsman and consumer advocate.\n"
            "You are provided with a 100% mathematically verified loan audit JSON.\n"
            "STRICT RULES:\n"
            "1. NEVER recalculate or alter any figures. Use the exact numbers provided in the audit JSON.\n"
            f"2. {lang_instructions.get(language, lang_instructions[Language.ENGLISH])}\n"
            "3. Format your breakdown into 3 distinct sections with Markdown headings:\n"
            "   - **🚨 The Verdict & Reality Check**: Stated rate vs True APR, net in-hand cash vs sanctioned loan.\n"
            "   - **🔍 Hidden Traps & Fee Breakdown**: 18% GST deductions, bundled insurance, and EMI anomalies.\n"
            "   - **📋 Actionable Checklist for the Borrower**: 3 specific questions or negotiation demands to take to the bank branch manager.\n"
        )

        user_content = f"Verified Deterministic Audit Data:\n```json\n{json.dumps(audit_summary_for_llm, indent=2)}\n```"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            explanation = await self._call_openrouter(
                messages=messages,
                temperature=0.3,
                custom_api_key=active_key,
            )
            return explanation
        except Exception:
            return self._local_template_explanation(audit, language)

    def _heuristic_fallback_extraction(self, text: str) -> ExtractedLoanTerms:
        """Intelligent regex and tabular extraction fallback for Indian loan letters."""
        # 1. Principal Amount (Handle Lakhs, Crores, or raw numbers)
        principal = 500000.0
        # Check for "X Lakhs / Lacs"
        lakh_match = re.search(r"([\d]+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs)", text, re.IGNORECASE)
        crore_match = re.search(r"([\d]+(?:\.\d+)?)\s*(?:crore|crores|cr)", text, re.IGNORECASE)

        if crore_match:
            try:
                principal = float(crore_match.group(1)) * 10000000.0
            except ValueError:
                pass
        elif lakh_match:
            try:
                principal = float(lakh_match.group(1)) * 100000.0
            except ValueError:
                pass
        else:
            p_match = re.search(
                r"(?:loan\s+amount|sanctioned\s+amount|principal|sanction\s+amount|facility\s+amount)[^\d\n\r]*([1-9][\d,]{3,}(?:\.\d+)?)",
                text,
                re.IGNORECASE,
            )
            if p_match:
                try:
                    principal = float(p_match.group(1).replace(",", ""))
                except ValueError:
                    pass

        # 2. Interest Rate (ROI)
        rate = 10.5
        r_match = re.search(
            r"(?:interest\s+rate|rate\s+of\s+interest|roi|applicable\s+rate)[^\d\n\r]*([\d]{1,2}(?:\.\d+)?)\s*%",
            text,
            re.IGNORECASE,
        )
        if r_match:
            try:
                rate = float(r_match.group(1))
            except ValueError:
                pass

        # 3. Tenure (Years vs Months)
        tenure = 36
        year_match = re.search(r"(?:tenure|period|term)[^\d\n\r]*(\d+)\s*(?:years|yrs)", text, re.IGNORECASE)
        month_match = re.search(r"(?:tenure|period|term)[^\d\n\r]*(\d+)\s*(?:months|mths|m)", text, re.IGNORECASE)

        if month_match:
            try:
                tenure = int(month_match.group(1))
            except ValueError:
                pass
        elif year_match:
            try:
                tenure = int(year_match.group(1)) * 12
            except ValueError:
                pass

        # 4. Stated EMI
        emi = None
        e_match = re.search(
            r"(?:emi|installment|monthly\s+installment)[^\d\n\r]*([1-9][\d,]{2,}(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if e_match:
            try:
                emi = float(e_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # 5. Processing Fee (Percentage vs Absolute)
        pf = 0.0
        pf_pct_match = re.search(
            r"(?:processing\s+fee|admin\s+fee|pf)[^\d\n\r]*([\d]+(?:\.\d+)?)\s*%",
            text,
            re.IGNORECASE,
        )
        if pf_pct_match:
            try:
                pct = float(pf_pct_match.group(1))
                pf = round((pct / 100.0) * principal, 2)
            except ValueError:
                pass
        else:
            pf_val_match = re.search(
                r"(?:processing\s+fee|admin\s+fee|pf)[^\d\n\r]*([1-9][\d,]{2,}(?:\.\d+)?)",
                text,
                re.IGNORECASE,
            )
            if pf_val_match:
                try:
                    pf = float(pf_val_match.group(1).replace(",", ""))
                except ValueError:
                    pass

        # 6. Documentation / Legal Charges
        doc_charges = 0.0
        doc_match = re.search(
            r"(?:documentation|stamp\s+duty|legal|cersai|login\s+fee)[^\d\n\r]*([1-9][\d,]{2,}(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if doc_match:
            try:
                doc_charges = float(doc_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # 7. Insurance
        ins = 0.0
        ins_match = re.search(
            r"(?:insurance|shield|protect|credit\s+life)[^\d\n\r]*([1-9][\d,]{2,}(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if ins_match:
            try:
                ins = float(ins_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # 8. Lender Name
        lender = "Indian Financial Institution"
        for bank in [
            "HDFC Bank",
            "State Bank of India",
            "SBI",
            "ICICI Bank",
            "Axis Bank",
            "Kotak Mahindra Bank",
            "Bajaj Finance",
            "Tata Capital",
            "Punjab National Bank",
            "Bank of Baroda",
            "Muthoot Finance",
        ]:
            if bank.lower() in text.lower():
                lender = bank
                break

        # 9. Foreclosure / Prepayment Penalty
        has_penalty = bool(
            re.search(r"foreclosure|pre-?payment\s+penalty|early\s+closure", text, re.IGNORECASE)
        )

        return ExtractedLoanTerms(
            lender_name=lender,
            principal_amount=principal,
            stated_interest_rate=rate,
            interest_type=InterestRateType.REDUCING,
            tenure_months=tenure,
            stated_emi=emi,
            processing_fee=pf,
            documentation_charges=doc_charges,
            bundled_insurance=ins,
            is_insurance_capitalized=False,
            has_prepayment_penalty=has_penalty,
            extraction_method="offline_heuristic_fallback",
        )

    def _local_template_explanation(self, audit: LoanAuditReport, language: Language) -> str:
        """Deterministic template fallback when OPENROUTER_API_KEY is not configured."""
        if language == Language.HINDI:
            return (
                f"### 🚨 ऋण विश्लेषण निष्कर्ष (LoanLens Audit)\n\n"
                f"- **स्वीकृत ऋण राशि**: ₹{audit.principal:,.2f}\n"
                f"- **वास्तविक प्राप्त राशि (Net Disbursal)**: ₹{audit.net_disbursed_amount:,.2f}\n"
                f"- **कथित ब्याज दर**: {audit.stated_annual_rate}% | **वास्तविक लागत (True APR)**: {audit.effective_annual_apr}%\n"
                f"- **कुल ब्याज एवं अतिरिक्त लागत**: ₹{audit.total_cost_of_borrowing:,.2f}\n\n"
                f"### 🔍 अतिरिक्त कटौती एवं शुल्क:\n"
                f"- प्रोसेसिंग शुल्क व 18% GST: ₹{audit.total_upfront_fees_incl_gst:,.2f}\n"
                f"- इंश्योरेंस प्रीमियम: ₹{audit.bundled_insurance_amount:,.2f}\n"
                f"- भारतीय रिजर्व बैंक (RBI) अनुपालन स्कोर: {audit.rbi_compliance_score}/100\n\n"
                f"### 📋 बैंक अधिकारी से पूछने योग्य बातें:\n"
                f"1. क्या इंश्योरेंस लेना अनिवार्य है? (RBI के अनुसार यह स्वैच्छिक होना चाहिए)\n"
                f"2. क्या 18% GST पहले से प्रोसेसिंग फीस में शामिल है या अलग से काटा गया है?\n"
                f"3. फ्लोटिंग रेट लोन पर किसी भी प्रकार का प्रीपेमेंट पेनल्टी शुल्क न लगाएं।"
            )
        elif language == Language.KANNADA:
            return (
                f"### 🚨 ಸಾಲ ಪರಿಶೀಲನಾ ವರದಿ (LoanLens Audit)\n\n"
                f"- **ಮಂಜೂರಾದ ಸಾಲದ ಮೊತ್ತ**: ₹{audit.principal:,.2f}\n"
                f"- **ನಿಮ್ಮ ಕೈಸೇರುವ ನೈಜ ಹಣ (Net In-Hand)**: ₹{audit.net_disbursed_amount:,.2f}\n"
                f"- **ಹೇಳಲಾದ ಬಡ್ಡಿದರ**: {audit.stated_annual_rate}% | **ನೈಜ ವಾರ್ಷಿಕ ಬಡ್ಡಿದರ (True APR)**: {audit.effective_annual_apr}%\n"
                f"- **ಸಾಲದ ಒಟ್ಟು ಹೆಚ್ಚುವರಿ ಹೊರೆ**: ₹{audit.total_cost_of_borrowing:,.2f}\n\n"
                f"### 🔍 ಗುಪ್ತ ಕಡಿತಗಳು ಮತ್ತು ಶುಲ್ಕಗಳು:\n"
                f"- ಪ್ರೊಸೆಸಿಂಗ್ ಶುಲ್ಕ ಮತ್ತು 18% GST: ₹{audit.total_upfront_fees_incl_gst:,.2f}\n"
                f"- ವಿಮೆ ಮೊತ್ತ: ₹{audit.bundled_insurance_amount:,.2f}\n"
                f"- RBI ನಿಯಮಾವಳಿ ಅನುಸರಣೆ ಅಂಕ: {audit.rbi_compliance_score}/100\n\n"
                f"### 📋 ಬ್ಯಾಂಕ್ ಅಧಿಕಾರಿಯೊಂದಿಗೆ ಚರ್ಚಿಸಬೇಕಾದ ಮುಖ್ಯಾಂಶಗಳು:\n"
                f"1. ವಿಮೆ ಕಡ್ಡಾಯವೇ? (ಆರ್‌ಬಿಐ ನಿಯಮಗಳ ಪ್ರಕಾರ ಸಾಲ ವಿಮೆ ಐಚ್ಛಿಕವಾಗಿರುತ್ತದೆ).\n"
                f"2. 18% GST ಯನ್ನು ಪ್ರೊಸೆಸಿಂಗ್ ಶುಲ್ಕದಲ್ಲಿ ಸೇರಿಸಲಾಗಿದೆಯೇ?\n"
                f"3. ಫ್ಲೋಟಿಂಗ್ ದರದ ಸಾಲಕ್ಕೆ ಮುಂಚಿತ ಮರುಪಾವತಿ ದಂಡ ವಿಧಿಸಬಾರದು."
            )
        else:
            return (
                f"### 🚨 The Verdict & Reality Check\n\n"
                f"- **Sanctioned Amount**: ₹{audit.principal:,.2f} | **In-Hand Cash Disbursed**: ₹{audit.net_disbursed_amount:,.2f}\n"
                f"- **Stated Interest Rate**: {audit.stated_annual_rate}% per annum\n"
                f"- **True Effective APR**: **{audit.effective_annual_apr}%** (Inflation spread: +{audit.rate_inflation_spread}%)\n"
                f"- **Total Cost of Borrowing**: ₹{audit.total_cost_of_borrowing:,.2f}\n\n"
                f"### 🔍 Hidden Traps & Fee Breakdown\n"
                f"- Upfront Processing & Documentation Fees: ₹{audit.raw_processing_fee + audit.raw_documentation_charges:,.2f}\n"
                f"- 18% Statutory GST on Banking Charges: ₹{audit.fee_gst_amount:,.2f}\n"
                f"- Bundled Insurance Premium: ₹{audit.bundled_insurance_amount:,.2f}\n"
                f"- RBI Fair Lending Score: **{audit.rbi_compliance_score} / 100**\n\n"
                f"### 📋 Actionable Checklist for the Borrower\n"
                f"1. **Opt-out of Mandatory Insurance**: Request the loan officer to unbundle the ₹{audit.bundled_insurance_amount:,.2f} insurance if you already possess adequate term cover.\n"
                f"2. **Fee Waivers**: Ask for a full or 50% waiver on processing fees to save ₹{audit.total_upfront_fees_incl_gst:,.2f}.\n"
                f"3. **KFS Verification**: Ensure the bank provides the mandatory RBI Key Fact Statement (KFS) matching these true APR numbers."
            )
