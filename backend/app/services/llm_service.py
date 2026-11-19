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

        # If API key is missing or placeholder, use enhanced pattern-matching extraction fallback
        if not active_key or "your_openrouter" in active_key.lower():
            logger.warning(
                "OPENROUTER_API_KEY is missing or placeholder. Using heuristic regex extraction fallback."
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
            clean_str = response_text.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            elif clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            clean_json = json.loads(clean_str.strip())
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
        if not active_key or "your_openrouter" in active_key.lower():
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
            Language.ENGLISH: "Write the response in authoritative, sharp, executive-level English with no corporate fluff.",
            Language.HINDI: "Write the complete response in natural, authoritative Hindi (Devanagari script). Use formal institutional finance terminology: ऋण (loan), किश्त (EMI), वास्तविक ब्याज दर (True APR), अग्रिम कटौती (upfront deduction), वित्तीय विसंगति (financial anomaly).",
            Language.KANNADA: "Write the complete response in formal, executive Kannada script (ಕನ್ನಡ). Use precise institutional financial terminology: ಸಾಲ (loan), ಮಾಸಿಕ ಕಂತು (EMI), ನೈಜ ಪರಿಣಾಮಕಾರಿ ಬಡ್ಡಿದರ (True Effective APR), ಮುಂಗಡ ಕಡಿತಗಳು (upfront deductions).",
        }

        system_prompt = (
            "You are a Senior Finance Manager, Chief Credit Risk Officer, and Elite Indian Banking & Loan Auditor with 20+ years of institutional treasury, corporate finance, and retail lending oversight.\n"
            "You are analyzing a 100% mathematically verified loan audit JSON on behalf of your client (the borrower).\n\n"
            "YOUR OBJECTIVE: Deliver a hard-hitting, sharp, executive-level financial analysis that dismantles deceptive bank terms, exposes mathematical traps, and arms the borrower with institutional-grade negotiation leverage.\n\n"
            "TONE & STYLE GUIDELINES:\n"
            "- Speak with the uncompromising authority, forensic precision, and clinical sharpness of a Senior Director of Finance / Chief Financial Analyst protecting a high-value client.\n"
            "- Call out predatory spreads, hidden margin padding, and unfair deductions with exact rupee impact and basis points.\n"
            "- Never sound timid or generic. Treat every rupee shaved off disbursal or padded into EMI as unacceptable leakages.\n"
            "- Reference statutory guidelines and regulatory directives where applicable (e.g., RBI Fair Lending Guidelines, April 2024 Key Fact Statement (KFS) Directions, RBI Circular RBI/2014-15/168 on floating rate prepayment penalty prohibition, GST SAC 9971 on banking charges).\n\n"
            "STRICT RULES:\n"
            "1. NEVER alter, round differently, or recalculate the numbers. Use the EXACT figures provided in the verified audit JSON.\n"
            f"2. {lang_instructions.get(language, lang_instructions[Language.ENGLISH])}\n"
            "3. Structure your executive report into these 3 mandatory sections with Markdown headings:\n\n"
            "### 🚨 Executive Verdict & Treasury Reality Check\n"
            "Provide an institutional-grade reality check contrasting the advertised nominal rate vs the True Effective APR. Detail the rate inflation spread in basis points/percentage, quantify the exact cash erosion between sanctioned principal and net in-hand cash, and state whether this loan represents acceptable market financing or aggressive lender rent-seeking.\n\n"
            "### 🔍 Forensic Breakdown of Hidden Charges & Traps\n"
            "Examine the 3 primary friction points:\n"
            "- **Statutory 18% GST (SAC 9971)**: Dissect upfront processing and documentation fees + GST impact on capital efficiency.\n"
            "- **Bundled Insurance & Ancillary Fees**: Evaluate credit shield/life covers. If bundled, flag that RBI mandates insurance cannot be made a mandatory tie-in condition for sanction.\n"
            "- **Amortization & EMI Anomalies**: Compare stated EMI vs true reducing EMI. If there is a variance, calculate the cumulative lifetime loss over the full tenure (e.g., ₹X/month × N months = ₹Total Rupee Drainage).\n\n"
            "### 📋 Institutional Negotiation Playbook for the Borrower\n"
            "Provide exactly 3 uncompromising, executive-level scripts for the borrower to demand from the Bank Branch Manager or Chief Credit Underwriter:\n"
            "1. **EMI Restructuring Demand**: An aggressive, mathematically armed question demanding immediate downward adjustment to true reducing amortization.\n"
            "2. **Upfront Fee & GST Clawback**: Script to demand a 50% to 100% processing fee waiver and recalculation of net in-hand disbursal.\n"
            "3. **Regulatory Non-Compliance & RBI Foreclosure Guarantee**: Formal demand invoking RBI Circular RBI/2014-15/168 (or KFS transparency norms) for written certification of zero penalty on prepayment and unbundling of any ancillary insurance.\n"
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
        """Deterministic Senior Finance Manager executive audit report fallback."""
        emi_diff = (audit.stated_emi or 0) - audit.calculated_reducing_emi
        lifetime_emi_loss = emi_diff * audit.tenure_months if emi_diff > 1 else 0

        if language == Language.HINDI:
            return (
                f"### 🚨 आधिकारिक ऋण विश्लेषण निष्कर्ष (LoanLens Forensic Audit)\n\n"
                f"कथित ब्याज दर **{audit.stated_annual_rate}%** के मुकाबले आपकी वास्तविक वार्षिक लागत (**True Effective APR**) **{audit.effective_annual_apr}%** है। "
                f"यह **+{audit.rate_inflation_spread}%** का स्प्रेड बैंक द्वारा छिपाए गए मार्जिन को दर्शाता है।\n\n"
                f"- **स्वीकृत ऋण राशि**: ₹{audit.principal:,.2f}\n"
                f"- **खाते में शुद्ध प्राप्त राशि (Net Disbursed)**: ₹{audit.net_disbursed_amount:,.2f} "
                f"(₹{audit.principal - audit.net_disbursed_amount:,.2f} की सीधी कटौती)\n"
                f"- **ऋण की कुल वास्तविक लागत**: ₹{audit.total_cost_of_borrowing:,.2f}\n"
                f"- **RBI अनुपालन रेटिंग**: **{audit.rbi_compliance_score}/100**\n\n"
                f"### 🔍 फोरेंसिक शुल्क विश्लेषण एवं वित्तीय विसंगतियां (Forensic Breakdown)\n\n"
                f"- **18% वैधानिक GST (SAC 9971)**: ₹{audit.raw_processing_fee + audit.raw_documentation_charges:,.2f} के बैंक शुल्क पर ₹{audit.fee_gst_amount:,.2f} का अतिरिक्त GST काटा गया।\n"
                f"- **बंडल इंश्योरेंस प्रीमियम**: ₹{audit.bundled_insurance_amount:,.2f} "
                f"(आरबीआई निर्देशों के तहत ऋण बीमा अनिवार्य नहीं हो सकता, यह पूर्णतः स्वैच्छिक है)।\n"
                + (f"- **EMI विसंगति (EMI Inflation)**: बैंक द्वारा मांगी गई ₹{audit.stated_emi:,.2f} की EMI गणितीय रूप से सही रिड्यूसिंग EMI (₹{audit.calculated_reducing_emi:,.2f}) से **₹{emi_diff:,.2f} प्रति माह अधिक** है! "
                   f"{audit.tenure_months} महीनों में यह कुल **₹{lifetime_emi_loss:,.2f} का अतिरिक्त वित्तीय नुकसान** है।\n\n" if emi_diff > 5 else "- **EMI मिलान**: स्वीकृत EMI रिड्यूसिंग बैलेंस फॉर्मूले से मेल खाती है।\n\n") +
                f"### 📋 बैंक शाखा प्रबंधक के समक्ष वार्ता रणनीति (Negotiation Playbook)\n\n"
                f"1. **EMI संशोधन की मांग**: \"बैंक की ₹{audit.stated_emi or audit.calculated_reducing_emi:,.2f} की EMI वास्तविक रिड्यूसिंग बैलेंस गणना (₹{audit.calculated_reducing_emi:,.2f}) से मेल नहीं खाती। कृपया इसे तुरंत संशोधित करें।\"\n"
                f"2. **प्रोसेसिंग फीस में छूट की मांग**: \"ऋण की वास्तविक APR {audit.effective_annual_apr}% तक पहुंच रही है। ₹{audit.total_upfront_fees_incl_gst:,.2f} के अग्रिम शुल्क पर 50% से 100% की छूट प्रदान करें।\"\n"
                f"3. **RBI सर्कुलर RBI/2014-15/168 का अनुपालन**: \"फ्लोटिंग रेट व्यक्तिगत ऋण पर प्रीपेमेंट या फोरक्लोजर पेनल्टी लगाना आरबीआई नियमों का उल्लंघन है। मुझे लिखित में शून्य पेनल्टी की पुष्टि दें।\""
            )
        elif language == Language.KANNADA:
            return (
                f"### 🚨 ಅಧಿಕೃತ ಸಾಲ ವಿಧಿವಿಜ್ಞಾನ ವರದಿ (LoanLens Forensic Audit)\n\n"
                f"ಬ್ಯಾಂಕ್ ಘೋಷಿಸಿದ ಬಡ್ಡಿದರ **{audit.stated_annual_rate}%** ಆಗಿದ್ದರೂ, ನಿಮ್ಮ ನೈಜ ಪರಿಣಾಮಕಾರಿ ವಾರ್ಷಿಕ ಬಡ್ಡಿದರ (**True Effective APR**) **{audit.effective_annual_apr}%** ಆಗಿದೆ. "
                f"ಇದು **+{audit.rate_inflation_spread}%** ನಷ್ಟು ದರ ಹಣದುಬ್ಬರವನ್ನು ತೋರಿಸುತ್ತದೆ.\n\n"
                f"- **ಮಂಜೂರಾದ ಒಟ್ಟು ಸಾಲ**: ₹{audit.principal:,.2f}\n"
                f"- **ನಿಮ್ಮ ಕೈಸೇರುವ ನಿವ್ವಳ ಮೊತ್ತ (Net Disbursed)**: ₹{audit.net_disbursed_amount:,.2f} (ಕಡಿತ: ₹{audit.principal - audit.net_disbursed_amount:,.2f})\n"
                f"- **ಸಾಲದ ಒಟ್ಟು ನೈಜ ಮರುಪಾವತಿ ವೆಚ್ಚ**: ₹{audit.total_cost_of_borrowing:,.2f}\n"
                f"- **ಆರ್‌ಬಿಐ ನ್ಯಾಯಯುತ ಸಾಲ ಅನುಸರಣೆ ಅಂಕ**: **{audit.rbi_compliance_score}/100**\n\n"
                f"### 🔍 ಗುಪ್ತ ಶುಲ್ಕಗಳ ಪರಿಶೀಲನೆ (Forensic Breakdown)\n\n"
                f"- **18% ಶಾಸನಬದ್ಧ GST (SAC 9971)**: ₹{audit.raw_processing_fee + audit.raw_documentation_charges:,.2f} ಶುಲ್ಕಕ್ಕೆ ₹{audit.fee_gst_amount:,.2f} GST ಕಡಿತ ಮಾಡಲಾಗಿದೆ.\n"
                f"- **ಬಂಡಲ್ ವಿಮೆ ಕಡಿತ**: ₹{audit.bundled_insurance_amount:,.2f} (ಆರ್‌ಬಿಐ ನಿಯಮಗಳ ಪ್ರಕಾರ ಸಾಲ ವಿಮೆ ಐಚ್ಛಿಕ, ಕಡ್ಡಾಯವಲ್ಲ).\n"
                + (f"- **EMI ಅಸಂಗತತೆ**: ಮಂಜೂರಾದ ಮಾಸಿಕ EMI ₹{audit.stated_emi:,.2f} ನೈಜ ರಿಡ್ಯೂಸಿಂಗ್ EMI ₹{audit.calculated_reducing_emi:,.2f} ಗಿಂತ ತಿಂಗಳಿಗೆ **₹{emi_diff:,.2f} ಹೆಚ್ಚಾಗಿದೆ**! {audit.tenure_months} ತಿಂಗಳುಗಳಲ್ಲಿ ಇದು ಒಟ್ಟು **₹{lifetime_emi_loss:,.2f} ಹೆಚ್ಚುವರಿ ನಷ್ಟ**!\n\n" if emi_diff > 5 else "- **EMI ಲೆಕ್ಕಾಚಾರ**: ಮಾಸಿಕ EMI ರಿಡ್ಯೂಸಿಂಗ್ ದರದೊಂದಿಗೆ ಹೊಂದಿಕೆಯಾಗುತ್ತದೆ.\n\n") +
                f"### 📋 ಬ್ಯಾಂಕ್ ಮ್ಯಾನೇಜರ್ ಅವರೊಂದಿಗೆ ಮಾತುಕತೆಯ ಕಾರ್ಯತಂತ್ರ (Negotiation Playbook)\n\n"
                f"1. **EMI ಮರುಲೆಕ್ಕಾಚಾರ**: \"ನೈಜ ರಿಡ್ಯೂಸಿಂಗ್ ಲೆಕ್ಕದ ಪ್ರಕಾರ ₹{audit.calculated_reducing_emi:,.2f} ಆಗಬೇಕಿರುವ EMI ಅನ್ನು ಸರಿಪಡಿಸಿ.\"\n"
                f"2. **ಪ್ರೊಸೆಸಿಂಗ್ ಶುಲ್ಕ ಮನ್ನಾ**: \"ಒಟ್ಟು ವೆಚ್ಚ ಕಡಿಮೆ ಮಾಡಲು ₹{audit.total_upfront_fees_incl_gst:,.2f} ಶುಲ್ಕದಲ್ಲಿ ಕನಿಷ್ಠ 50% ರಿಯಾಯಿತಿ ನೀಡಿ.\"\n"
                f"3. **ಆರ್‌ಬಿಐ ಸುತ್ತೋಲೆ RBI/2014-15/168 ಖಾತರಿ**: \"ಫ್ಲೋಟಿಂಗ್ ದರದ ಸಾಲಕ್ಕೆ ಯಾವುದೇ ಮುಂಚಿತ ತೀರುವಳಿ ಶುಲ್ಕ ಇರುವುದಿಲ್ಲ ಎಂದು ಲಿಖಿತ ಭರವಸೆ ನೀಡಿ.\""
            )
        else:
            return (
                f"### 🚨 Executive Audit Verdict & Reality Check\n\n"
                f"Based on our verified financial audit and forensic risk analysis, the numbers indicate a significant divergence from the headline offer. "
                f"While the bank marketed an annual interest rate of **{audit.stated_annual_rate:.2f}%**, your **True Effective APR (via Newton-Raphson IRR) is {audit.effective_annual_apr:.2f}%**. "
                f"This represents an aggressive **+{audit.rate_inflation_spread:.2f}% rate spread padding** in the lender's favor.\n\n"
                f"- **Sanctioned Facility Amount**: ₹{audit.principal:,.2f}\n"
                f"- **Net Capital Realized (In-Hand Cash)**: **₹{audit.net_disbursed_amount:,.2f}** (Immediate cash leakage of ₹{audit.principal - audit.net_disbursed_amount:,.2f})\n"
                f"- **Total Lifetime Borrowing Cost**: ₹{audit.total_cost_of_borrowing:,.2f}\n"
                f"- **Institutional RBI Compliance Score**: **{audit.rbi_compliance_score} / 100**\n\n"
                f"### 🔍 Forensic Breakdown of Hidden Charges & Traps\n\n"
                f"- **18% Statutory GST (SAC 9971)**: The lender levies ₹{audit.raw_processing_fee + audit.raw_documentation_charges:,.2f} in origination charges, attracting ₹{audit.fee_gst_amount:,.2f} in GST. This immediately reduces your working capital at day zero.\n"
                f"- **Bundled Insurance Policy**: ₹{audit.bundled_insurance_amount:,.2f} deducted upfront. *Regulatory note*: Under RBI fair-practice directives, credit-life insurance cannot be legally mandated as a pre-condition for loan sanction.\n"
                + (f"- **EMI Amortization Discrepancy**: The sanctioned EMI of ₹{audit.stated_emi:,.2f} exceeds the true actuarial reducing EMI of ₹{audit.calculated_reducing_emi:,.2f} by **₹{emi_diff:,.2f} per month**. Over the {audit.tenure_months}-month tenure, this quietly drains **₹{lifetime_emi_loss:,.2f}** from your net worth.\n\n" if emi_diff > 5 else f"- **EMI Verification**: Stated EMI (₹{audit.stated_emi or audit.calculated_reducing_emi:,.2f}) matches exact mathematical reducing balance amortization.\n\n") +
                f"### 📋 Institutional Negotiation Playbook for the Borrower\n\n"
                f"Take these exact, non-negotiable points to the Bank Branch Manager or Chief Credit Underwriter:\n\n"
                f"1. **Rectify EMI Calculation**: *\"Our independent audit shows the actuarial reducing balance EMI on ₹{audit.principal:,.2f} at {audit.stated_annual_rate}% is ₹{audit.calculated_reducing_emi:,.2f}, not ₹{audit.stated_emi or audit.calculated_reducing_emi:,.2f}. Adjust our repayment schedule immediately before disbursal.\"*\n"
                f"2. **Demand 50%–100% Fee Clawback**: *\"Because the True APR reaches {audit.effective_annual_apr:.2f}%, we require a minimum 50% waiver on the ₹{audit.raw_processing_fee + audit.raw_documentation_charges:,.2f} upfront charges to bring the effective cost in line with market benchmarks.\"*\n"
                f"3. **Zero Foreclosure Guarantee (RBI Circular RBI/2014-15/168)**: *\"Under RBI mandate, banks are strictly barred from levying prepayment or foreclosure charges on individual floating-rate loans. Provide explicit written confirmation in the sanction annexure that prepayment carries 0% penalty.\"*"
            )
