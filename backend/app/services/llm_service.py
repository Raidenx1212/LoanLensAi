import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
import httpx
from app.schemas.loan import (
    ExtractedLoanTerms,
    ExtractedFieldInfo,
    ExtractedCharge,
    FieldStatus,
    ChargeCategory,
    LoanAuditReport,
    Language,
    InterestRateType,
    LoanCategory,
)
from app.config import settings

logger = logging.getLogger(__name__)

# Resilient model aliases mapping to active OpenRouter models
MODEL_ALIASES = {
    "anthropic/claude-3.5-sonnet": "google/gemini-2.5-flash",
    "anthropic/claude-3-5-sonnet": "google/gemini-2.5-flash",
    "mistralai/mistral-large-2411": "mistralai/mistral-large-2512",
    "google/gemini-2.0-flash-001": "google/gemini-2.5-flash",
    "google/gemini-2.0-flash": "google/gemini-2.5-flash",
}

WORDS_NUM_MAP = {
    "ninety five": 95, "eighty five": 85, "seventy five": 75, "sixty five": 65,
    "fifty five": 55, "forty five": 45, "thirty five": 35, "twenty five": 25,
    "nineteen": 19, "eighteen": 18, "seventeen": 17, "sixteen": 16, "fifteen": 15,
    "fourteen": 14, "thirteen": 13, "twelve": 12, "eleven": 11,
    "ninety": 90, "eighty": 80, "seventy": 70, "sixty": 60, "fifty": 50,
    "forty": 40, "thirty": 30, "twenty": 20, "ten": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


class OpenRouterLLMService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model or getattr(settings, "DEFAULT_LLM_MODEL", "google/gemini-2.5-flash")

    async def _call_openrouter(
        self,
        messages: list,
        temperature: float = 0.0,
        response_format: dict = None,
        custom_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        key = custom_api_key or self.api_key or settings.OPENROUTER_API_KEY
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")

        requested_model = model_name or self.model
        chosen_model = MODEL_ALIASES.get(requested_model, requested_model)

        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://loanlens.ai",
            "X-Title": "LoanLens AI Auditor",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=5.0)) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def extract_loan_parameters(
        self,
        raw_document_text: str,
        custom_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        pages: Optional[List[Dict[str, Any]]] = None,
    ) -> ExtractedLoanTerms:
        """
        Extracts structured loan terms across all pages, tables, annexures, and footnotes.
        Zero False Zeros & Zero Hallucination guarantee:
        Missing or unstated values MUST be None with status NOT_FOUND, never 0.0.
        """
        parsed_pages = pages or self._split_text_into_pages(raw_document_text)
        active_key = custom_api_key or self.api_key or settings.OPENROUTER_API_KEY

        if not active_key or "your_openrouter" in active_key.lower():
            logger.info("Using deterministic multi-page forensic extractor (no active OpenRouter key).")
            return self._deterministic_page_extract(parsed_pages)

        system_prompt = (
            "You are a Forensic Financial Document Extractor for Indian Banking documents.\n"
            "Extract verbatim loan terms, charges, and repayment schedule cues across ALL pages, "
            "tables, annexures, paragraphs, and footnotes.\n\n"
            "CRITICAL ANTI-HALLUCINATION & NO-FALSE-ZERO RULES:\n"
            "1. Extract ONLY values explicitly written in the document. NEVER guess, estimate, or compute missing numbers.\n"
            "2. NEVER convert a missing field into 0 or 0.0. If a value is NOT stated, return null with status 'NOT_FOUND'.\n"
            "3. Only return 0 or 0.0 if the document explicitly states '0', 'Zero', or 'Nil'.\n"
            "4. Distinguish Sanctioned Principal vs Opening Financed Balance (e.g. principal + financed insurance).\n"
            "5. Distinguish Interest Rate vs Reference/Benchmark Rate or Penal/Prepayment %.\n"
            "6. Separate ALL charges across all pages/footnotes into categories:\n"
            "   - 'confirmed_upfront': confirmed upfront cash fees/recoveries (processing fee, other cash recoveries, CERSAI, valuation, etc.)\n"
            "   - 'financed': insurance or fees capitalized/financed into loan balance\n"
            "   - 'conditional': conditional/contingent fees (e.g. mandate/NACH setup or bounce fee applicable conditionally)\n"
            "   - 'third_party': third-party charges payable outside lender deductions\n"
            "7. GST: Extract stated_gst_amount ONLY if an exact rupee GST amount is written; extract stated_gst_rate_pct ONLY if an explicit % like '18% GST' is written. If it only says 'plus applicable GST', return null for both and put 'plus applicable GST' in gst_status_note. NEVER assume 18%.\n"
            "8. Net Disbursement: Extract stated_net_disbursement ONLY if explicitly stated. Otherwise return null.\n"
            "9. Every field and charge MUST include source_page (int) and verbatim evidence_snippet.\n\n"
            "Return ONLY valid JSON matching:\n"
            "{\n"
            '  "lender_name": {"value": "string or null", "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "loan_category": "personal | home | vehicle | business | education | other",\n'
            '  "principal_amount": {"value": float_or_null, "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "opening_financed_balance": {"value": float_or_null, "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "stated_interest_rate": {"value": float_or_null, "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "interest_type": "reducing | flat | floating | fixed | unknown",\n'
            '  "tenure_months": {"value": int_or_null, "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "stated_emi": {"value": float_or_null, "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "stated_gst_amount": {"value": float_or_null, "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "stated_gst_rate_pct": {"value": float_or_null, "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "gst_status_note": "string or null",\n'
            '  "stated_net_disbursement": {"value": float_or_null, "status": "CONFIRMED|NOT_FOUND|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string or null"},\n'
            '  "prepayment_penalty_terms": "string or null",\n'
            '  "has_prepayment_penalty": true_false_or_null,\n'
            '  "schedule_instalment_count": int_or_null,\n'
            '  "charges": [\n'
            '    {"name": "string", "amount": float_or_null, "category": "confirmed_upfront|conditional|financed|third_party", "status": "CONFIRMED|UNCERTAIN", "source_page": int_or_null, "evidence_snippet": "string", "tax_note": "string or null"}\n'
            "  ]\n"
            "}"
        )

        user_content = f"Sanction Letter / KFS Content:\n```markdown\n{raw_document_text[:60000]}\n```"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            chosen_model = model_name or self.model
            response_text = await self._call_openrouter(
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                custom_api_key=active_key,
                model_name=chosen_model,
            )
            clean_str = response_text.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            elif clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            raw_json = json.loads(clean_str.strip())

            # Parse LLM response and cross-check with deterministic page extractor
            llm_terms = self._build_terms_from_llm_json(raw_json, chosen_model)
            det_terms = self._deterministic_page_extract(parsed_pages)
            return self._merge_llm_and_deterministic(llm_terms, det_terms)
        except Exception as e:
            logger.error(f"OpenRouter extraction failed: {e}. Using deterministic page extractor.")
            fallback = self._deterministic_page_extract(parsed_pages)
            fallback.extraction_method = f"deterministic_extractor (LLM fallback: {str(e)[:50]})"
            return fallback

    def _build_terms_from_llm_json(self, raw_json: Dict[str, Any], chosen_model: str) -> ExtractedLoanTerms:
        """Converts structured LLM JSON (nested {value, status, source_page, evidence_snippet} or flat) into ExtractedLoanTerms."""
        field_metadata: Dict[str, ExtractedFieldInfo] = {}

        def unwrap(field_name: str, default=None):
            entry = raw_json.get(field_name)
            if isinstance(entry, dict) and ("value" in entry or "status" in entry):
                val = entry.get("value", default)
                raw_status = str(entry.get("status") or ("CONFIRMED" if val is not None else "NOT_FOUND")).upper()
                try:
                    status = FieldStatus(raw_status)
                except ValueError:
                    status = FieldStatus.CONFIRMED if val is not None else FieldStatus.NOT_FOUND
                if val is None and status == FieldStatus.CONFIRMED:
                    status = FieldStatus.NOT_FOUND
                field_metadata[field_name] = ExtractedFieldInfo(
                    status=status,
                    source_page=entry.get("source_page"),
                    evidence_snippet=entry.get("evidence_snippet"),
                )
                return val
            val = entry if entry is not None else default
            field_metadata[field_name] = ExtractedFieldInfo(
                status=FieldStatus.CONFIRMED if val is not None else FieldStatus.NOT_FOUND
            )
            return val

        lender = unwrap("lender_name")
        principal = unwrap("principal_amount")
        opening_bal = unwrap("opening_financed_balance")
        rate = unwrap("stated_interest_rate")
        tenure = unwrap("tenure_months")
        emi = unwrap("stated_emi")
        gst_amt = unwrap("stated_gst_amount")
        gst_rate = unwrap("stated_gst_rate_pct")
        net_disb = unwrap("stated_net_disbursement")

        charges: List[ExtractedCharge] = []
        for ch in raw_json.get("charges") or []:
            if not isinstance(ch, dict):
                continue
            try:
                cat = ChargeCategory(str(ch.get("category", "confirmed_upfront")).lower())
            except ValueError:
                cat = ChargeCategory.CONFIRMED_UPFRONT
            try:
                st = FieldStatus(str(ch.get("status", "CONFIRMED")).upper())
            except ValueError:
                st = FieldStatus.CONFIRMED
            amt = ch.get("amount")
            if isinstance(amt, str):
                amt = self._parse_plain_number(amt)
            charges.append(
                ExtractedCharge(
                    name=str(ch.get("name") or "Charge"),
                    amount=float(amt) if amt is not None else None,
                    category=cat,
                    status=st,
                    source_page=ch.get("source_page"),
                    evidence_snippet=ch.get("evidence_snippet"),
                    tax_note=ch.get("tax_note"),
                )
            )

        upfront_charges = [c for c in charges if c.category == ChargeCategory.CONFIRMED_UPFRONT and c.amount is not None]
        financed_charges = [c for c in charges if c.category == ChargeCategory.FINANCED and c.amount is not None]

        # Derive processing/other totals strictly from extracted confirmed upfront items (None if none found)
        pf_val = None
        other_val = None
        doc_val = None
        for c in upfront_charges:
            lname = c.name.lower()
            if "processing" in lname:
                pf_val = (pf_val or 0.0) + c.amount
            elif any(k in lname for k in ("doc", "stamp", "legal", "cersai")):
                doc_val = (doc_val or 0.0) + c.amount
            else:
                other_val = (other_val or 0.0) + c.amount

        ins_val = sum(c.amount for c in financed_charges) if financed_charges else None
        is_ins_cap = bool(financed_charges)

        # Fallback to flat legacy keys if LLM did not populate charges array
        if pf_val is None and raw_json.get("processing_fee") is not None:
            pf_val = raw_json.get("processing_fee")
        if doc_val is None and raw_json.get("documentation_charges") is not None:
            doc_val = raw_json.get("documentation_charges")
        if ins_val is None and raw_json.get("bundled_insurance") is not None:
            ins_val = raw_json.get("bundled_insurance")
            is_ins_cap = bool(raw_json.get("is_insurance_capitalized", False))

        missing = [
            fname for fname in ("principal_amount", "stated_interest_rate", "tenure_months")
            if field_metadata.get(fname, ExtractedFieldInfo()).status in (FieldStatus.NOT_FOUND, FieldStatus.UNCERTAIN)
        ]

        return ExtractedLoanTerms(
            lender_name=lender,
            loan_category=raw_json.get("loan_category", "personal"),
            principal_amount=principal,
            opening_financed_balance=opening_bal,
            stated_interest_rate=rate,
            interest_type=raw_json.get("interest_type", "unknown"),
            tenure_months=tenure,
            stated_emi=emi,
            processing_fee=pf_val,
            documentation_charges=doc_val,
            other_cash_recoveries=other_val,
            bundled_insurance=ins_val,
            is_insurance_capitalized=is_ins_cap,
            stated_gst_amount=gst_amt,
            stated_gst_rate_pct=gst_rate,
            stated_net_disbursement=net_disb,
            gst_status_note=raw_json.get("gst_status_note"),
            prepayment_penalty_terms=raw_json.get("prepayment_penalty_terms"),
            has_prepayment_penalty=raw_json.get("has_prepayment_penalty"),
            charges=charges,
            schedule_instalment_count=raw_json.get("schedule_instalment_count"),
            field_metadata=field_metadata,
            is_verified=len(missing) == 0,
            missing_fields=missing,
            extraction_method=f"ai_model ({chosen_model})",
        )

    def _merge_llm_and_deterministic(
        self, llm_terms: ExtractedLoanTerms, det_terms: ExtractedLoanTerms
    ) -> ExtractedLoanTerms:
        """
        Backfills any core fields or charges missed by the LLM using verified deterministic matches,
        and preserves evidence snippets + page numbers without inventing missing data.
        """
        core_fields = (
            "principal_amount",
            "opening_financed_balance",
            "stated_interest_rate",
            "tenure_months",
            "stated_emi",
            "stated_gst_amount",
            "stated_gst_rate_pct",
            "stated_net_disbursement",
        )
        for fname in core_fields:
            llm_val = getattr(llm_terms, fname)
            det_val = getattr(det_terms, fname)
            llm_meta = llm_terms.field_metadata.get(fname)
            det_meta = det_terms.field_metadata.get(fname)

            if llm_val is None and det_val is not None:
                setattr(llm_terms, fname, det_val)
                if det_meta:
                    llm_terms.field_metadata[fname] = det_meta
            elif llm_val is not None and llm_meta and not llm_meta.evidence_snippet and det_meta and det_meta.evidence_snippet:
                llm_meta.source_page = llm_meta.source_page or det_meta.source_page
                llm_meta.evidence_snippet = det_meta.evidence_snippet

        if llm_terms.interest_type == InterestRateType.UNKNOWN and det_terms.interest_type != InterestRateType.UNKNOWN:
            llm_terms.interest_type = det_terms.interest_type
            if "interest_type" in det_terms.field_metadata:
                llm_terms.field_metadata["interest_type"] = det_terms.field_metadata["interest_type"]

        if not llm_terms.charges and det_terms.charges:
            llm_terms.charges = det_terms.charges
            llm_terms.processing_fee = det_terms.processing_fee
            llm_terms.documentation_charges = det_terms.documentation_charges
            llm_terms.other_cash_recoveries = det_terms.other_cash_recoveries
            llm_terms.bundled_insurance = det_terms.bundled_insurance
            llm_terms.is_insurance_capitalized = det_terms.is_insurance_capitalized

        if not llm_terms.gst_status_note and det_terms.gst_status_note:
            llm_terms.gst_status_note = det_terms.gst_status_note

        if llm_terms.schedule_instalment_count is None and det_terms.schedule_instalment_count is not None:
            llm_terms.schedule_instalment_count = det_terms.schedule_instalment_count
            llm_terms.schedule_emis = det_terms.schedule_emis

        missing = [
            fname for fname in ("principal_amount", "stated_interest_rate", "tenure_months")
            if getattr(llm_terms, fname) is None
        ]
        llm_terms.missing_fields = missing
        llm_terms.is_verified = len(missing) == 0
        return llm_terms

    # =========================================================================
    # GENERAL MULTI-PAGE DETERMINISTIC EXTRACTOR (ZERO FALSE ZEROS)
    # =========================================================================

    def _split_text_into_pages(self, text: str) -> List[Dict[str, Any]]:
        """Splits combined document text into page-indexed records using --- Page N --- headers."""
        if not text:
            return []
        markers = list(re.finditer(r"---\s*Page\s+(\d+)[^\n]*---", text, re.IGNORECASE))
        if not markers:
            return [{"page": 1, "text": text}]

        page_buckets: Dict[int, List[str]] = {}
        for idx, match in enumerate(markers):
            page_num = int(match.group(1))
            start = match.end()
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                page_buckets.setdefault(page_num, []).append(chunk)

        return [{"page": p, "text": "\n".join(chunks)} for p, chunks in sorted(page_buckets.items())]

    @staticmethod
    def _normalize_text(text: str) -> str:
        norm = re.sub(r"[\u20B9₹]|`(?=\s*\d)|\bRs\.?\s*|\bINR\b\s*", "INR ", text)
        norm = re.sub(r"(\d)\s+,\s*(\d)", r"\1,\2", norm)
        norm = re.sub(r"(\d),\s+(\d)", r"\1,\2", norm)
        norm = re.sub(r"(\d)\s+,\s*(\d)", r"\1,\2", norm)
        norm = re.sub(r"(\d)\s*\.\s*(\d+)", r"\1.\2", norm)
        return norm

    @staticmethod
    def _parse_plain_number(raw: str) -> Optional[float]:
        if raw is None:
            return None
        cleaned = re.sub(r"[^\d.]", "", str(raw))
        if not cleaned or cleaned == ".":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    @classmethod
    def _extract_inr_from_window(cls, window: str) -> Optional[float]:
        """Parses numeric INR figures or Indian word denominations from a short text snippet."""
        # Prefer explicit INR/currency-tagged or comma-grouped or decimal figures
        m = re.search(r"INR\s*([\d,]+(?:\.\d+)?)", window, re.IGNORECASE)
        if m:
            val = cls._parse_plain_number(m.group(1))
            if val is not None:
                # Check if immediately followed by lakh/crore multiplier
                after = window[m.end():m.end() + 20].lower()
                if re.match(r"\s*(?:lakhs?|lacs?)\b", after):
                    return val * 100000.0
                if re.match(r"\s*(?:crores?|cr)\b", after):
                    return val * 10000000.0
                return val

        # Fallback: numeric value (must not be followed by %, days, months, years)
        m2 = re.search(r"(?<!\d)(\d[\d,]*(?:\.\d+)?)(?!\s*(?:%|days?\b|months?\b|years?\b|yrs?\b|mths?\b))", window, re.IGNORECASE)
        if m2:
            val = cls._parse_plain_number(m2.group(1))
            if val is not None:
                after = window[m2.end():m2.end() + 20].lower()
                if re.match(r"\s*(?:lakhs?|lacs?)\b", after):
                    return val * 100000.0
                if re.match(r"\s*(?:crores?|cr)\b", after):
                    return val * 10000000.0
                return val

        # Word-based Indian denominations e.g. "Forty Five Lakh"
        for w, num in WORDS_NUM_MAP.items():
            if re.search(rf"\b{w}\s+(?:lakhs?|lacs?)\b", window, re.IGNORECASE):
                return num * 100000.0
            if re.search(rf"\b{w}\s+(?:crores?|cr)\b", window, re.IGNORECASE):
                return num * 10000000.0
        return None

    def _heuristic_fallback_extraction(self, text: str) -> ExtractedLoanTerms:
        """Backwards-compatible entry point delegating to the page-aware deterministic extractor."""
        return self._deterministic_page_extract(self._split_text_into_pages(text))

    def _deterministic_page_extract(self, pages: List[Dict[str, Any]]) -> ExtractedLoanTerms:
        """
        General multi-page deterministic extractor scanning every page, table, annexure,
        paragraph, and footnote. Never fabricates values or turns missing fields into zero.
        """
        field_metadata: Dict[str, ExtractedFieldInfo] = {}
        full_norm = "\n\n".join(self._normalize_text(p["text"]) for p in pages)

        # 1. Lender Name & Category
        lender = None
        lender_match = re.search(
            r"(Hero\s+Housing\s+Finance(?:\s+Limited)?|HHFL|State\s+Bank\s+of\s+India|SBI|HDFC(?:\s+Bank(?:\s+Limited)?)?|ICICI(?:\s+Bank(?:\s+Limited)?)?|Axis\s+Bank|Bajaj\s+Finance|Tata\s+Capital|Kotak(?:\s+Mahindra(?:\s+Bank)?)?|Bank\s+of\s+Baroda|Punjab\s+National\s+Bank|PNB(?:\s+Housing)?|LIC\s+Housing|Canara\s+Bank|Union\s+Bank|IndusInd|IDFC\s+FIRST|Piramal|Aavas|Home\s+First|Lendingkart|Fedbank|Aditya\s+Birla\s+Finance|L&T\s+Finance|Godrej\s+Capital|Fullerton|SMFG\s+India)",
            full_norm,
            re.IGNORECASE,
        )
        if lender_match:
            raw_l = lender_match.group(1).strip()
            lender = "Hero Housing Finance Limited" if raw_l.upper() == "HHFL" else raw_l
        else:
            # Generic fallback: "<Word> <Word> Bank / Finance / Housing / Capital Limited"
            gen_lender = re.search(
                r"\b([A-Z][A-Za-z&\s]{2,40}?(?:Bank|Finance|Housing\s+Finance|Capital|Financial\s+Services)(?:\s+Limited|\s+Ltd\.?)?)\b",
                full_norm,
            )
            if gen_lender:
                lender = gen_lender.group(1).strip()

        category = LoanCategory.PERSONAL
        if re.search(r"\bhome\s*loan\b|\bhousing\s*loan\b", full_norm, re.IGNORECASE):
            category = LoanCategory.HOME
        elif re.search(r"\bauto\s*loan\b|\bvehicle\s*loan\b|\bcar\s*loan\b|\btwo[-\s]*wheeler\b", full_norm, re.IGNORECASE):
            category = LoanCategory.VEHICLE
        elif re.search(r"\bbusiness\s*loan\b|\bmsme\b|\bworking\s*capital\b", full_norm, re.IGNORECASE):
            category = LoanCategory.BUSINESS
        elif re.search(r"\beducation\s*loan\b", full_norm, re.IGNORECASE):
            category = LoanCategory.EDUCATION
        elif re.search(r"\bloan\s+against\s+property\b|\blap\b", full_norm, re.IGNORECASE):
            category = LoanCategory.OTHER

        # Helper to scan lines across all pages with a 2-line window
        def scan_pages_for_amount(
            label_patterns: List[str],
            exclude_pattern: Optional[str] = None,
            min_val: float = 0.0,
        ) -> Tuple[Optional[float], ExtractedFieldInfo]:
            for p in pages:
                p_num = p["page"]
                lines = [ln.strip() for ln in self._normalize_text(p["text"]).splitlines() if ln.strip()]
                for i, line in enumerate(lines):
                    if exclude_pattern and re.search(exclude_pattern, line, re.IGNORECASE):
                        continue
                    for pat in label_patterns:
                        m = re.search(pat, line, re.IGNORECASE)
                        if m:
                            snippet = line
                            search_win = line[m.end():]
                            if (
                                not re.search(r"\d|lakh|crore|nil|zero", search_win, re.IGNORECASE)
                                and i + 1 < len(lines)
                                and not re.match(r"^(?:\d+\.|[A-Z]\.|[-•*]|\(?[a-z]\)|footnote|note\b)", lines[i + 1], re.IGNORECASE)
                                and not (exclude_pattern and re.search(exclude_pattern, lines[i + 1], re.IGNORECASE))
                            ):
                                search_win += " " + lines[i + 1]
                                snippet = line + " | " + lines[i + 1]
                            val = self._extract_inr_from_window(search_win)
                            if val is None and re.search(r"\b(?:nil|zero)\b", search_win, re.IGNORECASE):
                                val = 0.0
                            if val is not None and val >= min_val:
                                return val, ExtractedFieldInfo(
                                    status=FieldStatus.CONFIRMED,
                                    source_page=p_num,
                                    evidence_snippet=snippet[:240],
                                )
            return None, ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)

        # 2. Principal Amount
        principal, principal_info = scan_pages_for_amount(
            label_patterns=[
                r"(?:sanctioned\s*(?:loan\s*)?principal|loan\s*amount\s*(?:approved|sanctioned)|approved\s*loan\s*amount|sanctioned\s*(?:loan\s*)?amount|facility\s*amount|principal\s*(?:loan\s*)?amount|sanction\s*amount|base\s*loan\s*amount)",
                r"(?:^|\|\s*)(?:principal|loan\s*amount)\b(?:\s*[:|\-])",
            ],
            exclude_pattern=r"principal\s*outstanding|opening\s*financed|financed\s*balance|capitalis?ed|prepayment\s*amount|total\s*financed",
            min_val=1000.0,
        )
        field_metadata["principal_amount"] = principal_info

        # 3. Opening Financed Balance (distinct from Principal)
        opening_bal, opening_bal_info = scan_pages_for_amount(
            label_patterns=[
                r"(?:opening\s*financed\s*balance|financed\s*(?:loan\s*)?balance|opening\s*principal\s*balance|capitalis?ed\s*(?:loan\s*)?(?:amount|balance|principal)|total\s*financed\s*amount|gross\s*loan\s*amount|total\s*loan\s*(?:amount\s*)?\(?\s*incl\.?\s*insurance\s*\)?|opening\s*balance)",
            ],
            exclude_pattern=r"bank\s*account|savings|balance\s*processing",
            min_val=1000.0,
        )
        field_metadata["opening_financed_balance"] = opening_bal_info

        # 4. Interest Rate & Interest Type
        rate: Optional[float] = None
        rate_info = ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)
        int_type = InterestRateType.UNKNOWN
        int_type_info = ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)

        roi_exclude = r"reference\s*rate|benchmark\s*rate|eblr|rllr|mclr|penal|default\s*interest|prepayment|foreclosure|conversion|repricing|gst"
        roi_labels = [
            r"(?:rate\s*of\s*interest\s*(?:\(?\s*roi\s*\)?)?|applicable\s*(?:interest\s*)?rate|effective\s*rate\s*of\s*interest|contractual\s*interest\s*rate|interest\s*rate(?:\s*\(%\s*(?:p\.?a\.?)?\))?|roi(?:\s*\(%\))?|floating\s*rate\s*of\s*interest|fixed\s*rate\s*of\s*interest)",
        ]

        for p in pages:
            p_num = p["page"]
            lines = [ln.strip() for ln in self._normalize_text(p["text"]).splitlines() if ln.strip()]
            for i, line in enumerate(lines):
                # Detect explicit ROI type row
                if re.search(r"(?:rate\s*of\s*interest\s*(?:\(\s*roi\s*\)\s*)?type|interest\s*(?:rate\s*)?type|nature\s*of\s*interest)", line, re.IGNORECASE):
                    win = line + (" " + lines[i + 1] if i + 1 < len(lines) else "")
                    detected = self._detect_interest_type(win)
                    if detected != InterestRateType.UNKNOWN:
                        int_type = detected
                        int_type_info = ExtractedFieldInfo(
                            status=FieldStatus.CONFIRMED,
                            source_page=p_num,
                            evidence_snippet=line[:240],
                        )

                if rate is None:
                    # Check if line has an ROI label (and doesn't START with an excluded benchmark label)
                    for pat in roi_labels:
                        m = re.search(pat, line, re.IGNORECASE)
                        if m:
                            prefix = line[:m.start()]
                            if re.search(roi_exclude, prefix, re.IGNORECASE) or re.search(r"^\s*(?:hhfl\s*)?(?:reference|benchmark)\s*rate", line, re.IGNORECASE):
                                continue
                            # Skip "Rate of Interest Type" lines unless a % also appears on that line
                            if re.search(r"interest\s*(?:\(\s*roi\s*\)\s*)?type", line, re.IGNORECASE) and "%" not in line:
                                continue
                            after_label = line[m.end():]
                            if not re.search(r"\d", after_label) and i + 1 < len(lines):
                                after_label += " " + lines[i + 1]
                            # Prefer number followed by %; allow bare decimal if label had (%)
                            pct_m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", after_label)
                            if not pct_m and "%" in line[:m.end()]:
                                pct_m = re.search(r"\b(\d{1,2}(?:\.\d+)?)\b", after_label)
                            if pct_m:
                                val = float(pct_m.group(1))
                                if 0.0 <= val <= 60.0:
                                    rate = val
                                    rate_info = ExtractedFieldInfo(
                                        status=FieldStatus.CONFIRMED,
                                        source_page=p_num,
                                        evidence_snippet=line[:240],
                                    )
                                    # Also infer interest type if written right alongside the rate e.g. "9.25% floating"
                                    if int_type == InterestRateType.UNKNOWN:
                                        detected = self._detect_interest_type(line + " " + after_label)
                                        if detected != InterestRateType.UNKNOWN:
                                            int_type = detected
                                            int_type_info = ExtractedFieldInfo(
                                                status=FieldStatus.CONFIRMED,
                                                source_page=p_num,
                                                evidence_snippet=line[:240],
                                            )
                                    break

            # Fallback prose match per page: e.g. "9.25% p.a. floating interest" or "interest @ 9.25% floating"
            if rate is None:
                for line in lines:
                    if re.search(roi_exclude, line, re.IGNORECASE):
                        continue
                    pm = re.search(
                        r"(?:interest(?:\s*rate)?[^%\n]{0,30}?(\d{1,2}(?:\.\d+)?)\s*%|\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:p\.?a\.?\s*)?(?:floating|fixed|flat|reducing))",
                        line,
                        re.IGNORECASE,
                    )
                    if pm:
                        val_str = pm.group(1) or pm.group(2)
                        val = float(val_str)
                        if 0.0 < val <= 60.0:
                            rate = val
                            rate_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                evidence_snippet=line[:240],
                            )
                            if int_type == InterestRateType.UNKNOWN:
                                detected = self._detect_interest_type(line)
                                if detected != InterestRateType.UNKNOWN:
                                    int_type = detected
                                    int_type_info = ExtractedFieldInfo(
                                        status=FieldStatus.CONFIRMED,
                                        source_page=p_num,
                                        evidence_snippet=line[:240],
                                    )
                            break

        if int_type == InterestRateType.UNKNOWN:
            # Global fallback scan for explicit floating/flat/fixed declarations
            detected = self._detect_interest_type(full_norm)
            if detected != InterestRateType.UNKNOWN:
                int_type = detected
                int_type_info = ExtractedFieldInfo(status=FieldStatus.CONFIRMED, note="Inferred from document text")

        field_metadata["stated_interest_rate"] = rate_info
        field_metadata["interest_type"] = int_type_info

        # 5. Tenure Detection (Months)
        tenure: Optional[int] = None
        tenure_info = ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)
        for p in pages:
            p_num = p["page"]
            lines = [ln.strip() for ln in self._normalize_text(p["text"]).splitlines() if ln.strip()]
            for i, line in enumerate(lines):
                win = line + (" " + lines[i + 1] if i + 1 < len(lines) else "")
                # Explicit label mentioning months in label header: "Tenor (in Months) | 120" or "Tenure (Months): 120"
                m_lbl = re.search(
                    r"(?:loan\s*)?(?:tenor|tenure|repayment\s*(?:period|tenure)|loan\s*term|facility\s*term|duration|number\s*of\s*install?ments|no\.?\s*of\s*emis)\s*\(?\s*(?:in\s*)?months?\s*\)?\s*[:|\-]?\s*(\d{1,3})\b",
                    win,
                    re.IGNORECASE,
                )
                if m_lbl:
                    tenure = int(m_lbl.group(1))
                    tenure_info = ExtractedFieldInfo(
                        status=FieldStatus.CONFIRMED,
                        source_page=p_num,
                        evidence_snippet=line[:240],
                    )
                    break

                # Label followed by N months/EMIs/instalments
                m_mo = re.search(
                    r"(?:tenor|tenure|term|period|duration|repayment|install?ments|emis)[\s\S]{0,60}?\b(\d{1,3})\s*(?:months?|mths?|monthly\s*install?ments?|emis?)\b",
                    win,
                    re.IGNORECASE,
                )
                if m_mo:
                    tenure = int(m_mo.group(1))
                    tenure_info = ExtractedFieldInfo(
                        status=FieldStatus.CONFIRMED,
                        source_page=p_num,
                        evidence_snippet=line[:240],
                    )
                    break

                # Label followed by N years
                m_yr = re.search(
                    r"(?:tenor|tenure|term|period|duration|repayment)[\s\S]{0,40}?\b(\d{1,2})\s*(?:years?|yrs?)\b",
                    win,
                    re.IGNORECASE,
                )
                if m_yr:
                    tenure = int(m_yr.group(1)) * 12
                    tenure_info = ExtractedFieldInfo(
                        status=FieldStatus.CONFIRMED,
                        source_page=p_num,
                        evidence_snippet=line[:240],
                    )
                    break
            if tenure is not None:
                break

        # Prose fallback: "120 monthly instalments" or "120 months"
        if tenure is None:
            for p in pages:
                p_num = p["page"]
                for line in [ln.strip() for ln in self._normalize_text(p["text"]).splitlines() if ln.strip()]:
                    m_p = re.search(r"\b(\d{2,3})\s+(?:monthly\s*install?ments?|emis?|months)\b", line, re.IGNORECASE)
                    if m_p:
                        tenure = int(m_p.group(1))
                        tenure_info = ExtractedFieldInfo(
                            status=FieldStatus.CONFIRMED,
                            source_page=p_num,
                            evidence_snippet=line[:240],
                        )
                        break
                if tenure is not None:
                    break
        field_metadata["tenure_months"] = tenure_info

        # 6. Stated Monthly EMI (supports both Instalment & Installment)
        emi: Optional[float] = None
        emi_info = ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)
        emi_exclude = r"emi\s*(?:frequency|due\s*date|start\s*date|cycle|bounce|dishonour|penal|amount\s*is\s*subject\s*to)|pre[-\s]*emi|minimum\s*pre"
        emi_labels = [
            r"(?:equated\s*monthly\s*install?ment|monthly\s*emi|sanctioned\s*emi|emi\s*\(\s*install?ment\s*amount\s*\)|emi\s*amount|monthly\s*install?ment|install?ment\s*amount)",
            r"(?:^|\|\s*)emi\b(?:\s*[:|\-])",
        ]
        for p in pages:
            p_num = p["page"]
            lines = [ln.strip() for ln in self._normalize_text(p["text"]).splitlines() if ln.strip()]
            for i, line in enumerate(lines):
                if re.search(emi_exclude, line, re.IGNORECASE):
                    continue
                for pat in emi_labels:
                    m = re.search(pat, line, re.IGNORECASE)
                    if m:
                        search_win = line[m.end():]
                        snippet = line
                        if not re.search(r"\d", search_win) and i + 1 < len(lines):
                            search_win += " " + lines[i + 1]
                            snippet += " | " + lines[i + 1]
                        val = self._extract_inr_from_window(search_win)
                        if val is not None and val >= 50.0 and (principal is None or principal == 0 or val < principal or tenure == 1):
                            emi = val
                            emi_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                evidence_snippet=snippet[:240],
                            )
                            break
                if emi is not None:
                    break
            if emi is not None:
                break
        field_metadata["stated_emi"] = emi_info

        # 7. Multi-Page Charge & Recovery Scanner (All Pages, Annexures, Tables, Footnotes)
        charges = self._scan_all_charges(pages, principal, opening_bal)

        upfront_charges = [c for c in charges if c.category == ChargeCategory.CONFIRMED_UPFRONT and c.amount is not None]
        financed_charges = [c for c in charges if c.category == ChargeCategory.FINANCED and c.amount is not None]

        pf_val: Optional[float] = None
        doc_val: Optional[float] = None
        other_val: Optional[float] = None
        for c in upfront_charges:
            lname = c.name.lower()
            if "processing" in lname:
                pf_val = (pf_val or 0.0) + c.amount
            elif any(k in lname for k in ("doc", "stamp", "legal", "cersai", "title")):
                doc_val = (doc_val or 0.0) + c.amount
            else:
                other_val = (other_val or 0.0) + c.amount

        ins_val: Optional[float] = None
        is_ins_cap = False
        if financed_charges:
            ins_val = round(sum(c.amount for c in financed_charges), 2)
            is_ins_cap = True
        else:
            # Check for non-financed bundled insurance if present
            ins_amt, ins_info = scan_pages_for_amount(
                label_patterns=[
                    r"(?:insurance\s*(?:amount|premium)(?:\s*and\s*additional\s*fees)?|credit\s*shield(?:\s*cover)?|life\s*insurance(?:\s*premium)?|loan\s*protection\s*(?:cover|insurance))"
                ],
                min_val=1.0,
            )
            if ins_amt is not None:
                ins_val = ins_amt
                # Check if opening_financed_balance = principal + insurance
                if principal is not None and opening_bal is not None and abs((principal + ins_amt) - opening_bal) <= 5.0:
                    is_ins_cap = True
                    charges.append(
                        ExtractedCharge(
                            name="Financed Insurance",
                            amount=ins_amt,
                            category=ChargeCategory.FINANCED,
                            status=FieldStatus.CONFIRMED,
                            source_page=ins_info.source_page,
                            evidence_snippet=ins_info.evidence_snippet,
                        )
                    )
                else:
                    field_metadata["bundled_insurance"] = ins_info

        # 8. Explicit GST Amount, GST Rate %, and Tax Status Note (NEVER invent 18%!)
        gst_amt: Optional[float] = None
        gst_rate: Optional[float] = None
        gst_note: Optional[str] = None
        gst_info = ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)

        for p in pages:
            p_num = p["page"]
            lines = [ln.strip() for ln in self._normalize_text(p["text"]).splitlines() if ln.strip()]
            for line in lines:
                # Explicit GST % e.g. "18% GST" or "GST @ 18%"
                if gst_rate is None:
                    rate_m = re.search(
                        r"(?:(\d{1,2}(?:\.\d+)?)\s*%\s*(?:applicable\s*)?gst\b|\bgst\s*(?:@|at)?\s*(\d{1,2}(?:\.\d+)?)\s*%)",
                        line,
                        re.IGNORECASE,
                    )
                    if rate_m:
                        gst_rate = float(rate_m.group(1) or rate_m.group(2))
                        gst_info = ExtractedFieldInfo(
                            status=FieldStatus.CONFIRMED,
                            source_page=p_num,
                            evidence_snippet=line[:240],
                        )

                # Explicit rupee GST amount e.g. "GST Amount : INR 2,700" (must NOT match "18% GST")
                if gst_amt is None:
                    amt_m = re.search(
                        r"\b(?:gst\s*amount|goods\s*(?:&|and)\s*services\s*tax|gst\s*\([^)]*%[^)]*\)|total\s*gst)\s*[:|\-]?\s*(?:INR\s*)([\d,]+(?:\.\d+)?)",
                        line,
                        re.IGNORECASE,
                    )
                    if amt_m:
                        val = self._parse_plain_number(amt_m.group(1))
                        if val is not None:
                            gst_amt = val
                            gst_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                evidence_snippet=line[:240],
                            )

                # Unquantified applicable GST phrase
                if gst_note is None and re.search(
                    r"(?:plus|exclusive\s*of|\+)\s*(?:applicable\s*)?(?:gst|taxes?)",
                    line,
                    re.IGNORECASE,
                ):
                    gst_note = line[:200]

        if gst_amt is None and gst_rate is None and gst_note:
            gst_info = ExtractedFieldInfo(
                status=FieldStatus.NOT_FOUND,
                note=f"Tax mentioned without explicit rupee amount or percentage: {gst_note}",
            )
        field_metadata["gst_amount"] = gst_info

        # 9. Explicit Final Net Disbursement Amount (ONLY if stated in document)
        net_disb, net_disb_info = scan_pages_for_amount(
            label_patterns=[
                r"(?:final\s*net\s*disbursement|net\s*disbursement\s*(?:amount)?|net\s*disbursed\s*(?:amount)?|final\s*disbursement\s*(?:amount)?|net\s*amount\s*(?:disbursed|credited|payable)|net\s*disbursal\s*(?:amount)?)"
            ],
            exclude_pattern=r"subject\s*to|post\s*deduction|will\s*be\s*made|pending|shall\s*be\s*communicated|to\s*be\s*communicated|tax\s*invoice|will\s*be\s*intimated",
            min_val=100.0,
        )
        field_metadata["net_disbursement"] = net_disb_info

        # 10. Foreclosure / Prepayment Penalty
        has_penalty: Optional[bool] = None
        penalty_terms: Optional[str] = None
        penalty_info = ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)
        for p in pages:
            p_num = p["page"]
            for line in [ln.strip() for ln in self._normalize_text(p["text"]).splitlines() if ln.strip()]:
                if re.search(r"minimum\s+pre[-\s]*payment\s+amount", line, re.IGNORECASE):
                    continue
                if re.search(r"(?:prepayment|pre-payment|foreclosure)", line, re.IGNORECASE):
                    if re.search(r"\b(?:nil|zero|no\s*(?:charges|penalty|fee))\b", line, re.IGNORECASE):
                        if has_penalty is None:
                            has_penalty = False
                            penalty_terms = line[:220]
                            penalty_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                evidence_snippet=line[:240],
                            )
                    elif re.search(r"(\d+(?:\.\d+)?\s*%|INR\s*\d+)", line, re.IGNORECASE):
                        if has_penalty is None:
                            has_penalty = True
                            penalty_terms = line[:220]
                            penalty_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                evidence_snippet=line[:240],
                            )
        field_metadata["prepayment_penalty"] = penalty_info

        # 11. Repayment Schedule Scan
        sched_count, sched_emis = self._scan_repayment_schedule(pages)

        missing = [
            fname for fname, val in (
                ("principal_amount", principal),
                ("stated_interest_rate", rate),
                ("tenure_months", tenure),
            )
            if val is None
        ]

        return ExtractedLoanTerms(
            lender_name=lender,
            loan_category=category,
            principal_amount=principal,
            opening_financed_balance=opening_bal,
            stated_interest_rate=rate,
            interest_type=int_type,
            tenure_months=tenure,
            stated_emi=emi,
            processing_fee=pf_val,
            documentation_charges=doc_val,
            other_cash_recoveries=other_val,
            bundled_insurance=ins_val,
            is_insurance_capitalized=is_ins_cap,
            stated_gst_amount=gst_amt,
            stated_gst_rate_pct=gst_rate,
            stated_net_disbursement=net_disb,
            gst_status_note=gst_note,
            prepayment_penalty_terms=penalty_terms,
            has_prepayment_penalty=has_penalty,
            charges=charges,
            schedule_instalment_count=sched_count,
            schedule_emis=sched_emis,
            field_metadata=field_metadata,
            is_verified=len(missing) == 0,
            missing_fields=missing,
            extraction_method="deterministic_multi_page_extractor",
        )

    @staticmethod
    def _detect_interest_type(text: str) -> InterestRateType:
        if re.search(r"\bflat\b", text, re.IGNORECASE):
            return InterestRateType.FLAT
        if re.search(r"\b(?:floating|variable|eblr|rllr|mclr|repo[-\s]*linked)\b", text, re.IGNORECASE):
            return InterestRateType.FLOATING
        if re.search(r"\bfixed\b", text, re.IGNORECASE):
            return InterestRateType.FIXED
        if re.search(r"\breducing\b", text, re.IGNORECASE):
            return InterestRateType.REDUCING
        return InterestRateType.UNKNOWN

    def _scan_all_charges(
        self,
        pages: List[Dict[str, Any]],
        principal: Optional[float],
        opening_bal: Optional[float],
    ) -> List[ExtractedCharge]:
        """
        Scans every page, table, annexure, paragraph, and footnote for charges and classifies them
        strictly into confirmed_upfront, financed, conditional, or third_party.
        Deduplicates identical items printed in both table and raw text sections on the same page.
        """
        found: List[ExtractedCharge] = []
        seen_keys = set()

        charge_patterns = [
            # Financed / Capitalized insurance or charges
            (
                "Financed Insurance",
                r"(?:financed\s*(?:credit\s*)?(?:shield|life|health|property)?\s*insurance|capitalis?ed\s*insurance|insurance\s*(?:premium\s*)?\(?\s*financed\s*\)?|insurance\s*(?:added|capitalis?ed)\s*to\s*(?:loan|principal))",
                ChargeCategory.FINANCED,
            ),
            # Conditional charges (must be checked BEFORE generic fee patterns)
            (
                "Conditional Mandate Charge",
                r"(?:conditional\s*(?:mandate|nach|ecs|e-mandate)?\s*(?:charge|fee)|(?:mandate|nach|ecs|e-mandate)\s*(?:registration|setup|processing|bounce|dishonour)?\s*(?:charge|fee)\s*(?:\([^)]*conditional[^)]*\)|[^\n]{0,40}?(?:if|upon|contingent|in\s*case))|(?:bounce|dishonour|cheque\s*return|delayed\s*payment|penal)\s*(?:charge|fee))",
                ChargeCategory.CONDITIONAL,
            ),
            # Third-party charges
            (
                "Third-Party Charge",
                r"(?:third[-\s]*party\s*(?:charge|fee|valuation|legal|insurance)|payable\s*directly\s*to\s*(?:third\s*party|advocate|valuer|sub-registrar))",
                ChargeCategory.THIRD_PARTY,
            ),
            # Processing fees (supports PDF table concatenated words like ProcessingFees)
            (
                "Processing Fee",
                r"(?:(?:b\)\s*)?processing\s*fees?|(?:c\)\s*)?balance\s*processing\s*fees?|processing\s*charges?|loan\s*processing\s*(?:fee|charge)|origination\s*(?:fee|charge)|admin(?:istrative)?\s*(?:fee|charge))",
                ChargeCategory.CONFIRMED_UPFRONT,
            ),
            # Other cash recoveries
            (
                "Other Cash Recoveries",
                r"(?:other\s*(?:upfront\s*)?cash\s*recover(?:y|ies)|other\s*upfront\s*recover(?:y|ies)|cash\s*recover(?:y|ies)|upfront\s*recover(?:y|ies)|other\s*deductions?)",
                ChargeCategory.CONFIRMED_UPFRONT,
            ),
            # Documentation / Legal / Valuation / CERSAI / Stamp Duty / Login / Title Search
            (
                "Documentation / Statutory Fee",
                r"(?:documentation\s*(?:&\s*stamp\s*)?(?:charges?|fees?)|legal\s*(?:&\s*technical\s*)?(?:charges?|fees?)|valuation\s*(?:charges?|fees?)|technical\s*(?:charges?|fees?)|login\s*(?:charges?|fees?)|cersai\s*(?:charges?|fees?)|stamp\s*duty|imd\b|title\s*search\s*(?:charges?|fees?))",
                ChargeCategory.CONFIRMED_UPFRONT,
            ),
        ]

        # Skip summary total rows so individual items aren't double-counted
        summary_skip = r"total\s*(?:upfront|cash)?\s*(?:fees?|charges?|recover(?:y|ies)|deductions?)|^prepayment|^foreclosure|switching/conversion"

        for p in pages:
            p_num = p["page"]
            lines = [ln.strip() for ln in self._normalize_text(p["text"]).splitlines() if ln.strip()]
            for i, line in enumerate(lines):
                if re.search(summary_skip, line, re.IGNORECASE):
                    continue
                # Skip section headers ending with ':' that contain no figures (e.g. "A. Upfront Recoveries from Borrower:")
                if re.match(r"^(?:[A-Z]\.|\d+\.)?\s*[A-Za-z\s/&(),-]+:\s*$", line):
                    continue
                # Skip percentage-only policy clauses (e.g. "1.00% + Applicable taxes", "0.5% of principal")
                if re.search(r"\b\d+(?:\.\d+)?\s*%", line) and not re.search(r"INR\s*[\d,]+", line, re.IGNORECASE):
                    continue

                for default_name, pat, cat in charge_patterns:
                    m = re.search(pat, line, re.IGNORECASE)
                    if not m:
                        continue

                    # Override category to CONDITIONAL if conditional phrasing appears anywhere on the line
                    effective_cat = cat
                    if effective_cat != ChargeCategory.FINANCED and re.search(
                        r"\b(?:conditional|if\s+applicable|upon\s+failure|in\s+case\s+of|contingent|if\s+mandate|if\s+dishonoured)\b",
                        line,
                        re.IGNORECASE,
                    ):
                        effective_cat = ChargeCategory.CONDITIONAL

                    search_win = line[m.end():]
                    snippet = line
                    if (
                        not re.search(r"\d", search_win)
                        and i + 1 < len(lines)
                        and not re.match(r"^(?:\d+\.|[A-Z]\.|[-•*]|\(?[a-z]\)|footnote|note\b)", lines[i + 1], re.IGNORECASE)
                    ):
                        search_win += " " + lines[i + 1]
                        snippet += " | " + lines[i + 1]

                    val = self._extract_inr_from_window(search_win)
                    if val is None or val <= 0:
                        continue

                    # Avoid confusing principal/balance amounts for charges
                    if principal is not None and val >= principal * 0.8 and effective_cat == ChargeCategory.CONFIRMED_UPFRONT:
                        continue

                    tax_note = None
                    if re.search(r"(?:plus|exclusive\s*of|\+)\s*(?:applicable\s*)?(?:gst|taxes?)", search_win, re.IGNORECASE):
                        tax_note = "plus applicable GST"
                    elif re.search(r"inclusive\s*of\s*(?:gst|taxes?)", search_win, re.IGNORECASE):
                        tax_note = "inclusive of GST"

                    # Derive clean item name from matched phrase
                    matched_label = re.sub(r"^(?:a\)|b\)|c\))\s*", "", m.group(0).strip(), flags=re.IGNORECASE).strip()
                    clean_name = matched_label[:60] if matched_label else default_name

                    # Deduplicate identical charges extracted from both Table and Document Text on the same page
                    dedup_key = (p_num, effective_cat.value, round(val, 2))
                    if dedup_key in seen_keys:
                        break
                    seen_keys.add(dedup_key)

                    found.append(
                        ExtractedCharge(
                            name=clean_name,
                            amount=round(val, 2),
                            category=effective_cat,
                            status=FieldStatus.CONFIRMED,
                            source_page=p_num,
                            evidence_snippet=snippet[:240],
                            tax_note=tax_note,
                        )
                    )
                    break

        # Handle Hero-style processing fee breakdown: if (a) IMD + (c) Balance == (b) Processing Fees, keep (b) once
        imd_items = [c for c in found if c.category == ChargeCategory.CONFIRMED_UPFRONT and "imd" in c.name.lower()]
        bal_items = [c for c in found if c.category == ChargeCategory.CONFIRMED_UPFRONT and "balance processing" in c.name.lower()]
        gross_pf_items = [
            c for c in found
            if c.category == ChargeCategory.CONFIRMED_UPFRONT
            and "processing" in c.name.lower()
            and "balance" not in c.name.lower()
        ]
        if imd_items and bal_items and gross_pf_items:
            imd_v = imd_items[0].amount or 0.0
            bal_v = bal_items[0].amount or 0.0
            gross_v = gross_pf_items[0].amount or 0.0
            if abs((imd_v + bal_v) - gross_v) <= 2.0:
                found = [c for c in found if c not in imd_items and c not in bal_items]

        return found

    def _scan_repayment_schedule(self, pages: List[Dict[str, Any]]) -> Tuple[Optional[int], List[float]]:
        """Extracts instalment count and EMIs from any repayment schedule table across pages."""
        max_inst_no: Optional[int] = None
        sched_emis: List[float] = []

        for p in pages:
            norm = self._normalize_text(p["text"])
            # Check explicit schedule range e.g. "Instalments 1 to 120" or "1 - 120"
            range_m = re.search(
                r"(?:instal?ments?|emis?)\s*(?:no\.?\s*)?1\s*(?:to|-|–)\s*(\d{2,3})\b",
                norm,
                re.IGNORECASE,
            )
            if range_m:
                cnt = int(range_m.group(1))
                max_inst_no = max(max_inst_no or 0, cnt)

            # Check pipe-delimited schedule table rows starting with instalment number
            if re.search(r"repayment\s*schedule|amortization\s*schedule|instal?ment\s*schedule", norm, re.IGNORECASE):
                for line in norm.splitlines():
                    row_m = re.match(r"^\s*(\d{1,3})\s*\|\s*(?:INR\s*)?([\d,]+(?:\.\d+)?)", line.strip(), re.IGNORECASE)
                    if row_m:
                        inst_no = int(row_m.group(1))
                        val = self._parse_plain_number(row_m.group(2))
                        max_inst_no = max(max_inst_no or 0, inst_no)
                        if val is not None and val >= 100 and len(sched_emis) < 240:
                            sched_emis.append(val)

        return max_inst_no, sched_emis

    # =========================================================================
    # EXPLANATION GENERATION (GATED ON VERIFICATION, NO INVENTED CITATIONS)
    # =========================================================================

    async def generate_multilingual_explanation(
        self,
        audit: LoanAuditReport,
        language: Language,
        custom_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Explains deterministic audit findings to an Indian borrower.
        Strictly respects Requirement 10 (no invented GST rates/SAC/circulars/scores)
        and Requirement 12 (blocks negotiation scripts when critical fields are unverified).
        """
        active_key = custom_api_key or self.api_key or settings.OPENROUTER_API_KEY
        if not active_key or "your_openrouter" in active_key.lower():
            return self._local_template_explanation(audit, language)

        apr_display = f"{audit.effective_annual_apr:.2f}%" if audit.effective_annual_apr is not None else "UNKNOWN"
        net_display = f"₹{audit.net_disbursed_amount:,.2f}" if audit.net_disbursed_amount is not None else "UNKNOWN"
        gst_display = f"₹{audit.fee_gst_amount:,.2f}" if audit.fee_gst_amount is not None else "UNKNOWN"

        audit_summary_for_llm = {
            "audit_status": audit.audit_status,
            "lender": audit.lender_name,
            "principal": audit.principal,
            "opening_financed_balance": audit.opening_financed_balance,
            "stated_rate": f"{audit.stated_annual_rate}%" if audit.stated_annual_rate is not None else "UNKNOWN",
            "interest_type": str(audit.interest_type.value if hasattr(audit.interest_type, "value") else audit.interest_type),
            "tenure_months": audit.tenure_months,
            "stated_emi": audit.stated_emi,
            "calculated_reducing_emi": audit.calculated_reducing_emi,
            "is_emi_matching": audit.is_emi_matching,
            "known_upfront_recoveries_excl_tax": audit.known_upfront_recoveries_excl_tax,
            "fee_gst_amount": gst_display,
            "gst_status_note": audit.gst_status_note,
            "bundled_insurance": audit.bundled_insurance_amount,
            "is_insurance_capitalized": audit.is_insurance_capitalized,
            "conditional_charges_total": audit.conditional_charges_total,
            "net_disbursed_amount": net_display,
            "effective_annual_apr": apr_display,
            "negotiation_script_blocked": audit.negotiation_script_blocked,
            "negotiation_block_reason": audit.negotiation_block_reason,
            "cross_validations": [
                {"check": c.check_name, "status": c.status, "summary": c.summary}
                for c in audit.cross_validations
            ],
            "discrepancies": [
                {
                    "code": d.code,
                    "title": d.title,
                    "description": d.description,
                    "severity": d.severity,
                    "source_page": d.source_page,
                    "evidence": d.evidence_snippet,
                }
                for d in audit.discrepancies
            ],
        }

        lang_instructions = {
            Language.ENGLISH: "Write the response in clear, sharp, forensic English.",
            Language.HINDI: "Write the complete response in formal, natural Hindi (Devanagari script).",
            Language.KANNADA: "Write the complete response in formal Kannada script (ಕನ್ನಡ).",
        }

        negotiation_instruction = (
            "CRITICAL RULE: negotiation_script_blocked is TRUE. "
            "DO NOT generate any negotiation scripts, fee-waiver demands, or APR arguments. "
            "Instead, state clearly that negotiation script generation is BLOCKED because critical financial fields are unverified or unknown, and list what the borrower must ask the lender to disclose in writing first."
            if audit.negotiation_script_blocked
            else "Provide 2-3 concise, verified borrower questions grounded strictly in the confirmed audit figures."
        )

        system_prompt = (
            "You are a Forensic Loan Document Analyst explaining a deterministic audit report to an Indian borrower.\n\n"
            "STRICT ZERO-HALLUCINATION RULES:\n"
            "1. Use ONLY the exact numbers and facts present in the audit JSON. NEVER invent, round differently, or guess missing figures.\n"
            "2. If a field is UNKNOWN / null (such as GST amount, Net Disbursement, or Effective APR), explicitly report it as UNKNOWN — NEVER substitute 0 or assume 18% GST.\n"
            "3. DO NOT invent or cite GST SAC codes, RBI circular numbers, or numeric compliance scores unless explicitly present in the audit data.\n"
            "4. Cite source page numbers when referencing findings.\n"
            f"5. {negotiation_instruction}\n"
            f"6. {lang_instructions.get(language, lang_instructions[Language.ENGLISH])}\n"
        )

        user_content = f"Verified Deterministic Audit Data:\n```json\n{json.dumps(audit_summary_for_llm, indent=2)}\n```"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            chosen_model = model_name or self.model
            return await self._call_openrouter(
                messages=messages,
                temperature=0.2,
                custom_api_key=active_key,
                model_name=chosen_model,
            )
        except Exception:
            return self._local_template_explanation(audit, language)

    def _local_template_explanation(self, audit: LoanAuditReport, language: Language) -> str:
        """
        Deterministic narrative summary strictly reporting only confirmed values,
        keeping unknown values explicitly UNKNOWN, and blocking negotiation scripts when gated.
        """
        p_str = f"₹{audit.principal:,.2f}" if audit.principal is not None else "UNKNOWN"
        r_str = f"{audit.stated_annual_rate:.2f}%" if audit.stated_annual_rate is not None else "UNKNOWN"
        t_str = f"{audit.tenure_months} months" if audit.tenure_months is not None else "UNKNOWN"
        emi_str = f"₹{audit.stated_emi:,.2f}" if audit.stated_emi is not None else (
            f"₹{audit.calculated_reducing_emi:,.2f} (calculated)" if audit.calculated_reducing_emi is not None else "UNKNOWN"
        )
        net_str = f"₹{audit.net_disbursed_amount:,.2f}" if audit.net_disbursed_amount is not None else "UNKNOWN"
        apr_str = f"{audit.effective_annual_apr:.2f}%" if audit.effective_annual_apr is not None else "UNKNOWN (required cash flows unavailable)"
        gst_str = f"₹{audit.fee_gst_amount:,.2f}" if audit.fee_gst_amount is not None else "UNKNOWN"
        upfront_excl_str = (
            f"₹{audit.known_upfront_recoveries_excl_tax:,.2f}"
            if audit.known_upfront_recoveries_excl_tax is not None
            else "UNKNOWN"
        )
        cond_str = (
            f"₹{audit.conditional_charges_total:,.2f}"
            if audit.conditional_charges_total is not None
            else "None detected"
        )

        lines = [
            "### Forensic Audit Summary",
            f"- **Audit Status**: `{audit.audit_status}`",
            f"- **Sanctioned Principal**: {p_str}",
        ]
        if audit.opening_financed_balance is not None:
            lines.append(f"- **Opening Financed Balance**: ₹{audit.opening_financed_balance:,.2f}")
        lines.extend([
            f"- **Stated Interest Rate**: {r_str} ({audit.interest_type.value})",
            f"- **Tenure**: {t_str}",
            f"- **Monthly EMI**: {emi_str}",
            f"- **Known Upfront Cash Recoveries (before tax)**: {upfront_excl_str}",
            f"- **Disclosed GST Amount**: {gst_str}" + (f" ({audit.gst_status_note})" if audit.gst_status_note else ""),
            f"- **Conditional Charges (excluded from upfront deductions)**: {cond_str}",
            f"- **Final Net Disbursement**: {net_str}",
            f"- **Effective Annual APR**: {apr_str}",
        ])

        if audit.discrepancies:
            lines.append("\n### Verified Findings & Evidence")
            for d in audit.discrepancies:
                page_tag = f" [Page {d.source_page}]" if d.source_page else ""
                ev_tag = f" — Evidence: \"{d.evidence_snippet}\"" if d.evidence_snippet else ""
                lines.append(f"- **[{d.severity.value.upper()}]{page_tag} {d.title}**: {d.description}{ev_tag}")

        if audit.negotiation_script_blocked:
            lines.append("\n### Negotiation Script Blocked")
            lines.append(
                f"Negotiation script generation is **blocked** because critical financial parameters remain unverified or incomplete: "
                f"{audit.negotiation_block_reason or ', '.join(audit.missing_fields) or 'unverified cash flows'}."
            )
            lines.append("Please obtain written disclosure of the missing amounts before proceeding.")
        else:
            lines.append("\n### Borrower Verification Questions")
            lines.append(
                f"1. Please confirm the full itemized fee schedule matching the verified upfront deductions and Net Disbursed amount of {net_str}."
            )
            lines.append(
                f"2. Please confirm the Effective Annual APR ({apr_str}) and repayment schedule alignment in the Key Fact Statement."
            )

        return "\n".join(lines)
