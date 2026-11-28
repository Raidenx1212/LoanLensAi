import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
import httpx
from app.schemas.loan import (
    ExtractedLoanTerms,
    ExtractedFieldInfo,
    ExtractedCharge,
    DocumentNote,
    FieldStatus,
    ChargeCategory,
    GSTClassification,
    ThirdPartyRecoveryMethod,
    LoanAuditReport,
    Language,
    InterestRateType,
    LoanCategory,
)
from app.config import settings

logger = logging.getLogger(__name__)

# Resilient model aliases mapping to active OpenRouter models
MODEL_ALIASES = {
    "anthropic/claude-3.5-sonnet": "anthropic/claude-sonnet-4.5",
    "anthropic/claude-3-5-sonnet": "anthropic/claude-sonnet-4.5",
    "mistralai/mistral-large-2411": "mistralai/mistral-large-2512",
    "google/gemini-2.0-flash-001": "openai/gpt-4o-mini",
    "google/gemini-2.0-flash": "openai/gpt-4o-mini",
}

FALLBACK_ZDR_MODELS = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
]

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
        self.model = model or getattr(settings, "DEFAULT_LLM_MODEL", "openai/gpt-4o-mini")
        self._last_used_model: Optional[str] = None

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
        candidates = [chosen_model] + [m for m in FALLBACK_ZDR_MODELS if m != chosen_model]

        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://loanlens.ai",
            "X-Title": "LoanLens AI Auditor",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=5.0)) as client:
            for candidate in candidates:
                payload: Dict[str, Any] = {
                    "model": candidate,
                    "messages": messages,
                    "temperature": temperature,
                }
                if response_format:
                    payload["response_format"] = response_format
                try:
                    response = await client.post(self.base_url, headers=headers, json=payload)
                    if response.status_code == 400 and response_format:
                        payload.pop("response_format", None)
                        response = await client.post(self.base_url, headers=headers, json=payload)
                    if response.status_code in (400, 404, 429):
                        logger.warning(f"OpenRouter model {candidate} returned HTTP {response.status_code}; trying next fallback model.")
                        last_err = httpx.HTTPStatusError(
                            f"HTTP {response.status_code}", request=response.request, response=response
                        )
                        continue
                    response.raise_for_status()
                    data = response.json()
                    self._last_used_model = candidate
                    return data["choices"][0]["message"]["content"]
                except httpx.HTTPStatusError as exc:
                    last_err = exc
                    if exc.response.status_code in (400, 404, 429):
                        continue
                    raise
            if last_err is not None:
                raise last_err
            raise ValueError("No OpenRouter model candidates available.")

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
            "6. Separate ALL charges across all pages/footnotes strictly by category:\n"
            "   - 'confirmed_upfront': mandatory upfront cash fees/recoveries in INR (NEVER put a % rate like '1.00%' into amount=1.0; if (a) IMD + (c) Balance Processing Fees == (b) Processing Fees, include ONLY (b) so it is not double-counted)\n"
            "   - 'financed': insurance or fees capitalized/financed into opening loan balance\n"
            "   - 'conditional': conditional fees (e.g. 'applicable only for Mumbai Metropolitan Region', assisted mandate setup, Cheque/NACH bounce INR 1000 per bounce, conversion/repricing)\n"
            "   - 'contingent': percentage-based default/penal charges (set amount=null, rate_pct=2.0, rate_basis='2% per month on overdue payment')\n"
            "   - 'third_party': third-party charges payable outside lender deductions\n"
            "7. GST: Extract stated_gst_amount ONLY if an exact rupee GST amount is written; extract stated_gst_rate_pct ONLY if an explicit % like '18% GST' is written. If it only says 'plus applicable GST' or '+ Applicable taxes', return null for both and put the phrase in gst_status_note. NEVER assume 18%.\n"
            "8. Net Disbursement: Extract stated_net_disbursement ONLY if explicitly stated. Otherwise return null.\n"
            "9. Every field and charge MUST include source_page (int) and minimal verbatim evidence_snippet.\n\n"
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
            effective_model = self._last_used_model or chosen_model
            llm_terms = self._build_terms_from_llm_json(raw_json, effective_model)
            det_terms = self._deterministic_page_extract(parsed_pages)
            return self._merge_llm_and_deterministic(llm_terms, det_terms)
        except Exception:
            logger.exception("OpenRouter extraction failed; using deterministic multi-page extractor fallback.")
            fallback = self._deterministic_page_extract(parsed_pages)
            fallback.extraction_method = "deterministic_multi_page_extractor"
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
            cname = str(ch.get("name") or "Charge")
            ev_text = str(ch.get("evidence_snippet") or "")
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
            amt_val = float(amt) if amt is not None else None
            rate_pct_val = ch.get("rate_pct")
            rate_basis_val = ch.get("rate_basis")

            # Guard against LLM converting percentage rule (e.g. "1.00% + Applicable taxes") into INR 1.0
            if amt_val is not None and amt_val <= 20.0 and "%" in f"{cname} {ev_text}":
                if re.search(r"penal|default", f"{cname} {ev_text}", re.IGNORECASE):
                    cat = ChargeCategory.CONTINGENT
                    rate_pct_val = rate_pct_val or amt_val
                    amt_val = None
                else:
                    continue

            # Guard against LLM putting penal/default rate under conditional amount instead of contingent rate_pct
            if re.search(r"penal|default\s*interest", f"{cname} {ev_text}", re.IGNORECASE) and (amt_val is None or amt_val <= 50.0):
                cat = ChargeCategory.CONTINGENT
                rate_pct_val = rate_pct_val or amt_val
                amt_val = None

            charges.append(
                ExtractedCharge(
                    name=cname,
                    amount=amt_val,
                    rate_pct=float(rate_pct_val) if rate_pct_val is not None else None,
                    rate_basis=rate_basis_val,
                    category=cat,
                    status=st,
                    source_page=ch.get("source_page"),
                    evidence_snippet=ev_text or None,
                    tax_note=ch.get("tax_note"),
                )
            )

        # Deduplicate Hero-style (a) IMD + (c) Balance == (b) Processing Fees if LLM emitted all three
        imd_items = [c for c in charges if c.category == ChargeCategory.CONFIRMED_UPFRONT and "imd" in c.name.lower()]
        bal_items = [c for c in charges if c.category == ChargeCategory.CONFIRMED_UPFRONT and "balance processing" in c.name.lower()]
        gross_pf_items = [
            c for c in charges
            if c.category == ChargeCategory.CONFIRMED_UPFRONT
            and "processing" in c.name.lower()
            and "balance" not in c.name.lower()
        ]
        if imd_items and bal_items and gross_pf_items:
            imd_v = imd_items[0].amount or 0.0
            bal_v = bal_items[0].amount or 0.0
            gross_v = gross_pf_items[0].amount or 0.0
            if abs((imd_v + bal_v) - gross_v) <= 2.0:
                charges = [c for c in charges if c not in imd_items and c not in bal_items]

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

        # Guard against hallucinated opening_financed_balance conflicting with deterministic document scan
        if llm_terms.opening_financed_balance is not None and det_terms.opening_financed_balance is None:
            llm_terms.opening_financed_balance = None
            llm_terms.field_metadata["opening_financed_balance"] = ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)

        # Backfill bundled_insurance & provenance when LLM omitted non-capitalized/customer-funded insurance
        if llm_terms.bundled_insurance is None and det_terms.bundled_insurance is not None:
            llm_terms.bundled_insurance = det_terms.bundled_insurance
            llm_terms.is_insurance_capitalized = det_terms.is_insurance_capitalized
            if "bundled_insurance" in det_terms.field_metadata:
                llm_terms.field_metadata["bundled_insurance"] = det_terms.field_metadata["bundled_insurance"]

        if det_terms.charges:
            # Prefer deterministic verified charge list enriched with any extra unique LLM charges,
            # preserving provenance (source_section, raw_text, GST classification, third-party method, dedup_id).
            merged_charges = list(det_terms.charges)
            det_sigs = {
                (round(c.amount, 2) if c.amount is not None else None, "".join(re.findall(r"[a-z]", c.name.lower()))[:6])
                for c in merged_charges
            }
            for lc in llm_terms.charges:
                lc_sig = (round(lc.amount, 2) if lc.amount is not None else None, "".join(re.findall(r"[a-z]", lc.name.lower()))[:6])
                if lc.amount is None and lc.rate_pct is None:
                    continue
                if any(
                    ds[0] == lc_sig[0] and ds[1] and lc_sig[1] and (ds[1] in lc_sig[1] or lc_sig[1] in ds[1] or ds[0] is not None)
                    for ds in det_sigs
                ):
                    continue
                merged_charges.append(lc)
            llm_terms.charges = merged_charges
            llm_terms.processing_fee = det_terms.processing_fee
            llm_terms.documentation_charges = det_terms.documentation_charges
            llm_terms.other_cash_recoveries = det_terms.other_cash_recoveries

        if not llm_terms.document_notes and det_terms.document_notes:
            llm_terms.document_notes = det_terms.document_notes

        if not llm_terms.gst_status_note and det_terms.gst_status_note:
            llm_terms.gst_status_note = det_terms.gst_status_note
        elif det_terms.stated_gst_amount is not None and det_terms.gst_status_note:
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

    @staticmethod
    def _detect_section_heading(line: str) -> Optional[str]:
        """Detects section, annexure, schedule, or footnote headings for provenance tracking."""
        clean = re.sub(r"\s+", " ", line).strip()
        if not clean or len(clean) > 100:
            return None
        # Ignore page headers/footers like "Sanction Letter Page 3 of 4"
        if re.search(r"\bPage\s+\d+\s+of\s+\d+\b", clean, re.IGNORECASE):
            return None
        if re.match(
            r"^(?:Section|Annexure|Schedule|Appendix|Part)\s+[A-Z0-9]+(?:\s*[-:–.]\s*.*)?$",
            clean,
            re.IGNORECASE,
        ):
            return clean.rstrip(":")
        if re.match(r"^(?:GENERAL|SPECIAL)\s+TERMS\s+AND\s+CONDITIONS$", clean, re.IGNORECASE):
            return clean
        if re.match(r"^Particulars\s+Description$", clean, re.IGNORECASE):
            return "Key Facility Particulars"
        if re.match(r"^(?:Footnotes?|General\s+Notes?|Key\s+Fact\s+Statement|Terms\s*(?:&|and)\s*Conditions)\s*[:\-–]?.*$", clean, re.IGNORECASE):
            if len(clean) <= 75 and "|" not in clean:
                return clean.rstrip(":")
        if re.match(r"^(?:[A-Z]|\d+)\.\s+[A-Z][A-Za-z0-9\s/&(),-]{2,65}$", clean) and "|" not in clean and not re.search(r"\b(?:INR|Rs\.?)\b|%", clean, re.IGNORECASE):
            return clean.rstrip(":")
        return None

    def _prepare_page_lines(
        self,
        page_text: str,
        initial_section: Optional[str] = None,
        raw_page_text: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[Optional[str]], Optional[str]]:
        """
        Splits a page into raw lines, normalized lines, and per-line source sections
        while preserving 1-to-1 line alignment.
        """
        clean_lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        norm_lines = [self._normalize_text(ln).strip() for ln in clean_lines]
        raw_src_lines = [ln.strip() for ln in (raw_page_text or page_text).splitlines() if ln.strip()]
        if len(raw_src_lines) != len(norm_lines):
            raw_src_lines = clean_lines
        sections: List[Optional[str]] = []
        current_section = initial_section
        for norm_ln in norm_lines:
            heading = self._detect_section_heading(norm_ln)
            if heading:
                current_section = heading
            sections.append(current_section)
        return raw_src_lines, norm_lines, sections, current_section

    @staticmethod
    def _pick_sentence_at_offset(text: str, pos: int) -> str:
        """Extracts the single minimal complete sentence containing character offset `pos`."""
        if not text:
            return ""
        if "|" in text:
            return text.strip()
        # Split only on real sentence endings, ignoring Rs., p.a., p.m., No., incl., excl., decimals
        split_pattern = re.compile(
            r"(?<!\bRs)(?<!\bNo)(?<!\bp\.a)(?<!\bp\.m)(?<!\bSr)(?<!\bSt)(?<!\bincl)(?<!\bexcl)(?<!\be\.g)(?<!\bi\.e)(?<!\d)[.!?](?=\s+[A-Z0-9(]|\s*$)"
        )
        boundaries = [0]
        for m in split_pattern.finditer(text):
            boundaries.append(m.end())
        boundaries.append(len(text))

        clamped_pos = max(0, min(pos, max(len(text) - 1, 0)))
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            if start <= clamped_pos <= end:
                seg = text[start:end].strip()
                if seg:
                    return seg
        return text.strip()

    def _extract_minimal_evidence(
        self,
        raw_lines: List[str],
        norm_lines: List[str],
        sections: List[Optional[str]],
        idx: int,
        match_start: int = 0,
        join_next: bool = False,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Returns (minimal_evidence_sentence, original_raw_text, source_section)
        without concatenating unrelated sentences.
        """
        norm_ln = norm_lines[idx]
        raw_ln = raw_lines[idx]
        sec = sections[idx] if idx < len(sections) else None

        if join_next and idx + 1 < len(norm_lines):
            # Join only if line doesn't already terminate with a complete sentence or table row
            if "|" in norm_ln:
                evidence = f"{norm_ln} | {norm_lines[idx + 1]}"
                raw_ev = f"{raw_ln} | {raw_lines[idx + 1]}"
            else:
                combined_norm = f"{norm_ln} {norm_lines[idx + 1]}".strip()
                combined_raw = f"{raw_ln} {raw_lines[idx + 1]}".strip()
                evidence = self._pick_sentence_at_offset(combined_norm, match_start)
                raw_ev = self._pick_sentence_at_offset(combined_raw, min(match_start, max(len(combined_raw) - 1, 0)))
        else:
            evidence = self._pick_sentence_at_offset(norm_ln, match_start)
            raw_ev = self._pick_sentence_at_offset(raw_ln, min(match_start, max(len(raw_ln) - 1, 0)))

        return evidence[:320], raw_ev[:320], sec

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

        # Fallback: numeric value (must not be part of code/reference like A-1, FAR-2950, Section 3, or followed by %)
        cleaned_win = re.sub(r"\b(?:section|clause|table|page|annexure|item|para|part|ref|code)\s+\d+", " ", window, flags=re.IGNORECASE)
        m2 = re.search(
            r"(?<![A-Za-z\d-])(\d[\d,]*(?:\.\d+)?)(?![\d.,]*\s*%|\s*(?:days?\b|months?\b|years?\b|yrs?\b|mths?\b|p\.?a\.?\b))",
            cleaned_win,
            re.IGNORECASE,
        )
        if m2:
            val = cls._parse_plain_number(m2.group(1))
            if val is not None:
                after = cleaned_win[m2.end():m2.end() + 20].lower()
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
            # Generic fallback: "<Word> <Word> Bank / Finance / Housing / Capital Limited" (supports ALL-CAPS & mixed case)
            gen_lender = re.search(
                r"\b([A-Z][A-Za-z&\s]{2,45}?(?:Bank|Housing\s+Finance|Finance|Capital|Financial\s+Services)(?:\s+(?:Limited|Ltd\.?))?)\b",
                full_norm,
                re.IGNORECASE,
            )
            if gen_lender:
                cand = gen_lender.group(1).strip()
                if not re.search(r"^(?:conditional|in-principle|home\s+loan|loan|sanction|facility)", cand, re.IGNORECASE):
                    lender = cand

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

        # Prepare page lines with section tracking and raw/normalized alignment
        page_records: List[Dict[str, Any]] = []
        active_sec: Optional[str] = None
        for p in pages:
            r_lines, n_lines, secs, active_sec = self._prepare_page_lines(p["text"], active_sec, p.get("raw_text"))
            page_records.append({
                "page": p["page"],
                "raw_lines": r_lines,
                "norm_lines": n_lines,
                "sections": secs,
            })

        # Helper to scan lines across all pages with clean minimal sentence evidence
        def scan_pages_for_amount(
            label_patterns: List[str],
            exclude_pattern: Optional[str] = None,
            min_val: float = 0.0,
        ) -> Tuple[Optional[float], ExtractedFieldInfo]:
            for pr in page_records:
                p_num = pr["page"]
                lines = pr["norm_lines"]
                raw_lines = pr["raw_lines"]
                sections = pr["sections"]
                for i, line in enumerate(lines):
                    if exclude_pattern and re.search(exclude_pattern, line, re.IGNORECASE):
                        continue
                    for pat in label_patterns:
                        m = re.search(pat, line, re.IGNORECASE)
                        if m:
                            join_next = False
                            search_win = line[m.end():]
                            if (
                                not re.search(r"\d|lakh|crore|nil|zero", search_win, re.IGNORECASE)
                                and i + 1 < len(lines)
                                and not re.match(r"^(?:\d+\.|[A-Z]\.|[-•*]|\(?[a-z]\)|footnote|note\b)", lines[i + 1], re.IGNORECASE)
                                and not (exclude_pattern and re.search(exclude_pattern, lines[i + 1], re.IGNORECASE))
                            ):
                                search_win += " " + lines[i + 1]
                                join_next = True
                            val = self._extract_inr_from_window(search_win)
                            if val is None and re.search(r"\b(?:nil|zero)\b", search_win, re.IGNORECASE):
                                val = 0.0
                            if val is not None and val >= min_val:
                                min_ev, raw_ev, sec = self._extract_minimal_evidence(
                                    raw_lines, lines, sections, i, m.start(), join_next=join_next
                                )
                                return val, ExtractedFieldInfo(
                                    status=FieldStatus.CONFIRMED,
                                    source_page=p_num,
                                    source_section=sec,
                                    evidence_snippet=min_ev,
                                    raw_text=raw_ev,
                                    extraction_confidence="HIGH",
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

        for pr in page_records:
            p_num = pr["page"]
            lines = pr["norm_lines"]
            raw_lines = pr["raw_lines"]
            sections = pr["sections"]
            for i, line in enumerate(lines):
                # Detect explicit ROI type row
                m_type = re.search(r"(?:rate\s*of\s*interest\s*(?:\(\s*roi\s*\)\s*)?type|interest\s*(?:rate\s*)?type|nature\s*of\s*interest)", line, re.IGNORECASE)
                if m_type:
                    win = line + (" " + lines[i + 1] if i + 1 < len(lines) else "")
                    detected = self._detect_interest_type(win)
                    if detected != InterestRateType.UNKNOWN:
                        int_type = detected
                        min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, m_type.start())
                        int_type_info = ExtractedFieldInfo(
                            status=FieldStatus.CONFIRMED,
                            source_page=p_num,
                            source_section=sec,
                            evidence_snippet=min_ev,
                            raw_text=raw_ev,
                            extraction_confidence="HIGH",
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
                                    min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, m.start())
                                    rate_info = ExtractedFieldInfo(
                                        status=FieldStatus.CONFIRMED,
                                        source_page=p_num,
                                        source_section=sec,
                                        evidence_snippet=min_ev,
                                        raw_text=raw_ev,
                                        extraction_confidence="HIGH",
                                    )
                                    # Also infer interest type if written right alongside the rate e.g. "9.25% floating"
                                    if int_type == InterestRateType.UNKNOWN:
                                        detected = self._detect_interest_type(line + " " + after_label)
                                        if detected != InterestRateType.UNKNOWN:
                                            int_type = detected
                                            int_type_info = ExtractedFieldInfo(
                                                status=FieldStatus.CONFIRMED,
                                                source_page=p_num,
                                                source_section=sec,
                                                evidence_snippet=min_ev,
                                                raw_text=raw_ev,
                                                extraction_confidence="HIGH",
                                            )
                                    break

            # Fallback prose match per page: e.g. "9.25% p.a. floating interest" or "interest @ 9.25% floating"
            if rate is None:
                for i, line in enumerate(lines):
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
                            min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, pm.start())
                            rate_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                source_section=sec,
                                evidence_snippet=min_ev,
                                raw_text=raw_ev,
                                extraction_confidence="MEDIUM",
                            )
                            if int_type == InterestRateType.UNKNOWN:
                                detected = self._detect_interest_type(line)
                                if detected != InterestRateType.UNKNOWN:
                                    int_type = detected
                                    int_type_info = ExtractedFieldInfo(
                                        status=FieldStatus.CONFIRMED,
                                        source_page=p_num,
                                        source_section=sec,
                                        evidence_snippet=min_ev,
                                        raw_text=raw_ev,
                                        extraction_confidence="MEDIUM",
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
        for pr in page_records:
            p_num = pr["page"]
            lines = pr["norm_lines"]
            raw_lines = pr["raw_lines"]
            sections = pr["sections"]
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
                    min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, m_lbl.start() if m_lbl.start() < len(line) else 0)
                    tenure_info = ExtractedFieldInfo(
                        status=FieldStatus.CONFIRMED,
                        source_page=p_num,
                        source_section=sec,
                        evidence_snippet=min_ev,
                        raw_text=raw_ev,
                        extraction_confidence="HIGH",
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
                    min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, m_mo.start() if m_mo.start() < len(line) else 0)
                    tenure_info = ExtractedFieldInfo(
                        status=FieldStatus.CONFIRMED,
                        source_page=p_num,
                        source_section=sec,
                        evidence_snippet=min_ev,
                        raw_text=raw_ev,
                        extraction_confidence="HIGH",
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
                    min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, m_yr.start() if m_yr.start() < len(line) else 0)
                    tenure_info = ExtractedFieldInfo(
                        status=FieldStatus.CONFIRMED,
                        source_page=p_num,
                        source_section=sec,
                        evidence_snippet=min_ev,
                        raw_text=raw_ev,
                        extraction_confidence="HIGH",
                    )
                    break
            if tenure is not None:
                break

        # Prose fallback: "120 monthly instalments" or "120 months"
        if tenure is None:
            for pr in page_records:
                p_num = pr["page"]
                lines = pr["norm_lines"]
                raw_lines = pr["raw_lines"]
                sections = pr["sections"]
                for i, line in enumerate(lines):
                    m_p = re.search(r"\b(\d{2,3})\s+(?:monthly\s*install?ments?|emis?|months)\b", line, re.IGNORECASE)
                    if m_p:
                        tenure = int(m_p.group(1))
                        min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, m_p.start())
                        tenure_info = ExtractedFieldInfo(
                            status=FieldStatus.CONFIRMED,
                            source_page=p_num,
                            source_section=sec,
                            evidence_snippet=min_ev,
                            raw_text=raw_ev,
                            extraction_confidence="MEDIUM",
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
            r"(?:indicative\s*(?:first\s*)?emi|first\s*emi|equated\s*monthly\s*install?ment|monthly\s*emi|sanctioned\s*emi|emi\s*\(\s*install?ment\s*amount\s*\)|emi\s*amount|monthly\s*install?ment|install?ment\s*amount)",
            r"(?:^|\|\s*)emi\b(?:\s*[:|\-])",
        ]
        for pr in page_records:
            p_num = pr["page"]
            lines = pr["norm_lines"]
            raw_lines = pr["raw_lines"]
            sections = pr["sections"]
            for i, line in enumerate(lines):
                if re.search(emi_exclude, line, re.IGNORECASE):
                    continue
                for pat in emi_labels:
                    m = re.search(pat, line, re.IGNORECASE)
                    if m:
                        search_win = line[m.end():]
                        join_next = False
                        if not re.search(r"\d", search_win) and i + 1 < len(lines):
                            search_win += " " + lines[i + 1]
                            join_next = True
                        val = self._extract_inr_from_window(search_win)
                        if val is not None and val >= 50.0 and (principal is None or principal == 0 or val < principal or tenure == 1):
                            emi = val
                            min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, m.start(), join_next=join_next)
                            emi_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                source_section=sec,
                                evidence_snippet=min_ev,
                                raw_text=raw_ev,
                                extraction_confidence="HIGH",
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
            if any(k in lname for k in ("doc", "custody", "instrument", "stamp", "legal", "cersai", "title", "registry")):
                doc_val = (doc_val or 0.0) + c.amount
            elif "processing" in lname or "file handling" in lname:
                pf_val = (pf_val or 0.0) + c.amount
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
                            trigger_condition="Optional insurance selected by borrower and capitalized into principal",
                            frequency="ONE_TIME_CAPITALIZED",
                            payment_timing="capitalized into opening loan balance",
                            caused_by_borrower_action=True,
                            gst_classification=GSTClassification.GST_NOT_APPLICABLE,
                            gst_related_service="Financed Insurance",
                            status=FieldStatus.CONFIRMED,
                            source_page=ins_info.source_page,
                            source_section=ins_info.source_section,
                            evidence_snippet=ins_info.evidence_snippet,
                            raw_text=ins_info.raw_text,
                            extraction_confidence=ins_info.extraction_confidence,
                        )
                    )
                else:
                    field_metadata["bundled_insurance"] = ins_info
                    is_cust_funded = bool(re.search(r"customer\s*funded", ins_info.evidence_snippet or "", re.IGNORECASE))
                    charges.append(
                        ExtractedCharge(
                            name="Customer-Funded Insurance" if is_cust_funded else "Bundled Insurance (Upfront)",
                            amount=ins_amt,
                            category=ChargeCategory.THIRD_PARTY if is_cust_funded else ChargeCategory.CONFIRMED_UPFRONT,
                            trigger_condition="Customer-funded insurance premium selected at onboarding" if is_cust_funded else "Upfront insurance premium deduction",
                            frequency="ONE_TIME",
                            payment_timing="paid/funded separately by customer" if is_cust_funded else "deducted upfront from disbursement",
                            caused_by_borrower_action=is_cust_funded,
                            mandatory_status="CUSTOMER_FUNDED" if is_cust_funded else "MANDATORY",
                            is_mandatory=not is_cust_funded,
                            is_capitalized=False,
                            gst_classification=GSTClassification.GST_UNKNOWN,
                            third_party_recovery_method=(
                                ThirdPartyRecoveryMethod.PAID_DIRECTLY_BY_BORROWER if is_cust_funded else None
                            ),
                            status=FieldStatus.CONFIRMED,
                            source_page=ins_info.source_page,
                            source_section=ins_info.source_section,
                            evidence_snippet=ins_info.evidence_snippet,
                            raw_text=ins_info.raw_text,
                            extraction_confidence=ins_info.extraction_confidence,
                            dedup_id=f"{'third_party' if is_cust_funded else 'confirmed_upfront'}:insurance:{round(ins_amt, 2)}",
                        )
                    )

        # 8. Explicit GST Amount, GST Rate %, and Tax Status Note (NEVER invent 18%!)
        gst_amt: Optional[float] = None
        gst_rate: Optional[float] = None
        gst_note: Optional[str] = None
        gst_note_page: Optional[int] = None
        gst_note_sec: Optional[str] = None
        gst_note_raw: Optional[str] = None
        gst_info = ExtractedFieldInfo(status=FieldStatus.NOT_FOUND)

        for pr in page_records:
            p_num = pr["page"]
            lines = pr["norm_lines"]
            raw_lines = pr["raw_lines"]
            sections = pr["sections"]
            for i, line in enumerate(lines):
                # Explicit GST % e.g. "18% GST" or "GST @ 18%"
                if gst_rate is None:
                    rate_m = re.search(
                        r"(?:(\d{1,2}(?:\.\d+)?)\s*%\s*(?:applicable\s*)?gst\b|\bgst\s*(?:@|at)?\s*(\d{1,2}(?:\.\d+)?)\s*%)",
                        line,
                        re.IGNORECASE,
                    )
                    if rate_m:
                        gst_rate = float(rate_m.group(1) or rate_m.group(2))
                        min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, rate_m.start())
                        gst_info = ExtractedFieldInfo(
                            status=FieldStatus.CONFIRMED,
                            source_page=p_num,
                            source_section=sec,
                            evidence_snippet=min_ev,
                            raw_text=raw_ev,
                            extraction_confidence="HIGH",
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
                            min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, amt_m.start())
                            gst_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                source_section=sec,
                                evidence_snippet=min_ev,
                                raw_text=raw_ev,
                                extraction_confidence="HIGH",
                            )

                # Unquantified applicable GST / indirect tax phrase (skip cap clauses like "upto 3% + applicable taxes")
                if gst_note is None and not re.search(r"(?:upto|up\s*to)\s*\d+(?:\.\d+)?\s*%", line, re.IGNORECASE):
                    tax_m = re.search(
                        r"(?:(?:plus|exclusive\s*of|\+)\s*(?:applicable\s*)?(?:gst|indirect\s*tax|taxes?)|applicable\s*indirect\s*tax|at\s*prevailing\s*rate|tax(?:es)?\s*(?:is\s*)?(?:calculated\s*separately|additional)|not\s*included\s*in\s*the\s*percentage)",
                        line,
                        re.IGNORECASE,
                    )
                    if tax_m:
                        min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, tax_m.start())
                        gst_note = min_ev
                        gst_note_page = p_num
                        gst_note_sec = sec
                        gst_note_raw = raw_ev

        # Reconcile when a document explicitly discloses BOTH a pre-tax % fee rule (e.g. "1.00% + Applicable taxes")
        # AND an itemized rupee fee matching exact 18% GST gross-up (e.g. 45,000 -> 53,100), provided tax isn't withheld separately.
        withholds_tax_or_disb = bool(
            re.search(
                r"calculated\s+separately\s+and\s+(?:is\s+)?not\s+included|final\s+disbursement\s+advice[^.]*will\s+evidence",
                full_norm,
                re.IGNORECASE,
            )
        )
        if gst_amt is None and gst_rate is None and not withholds_tax_or_disb and principal is not None and principal > 0 and pf_val is not None:
            for pct_rule_m in re.finditer(
                r"(?:processing\s*(?:charges?|fees?)|origination\s*(?:charges?|fees?))[^\n%]{0,40}?\b(\d+(?:\.\d+)?)\s*%\s*\+?\s*(?:applicable\s*)?(?:gst|indirect\s*tax|taxes?)",
                full_norm,
                re.IGNORECASE,
            ):
                if re.search(r"(?:upto|up\s*to|max(?:imum)?)\s*\d", pct_rule_m.group(0), re.IGNORECASE):
                    continue
                pct_val = float(pct_rule_m.group(1))
                base_fee = round((pct_val / 100.0) * principal, 2)
                expected_gross = round(base_fee * 1.18, 2)
                if base_fee > 0 and abs(pf_val - expected_gross) <= 5.0:
                    reconciled_gst = round(pf_val - base_fee, 2)
                    gst_amt = reconciled_gst
                    gst_rate = 18.0
                    gross_pf_val = pf_val
                    pf_val = base_fee
                    for ch in charges:
                        if ch.category == ChargeCategory.CONFIRMED_UPFRONT and ("processing" in ch.name.lower() or "file handling" in ch.name.lower()):
                            ch.amount = base_fee
                            ch.gst_classification = GSTClassification.GST_AMOUNT_CONFIRMED
                            ch.gst_related_service = ch.name
                            ch.tax_note = f"Base ₹{base_fee:,.2f} ({pct_val:.2f}%) + ₹{reconciled_gst:,.2f} GST (18%) = ₹{gross_pf_val:,.2f} total"
                            break
                    gst_note = f"Reconciled from {pct_val:.2f}% base (₹{base_fee:,.2f}) vs ₹{gross_pf_val:,.2f} total processing fee (GST ₹{reconciled_gst:,.2f} @ 18%)"
                    gst_info = ExtractedFieldInfo(
                        status=FieldStatus.CONFIRMED,
                        source_page=gst_note_page or 1,
                        source_section=gst_note_sec or "Key Facility Particulars",
                        evidence_snippet=gst_note,
                        raw_text=gst_note_raw or gst_note,
                        extraction_confidence="HIGH",
                    )
                    break

        if gst_amt is None and gst_rate is None and gst_note:
            gst_info = ExtractedFieldInfo(
                status=FieldStatus.NOT_FOUND,
                source_page=gst_note_page,
                source_section=gst_note_sec,
                evidence_snippet=gst_note,
                raw_text=gst_note_raw,
                extraction_confidence="HIGH",
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
        for pr in page_records:
            p_num = pr["page"]
            lines = pr["norm_lines"]
            raw_lines = pr["raw_lines"]
            sections = pr["sections"]
            for i, line in enumerate(lines):
                if re.search(r"minimum\s+pre[-\s]*payment\s+amount", line, re.IGNORECASE):
                    continue
                pm_prep = re.search(r"(?:prepayment|pre-payment|foreclosure)", line, re.IGNORECASE)
                if pm_prep:
                    if re.search(r"\b(?:nil|zero|no\s*(?:charges|penalty|fee))\b", line, re.IGNORECASE):
                        if has_penalty is None:
                            has_penalty = False
                            min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, pm_prep.start())
                            penalty_terms = min_ev
                            penalty_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                source_section=sec,
                                evidence_snippet=min_ev,
                                raw_text=raw_ev,
                                extraction_confidence="HIGH",
                            )
                    elif re.search(r"(\d+(?:\.\d+)?\s*%|INR\s*\d+)", line, re.IGNORECASE):
                        if has_penalty is None:
                            has_penalty = True
                            min_ev, raw_ev, sec = self._extract_minimal_evidence(raw_lines, lines, sections, i, pm_prep.start())
                            penalty_terms = min_ev
                            penalty_info = ExtractedFieldInfo(
                                status=FieldStatus.CONFIRMED,
                                source_page=p_num,
                                source_section=sec,
                                evidence_snippet=min_ev,
                                raw_text=raw_ev,
                                extraction_confidence="HIGH",
                            )
        field_metadata["prepayment_penalty"] = penalty_info

        # 11. Repayment Schedule & Annexure / Narrative Notes Scan
        sched_count, sched_emis = self._scan_repayment_schedule(pages)
        document_notes = self._scan_document_notes(pages, charges)

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
            document_notes=document_notes,
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
        strictly into confirmed_upfront, financed, conditional, contingent, or third_party.
        Retains minimal evidence sentence, source_section, raw_text, extraction_confidence,
        GST classification + service linkage, and third-party recovery method.
        """
        found: List[ExtractedCharge] = []
        seen_keys = set()

        page_records: List[Dict[str, Any]] = []
        active_sec: Optional[str] = None
        for p in pages:
            r_lines, n_lines, secs, active_sec = self._prepare_page_lines(p["text"], active_sec, p.get("raw_text"))
            page_records.append({
                "page": p["page"],
                "raw_lines": r_lines,
                "norm_lines": n_lines,
                "sections": secs,
            })

        charge_patterns = [
            # Financed / Capitalized insurance or credit-protection premium
            (
                "Financed Insurance",
                r"(?:(?:optional\s*)?credit[-\s]*protection\s*premium|financed\s*(?:credit\s*)?(?:shield|life|health|property|protection)?\s*(?:insurance|premium)|capitalis?ed\s*(?:insurance|premium)|insurance\s*(?:premium\s*)?\(?\s*financed\s*\)?|insurance\s*(?:added|capitalis?ed)\s*to\s*(?:loan|principal|facility))",
                ChargeCategory.FINANCED,
            ),
            # Conditional charges (must be checked BEFORE generic fee patterns)
            (
                "Conditional Mandate Charge",
                r"(?:conditional\s*(?:mandate|nach|ecs|e-mandate)?\s*(?:charge|fee|cost)|(?:mandate|nach|ecs|e-mandate)\s*(?:setup|registration|processing|verification|bounce|dishonour)?(?:\s*(?:and|&)\s*\w+)?\s*(?:charge|fee|cost)|(?:overdue\s*servicing|bounce|dishonour|cheque\s*return|delayed\s*payment|penal)\s*(?:charge|fee|cost))",
                ChargeCategory.CONDITIONAL,
            ),
            # Third-party charges
            (
                "Third-Party Charge",
                r"(?:third[-\s]*party\s*(?:charge|fee|valuation|legal|insurance)|payable\s*directly\s*to\s*(?:third\s*party|advocate|valuer|sub-registrar))",
                ChargeCategory.THIRD_PARTY,
            ),
            # Processing / File handling fees (supports PDF table concatenated words)
            (
                "Processing Fee",
                r"(?:(?:b\)\s*)?processing\s*fees?|(?:c\)\s*)?balance\s*processing\s*fees?|processing\s*charges?|loan\s*processing\s*(?:fee|charge)|file\s*handling(?:\s*(?:and|&)\s*credit\s*assessment|\s*fee|\s*charge)?|origination\s*(?:fee|charge)|admin(?:istrative)?\s*(?:fee|charge))",
                ChargeCategory.CONFIRMED_UPFRONT,
            ),
            # Other cash recoveries & activation reserves
            (
                "Other Cash Recoveries",
                r"(?:other\s*(?:upfront\s*)?cash\s*recover(?:y|ies)|other\s*upfront\s*recover(?:y|ies)|cash\s*recover(?:y|ies)|upfront\s*recover(?:y|ies)|other\s*deductions?|facility\s*activation\s*reserve|activation\s*reserve)",
                ChargeCategory.CONFIRMED_UPFRONT,
            ),
            # Documentation / Custody / Valuation / Title Search / Registry / CERSAI / Stamp Duty / Login
            (
                "Documentation / Statutory Fee",
                r"(?:document\s*custody(?:\s*(?:and|/|&)\s*instrument\s*processing)?|documentation\s*(?:&\s*stamp\s*)?(?:charges?|fees?)|legal\s*(?:&\s*technical\s*)?(?:charges?|fees?)|(?:property\s*)?title\s*search(?:\s*(?:and|&)\s*technical\s*panel\s*visit|\s*(?:charges?|fees?))?|technical\s*panel\s*visit|valuation\s*(?:charges?|fees?)|technical\s*(?:charges?|fees?)|login\s*(?:charges?|fees?)|(?:electronic\s*)?registry\s*filing(?:\s*(?:and|&)\s*security[-\s]*interest\s*notation)?|cersai\s*(?:charges?|fees?)|stamp\s*duty|imd\b)",
                ChargeCategory.CONFIRMED_UPFRONT,
            ),
        ]

        # Skip summary total rows so individual items aren't double-counted
        summary_skip = r"total\s*(?:upfront|cash)?\s*(?:fees?|charges?|recover(?:y|ies)|deductions?)|^prepayment|^foreclosure|switching/conversion"

        for pr in page_records:
            p_num = pr["page"]
            lines = pr["norm_lines"]
            raw_lines = pr["raw_lines"]
            sections = pr["sections"]
            for i, line in enumerate(lines):
                if re.search(summary_skip, line, re.IGNORECASE):
                    continue
                # Skip section headers ending with ':' that contain no figures
                if re.match(r"^(?:[A-Z]\.|\d+\.)?\s*[A-Za-z\s/&(),-]+:\s*$", line):
                    continue
                # Allow lines with "% of ... principal" even without INR amount; skip other %-only policy clauses
                has_pct_of_principal = bool(
                    re.search(r"\d+(?:\.\d+)?\s*%\s*of\s*(?:sanctioned\s*|approved\s*|loan\s*)?principal", line, re.IGNORECASE)
                )
                if re.search(r"\b\d+(?:\.\d+)?\s*%", line) and not re.search(r"INR\s*[\d,]+", line, re.IGNORECASE) and not has_pct_of_principal:
                    continue

                for default_name, pat, cat in charge_patterns:
                    m = re.search(pat, line, re.IGNORECASE)
                    if not m:
                        continue

                    next_line = lines[i + 1] if i + 1 < len(lines) else ""
                    next_is_separate_item = bool(
                        not next_line
                        or re.match(r"^(?:\d+\.|[A-Z]\.|[a-z]\)\s+|[A-Z]-\d)", next_line)
                        or re.search(r"INR\s*[\d,]+", next_line, re.IGNORECASE)
                    )
                    cond_ctx = line if next_is_separate_item else f"{line} {next_line}"

                    # Override category to CONDITIONAL only when conditional applicability governs the fee itself
                    # (ignore account-opening debits and balance-reconciliation caveats like "any unutilised balance, if applicable")
                    cond_clean_ctx = re.sub(
                        r"any\s+unutilis?ed\s+balance[^.]*\.?",
                        " ",
                        cond_ctx,
                        flags=re.IGNORECASE,
                    )
                    effective_cat = cat
                    is_account_opening_debit = bool(
                        re.search(r"when\s+the\s+loan\s+account\s+is\s+opened|account[-\s]*opening\s*debit", cond_ctx, re.IGNORECASE)
                    )
                    if effective_cat != ChargeCategory.FINANCED and not is_account_opening_debit and re.search(
                        r"\b(?:conditional|if\s+applicable|applicable\s+only\s+(?:for|in|to|if|where|when)|upon\s+failure|in\s+case\s+of|contingent|if\s+mandate|if\s+the\s+account|if\s+an\s+install?ment|per\s+event|per\s+bounce|if\s+dishonour|may\s+be\s+collected\s+if|may\s+be\s+applied|applies\s+only\s+where)\b",
                        cond_clean_ctx,
                        re.IGNORECASE,
                    ):
                        effective_cat = ChargeCategory.CONDITIONAL

                    # Verify FINANCED insurance cues when matched via generic protection label
                    if effective_cat == ChargeCategory.FINANCED:
                        is_cap_ctx = bool(re.search(
                            r"(?:added\s*to\s*(?:the\s*)?(?:facility|loan|principal|opening)|financed|capitalis?ed|rather\s*than\s*collected\s*separately|booked\s*as\s*a\s*financed)",
                            cond_ctx,
                            re.IGNORECASE,
                        ))
                        if not is_cap_ctx and principal is not None and opening_bal is not None:
                            is_cap_ctx = True  # Checked via balance reconciliation below
                        if not is_cap_ctx:
                            effective_cat = ChargeCategory.CONFIRMED_UPFRONT

                    search_win = line[m.end():]
                    join_next = False
                    if (
                        "|" not in line
                        and not re.search(r"\d", search_win)
                        and not re.search(r"INR\s*\d", line[:m.start()], re.IGNORECASE)
                        and i + 1 < len(lines)
                        and "|" not in lines[i + 1]
                        and not re.match(r"^(?:\d+\.|[A-Z]\.|[-•*]|\(?[a-z]\)|footnote|note\b)", lines[i + 1], re.IGNORECASE)
                    ):
                        search_win += " " + lines[i + 1]
                        join_next = True

                    # Calculate percentage-of-principal fee when stated (e.g. "1.25% of sanctioned principal; minimum INR 4,000")
                    val: Optional[float] = None
                    pct_p_m = re.search(
                        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(?:sanctioned\s*|approved\s*|loan\s*)?principal",
                        search_win,
                        re.IGNORECASE,
                    )
                    if pct_p_m and principal is not None and principal > 0:
                        val = round((float(pct_p_m.group(1)) / 100.0) * principal, 2)
                        min_m = re.search(r"minimum\s*(?:INR\s*)?([\d,]+(?:\.\d+)?)", search_win, re.IGNORECASE)
                        if min_m:
                            min_v = self._parse_plain_number(min_m.group(1))
                            if min_v is not None:
                                val = max(val, min_v)
                        max_m = re.search(r"(?:maximum|upto)\s*(?:INR\s*)?([\d,]+(?:\.\d+)?)", search_win, re.IGNORECASE)
                        if max_m:
                            max_v = self._parse_plain_number(max_m.group(1))
                            if max_v is not None:
                                val = min(val, max_v)
                    elif re.search(r"INR\s*\d", search_win, re.IGNORECASE) or "|" in line:
                        val = self._extract_inr_from_window(search_win)

                    if val is None or val <= 0:
                        continue

                    # Avoid confusing principal/balance amounts for charges
                    if principal is not None and val >= principal * 0.8 and effective_cat == ChargeCategory.CONFIRMED_UPFRONT:
                        continue

                    matched_label = re.sub(r"^(?:a\)|b\)|c\))\s*", "", m.group(0).strip(), flags=re.IGNORECASE).strip()
                    clean_name = matched_label[:60] if matched_label else default_name

                    # Minimal evidence sentence + unnormalized raw_text + source_section
                    min_ev, raw_ev, sec = self._extract_minimal_evidence(
                        raw_lines, lines, sections, i, m.start(), join_next=join_next
                    )

                    # GST classification and exact service linkage
                    tax_note = None
                    gst_cls = GSTClassification.GST_UNKNOWN
                    gst_service: Optional[str] = None
                    ev_search = f"{line} {search_win}"
                    if re.search(r"(?:including|inclusive\s*of)\s*(?:applicable\s*)?(?:gst|indirect\s*tax|tax(?:es)?)", ev_search, re.IGNORECASE):
                        tax_note = "inclusive of tax"
                        gst_cls = GSTClassification.GST_INCLUDED
                        gst_service = clean_name
                    elif re.search(r"(?:(\d{1,2}(?:\.\d+)?)\s*%\s*(?:applicable\s*)?gst|\bgst\s*(?:@|at)?\s*(\d{1,2}(?:\.\d+)?)\s*%)", ev_search, re.IGNORECASE):
                        tax_note = "explicit GST rate disclosed"
                        gst_cls = GSTClassification.GST_RATE_CONFIRMED
                        gst_service = clean_name
                    elif re.search(r"(?:plus|exclusive\s*of|\+)\s*(?:applicable\s*)?(?:gst|indirect\s*tax|taxes?)|tax(?:es)?\s*(?:additional|calculated\s*separately)|not\s*included\s*in\s*the\s*percentage", ev_search, re.IGNORECASE):
                        tax_note = "plus applicable GST/tax"
                        gst_cls = GSTClassification.GST_POTENTIALLY_APPLICABLE
                        gst_service = clean_name
                    elif re.search(r"(?:exempt\s*from\s*gst|no\s*gst\s*applicable|not\s*subject\s*to\s*gst)", ev_search, re.IGNORECASE):
                        gst_cls = GSTClassification.GST_NOT_APPLICABLE
                        gst_service = clean_name
                    elif effective_cat == ChargeCategory.FINANCED:
                        gst_cls = GSTClassification.GST_NOT_APPLICABLE

                    # Third-party recovery method detection
                    tp_method: Optional[ThirdPartyRecoveryMethod] = None
                    if re.search(r"recovered\s*(?:by\s*(?:the\s*)?lender\s*)?against\s*(?:the\s*)?(?:third[-\s]*party\s*)?agency\s*invoice", ev_search, re.IGNORECASE):
                        tp_method = ThirdPartyRecoveryMethod.RECOVERED_AGAINST_AGENCY_INVOICE
                    elif re.search(r"paid\s*directly\s*by\s*(?:the\s*)?(?:applicant|borrower)|payable\s*directly\s*to\s*(?:sub-registrar|authority|third\s*party|bank)|levied\s*by\s*(?:the\s*)?borrower'?s\s*bank", ev_search, re.IGNORECASE):
                        tp_method = ThirdPartyRecoveryMethod.PAID_DIRECTLY_BY_BORROWER
                        # Direct borrower payments must NOT reduce lender disbursement unless explicitly deducted
                        if not re.search(r"deducted\s*from\s*(?:the\s*)?(?:facility|loan|disbursement)", ev_search, re.IGNORECASE):
                            effective_cat = ChargeCategory.THIRD_PARTY
                    elif re.search(r"collected\s*by\s*(?:the\s*)?lender\s*(?:on\s*behalf|for\s*(?:the\s*)?(?:agency|third[-\s]*party))", ev_search, re.IGNORECASE):
                        tp_method = ThirdPartyRecoveryMethod.COLLECTED_BY_LENDER
                    elif effective_cat == ChargeCategory.THIRD_PARTY or re.search(r"empanelled\s*agency|third[-\s]*party|advocate|valuer", ev_search, re.IGNORECASE):
                        tp_method = ThirdPartyRecoveryMethod.RECOVERY_METHOD_UNKNOWN

                    # Trigger condition, frequency, timing, borrower-action flag & mandatory status
                    trigger_cond: Optional[str] = None
                    frequency: Optional[str] = None
                    caused_by_action = False
                    if effective_cat == ChargeCategory.FINANCED:
                        mandatory_status = "OPTIONAL_FINANCED" if "optional" in cond_ctx.lower() else "FINANCED"
                        is_mandatory = "optional" not in cond_ctx.lower()
                        is_cap = True
                        payment_timing = "capitalized into opening loan balance"
                        frequency = "ONE_TIME_CAPITALIZED"
                        trigger_cond = "Selected at loan origination and financed into opening balance"
                        caused_by_action = "optional" in cond_ctx.lower()
                    elif effective_cat == ChargeCategory.CONDITIONAL:
                        mandatory_status = "CONDITIONAL"
                        is_mandatory = False
                        is_cap = False
                        cond_clause_m = re.search(r"(applicable\s+only\s+(?:for|in|to|if|where|when)\s+[^.)]+)", cond_ctx, re.IGNORECASE)
                        if "bounce" in clean_name.lower() or "dishonour" in clean_name.lower() or "return" in clean_name.lower() or "per bounce" in cond_ctx.lower():
                            caused_by_action = True
                            payment_timing = "per bounce event"
                            frequency = "PER_BOUNCE_EVENT"
                            trigger_cond = "Triggered upon cheque or NACH/ECS bounce or dishonour"
                        elif "mandate" in clean_name.lower() or "nach" in clean_name.lower():
                            caused_by_action = True
                            payment_timing = "at mandate setup if assisted channel used"
                            frequency = "ONE_TIME"
                            trigger_cond = "Applies only when assisted/manual mandate setup route is used instead of self-service"
                        elif "overdue" in clean_name.lower():
                            caused_by_action = True
                            payment_timing = "per overdue event"
                            frequency = "PER_OVERDUE_EVENT"
                            trigger_cond = "Triggered when an instalment remains unpaid beyond due date"
                        elif cond_clause_m:
                            caused_by_action = False
                            payment_timing = "conditional upon geographic / transaction applicability"
                            frequency = "CONDITIONAL_ONE_TIME"
                            trigger_cond = cond_clause_m.group(1).strip()
                        else:
                            caused_by_action = True
                            payment_timing = "conditional upon trigger event"
                            frequency = "CONDITIONAL"
                            trigger_cond = "Applies only when the specified conditional trigger occurs"
                    elif effective_cat == ChargeCategory.THIRD_PARTY:
                        mandatory_status = "THIRD_PARTY_DIRECT" if tp_method == ThirdPartyRecoveryMethod.PAID_DIRECTLY_BY_BORROWER else "THIRD_PARTY"
                        is_mandatory = False
                        is_cap = False
                        payment_timing = (
                            "paid directly by borrower"
                            if tp_method == ThirdPartyRecoveryMethod.PAID_DIRECTLY_BY_BORROWER
                            else "payable to third party"
                        )
                        frequency = "AS_APPLICABLE"
                        trigger_cond = "Third-party agency or statutory obligation"
                    else:
                        mandatory_status = "MANDATORY"
                        is_mandatory = True
                        is_cap = False
                        frequency = "ONE_TIME"
                        if is_account_opening_debit:
                            payment_timing = "at account opening"
                        elif "one time" in search_win.lower():
                            payment_timing = "one-time upfront recovery"
                        else:
                            payment_timing = "at onboarding / first disbursement"

                    canon_prefix = "".join(re.findall(r"[a-z]", clean_name.lower()))[:10]
                    dedup_key = (effective_cat.value, round(val, 2), canon_prefix)
                    dedup_id = f"{effective_cat.value}:{canon_prefix}:{round(val, 2)}"

                    # Deduplicate across table+text AND summary/reconciliation annexures; record repeated page refs
                    existing_match = next(
                        (c for c in found if c.dedup_id == dedup_id),
                        None,
                    )
                    if existing_match is not None:
                        if p_num not in existing_match.repeated_on_pages and p_num != existing_match.source_page:
                            existing_match.repeated_on_pages.append(p_num)
                        if tax_note and not existing_match.tax_note:
                            existing_match.tax_note = tax_note
                        if gst_cls != GSTClassification.GST_UNKNOWN and existing_match.gst_classification == GSTClassification.GST_UNKNOWN:
                            existing_match.gst_classification = gst_cls
                            existing_match.gst_related_service = gst_service
                        if tp_method and existing_match.third_party_recovery_method in (None, ThirdPartyRecoveryMethod.RECOVERY_METHOD_UNKNOWN):
                            existing_match.third_party_recovery_method = tp_method
                        break
                    if dedup_key in seen_keys:
                        break
                    seen_keys.add(dedup_key)

                    note_parts = []
                    if re.search(r"payable\s*to\s*empanelled\s*agency", search_win, re.IGNORECASE):
                        note_parts.append("Payable to empanelled agency / third-party provider")
                    if is_account_opening_debit:
                        note_parts.append("Debited at account opening")

                    found.append(
                        ExtractedCharge(
                            name=clean_name,
                            amount=round(val, 2),
                            currency="INR",
                            category=effective_cat,
                            trigger_condition=trigger_cond,
                            frequency=frequency,
                            payment_timing=payment_timing,
                            caused_by_borrower_action=caused_by_action,
                            mandatory_status=mandatory_status,
                            is_mandatory=is_mandatory,
                            is_capitalized=is_cap,
                            gst_classification=gst_cls,
                            gst_related_service=gst_service,
                            third_party_recovery_method=tp_method,
                            status=FieldStatus.CONFIRMED,
                            source_page=p_num,
                            source_section=sec,
                            evidence_snippet=min_ev,
                            raw_text=raw_ev,
                            extraction_confidence="HIGH",
                            dedup_id=dedup_id,
                            tax_note=tax_note,
                            notes="; ".join(note_parts) if note_parts else None,
                        )
                    )
                    break

        # Pass B: Contingent percentage-based penal charges across multi-line paragraphs (stop at next numbered clause!)
        for pr in page_records:
            p_num = pr["page"]
            lines = pr["norm_lines"]
            raw_lines = pr["raw_lines"]
            sections = pr["sections"]
            for i, line in enumerate(lines):
                pm_penal = re.search(r"\bpenal\s*(?:charges?|interest)|default\s*(?:&|and)?\s*penal\s*charges?|default\s*interest", line, re.IGNORECASE)
                if pm_penal:
                    win_lines = [line]
                    raw_win_lines = [raw_lines[i]]
                    for step in (1, 2):
                        if i + step < len(lines):
                            nxt = lines[i + step]
                            # Do not bleed into a subsequent numbered/lettered clause like "26. If a change..."
                            if re.match(r"^(?:\d{1,2}\.|[A-Z]\.|[a-z]\))\s+", nxt):
                                break
                            win_lines.append(nxt)
                            raw_win_lines.append(raw_lines[i + step])
                    para_win = " ".join(win_lines)
                    raw_para_win = " ".join(raw_win_lines)
                    penal_pos = para_win.lower().find("penal")
                    if penal_pos < 0:
                        penal_pos = pm_penal.start()
                    single_sent_norm = self._pick_sentence_at_offset(para_win, penal_pos)
                    single_sent_raw = self._pick_sentence_at_offset(raw_para_win, min(penal_pos, max(len(raw_para_win) - 1, 0)))
                    pct_m = re.search(
                        r"(\d+(?:\.\d+)?)\s*%\s*(?:per\s*annum|p\.?\s*a\.?|per\s*month|p\.?\s*m\.?)?",
                        single_sent_norm[single_sent_norm.lower().find("penal"):] if "penal" in single_sent_norm.lower() else single_sent_norm,
                        re.IGNORECASE,
                    )
                    if pct_m:
                        rate_val = float(pct_m.group(1))
                        is_monthly_rate = bool(re.search(r"per\s*month|p\.?\s*m\.?", pct_m.group(0), re.IGNORECASE))
                        freq_val = "PER_MONTH_ON_OVERDUE" if is_monthly_rate else "PER_ANNUM_ON_OVERDUE"
                        default_basis = (
                            f"{rate_val:.2f}% per month on overdue payment ({rate_val * 12.0:.2f}% p.a. annualized)"
                            if is_monthly_rate
                            else f"{rate_val:.2f}% per annum on overdue amount"
                        )
                        dedup_id = f"contingent:penalcharg:{round(rate_val, 2)}:{'m' if is_monthly_rate else 'a'}"
                        if not any(c.dedup_id == dedup_id for c in found):
                            not_cap = bool(re.search(r"penal\s*charges?\s*(?:are|is)\s*not\s*capitalis?ed", para_win, re.IGNORECASE))
                            basis_m = re.search(
                                r"(?:calculated\s*on\s*[^.]+|is\s*\d+(?:\.\d+)?\s*%[^.]+|at\s*\d+(?:\.\d+)?\s*%[^.]+)",
                                single_sent_norm,
                                re.IGNORECASE,
                            )
                            rate_basis_txt = default_basis if is_monthly_rate else (basis_m.group(0).strip()[:120] if basis_m else default_basis)
                            found.append(
                                ExtractedCharge(
                                    name="Penal Charge",
                                    amount=None,
                                    rate_pct=rate_val,
                                    rate_basis=rate_basis_txt,
                                    currency="INR",
                                    category=ChargeCategory.CONTINGENT,
                                    trigger_condition=(
                                        "Calculated monthly on overdue payment for the period of default"
                                        if is_monthly_rate
                                        else "Calculated on overdue amount for the period of default"
                                    ),
                                    frequency=freq_val,
                                    payment_timing="upon default / overdue period",
                                    caused_by_borrower_action=True,
                                    mandatory_status="CONTINGENT",
                                    is_mandatory=False,
                                    is_capitalized=False,
                                    gst_classification=GSTClassification.GST_NOT_APPLICABLE,
                                    gst_related_service="Penal Charge",
                                    status=FieldStatus.CONFIRMED,
                                    source_page=p_num,
                                    source_section=sections[i] if i < len(sections) else None,
                                    evidence_snippet=single_sent_norm[:320],
                                    raw_text=single_sent_raw[:320],
                                    extraction_confidence="HIGH",
                                    dedup_id=dedup_id,
                                    notes="Penal charges are not capitalised" if not_cap else "Contingent upon payment default",
                                )
                            )
                        else:
                            existing_penal = next(c for c in found if c.dedup_id == dedup_id)
                            if p_num != existing_penal.source_page and p_num not in existing_penal.repeated_on_pages:
                                existing_penal.repeated_on_pages.append(p_num)

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

    def _scan_document_notes(
        self, pages: List[Dict[str, Any]], charges: List[ExtractedCharge]
    ) -> List[DocumentNote]:
        """
        Scans all pages, annexures, footnotes, and narrative paragraphs for operational,
        tax, third-party agency, financed-insurance, mandate, and failed-debit notes,
        enriching matching charge items and recording repeated disclosure pages.
        """
        notes: List[DocumentNote] = []
        seen_topics = set()

        page_records: List[Dict[str, Any]] = []
        active_sec: Optional[str] = None
        for p in pages:
            r_lines, n_lines, secs, active_sec = self._prepare_page_lines(p["text"], active_sec, p.get("raw_text"))
            page_records.append({
                "page": p["page"],
                "raw_lines": r_lines,
                "norm_lines": n_lines,
                "sections": secs,
            })

        def add_note(
            topic: str,
            summary: str,
            page_num: int,
            min_ev: str,
            raw_ev: str,
            sec: Optional[str],
            confidence: str = "HIGH",
        ):
            key = (topic, page_num)
            if key not in seen_topics:
                seen_topics.add(key)
                notes.append(
                    DocumentNote(
                        topic=topic,
                        summary=summary,
                        source_page=page_num,
                        source_section=sec,
                        evidence_snippet=min_ev[:320],
                        raw_text=raw_ev[:320],
                        extraction_confidence=confidence,
                    )
                )

        for pr in page_records:
            p_num = pr["page"]
            lines = pr["norm_lines"]
            raw_lines = pr["raw_lines"]
            sections = pr["sections"]
            for i, line in enumerate(lines):
                win = line + (" " + lines[i + 1] if i + 1 < len(lines) else "")
                raw_win = raw_lines[i] + (" " + raw_lines[i + 1] if i + 1 < len(raw_lines) else "")

                # 1. Facility activation reserve FAR-2950 / non-refundable note
                m_far = re.search(r"FAR-\d+|facility\s+activation\s+reserve", line, re.IGNORECASE)
                if m_far:
                    code_m = re.search(r"\b(FAR-\d+)\b", win, re.IGNORECASE)
                    if code_m or re.search(r"not\s+a\s+refundable\s+security\s+deposit|reconciliation\s+is\s+limited", win, re.IGNORECASE):
                        code_txt = f"Internal code {code_m.group(1)}" if code_m else "Operational reserve"
                        min_ev = self._pick_sentence_at_offset(win, m_far.start())
                        raw_ev = self._pick_sentence_at_offset(raw_win, min(m_far.start(), max(len(raw_win) - 1, 0)))
                        add_note(
                            "Facility Activation Reserve",
                            f"{code_txt}; non-refundable account-opening debit.",
                            p_num,
                            min_ev,
                            raw_ev,
                            sections[i],
                        )
                        for c in charges:
                            if "activation reserve" in c.name.lower():
                                if p_num != c.source_page and p_num not in c.repeated_on_pages:
                                    c.repeated_on_pages.append(p_num)
                                extra = f"{code_txt}; not a refundable security deposit"
                                if not c.notes or extra not in c.notes:
                                    c.notes = f"{c.notes}; {extra}" if c.notes else extra

                # 2. Financed insurance explanation
                m_ins = re.search(r"(?:credit[-\s]*protection\s*premium|optional\s*protection\s*product)", line, re.IGNORECASE)
                if m_ins and re.search(
                    r"(?:added\s*to|financed\s*component|forms\s*part\s*of\s*the\s*opening|insurer'?s\s*policy)",
                    win,
                    re.IGNORECASE,
                ):
                    min_ev = self._pick_sentence_at_offset(win, m_ins.start())
                    raw_ev = self._pick_sentence_at_offset(raw_win, min(m_ins.start(), max(len(raw_win) - 1, 0)))
                    add_note(
                        "Financed Insurance Explanation",
                        "Optional credit-protection premium is capitalized into the opening loan balance rather than collected upfront.",
                        p_num,
                        min_ev,
                        raw_ev,
                        sections[i],
                    )
                    for c in charges:
                        if c.category == ChargeCategory.FINANCED:
                            if p_num != c.source_page and p_num not in c.repeated_on_pages:
                                c.repeated_on_pages.append(p_num)

                # 3. Tax treatment of processing / file handling fees (exact service linkage!)
                m_tax = re.search(r"(?:tax\s+note|file\s+handling\s+fee\s+A-1|applicable\s+indirect\s+tax\s+on\s+A-1)", line, re.IGNORECASE)
                if m_tax and re.search(
                    r"(?:calculated\s+separately|prevailing\s+rate|not\s+included\s+in\s+the\s+percentage)",
                    win,
                    re.IGNORECASE,
                ):
                    min_ev = self._pick_sentence_at_offset(win, m_tax.start())
                    raw_ev = self._pick_sentence_at_offset(raw_win, min(m_tax.start(), max(len(raw_win) - 1, 0)))
                    add_note(
                        "Tax Treatment of Processing Fee",
                        "Indirect tax on processing / file handling fee is calculated separately at prevailing rate (not quantified).",
                        p_num,
                        min_ev,
                        raw_ev,
                        sections[i],
                    )
                    for c in charges:
                        if c.category == ChargeCategory.CONFIRMED_UPFRONT and any(k in c.name.lower() for k in ("processing", "file handling")):
                            c.tax_note = "plus applicable indirect tax (calculated separately at prevailing rate; amount not disclosed)"
                            if c.gst_classification == GSTClassification.GST_UNKNOWN:
                                c.gst_classification = GSTClassification.GST_POTENTIALLY_APPLICABLE
                            c.gst_related_service = c.name
                            if p_num != c.source_page and p_num not in c.repeated_on_pages:
                                c.repeated_on_pages.append(p_num)

                # 4. Third-party agency recovery arrangements
                m_tp = re.search(r"(?:third[-\s]*party\s*providers|payable\s*to\s*empanelled\s*agency|agency\s*invoice|paid\s*directly\s*by\s*the\s*applicant)", win, re.IGNORECASE)
                if m_tp:
                    min_ev = self._pick_sentence_at_offset(win, m_tp.start())
                    raw_ev = self._pick_sentence_at_offset(raw_win, min(m_tp.start(), max(len(raw_win) - 1, 0)))
                    add_note(
                        "Third-Party Agency Recovery Arrangement",
                        "Agency/panel fees may be paid directly by borrower or recovered by lender against third-party agency invoice.",
                        p_num,
                        min_ev,
                        raw_ev,
                        sections[i],
                    )
                    for c in charges:
                        if any(k in c.name.lower() for k in ("title search", "technical panel")):
                            if p_num != c.source_page and p_num not in c.repeated_on_pages:
                                c.repeated_on_pages.append(p_num)
                            if re.search(r"agency\s*invoice", win, re.IGNORECASE):
                                c.third_party_recovery_method = ThirdPartyRecoveryMethod.RECOVERED_AGAINST_AGENCY_INVOICE
                            elif re.search(r"collected\s*by\s*(?:the\s*)?lender", win, re.IGNORECASE):
                                c.third_party_recovery_method = ThirdPartyRecoveryMethod.COLLECTED_BY_LENDER
                            arr = "Third-party agency recovery arrangement (paid directly or recovered against agency invoice)"
                            if not c.notes or "agency invoice" not in c.notes:
                                c.notes = f"{c.notes}; {arr}" if c.notes else arr

                # 5. Assisted-channel mandate fee note
                m_mand = re.search(r"assisted[-\s]*channel\s*mandate\s*fee|self[-\s]*service\s*mandate\s*route", line, re.IGNORECASE)
                if m_mand:
                    min_ev = self._pick_sentence_at_offset(win, m_mand.start())
                    raw_ev = self._pick_sentence_at_offset(raw_win, min(m_mand.start(), max(len(raw_win) - 1, 0)))
                    add_note(
                        "Assisted-Channel Mandate Fee",
                        "Mandate fee applies only if assisted/manual verification channel is used; self-service route carries zero fee.",
                        p_num,
                        min_ev,
                        raw_ev,
                        sections[i],
                    )
                    for c in charges:
                        if "mandate" in c.name.lower() and c.category == ChargeCategory.CONDITIONAL:
                            if p_num != c.source_page and p_num not in c.repeated_on_pages:
                                c.repeated_on_pages.append(p_num)

                # 6. Bank charges on failed debits (unquantified third-party charges paid directly via borrower's bank)
                m_fail = re.search(r"failed\s*debit", line, re.IGNORECASE)
                if m_fail and re.search(
                    r"(?:bank\s*charges|borrower'?s\s*bank|not\s*quantified)",
                    win,
                    re.IGNORECASE,
                ):
                    min_ev = self._pick_sentence_at_offset(win, m_fail.start())
                    raw_ev = self._pick_sentence_at_offset(raw_win, min(m_fail.start(), max(len(raw_win) - 1, 0)))
                    add_note(
                        "Bank Charges on Failed Debits",
                        "Failed debits may trigger third-party bank charges levied by the borrower's bank (unquantified).",
                        p_num,
                        min_ev,
                        raw_ev,
                        sections[i],
                    )
                    dedup_id = "third_party:faileddebitbank:unquantified"
                    if not any(c.dedup_id == dedup_id for c in charges):
                        charges.append(
                            ExtractedCharge(
                                name="Borrower Bank Failed-Debit Charges (Unquantified)",
                                amount=None,
                                currency="INR",
                                category=ChargeCategory.THIRD_PARTY,
                                trigger_condition="Upon failed NACH/e-mandate debit presentation",
                                frequency="PER_FAILED_DEBIT",
                                payment_timing="upon failed debit (paid via borrower's bank)",
                                caused_by_borrower_action=True,
                                mandatory_status="THIRD_PARTY_DIRECT",
                                is_mandatory=False,
                                is_capitalized=False,
                                gst_classification=GSTClassification.GST_UNKNOWN,
                                third_party_recovery_method=ThirdPartyRecoveryMethod.PAID_DIRECTLY_BY_BORROWER,
                                status=FieldStatus.NOT_FOUND,
                                source_page=p_num,
                                source_section=sections[i],
                                evidence_snippet=min_ev[:320],
                                raw_text=raw_ev[:320],
                                extraction_confidence="HIGH",
                                dedup_id=dedup_id,
                                notes="Third-party bank charges levied directly by borrower's bank on failed debits; not quantified",
                            )
                        )

        return notes

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
        gst_cls_val = (
            audit.gst_classification.value
            if hasattr(audit.gst_classification, "value")
            else str(audit.gst_classification)
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
            f"- **GST Classification**: `{gst_cls_val}` — Disclosed Amount: {gst_str}" + (f" ({audit.gst_status_note})" if audit.gst_status_note else ""),
            f"- **Final Net Disbursement**: {net_str}",
            f"- **Effective Annual APR**: {apr_str}",
        ])

        if audit.conditional_charges:
            lines.append("\n### Conditional & Contingent Charges (Tracked Individually — Not Summed)")
            for c in audit.conditional_charges:
                amt_disp = f"₹{c.amount:,.2f} {c.currency}" if c.amount is not None else (
                    f"{c.rate_pct:.2f}% ({c.rate_basis or 'contingent'})" if c.rate_pct is not None else "Unquantified"
                )
                pages_tag = f"Pages {', '.join(str(x) for x in [c.source_page] + c.repeated_on_pages if x)}"
                sec_tag = f" | Section: {c.source_section}" if c.source_section else ""
                lines.append(
                    f"- **{c.name}**: **{amt_disp}** | Trigger: {c.trigger_condition or 'Upon trigger'} | "
                    f"Frequency: `{c.frequency or 'CONDITIONAL'}` | Timing: {c.payment_timing or 'unspecified'} | "
                    f"Caused by borrower action: `{c.caused_by_borrower_action}` [{pages_tag}{sec_tag}]"
                )

        if audit.charges_breakdown:
            lines.append("\n### Individual Charge Reconciliation & Evidence")
            for c in audit.charges_breakdown:
                amt_disp = f"₹{c.amount:,.2f} {c.currency}" if c.amount is not None else (
                    f"{c.rate_pct:.2f}% ({c.rate_basis})" if c.rate_pct is not None else "Unquantified"
                )
                rep_tag = f" (also disclosed on p.{', p.'.join(str(x) for x in c.repeated_on_pages)})" if c.repeated_on_pages else ""
                note_tag = f" — {c.notes}" if c.notes else ""
                gst_tag = f" | GST: `{c.gst_classification.value}`" + (f" ({c.gst_related_service})" if c.gst_related_service else "")
                tp_tag = f" | Recovery: `{c.third_party_recovery_method.value}`" if c.third_party_recovery_method else ""
                sec_tag = f" | {c.source_section}" if c.source_section else ""
                ev_tag = f" — Evidence: \"{c.evidence_snippet}\"" if c.evidence_snippet else ""
                lines.append(
                    f"- **{c.name}** (`{c.category.value}` | `{c.mandatory_status}` | {c.payment_timing or 'unspecified'}{gst_tag}{tp_tag}): "
                    f"**{amt_disp}** [Page {c.source_page or '?'}{sec_tag}]{rep_tag}{note_tag}{ev_tag}"
                )

        if audit.document_notes:
            lines.append("\n### Operational, Tax & Annexure Notes")
            for n in audit.document_notes:
                sec_tag = f" | {n.source_section}" if n.source_section else ""
                lines.append(f"- **[Page {n.source_page}{sec_tag}] {n.topic}**: {n.summary} — Evidence: \"{n.evidence_snippet}\"")

        if audit.discrepancies:
            lines.append("\n### Verified Findings & Evidence")
            for d in audit.discrepancies:
                pages_list = d.source_pages or ([d.source_page] if d.source_page else [])
                page_tag = f" [Pages {', '.join(str(p) for p in pages_list)}]" if pages_list else ""
                sec_tag = f" ({d.source_section})" if d.source_section else ""
                ev_tag = f" — Evidence: \"{d.evidence_snippet}\"" if d.evidence_snippet else ""
                lines.append(f"- **[{d.severity.value.upper()}]{page_tag}{sec_tag} {d.title}**: {d.description}{ev_tag}")

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
