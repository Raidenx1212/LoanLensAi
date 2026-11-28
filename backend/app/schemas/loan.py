import re
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class InterestRateType(str, Enum):
    REDUCING = "reducing"
    FLAT = "flat"
    FLOATING = "floating"
    FIXED = "fixed"
    UNKNOWN = "unknown"


class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    KANNADA = "kn"


class LoanCategory(str, Enum):
    PERSONAL = "personal"
    HOME = "home"
    VEHICLE = "vehicle"
    BUSINESS = "business"
    EDUCATION = "education"
    OTHER = "other"


class FieldStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    NOT_FOUND = "NOT_FOUND"
    UNCERTAIN = "UNCERTAIN"
    USER_VERIFIED = "USER_VERIFIED"


class ChargeCategory(str, Enum):
    CONFIRMED_UPFRONT = "confirmed_upfront"
    CONDITIONAL = "conditional"
    CONTINGENT = "contingent"
    FINANCED = "financed"
    THIRD_PARTY = "third_party"


class GSTClassification(str, Enum):
    GST_AMOUNT_CONFIRMED = "GST_AMOUNT_CONFIRMED"
    GST_RATE_CONFIRMED = "GST_RATE_CONFIRMED"
    GST_INCLUDED = "GST_INCLUDED"
    GST_POTENTIALLY_APPLICABLE = "GST_POTENTIALLY_APPLICABLE"
    GST_UNKNOWN = "GST_UNKNOWN"
    GST_NOT_APPLICABLE = "GST_NOT_APPLICABLE"


class ThirdPartyRecoveryMethod(str, Enum):
    PAID_DIRECTLY_BY_BORROWER = "PAID_DIRECTLY_BY_BORROWER"
    COLLECTED_BY_LENDER = "COLLECTED_BY_LENDER"
    RECOVERED_AGAINST_AGENCY_INVOICE = "RECOVERED_AGAINST_AGENCY_INVOICE"
    RECOVERY_METHOD_UNKNOWN = "RECOVERY_METHOD_UNKNOWN"


class CrossValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED_MISSING_DATA = "SKIPPED_MISSING_DATA"


class ExtractedFieldInfo(BaseModel):
    status: FieldStatus = FieldStatus.NOT_FOUND
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    evidence_snippet: Optional[str] = None
    raw_text: Optional[str] = None
    extraction_confidence: str = "HIGH"
    note: Optional[str] = None


class ExtractedCharge(BaseModel):
    name: str
    amount: Optional[float] = Field(default=None, ge=0)
    rate_pct: Optional[float] = Field(default=None, ge=0)
    rate_basis: Optional[str] = None
    currency: str = "INR"
    category: ChargeCategory
    trigger_condition: Optional[str] = None
    frequency: Optional[str] = None
    payment_timing: Optional[str] = None
    caused_by_borrower_action: bool = False
    mandatory_status: str = "MANDATORY"
    is_mandatory: bool = True
    is_capitalized: bool = False
    gst_classification: GSTClassification = GSTClassification.GST_UNKNOWN
    gst_related_service: Optional[str] = None
    third_party_recovery_method: Optional[ThirdPartyRecoveryMethod] = None
    status: FieldStatus = FieldStatus.CONFIRMED
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    evidence_snippet: Optional[str] = None
    raw_text: Optional[str] = None
    extraction_confidence: str = "HIGH"
    dedup_id: Optional[str] = None
    repeated_on_pages: List[int] = Field(default_factory=list)
    tax_note: Optional[str] = None
    notes: Optional[str] = None


class DocumentNote(BaseModel):
    topic: str
    summary: str
    source_page: int
    source_section: Optional[str] = None
    evidence_snippet: str
    raw_text: Optional[str] = None
    extraction_confidence: str = "HIGH"


class CrossValidationResult(BaseModel):
    check_name: str
    status: CrossValidationStatus
    summary: str
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    source_page: Optional[int] = None
    evidence_snippet: Optional[str] = None


# =====================================================================
# 1. EXTRACTED DATA SCHEMA (Zero False Zeros, Evidence-Backed)
# =====================================================================
class ExtractedLoanTerms(BaseModel):
    lender_name: Optional[str] = Field(
        default=None, description="Name of the bank or NBFC"
    )
    loan_category: LoanCategory = Field(
        default=LoanCategory.PERSONAL, description="Type of loan"
    )
    principal_amount: Optional[float] = Field(
        default=None, ge=0, description="Sanctioned principal loan amount in INR; None if NOT_FOUND"
    )
    opening_financed_balance: Optional[float] = Field(
        default=None, ge=0, description="Opening financed/capitalized loan balance if stated; None if NOT_FOUND"
    )
    stated_interest_rate: Optional[float] = Field(
        default=None, ge=0, description="Annual nominal interest rate as stated; None if NOT_FOUND"
    )
    interest_type: InterestRateType = Field(
        default=InterestRateType.UNKNOWN,
        description="Whether interest is calculated on reducing balance, flat balance, fixed, or floating",
    )
    tenure_months: Optional[int] = Field(
        default=None, ge=0, description="Total loan tenure in calendar months; None if NOT_FOUND"
    )
    stated_emi: Optional[float] = Field(
        default=None, ge=0, description="Monthly EMI stated in sanction letter / KFS; None if NOT_FOUND"
    )
    processing_fee: Optional[float] = Field(
        default=None, ge=0, description="Upfront processing fee before tax; None if NOT_FOUND"
    )
    documentation_charges: Optional[float] = Field(
        default=None, ge=0, description="Legal, stamp duty, or documentation fees; None if NOT_FOUND"
    )
    other_cash_recoveries: Optional[float] = Field(
        default=None, ge=0, description="Other confirmed upfront cash recoveries before tax; None if NOT_FOUND"
    )
    bundled_insurance: Optional[float] = Field(
        default=None, ge=0, description="Credit shield, health or life insurance charged; None if NOT_FOUND"
    )
    is_insurance_capitalized: bool = Field(
        default=False,
        description="True if insurance was financed/added into principal; False if deducted upfront",
    )
    stated_gst_amount: Optional[float] = Field(
        default=None, ge=0, description="Exact GST rupee amount ONLY if explicitly stated; None if UNKNOWN"
    )
    stated_gst_rate_pct: Optional[float] = Field(
        default=None, ge=0, description="Exact GST % ONLY if explicitly stated in document; None if NOT_FOUND"
    )
    stated_net_disbursement: Optional[float] = Field(
        default=None, ge=0, description="Exact final net disbursement amount ONLY if stated; None if UNKNOWN"
    )
    gst_status_note: Optional[str] = Field(
        default=None, description="Verbatim tax wording if applicable tax is stated without amount"
    )
    prepayment_penalty_terms: Optional[str] = Field(
        default=None,
        description="Text clause detailing prepayment or foreclosure charges",
    )
    has_prepayment_penalty: Optional[bool] = Field(
        default=None,
        description="True if penalty applies, False if explicitly Nil, None if not disclosed",
    )

    # Multi-page granular charge & schedule tracking
    charges: List[ExtractedCharge] = Field(
        default_factory=list,
        description="All extracted charges separated by confirmed_upfront, conditional, contingent, financed, third_party",
    )
    document_notes: List[DocumentNote] = Field(
        default_factory=list,
        description="Supporting annexure, tax, and servicing notes with source pages and snippets",
    )
    schedule_instalment_count: Optional[int] = Field(
        default=None, ge=0, description="Number of instalments found in repayment schedule if present"
    )
    schedule_emis: List[float] = Field(
        default_factory=list, description="Sample or full instalment amounts extracted from repayment schedule"
    )

    # Explicit field-level verification status and provenance
    field_metadata: Dict[str, ExtractedFieldInfo] = Field(
        default_factory=dict,
        description="Verification status, source page, and evidence snippet per extracted field",
    )
    is_verified: bool = Field(
        default=True,
        description="Whether core terms were extracted with certainty",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of core terms not detected in document",
    )
    extraction_method: str = Field(
        default="openrouter_llm",
        description="Method used for extraction",
    )

    @field_validator("interest_type", mode="before")
    @classmethod
    def normalize_interest_type(cls, v):
        if not v or not isinstance(v, str):
            return InterestRateType.UNKNOWN
        v_lower = v.strip().lower()
        if "flat" in v_lower:
            return InterestRateType.FLAT
        if "float" in v_lower:
            return InterestRateType.FLOATING
        if "reduc" in v_lower:
            return InterestRateType.REDUCING
        if "fix" in v_lower:
            return InterestRateType.FIXED
        return InterestRateType.UNKNOWN

    @field_validator("loan_category", mode="before")
    @classmethod
    def normalize_loan_category(cls, v):
        if not v or not isinstance(v, str):
            return LoanCategory.PERSONAL
        v_lower = v.strip().lower()
        for cat in LoanCategory:
            if cat.value in v_lower:
                return cat
        return LoanCategory.PERSONAL

    @field_validator(
        "principal_amount",
        "opening_financed_balance",
        "stated_interest_rate",
        "stated_emi",
        "processing_fee",
        "documentation_charges",
        "other_cash_recoveries",
        "bundled_insurance",
        "stated_gst_amount",
        "stated_gst_rate_pct",
        "stated_net_disbursement",
        mode="before",
    )
    @classmethod
    def clean_optional_float(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("", "null", "none", "unknown", "n/a", "na", "not found", "not_found"):
                return None
            cleaned = re.sub(r"[^\d.]", "", v)
            if not cleaned or cleaned == ".":
                return None
            return float(cleaned)
        return float(v)

    @field_validator("tenure_months", "schedule_instalment_count", mode="before")
    @classmethod
    def clean_optional_int(cls, v):
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("", "null", "none", "unknown", "n/a", "na", "not found", "not_found"):
                return None
            cleaned = re.sub(r"[^\d]", "", v)
            return int(cleaned) if cleaned else None
        return int(v)


# =====================================================================
# 2. DETERMINISTIC AUDIT OUTPUT SCHEMA (Computed by auditor.py)
# =====================================================================
class DiscrepancySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DiscrepancyItem(BaseModel):
    code: str
    severity: DiscrepancySeverity
    title: str
    description: str
    impact_amount: Optional[float] = None
    source_page: Optional[int] = None
    source_pages: List[int] = Field(default_factory=list)
    source_section: Optional[str] = None
    evidence_snippet: Optional[str] = None
    raw_text: Optional[str] = None
    extraction_confidence: str = "HIGH"


class LoanAuditReport(BaseModel):
    lender_name: Optional[str] = None
    principal: Optional[float] = None
    opening_financed_balance: Optional[float] = None
    tenure_months: Optional[int] = None
    stated_annual_rate: Optional[float] = None
    interest_type: InterestRateType = InterestRateType.UNKNOWN

    # Deterministic EMI validation
    stated_emi: Optional[float] = None
    calculated_reducing_emi: Optional[float] = None
    calculated_flat_emi: Optional[float] = None
    emi_difference: Optional[float] = None
    is_emi_matching: Optional[bool] = None

    # Fee, recovery, insurance & tax breakdown (Never invented)
    raw_processing_fee: Optional[float] = None
    raw_documentation_charges: Optional[float] = None
    other_cash_recoveries: Optional[float] = None
    known_upfront_recoveries_excl_tax: Optional[float] = None
    fee_gst_amount: Optional[float] = None
    stated_gst_rate_pct: Optional[float] = None
    gst_classification: GSTClassification = GSTClassification.GST_UNKNOWN
    gst_status_note: Optional[str] = None
    total_upfront_fees_incl_gst: Optional[float] = None
    bundled_insurance_amount: Optional[float] = None
    is_insurance_capitalized: bool = False
    conditional_charges_total: Optional[float] = None
    contingent_charges_total: Optional[float] = None
    third_party_charges_total: Optional[float] = None
    penal_charge_rate_pct: Optional[float] = None
    penal_charges_capitalized: Optional[bool] = None
    conditional_charges: List[ExtractedCharge] = Field(
        default_factory=list,
        description="Individual conditional/contingent charges kept separate and never summed as expected expenses",
    )

    # Cash flows and true cost (None / UNKNOWN if inputs are missing)
    gross_loan_amount: Optional[float] = None
    net_disbursed_amount: Optional[float] = None
    total_repayment_amount: Optional[float] = None
    total_interest_payable: Optional[float] = None
    total_cost_of_borrowing: Optional[float] = None

    # Deterministic True APR & IRR (None if required cash flows are unavailable)
    is_irr_computable: bool = Field(
        default=False,
        description="True only when net disbursement, EMI, and tenure are all verified and valid",
    )
    true_monthly_irr: Optional[float] = None
    nominal_apr: Optional[float] = None
    effective_annual_apr: Optional[float] = None
    rate_inflation_spread: Optional[float] = None

    # Evidence, verification & cross-validation
    audit_status: str = Field(
        default="VERIFIED",
        description="VERIFIED | PARTIAL_MISSING_DATA | REJECTED_INVALID_COMBINATION",
    )
    is_verified: bool = Field(
        default=True,
        description="Whether all core loan parameters were verified",
    )
    missing_fields: List[str] = Field(default_factory=list)
    field_statuses: Dict[str, ExtractedFieldInfo] = Field(default_factory=dict)
    charges_breakdown: List[ExtractedCharge] = Field(default_factory=list)
    document_notes: List[DocumentNote] = Field(default_factory=list)
    cross_validations: List[CrossValidationResult] = Field(default_factory=list)

    # Negotiation gating
    negotiation_script_blocked: bool = Field(
        default=False,
        description="True when critical financial fields are unverified/unknown or impossible combination rejected",
    )
    negotiation_block_reason: Optional[str] = None

    # Flags & Findings
    discrepancies: List[DiscrepancyItem] = Field(default_factory=list)
    rbi_compliance_score: Optional[int] = Field(
        default=None, ge=0, le=100, description="None unless explicitly computed without invented heuristics"
    )

    # Narrative explanation (populated by LLM explainer)
    explanation: Optional[str] = None

    # Debug & Transparency Metadata
    extraction_method: str = "openrouter_llm"
    total_pages_analyzed: int = Field(default=1, description="Total number of document pages extracted and audited")
    raw_extracted_text_preview: Optional[str] = None


# =====================================================================
# 3. COMPARISON SCHEMAS (/compare)
# =====================================================================
class LoanComparisonRequest(BaseModel):
    loan_a: ExtractedLoanTerms
    loan_b: ExtractedLoanTerms


class RankedLoanItem(BaseModel):
    rank: int
    identifier: str
    lender_name: Optional[str]
    effective_annual_apr: Optional[float] = None
    total_cost_of_borrowing: Optional[float] = None
    net_disbursed_amount: Optional[float] = None
    total_repayment_amount: Optional[float] = None
    cost_difference_vs_winner: Optional[float] = None
    key_takeaway: str


class LoanComparisonResponse(BaseModel):
    recommendation: str
    ranked_loans: List[RankedLoanItem]
