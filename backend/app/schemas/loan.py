import re
from enum import Enum
from typing import List, Optional
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


# =====================================================================
# 1. EXTRACTED DATA SCHEMA (Produced by OpenRouter LLM, Zero Math)
# =====================================================================
class ExtractedLoanTerms(BaseModel):
    lender_name: Optional[str] = Field(
        default=None, description="Name of the bank or NBFC"
    )
    loan_category: LoanCategory = Field(
        default=LoanCategory.PERSONAL, description="Type of loan"
    )
    principal_amount: float = Field(
        ..., gt=0, description="Sanctioned principal loan amount in INR"
    )
    stated_interest_rate: float = Field(
        ..., ge=0, description="Annual nominal interest rate as stated (e.g. 10.5 for 10.5%)"
    )
    interest_type: InterestRateType = Field(
        default=InterestRateType.REDUCING,
        description="Whether interest is calculated on reducing balance or flat balance",
    )
    tenure_months: int = Field(
        ..., gt=0, description="Total loan tenure in calendar months"
    )
    stated_emi: Optional[float] = Field(
        default=None, ge=0, description="Monthly EMI stated in sanction letter / KFS"
    )
    processing_fee: float = Field(
        default=0.0, ge=0, description="Upfront processing fee before GST"
    )
    documentation_charges: float = Field(
        default=0.0, ge=0, description="Legal, stamp duty, or documentation fees"
    )
    bundled_insurance: float = Field(
        default=0.0, ge=0, description="Credit shield, health or life insurance charged"
    )
    is_insurance_capitalized: bool = Field(
        default=False,
        description="True if insurance was added into principal; False if deducted upfront from disbursal",
    )
    prepayment_penalty_terms: Optional[str] = Field(
        default=None,
        description="Text clause detailing prepayment or foreclosure charges",
    )
    has_prepayment_penalty: bool = Field(
        default=False,
        description="True if early foreclosure or part-prepayment incurs fees",
    )
    extraction_method: str = Field(
        default="openrouter_llm",
        description="Method used for extraction (openrouter_llm or offline_heuristic_fallback)",
    )

    @field_validator("interest_type", mode="before")
    @classmethod
    def normalize_interest_type(cls, v):
        if not v or not isinstance(v, str):
            return InterestRateType.REDUCING
        v_lower = v.strip().lower()
        if "flat" in v_lower:
            return InterestRateType.FLAT
        if "reduc" in v_lower:
            return InterestRateType.REDUCING
        if "float" in v_lower:
            return InterestRateType.FLOATING
        if "fix" in v_lower:
            return InterestRateType.FIXED
        return InterestRateType.REDUCING

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
        "stated_interest_rate",
        "processing_fee",
        "documentation_charges",
        "bundled_insurance",
        mode="before",
    )
    @classmethod
    def clean_floats(cls, v):
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = re.sub(r"[^\d.]", "", v)
            return float(cleaned) if cleaned else 0.0
        return float(v)

    @field_validator("stated_emi", mode="before")
    @classmethod
    def clean_optional_float(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = re.sub(r"[^\d.]", "", v)
            return float(cleaned) if cleaned else None
        return float(v)


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


class LoanAuditReport(BaseModel):
    lender_name: Optional[str]
    principal: float
    tenure_months: int
    stated_annual_rate: float
    interest_type: InterestRateType

    # Deterministic EMI validation
    stated_emi: Optional[float]
    calculated_reducing_emi: float
    calculated_flat_emi: float
    emi_difference: float
    is_emi_matching: bool

    # Fee and GST breakdown (18% Statutory Indian GST on Banking Services)
    raw_processing_fee: float
    raw_documentation_charges: float
    fee_gst_amount: float
    total_upfront_fees_incl_gst: float
    bundled_insurance_amount: float

    # Cash flows and true cost
    gross_loan_amount: float
    net_disbursed_amount: float
    total_repayment_amount: float
    total_interest_payable: float
    total_cost_of_borrowing: float  # (Total Repayment + Upfront Fees + GST) - Principal

    # Deterministic True APR & IRR
    true_monthly_irr: float
    nominal_apr: float
    effective_annual_apr: float
    rate_inflation_spread: float

    # Flags & Findings
    discrepancies: List[DiscrepancyItem] = []
    rbi_compliance_score: int = Field(
        ..., ge=0, le=100, description="Heuristic compliance score out of 100"
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
    effective_annual_apr: float
    total_cost_of_borrowing: float
    net_disbursed_amount: float
    total_repayment_amount: float
    cost_difference_vs_winner: float
    key_takeaway: str


class LoanComparisonResponse(BaseModel):
    recommendation: str
    ranked_loans: List[RankedLoanItem]
