import datetime
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, JSON
from app.core.database import Base


class LoanAuditModel(Base):
    __tablename__ = "loan_audits"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    lender_name = Column(String(255), nullable=True)
    principal_amount = Column(Float, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    stated_rate = Column(Float, nullable=False)
    effective_annual_apr = Column(Float, nullable=True)
    net_disbursed_amount = Column(Float, nullable=False)
    total_cost_of_borrowing = Column(Float, nullable=False)
    rbi_compliance_score = Column(Integer, nullable=False)
    audit_data = Column(JSON, nullable=False)
    explanation = Column(Text, nullable=True)
