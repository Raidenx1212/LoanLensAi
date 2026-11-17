import React from "react";
import { Wallet, Percent, Receipt, Shield, AlertTriangle, Sparkles } from "lucide-react";
import { TrueAprGauge } from "./TrueAprGauge";

export const AuditDashboard = ({ auditData }) => {
  if (!auditData) return null;

  const {
    principal,
    net_disbursed_amount,
    stated_annual_rate,
    effective_annual_apr,
    rate_inflation_spread,
    total_upfront_fees_incl_gst,
    fee_gst_amount,
    bundled_insurance_amount,
    rbi_compliance_score,
    discrepancies,
    explanation,
  } = auditData;

  const totalDeductions = principal - net_disbursed_amount;

  return (
    <div className="space-y-8">
      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Net In-Hand */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-400">Net In-Hand Disbursal</span>
            <Wallet className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-extrabold text-white">
            ₹{net_disbursed_amount.toLocaleString("en-IN")}
          </div>
          <div className="mt-2 text-xs text-slate-400 flex items-center justify-between">
            <span>Sanctioned: ₹{principal.toLocaleString("en-IN")}</span>
            <span className="text-rose-400 font-medium">-₹{totalDeductions.toLocaleString("en-IN")}</span>
          </div>
        </div>

        {/* True APR Component */}
        <TrueAprGauge
          statedRate={stated_annual_rate}
          trueApr={effective_annual_apr}
          spread={rate_inflation_spread}
        />

        {/* Upfront Fees & 18% GST */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-400">Upfront Fees & 18% GST</span>
            <Receipt className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-extrabold text-white">
            ₹{total_upfront_fees_incl_gst.toLocaleString("en-IN")}
          </div>
          <div className="mt-2 text-xs text-slate-400 flex items-center justify-between">
            <span>GST: ₹{fee_gst_amount.toLocaleString("en-IN")}</span>
            <span>Insurance: ₹{bundled_insurance_amount.toLocaleString("en-IN")}</span>
          </div>
        </div>

        {/* RBI Compliance Score */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-400">RBI Fair Lending Score</span>
            <Shield className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-extrabold text-emerald-400">
            {rbi_compliance_score} / 100
          </div>
          <div className="mt-2 text-xs text-slate-400">
            {rbi_compliance_score >= 80 ? "Compliant loan terms" : "Caution: predatory practices"}
          </div>
        </div>
      </div>

      {/* Discrepancies */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-bold text-white flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            <span>Auditor Discrepancies & Regulatory Findings</span>
          </h3>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-800 text-slate-300">
            {discrepancies.length} Findings
          </span>
        </div>

        <div className="space-y-3">
          {discrepancies.map((d, idx) => (
            <div
              key={idx}
              className={`p-4 rounded-xl border ${
                d.severity === "critical"
                  ? "border-rose-500/30 bg-rose-500/10 text-rose-200"
                  : d.severity === "warning"
                  ? "border-amber-500/30 bg-amber-500/10 text-amber-200"
                  : "border-indigo-500/30 bg-indigo-500/10 text-indigo-200"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <h4 className="text-xs font-bold text-white">{d.title}</h4>
                <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border border-current">
                  {d.severity}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">{d.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Multilingual Narrative Explanation */}
      {explanation && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            <h3 className="text-base font-bold text-white">
              Borrower Explanation & Actionable Checklist
            </h3>
          </div>
          <div className="text-xs text-slate-300 whitespace-pre-wrap leading-relaxed space-y-2">
            {explanation}
          </div>
        </div>
      )}
    </div>
  );
};
