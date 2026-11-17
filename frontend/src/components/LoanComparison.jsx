import React, { useState } from "react";
import { Scale, Trophy } from "lucide-react";

export const LoanComparison = () => {
  const [loanA, setLoanA] = useState({
    lender_name: "State Bank of India",
    principal_amount: 500000,
    stated_interest_rate: 10.5,
    tenure_months: 36,
    processing_fee: 1500,
    bundled_insurance: 0,
  });

  const [loanB, setLoanB] = useState({
    lender_name: "Aggressive NBFC",
    principal_amount: 500000,
    stated_interest_rate: 10.5,
    tenure_months: 36,
    processing_fee: 12500,
    bundled_insurance: 14000,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCompare = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ loan_a: loanA, loan_b: loanB }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert(`Comparison failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loan A */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
          <div className="flex items-center space-x-2 mb-2">
            <span className="w-5 h-5 rounded-full bg-emerald-500 text-slate-950 flex items-center justify-center text-xs font-bold">
              A
            </span>
            <span className="text-sm font-bold text-white">Loan Offer A</span>
          </div>
          <input
            type="text"
            value={loanA.lender_name}
            onChange={(e) => setLoanA({ ...loanA, lender_name: e.target.value })}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
          />
          <div className="grid grid-cols-2 gap-3">
            <input
              type="number"
              value={loanA.principal_amount}
              onChange={(e) => setLoanA({ ...loanA, principal_amount: +e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
            />
            <input
              type="number"
              step="0.1"
              value={loanA.stated_interest_rate}
              onChange={(e) => setLoanA({ ...loanA, stated_interest_rate: +e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <input
              type="number"
              value={loanA.processing_fee}
              onChange={(e) => setLoanA({ ...loanA, processing_fee: +e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
            />
            <input
              type="number"
              value={loanA.bundled_insurance}
              onChange={(e) => setLoanA({ ...loanA, bundled_insurance: +e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
            />
          </div>
        </div>

        {/* Loan B */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
          <div className="flex items-center space-x-2 mb-2">
            <span className="w-5 h-5 rounded-full bg-indigo-500 text-white flex items-center justify-center text-xs font-bold">
              B
            </span>
            <span className="text-sm font-bold text-white">Loan Offer B</span>
          </div>
          <input
            type="text"
            value={loanB.lender_name}
            onChange={(e) => setLoanB({ ...loanB, lender_name: e.target.value })}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
          />
          <div className="grid grid-cols-2 gap-3">
            <input
              type="number"
              value={loanB.principal_amount}
              onChange={(e) => setLoanB({ ...loanB, principal_amount: +e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
            />
            <input
              type="number"
              step="0.1"
              value={loanB.stated_interest_rate}
              onChange={(e) => setLoanB({ ...loanB, stated_interest_rate: +e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <input
              type="number"
              value={loanB.processing_fee}
              onChange={(e) => setLoanB({ ...loanB, processing_fee: +e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
            />
            <input
              type="number"
              value={loanB.bundled_insurance}
              onChange={(e) => setLoanB({ ...loanB, bundled_insurance: +e.target.value })}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"
            />
          </div>
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={handleCompare}
          disabled={loading}
          className="px-8 py-3 rounded-xl bg-emerald-500 text-slate-950 text-xs font-bold hover:bg-emerald-400 transition-colors shadow-lg shadow-emerald-500/20 inline-flex items-center space-x-2"
        >
          <Scale className="w-4 h-4" />
          <span>{loading ? "Comparing..." : "Rank by True Cost of Borrowing"}</span>
        </button>
      </div>

      {result && (
        <div className="max-w-3xl mx-auto bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
            <div className="flex items-center space-x-2 text-emerald-400 font-bold text-xs mb-1">
              <Trophy className="w-4 h-4" />
              <span>Auditor Recommendation</span>
            </div>
            <p className="text-xs text-slate-200 leading-relaxed">{result.recommendation}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {result.ranked_loans.map((item) => (
              <div
                key={item.rank}
                className={`p-4 rounded-xl border ${
                  item.rank === 1
                    ? "bg-emerald-500/10 border-emerald-500/30"
                    : "bg-slate-950 border-slate-800"
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-white">
                    {item.identifier} ({item.lender_name})
                  </span>
                  <span className="text-[10px] font-extrabold uppercase px-2 py-0.5 rounded bg-emerald-500 text-slate-950">
                    Rank #{item.rank}
                  </span>
                </div>
                <div className="space-y-1 text-xs text-slate-300">
                  <div className="flex justify-between">
                    <span>True APR:</span>
                    <strong>{item.effective_annual_apr.toFixed(2)}%</strong>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Borrowing Cost:</span>
                    <strong className={item.rank === 1 ? "text-emerald-400" : "text-rose-400"}>
                      ₹{item.total_cost_of_borrowing.toLocaleString("en-IN")}
                    </strong>
                  </div>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-800 text-[11px] text-slate-400">
                  {item.key_takeaway}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
