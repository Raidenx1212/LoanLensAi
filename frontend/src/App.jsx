import React, { useState } from "react";
import { ShieldCheck, FileText, Scale, Calculator } from "lucide-react";
import { LanguageSwitcher } from "./components/LanguageSwitcher";
import { FileUpload } from "./components/FileUpload";
import { AuditDashboard } from "./components/AuditDashboard";
import { LoanComparison } from "./components/LoanComparison";

export default function App() {
  const [activeTab, setActiveTab] = useState("audit");
  const [currentLang, setCurrentLang] = useState("en");
  const [auditData, setAuditData] = useState(null);

  return (
    <div className="bg-slate-900 text-slate-100 min-h-screen">
      {/* Navbar */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <ShieldCheck className="w-6 h-6 text-slate-950" />
            </div>
            <div>
              <span className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 via-teal-200 to-indigo-300 bg-clip-text text-transparent">
                LoanLens
              </span>
              <span className="ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                RBI KFS Auditor
              </span>
            </div>
          </div>

          <LanguageSwitcher currentLang={currentLang} onLanguageChange={setCurrentLang} />
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        <div className="text-center max-w-3xl mx-auto">
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white mb-2">
            Uncover Hidden Charges in Your <span className="text-emerald-400">Loan Sanction Letter</span>
          </h1>
          <p className="text-slate-400 text-sm">
            Deterministic mathematical audit engine verifying reducing vs. flat EMIs, upfront fees, 18% GST, and True APR.
          </p>
        </div>

        {/* Tab Buttons */}
        <div className="flex justify-center border-b border-slate-800">
          <div className="flex space-x-8">
            <button
              onClick={() => setActiveTab("audit")}
              className={`py-3 px-2 text-xs font-bold flex items-center space-x-2 border-b-2 transition-all ${
                activeTab === "audit"
                  ? "border-emerald-500 text-emerald-400"
                  : "border-transparent text-slate-400 hover:text-white"
              }`}
            >
              <FileText className="w-4 h-4" />
              <span>Sanction Letter PDF Auditor</span>
            </button>
            <button
              onClick={() => setActiveTab("compare")}
              className={`py-3 px-2 text-xs font-bold flex items-center space-x-2 border-b-2 transition-all ${
                activeTab === "compare"
                  ? "border-emerald-500 text-emerald-400"
                  : "border-transparent text-slate-400 hover:text-white"
              }`}
            >
              <Scale className="w-4 h-4" />
              <span>Compare 2 Loans (/compare)</span>
            </button>
          </div>
        </div>

        {/* Tab Contents */}
        {activeTab === "audit" && (
          <div className="space-y-8">
            <FileUpload onAuditComplete={setAuditData} currentLang={currentLang} />
            <AuditDashboard auditData={auditData} />
          </div>
        )}

        {activeTab === "compare" && <LoanComparison />}
      </main>
    </div>
  );
}
