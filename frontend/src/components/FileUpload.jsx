import React, { useState } from "react";
import { UploadCloud, FileCode, Cpu } from "lucide-react";

export const FileUpload = ({ onAuditComplete, currentLang }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleAudit = async () => {
    if (!file) return;
    setLoading(true);
    setLoadingStep("Extracting text via PyMuPDF / OCR...");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", currentLang);

    try {
      setLoadingStep("Running Deterministic Financial Audit & RBI Rules Engine...");
      const res = await fetch("/api/v1/audit", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Audit execution failed");
      }

      const data = await res.json();
      onAuditComplete(data);
    } catch (err) {
      alert(`Audit Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadSample = async () => {
    try {
      const res = await fetch("/sample_sanction_letter.pdf");
      const blob = await res.blob();
      const sampleFile = new File([blob], "HDFC_Sanction_Letter.pdf", {
        type: "application/pdf",
      });
      setFile(sampleFile);
    } catch {
      alert("Sample PDF file ready on backend.");
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <div className="border-2 border-dashed border-slate-700 hover:border-emerald-500/50 bg-slate-900/50 rounded-2xl p-8 text-center relative group">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
        <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center mx-auto mb-4 group-hover:scale-105 transition-transform">
          <UploadCloud className="w-8 h-8" />
        </div>
        <h3 className="text-base font-semibold text-white mb-1">
          Upload Bank Sanction Letter or KFS (PDF)
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          All major Indian banks supported (SBI, HDFC, ICICI, Axis, NBFCs)
        </p>
        <span className="inline-flex items-center px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-500 text-slate-950">
          Browse PDF Document
        </span>
        {file && (
          <div className="mt-3 text-xs text-emerald-400 font-medium">
            Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
          </div>
        )}
      </div>

      <div className="flex items-center justify-between">
        <button
          onClick={loadSample}
          type="button"
          className="text-xs text-slate-400 hover:text-emerald-400 flex items-center space-x-1.5"
        >
          <FileCode className="w-3.5 h-3.5" />
          <span>Load Sample Bank KFS Letter</span>
        </button>

        <button
          onClick={handleAudit}
          disabled={!file || loading}
          className={`px-5 py-2.5 rounded-xl text-xs font-bold flex items-center space-x-2 transition-all ${
            file && !loading
              ? "bg-emerald-500 text-slate-950 hover:bg-emerald-400 cursor-pointer shadow-lg shadow-emerald-500/20"
              : "bg-slate-800 text-slate-500 cursor-not-allowed"
          }`}
        >
          <Cpu className="w-4 h-4" />
          <span>{loading ? "Auditing..." : "Run Deterministic Audit"}</span>
        </button>
      </div>

      {loading && (
        <div className="text-center py-6">
          <div className="w-8 h-8 border-3 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin mx-auto mb-2" />
          <p className="text-xs text-slate-400">{loadingStep}</p>
        </div>
      )}
    </div>
  );
};
