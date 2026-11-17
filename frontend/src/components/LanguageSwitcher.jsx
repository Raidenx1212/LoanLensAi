import React from "react";

export const LanguageSwitcher = ({ currentLang, onLanguageChange }) => {
  const languages = [
    { code: "en", label: "English" },
    { code: "hi", label: "हिन्दी" },
    { code: "kn", label: "ಕನ್ನಡ" },
  ];

  return (
    <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
      {languages.map((lang) => (
        <button
          key={lang.code}
          onClick={() => onLanguageChange(lang.code)}
          className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
            currentLang === lang.code
              ? "bg-emerald-500 text-slate-950 font-bold"
              : "text-slate-400 hover:text-white"
          }`}
        >
          {lang.label}
        </button>
      ))}
    </div>
  );
};
