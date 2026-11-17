import React from "react";
import { TrendingUp, AlertTriangle } from "lucide-react";

export const TrueAprGauge = ({ statedRate, trueApr, spread }) => {
  const isHighSpread = spread >= 2.0;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-400">True Effective APR</span>
        {isHighSpread ? (
          <AlertTriangle className="w-4 h-4 text-rose-400" />
        ) : (
          <TrendingUp className="w-4 h-4 text-emerald-400" />
        )}
      </div>
      <div className={`text-2xl font-extrabold ${isHighSpread ? "text-rose-400" : "text-emerald-400"}`}>
        {trueApr.toFixed(2)}%
      </div>
      <div className="mt-2 text-xs text-slate-400 flex items-center justify-between">
        <span>Stated Rate: <strong className="text-slate-200">{statedRate.toFixed(2)}%</strong></span>
        <span className={isHighSpread ? "text-rose-400 font-semibold" : "text-slate-400"}>
          +{spread.toFixed(2)}% hidden spread
        </span>
      </div>
    </div>
  );
};
