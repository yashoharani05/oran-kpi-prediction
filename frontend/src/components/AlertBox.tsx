/*
  components/AlertBox.tsx

  Shows the final prediction result:
    - Green box for Normal
    - Red pulsing box for Degraded

  Also displays the recommendation text from the backend.

  Props:
    riskLabel      — "Normal" | "Degraded"
    riskCode       — 0 | 1
    probability    — 0.0 – 1.0
    recommendation — plain-English advice string from the API
    modelUsed      — name of the ML model that made this prediction
*/

"use client";

import { CheckCircle, AlertTriangle } from "lucide-react";
import type { PredictionResult } from "@/types";

interface Props {
  result: PredictionResult;
}

export default function AlertBox({ result }: Props) {
  const isNormal = result.risk_code === 0;

  // Colour scheme switches entirely based on the prediction
  const scheme = isNormal
    ? {
        border: "border-green-500/40",
        bg: "bg-green-950/30",
        icon: <CheckCircle className="w-6 h-6 text-green-400 flex-shrink-0" />,
        label: "text-green-400",
        pulse: "",
      }
    : {
        border: "border-red-500/40",
        bg: "bg-red-950/30",
        icon: <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0" />,
        label: "text-red-400",
        pulse: "alert-degraded", // CSS animation in globals.css
      };

  return (
    <div
      className={`
        rounded-xl border ${scheme.border} ${scheme.bg} ${scheme.pulse}
        p-5 flex flex-col gap-4
      `}
    >
      {/* Header row: icon + big risk label */}
      <div className="flex items-center gap-3">
        {scheme.icon}
        <div>
          <p className="text-xs font-mono text-slate-500 uppercase tracking-widest mb-0.5">
            Prediction Result
          </p>
          <p className={`text-2xl font-bold font-mono ${scheme.label}`}>
            {result.risk_label}
          </p>
        </div>

        {/* Probability pill — right side */}
        <div className="ml-auto text-right">
          <p className="text-xs text-slate-500 mb-0.5">Confidence</p>
          <p className={`text-xl font-mono font-bold ${scheme.label}`}>
            {(result.probability * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-slate-700/60" />

      {/* Recommendation text */}
      <div>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2">
          Recommendation
        </p>
        <p className="text-sm text-slate-200 leading-relaxed">
          {result.recommendation}
        </p>
      </div>

      {/* Footer: model tag */}
      <p className="text-xs text-slate-600 font-mono">
        Model: {result.model_used}
      </p>
    </div>
  );
}
