/*
  components/AlertBox.tsx

  Shows the final prediction result:
    - Green box for Normal
    - Red pulsing box for Degraded

  Also displays the recommendation text from the backend.

  METHODOLOGY NOTE (forecasting correction):
  result.risk_label/risk_code/probability now describe the FORECAST state
  ~result.forecast_horizon_seconds ahead — not "right now". If the backend
  also supplied result.current_status (the deterministic rule-based
  "right now" assessment, independent of the ML model), we show both,
  clearly labelled, so the distinction in docs/FORECASTING_METHODOLOGY_UPDATE.md
  is visible in the UI. If current_status is absent (older backend), the
  component falls back to its original single-box behaviour.

  Props:
    result — the full PredictionResult from the API
*/

"use client";

import { CheckCircle, AlertTriangle, Radar } from "lucide-react";
import type { PredictionResult } from "@/types";

interface Props {
  result: PredictionResult;
}

export default function AlertBox({ result }: Props) {
  const isNormal = result.risk_code === 0;
  const hasCurrentStatus = result.current_status != null;
  const horizon = result.forecast_horizon_seconds ?? 5;

  // Colour scheme switches entirely based on the FORECAST prediction
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

  const currentIsDegraded = result.current_status === "Degraded";

  return (
    <div
      className={`
        rounded-xl border ${scheme.border} ${scheme.bg} ${scheme.pulse}
        p-5 flex flex-col gap-4
      `}
    >
      {/* Early-warning banner: current Normal, forecast Degraded */}
      {result.early_warning && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-500/50 bg-amber-950/40 px-3 py-2">
          <Radar className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <p className="text-xs font-mono text-amber-300">
            EARLY WARNING — network is normal now but degradation is forecast within {horizon}s
          </p>
        </div>
      )}

      {/* Current status vs forecast, side by side, when the backend provides both */}
      {hasCurrentStatus && (
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-3">
            <p className="text-xs font-mono text-slate-500 uppercase tracking-widest mb-1">
              Current Status
            </p>
            <p className={`text-lg font-bold font-mono ${currentIsDegraded ? "text-red-400" : "text-green-400"}`}>
              {result.current_status}
            </p>
            <p className="text-[10px] text-slate-600 mt-0.5">rule-based, right now</p>
          </div>
          <div className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-3">
            <p className="text-xs font-mono text-slate-500 uppercase tracking-widest mb-1">
              {horizon}s Forecast
            </p>
            <p className={`text-lg font-bold font-mono ${scheme.label}`}>
              {result.risk_label}
            </p>
            <p className="text-[10px] text-slate-600 mt-0.5">ML model, {horizon}s ahead</p>
          </div>
        </div>
      )}

      {/* Header row: icon + big risk label */}
      <div className="flex items-center gap-3">
        {scheme.icon}
        <div>
          <p className="text-xs font-mono text-slate-500 uppercase tracking-widest mb-0.5">
            {hasCurrentStatus ? `Forecast (${horizon}s ahead)` : "Prediction Result"}
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
