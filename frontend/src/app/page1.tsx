/*
  page.tsx  (route: /)
  ============================================================
  Main dashboard page for the O-RAN KPI Prediction xApp.

  LAYOUT (two-column on desktop, stacked on mobile):
  ┌──────────────────────────────────────────────────────────┐
  │  Header (sticky — status indicator + clock)              │
  ├──────────────────────────┬───────────────────────────────┤
  │  LEFT PANEL              │  RIGHT PANEL                  │
  │  ─ KPI input form        │  ─ Probability gauge          │
  │  ─ Preset buttons        │  ─ Alert box (result)         │
  │  ─ KPI metric cards      │  ─ KPI health bar chart       │
  │  ─ Predict button        │                               │
  └──────────────────────────┴───────────────────────────────┘

  STATE MANAGED HERE:
    kpiValues      — the 18 input numbers (controlled form)
    prediction     — the last result from POST /api/predict
    status         — "idle" | "loading" | "success" | "error"
    modelLoaded    — health check result for the header indicator
    errorMessage   — string shown when the API call fails
*/

"use client";

import { useState, useEffect, useCallback } from "react";
import { Zap, Loader2, Radio } from "lucide-react";
import Link from "next/link";

// Components
import Header     from "@/components/Header";
import KpiCard    from "@/components/KpiCard";
import KpiChart   from "@/components/KpiChart";
import AlertBox   from "@/components/AlertBox";
import KpiInputForm    from "@/components/KpiInputForm";
import ProbabilityGauge from "@/components/ProbabilityGauge";

// API + types
import { checkHealth, predictRisk, fetchComparison } from "@/lib/api";
import type { ModelMetrics } from "@/lib/api";
import ModelComparison, { ModelDifferences } from "@/components/ModelComparison";
import type { KpiValues, PredictionResult, ApiStatus } from "@/types";


// ============================================================
// DEFAULT KPI VALUES — a "healthy" baseline from the dataset
// ============================================================
const HEALTHY_SAMPLE: KpiValues = {
  // These values are clearly in the healthy range:
  // High MCS (15), high CQI (12), zero errors, good throughput
  dl_mcs: 15.0,
  dl_n_samples: 200,
  dl_buffer_bytes: 0,
  tx_brate_downlink_mbps: 0.50,
  tx_pkts_downlink: 80,
  dl_cqi: 12.0,
  ul_mcs: 20.0,
  ul_n_samples: 10,
  ul_buffer_bytes: 0,
  rx_brate_uplink_mbps: 0.3,
  rx_pkts_uplink: 8,
  rx_errors_uplink_pct: 0.0,
  ul_sinr: 25.0,
  phr: 30,
  sum_requested_prbs: 900,
  sum_granted_prbs: 800,
  ul_turbo_iters: 1.0,
  prb_grant_ratio: 0.89,
};

// A sample row with multiple bad KPIs — should return Degraded
// Very low MCS (0.18), very low CQI (3.0), 100% error rate
const DEGRADED_SAMPLE: KpiValues = {
  dl_mcs: 0.18,
  dl_n_samples: 10,
  dl_buffer_bytes: 0,
  tx_brate_downlink_mbps: 0.001,
  tx_pkts_downlink: 3,
  dl_cqi: 3.0,
  ul_mcs: 0.0,
  ul_n_samples: 5,
  ul_buffer_bytes: 0,
  rx_brate_uplink_mbps: 0.01,
  rx_pkts_uplink: 2,
  rx_errors_uplink_pct: 100.0,
  ul_sinr: 0.81,
  phr: 0,
  sum_requested_prbs: 56,
  sum_granted_prbs: 28,
  ul_turbo_iters: 9.0,
  prb_grant_ratio: 0.49,
};


// ============================================================
// KPI CARDS CONFIG — the 8 tiles shown in the quick-glance grid
// ============================================================
// Each entry defines which field to read + when to warn/alert.
// 'critical' and 'warn' are computed from the value at render time.
function buildKpiCards(v: KpiValues) {
  return [
    {
      label:    "DL MCS",
      value:    v.dl_mcs,
      unit:     "",
      decimals: 2,
      critical: v.dl_mcs < 6.57,
      warn:     v.dl_mcs < 8.0 && v.dl_mcs >= 6.57,
    },
    {
      label:    "DL CQI",
      value:    v.dl_cqi,
      unit:     "",
      decimals: 2,
      critical: v.dl_cqi < 6.33,
      warn:     v.dl_cqi < 7.0 && v.dl_cqi >= 6.33,
    },
    {
      label:    "DL Throughput",
      value:    v.tx_brate_downlink_mbps,
      unit:     "Mbps",
      decimals: 4,
      critical: v.tx_brate_downlink_mbps < 0.003,
      warn:     v.tx_brate_downlink_mbps < 0.05 && v.tx_brate_downlink_mbps >= 0.003,
    },
    {
      label:    "UL Error Rate",
      value:    v.rx_errors_uplink_pct,
      unit:     "%",
      decimals: 1,
      critical: v.rx_errors_uplink_pct > 57.1,
      warn:     v.rx_errors_uplink_pct > 20 && v.rx_errors_uplink_pct <= 57.1,
    },
    {
      label:    "UL SINR",
      value:    v.ul_sinr,
      unit:     "dB",
      decimals: 2,
      critical: v.ul_sinr > 0 && v.ul_sinr < 5.05,
      warn:     false,
    },
    {
      label:    "PRB Grant Ratio",
      value:    v.prb_grant_ratio,
      unit:     "",
      decimals: 3,
      critical: v.prb_grant_ratio < 0.15,
      warn:     v.prb_grant_ratio < 0.21 && v.prb_grant_ratio >= 0.15,
    },
    {
      label:    "Turbo Iters",
      value:    v.ul_turbo_iters,
      unit:     "",
      decimals: 2,
      critical: v.ul_turbo_iters > 6.85,
      warn:     v.ul_turbo_iters > 3 && v.ul_turbo_iters <= 6.85,
    },
    {
      label:    "UL Throughput",
      value:    v.rx_brate_uplink_mbps,
      unit:     "Mbps",
      decimals: 4,
      critical: false,
      warn:     false,
    },
  ];
}


// ============================================================
// PAGE COMPONENT
// ============================================================
export default function DashboardPage() {
  const [kpiValues,      setKpiValues]      = useState<KpiValues>(HEALTHY_SAMPLE);
  const [prediction,     setPrediction]     = useState<PredictionResult | null>(null);
  const [status,         setStatus]         = useState<ApiStatus>("idle");
  const [modelLoaded,    setModelLoaded]    = useState<boolean | null>(null);
  const [errorMsg,       setErrorMsg]       = useState<string>("");
  const [selectedModel,  setSelectedModel]  = useState<"random_forest" | "xgboost" | "lstm">("random_forest");
  const [comparisonData, setComparisonData] = useState<ModelMetrics[]>([]);

  // --- Health check + comparison data on mount ---
  useEffect(() => {
    checkHealth()
      .then((res) => setModelLoaded(res.model_loaded))
      .catch(() => setModelLoaded(false));
    fetchComparison()
      .then((res) => setComparisonData(res.models))
      .catch(() => {}); // comparison is optional — don't block the page
  }, []);

  // --- Handle single field change ---
  const handleFieldChange = useCallback(
    (field: keyof KpiValues, value: number) => {
      setKpiValues((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  // --- Load a preset ---
  const handlePreset = useCallback((preset: "healthy" | "degraded") => {
    setKpiValues(preset === "healthy" ? HEALTHY_SAMPLE : DEGRADED_SAMPLE);
    setPrediction(null);
    setStatus("idle");
    setErrorMsg("");
  }, []);

  // --- Run prediction ---
  const handlePredict = async () => {
    setStatus("loading");
    setErrorMsg("");
    try {
      const result = await predictRisk(kpiValues, selectedModel);
      setPrediction(result);
      setStatus("success");
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "Could not reach the API. Make sure the backend is running.";
      setErrorMsg(msg);
      setStatus("error");
    }
  };

  const kpiCards = buildKpiCards(kpiValues);

  return (
    <div className="min-h-screen bg-[#0f172a]">
      {/* Sticky header */}
      <Header modelLoaded={modelLoaded} />

      {/* Page body */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex flex-col gap-8">

        {/* Page title row */}
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Network Degradation Dashboard
            </h1>
            <p className="text-sm text-slate-500 mt-1">
              Enter KPI values or load a sample, then run the Random Forest prediction.
            </p>
          </div>

          {/* Link to live monitoring page */}
          <Link
            href="/live"
            className="
              flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold
              bg-sky-500/10 border border-sky-500/30 text-sky-400
              hover:bg-sky-500/20 hover:border-sky-500/50
              transition-colors duration-150 w-fit flex-shrink-0
            "
          >
            <Radio className="w-4 h-4 animate-pulse" />
            Live Monitoring
          </Link>
        </div>

        {/* ── MAIN GRID ──────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-8 items-start">

          {/* ── LEFT PANEL ─────────────────────────────────────── */}
          <div className="flex flex-col gap-6">

            {/* KPI input form */}
            <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-6">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-5">
                KPI Input Values
              </p>
              <KpiInputForm
                values={kpiValues}
                onChange={handleFieldChange}
                onPreset={handlePreset}
              />
            </div>

            {/* Quick-glance KPI cards */}
            <div>
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">
                Key Metric Summary
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {kpiCards.map((card) => (
                  <KpiCard key={card.label} {...card} />
                ))}
              </div>
            </div>

            {/* KPI health bar chart */}
            <KpiChart kpiValues={kpiValues} />

            {/* PREDICT BUTTON */}
            <button
              onClick={handlePredict}
              disabled={status === "loading"}
              className="
                w-full flex items-center justify-center gap-2.5
                py-3.5 rounded-xl font-semibold text-sm
                bg-sky-500 hover:bg-sky-400 active:bg-sky-600
                disabled:opacity-50 disabled:cursor-not-allowed
                text-white transition-colors duration-150
                shadow-lg shadow-sky-500/20
              "
            >
              {status === "loading" ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running prediction…
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Predict Network Risk
                </>
              )}
            </button>

            {/* Error message */}
            {status === "error" && (
              <div className="rounded-xl border border-red-500/30 bg-red-950/20 px-4 py-3">
                <p className="text-xs font-semibold text-red-400 mb-1">API Error</p>
                <p className="text-xs text-slate-400">{errorMsg}</p>
              </div>
            )}
          </div>

          {/* ── RIGHT PANEL ────────────────────────────────────── */}
          <div className="flex flex-col gap-6">

            {/* Probability gauge + result — shown after first prediction */}
            {prediction ? (
              <>
                {/* Gauge */}
                <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-6 flex flex-col items-center gap-4">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest self-start">
                    Risk Probability
                  </p>
                  <ProbabilityGauge
                    probability={prediction.probability}
                    riskCode={prediction.risk_code}
                  />

                  {/* Big risk label below the gauge */}
                  <div className="text-center">
                    <p
                      className={`text-4xl font-bold font-mono tracking-tight ${
                        prediction.risk_code === 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {prediction.risk_label.toUpperCase()}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {prediction.risk_code === 0
                        ? "Network state is within normal operating parameters."
                        : "Network is experiencing degraded conditions."}
                    </p>
                  </div>
                </div>

                {/* Alert box with recommendation */}
                <AlertBox result={prediction} />
              </>
            ) : (
              /* Placeholder shown before first prediction */
              <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-8 flex flex-col items-center justify-center gap-3 text-center min-h-[280px]">
                <div className="w-12 h-12 rounded-full border border-slate-700 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-slate-600" />
                </div>
                <p className="text-sm font-semibold text-slate-500">
                  No prediction yet
                </p>
                <p className="text-xs text-slate-600 max-w-[200px]">
                  Enter KPI values or load a sample, then click{" "}
                  <span className="text-slate-400">Predict Network Risk</span>.
                </p>
              </div>
            )}

            {/* How it works — explanation panel (helps in viva demos) */}
            <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-5 flex flex-col gap-3">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">
                How It Works
              </p>
              <ol className="flex flex-col gap-2.5">
                {[
                  "18 KPI values are sent to the FastAPI backend via POST /api/predict.",
                  "The pre-trained Random Forest model evaluates all features simultaneously.",
                  "The model outputs a binary label (Normal / Degraded) and a probability.",
                  "A recommendation is generated based on the label and confidence level.",
                ].map((step, i) => (
                  <li key={i} className="flex gap-2.5 text-xs text-slate-400">
                    <span className="font-mono text-slate-600 flex-shrink-0 w-4">{i + 1}.</span>
                    {step}
                  </li>
                ))}
              </ol>
            </div>

            {/* Model info card */}
            <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-5">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">
                Model Info
              </p>
              <div className="grid grid-cols-2 gap-y-2 text-xs font-mono">
                {[
                  ["Algorithm",   "Random Forest"],
                  ["Trees",       "100 estimators"],
                  ["Training",    "1,672 rows"],
                  ["Test",        "418 rows"],
                  ["F1 Score",    "97.37%"],
                  ["Accuracy",    "98.56%"],
                  ["Precision",   "100.00%"],
                  ["Recall",      "94.87%"],
                ].map(([k, v]) => (
                  <div key={k} className="flex flex-col">
                    <span className="text-slate-600">{k}</span>
                    <span className="text-slate-300">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
        {/* ── MODEL COMPARISON PANEL ─────────────────────────────── */}
        {comparisonData.length > 0 && (
          <ModelComparison models={comparisonData} />
        )}

        {/* ── MODEL DIFFERENCES EXPLANATION ──────────────────────── */}
        <ModelDifferences />

      </main>
    </div>
  );
}
