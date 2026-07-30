/*
  app/live/page.tsx  (route: /live)
  ============================================================
  Live KPI Monitoring Dashboard.

  Polls GET /api/stream/next every 2 seconds.
  Each response contains one row of KPI data + the model's prediction.
  The page updates all displays — cards, chart, gauge, alert — in place.

  STATE:
    currentRow  — the most recent StreamRow from the API
    history     — last 60 rows (used for the line chart)
    isRunning   — whether the poller is active
    status      — "idle" | "loading" | "error"
    stats       — running counts of normal/degraded rows seen

  POLLING STRATEGY:
    We use setInterval (not useEffect with a dependency) so the timer
    fires every 2000 ms regardless of render cycles. The interval is
    cleared when the component unmounts or the user pauses.
*/

"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { Play, Pause, RotateCcw, ArrowLeft } from "lucide-react";

import Header from "@/components/Header";
import KpiCard from "@/components/KpiCard";
import AlertBox from "@/components/AlertBox";
import ProbabilityGauge from "@/components/ProbabilityGauge";
import LiveChart from "@/components/LiveChart";
import StreamProgress from "@/components/StreamProgress";

import { fetchNextRow, resetStream, checkHealth } from "@/lib/api";
import type { StreamRow } from "@/lib/api";
import type { PredictionResult } from "@/types";

// How often to fetch a new row (milliseconds)
const POLL_INTERVAL_MS = 2000;

// How many historical rows to keep for the chart
const MAX_HISTORY = 60;

// =============================================================================
// Helper: convert a StreamRow into the PredictionResult shape AlertBox expects
// =============================================================================
function toPredictionResult(row: StreamRow): PredictionResult {
  return {
    risk_label: row.risk_label,
    risk_code: row.predicted_risk as 0 | 1,
    probability: row.probability,
    recommendation: row.recommendation,
    model_used: "Random Forest",
  };
}

// =============================================================================
// Helper: determine warning level for a KPI card
// (same thresholds used in the manual dashboard)
// =============================================================================
function kpiState(
  key: string,
  value: number,
): { critical: boolean; warn: boolean } {
  switch (key) {
    case "dl_mcs":
      return { critical: value < 6.57, warn: value < 8.0 };
    case "dl_cqi":
      return { critical: value < 6.33, warn: value < 7.0 };
    case "rx_errors_uplink_pct":
      return { critical: value > 57.1, warn: value > 20 };
    case "tx_brate_downlink_mbps":
      return { critical: value < 0.003, warn: value < 0.05 };
    case "prb_grant_ratio":
      return { critical: value < 0.15, warn: value < 0.21 };
    case "ul_turbo_iters":
      return { critical: value > 6.85, warn: value > 3 };
    case "ul_sinr":
      return { critical: value > 0 && value < 5.05, warn: false };
    default:
      return { critical: false, warn: false };
  }
}

// =============================================================================
// PAGE
// =============================================================================
export default function LiveMonitorPage() {
  const [currentRow, setCurrentRow] = useState<StreamRow | null>(null);
  const [history, setHistory] = useState<StreamRow[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [stats, setStats] = useState({ total: 0, degraded: 0, normal: 0 });

  // Ref to hold the interval ID so it survives re-renders without triggering effects
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // --- Health check on mount ---
  useEffect(() => {
    checkHealth()
      .then((res) => setModelLoaded(res.model_loaded))
      .catch(() => setModelLoaded(false));
  }, []);

  // --- Fetch one row and update all state ---
  const fetchRow = useCallback(async () => {
    try {
      const row = await fetchNextRow();
      setCurrentRow(row);
      setHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), row]);
      setStats((prev) => ({
        total: prev.total + 1,
        degraded: prev.degraded + (row.predicted_risk === 1 ? 1 : 0),
        normal: prev.normal + (row.predicted_risk === 0 ? 1 : 0),
      }));
      setErrorMsg("");
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "Could not reach the backend. Is it running?";
      setErrorMsg(msg);
    }
  }, []);

  // --- Start / stop the poller ---
  const startPolling = useCallback(() => {
    if (intervalRef.current) return; // already running
    fetchRow(); // immediate first fetch
    intervalRef.current = setInterval(fetchRow, POLL_INTERVAL_MS);
    setIsRunning(true);
  }, [fetchRow]);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsRunning(false);
  }, []);

  // Clean up on unmount
  useEffect(() => () => stopPolling(), [stopPolling]);

  // --- Reset stream ---
  const handleReset = async () => {
    stopPolling();
    await resetStream();
    setCurrentRow(null);
    setHistory([]);
    setStats({ total: 0, degraded: 0, normal: 0 });
    setErrorMsg("");
  };

  // Derived data for display
  const prediction = currentRow ? toPredictionResult(currentRow) : null;
  const degradedPct =
    stats.total > 0 ? ((stats.degraded / stats.total) * 100).toFixed(1) : "0.0";

  return (
    <div className="min-h-screen bg-[#0f172a]">
      <Header modelLoaded={modelLoaded} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6">
        {/* ── TOP BAR ─────────────────────────────────────────── */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          {/* Back link */}
          <Link
            href="/"
            className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors w-fit"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Manual Predict
          </Link>

          <div className="sm:ml-auto flex items-center gap-3 flex-wrap">
            {/* Play / Pause */}
            <button
              onClick={isRunning ? stopPolling : startPolling}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold
                transition-colors duration-150
                ${
                  isRunning
                    ? "bg-amber-500/20 border border-amber-500/40 text-amber-400 hover:bg-amber-500/30"
                    : "bg-sky-500 text-white hover:bg-sky-400 shadow-lg shadow-sky-500/20"
                }
              `}
            >
              {isRunning ? (
                <>
                  <Pause className="w-4 h-4" /> Pause
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" /> Start Monitoring
                </>
              )}
            </button>

            {/* Reset */}
            <button
              onClick={handleReset}
              className="
                flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold
                border border-slate-700 text-slate-400
                hover:border-slate-500 hover:text-slate-200
                transition-colors duration-150
              "
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          </div>
        </div>

        {/* ── PAGE TITLE ──────────────────────────────────────── */}
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Live KPI Monitoring
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Replaying{" "}
            <span className="font-mono text-slate-400">
              labeled_dataset.csv
            </span>{" "}
            — one row every 2 seconds.{" "}
            {currentRow && (
              <span className="font-mono text-slate-400">
                {currentRow.total_rows.toLocaleString()} rows total.
              </span>
            )}
          </p>
        </div>

        {/* ── STREAM PROGRESS BAR ─────────────────────────────── */}
        {currentRow && (
          <StreamProgress
            rowIndex={currentRow.row_index}
            totalRows={currentRow.total_rows}
          />
        )}

        {/* ── ERROR ───────────────────────────────────────────── */}
        {errorMsg && (
          <div className="rounded-xl border border-red-500/30 bg-red-950/20 px-4 py-3">
            <p className="text-xs font-semibold text-red-400 mb-1">
              Stream Error
            </p>
            <p className="text-xs text-slate-400">{errorMsg}</p>
          </div>
        )}

        {/* ── SESSION STATS ───────────────────────────────────── */}
        <div className="grid grid-cols-3 gap-3">
          {[
            {
              label: "Rows Seen",
              value: stats.total.toLocaleString(),
              color: "text-slate-200",
            },
            {
              label: "Normal",
              value: stats.normal.toLocaleString(),
              color: "text-green-400",
            },
            {
              label: "Degraded",
              value: `${stats.degraded} (${degradedPct}%)`,
              color: "text-red-400",
            },
          ].map(({ label, value, color }) => (
            <div
              key={label}
              className="bg-[#1e293b] border border-slate-700/50 rounded-xl px-4 py-3"
            >
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">
                {label}
              </p>
              <p className={`font-mono text-xl font-bold ${color}`}>{value}</p>
            </div>
          ))}
        </div>

        {/* ── MAIN GRID ───────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6 items-start">
          {/* LEFT — KPI cards + chart */}
          <div className="flex flex-col gap-6">
            {/* KPI metric cards — 8 tiles */}
            {currentRow ? (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">
                    Current KPI Values
                  </p>
                  <p className="text-[10px] font-mono text-slate-600">
                    {new Date(currentRow.timestamp).toLocaleTimeString()}
                  </p>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    { key: "dl_mcs", label: "DL MCS", unit: "", decimals: 2 },
                    { key: "dl_cqi", label: "DL CQI", unit: "", decimals: 2 },
                    {
                      key: "tx_brate_downlink_mbps",
                      label: "DL Throughput",
                      unit: "Mbps",
                      decimals: 4,
                    },
                    {
                      key: "rx_errors_uplink_pct",
                      label: "UL Error Rate",
                      unit: "%",
                      decimals: 1,
                    },
                    {
                      key: "ul_sinr",
                      label: "UL SINR",
                      unit: "dB",
                      decimals: 2,
                    },
                    {
                      key: "prb_grant_ratio",
                      label: "PRB Ratio",
                      unit: "",
                      decimals: 3,
                    },
                    {
                      key: "ul_turbo_iters",
                      label: "Turbo Iters",
                      unit: "",
                      decimals: 2,
                    },
                    {
                      key: "rx_brate_uplink_mbps",
                      label: "UL Throughput",
                      unit: "Mbps",
                      decimals: 4,
                    },
                  ].map(({ key, label, unit, decimals }) => {
                    const value = currentRow[key as keyof StreamRow] as number;
                    const { critical, warn } = kpiState(key, value);
                    return (
                      <KpiCard
                        key={key}
                        label={label}
                        value={value}
                        unit={unit}
                        decimals={decimals}
                        critical={critical}
                        warn={warn}
                      />
                    );
                  })}
                </div>
              </div>
            ) : (
              // Placeholder before first row arrives
              <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-8 flex flex-col items-center justify-center gap-3 min-h-[140px]">
                <p className="text-sm text-slate-500">
                  {isRunning
                    ? "Waiting for first row…"
                    : "Press Start Monitoring to begin."}
                </p>
              </div>
            )}

            {/* Live throughput + error rate chart */}
            <LiveChart history={history} maxPoints={30} />
          </div>

          {/* RIGHT — gauge + alert */}
          <div className="flex flex-col gap-6">
            {prediction ? (
              <>
                {/* Probability gauge */}
                <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-6 flex flex-col items-center gap-4">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest self-start">
                    Risk Probability
                  </p>
                  <ProbabilityGauge
                    probability={prediction.probability}
                    riskCode={prediction.risk_code}
                  />
                  {/* Big label */}
                  <div className="text-center">
                    <p
                      className={`text-4xl font-bold font-mono tracking-tight ${
                        prediction.risk_code === 0
                          ? "text-green-400"
                          : "text-red-400"
                      }`}
                    >
                      {prediction.risk_label.toUpperCase()}
                    </p>
                    <p className="text-xs text-slate-500 mt-1 font-mono">
                      Row {currentRow?.row_index} —{" "}
                      {(prediction.probability * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                </div>

                {/* Alert box */}
                <AlertBox result={prediction} />

                {/* Ground truth comparison */}
                {currentRow && (
                  <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-4">
                    <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">
                      Ground Truth vs Prediction
                    </p>
                    <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                      <div className="flex flex-col gap-0.5">
                        <span className="text-slate-600">Actual label</span>
                        <span
                          className={
                            currentRow.actual_risk === 0
                              ? "text-green-400"
                              : "text-red-400"
                          }
                        >
                          {currentRow.actual_risk === 0 ? "Normal" : "Degraded"}
                        </span>
                      </div>
                      <div className="flex flex-col gap-0.5">
                        <span className="text-slate-600">Predicted</span>
                        <span
                          className={
                            currentRow.predicted_risk === 0
                              ? "text-green-400"
                              : "text-red-400"
                          }
                        >
                          {currentRow.predicted_risk === 0
                            ? "Normal"
                            : "Degraded"}
                        </span>
                      </div>
                      <div className="col-span-2 flex flex-col gap-0.5 border-t border-slate-700 pt-2 mt-1">
                        <span className="text-slate-600">Match?</span>
                        <span
                          className={
                            currentRow.actual_risk === currentRow.predicted_risk
                              ? "text-green-400"
                              : "text-amber-400"
                          }
                        >
                          {currentRow.actual_risk === currentRow.predicted_risk
                            ? "✓ Correct prediction"
                            : "✗ Mismatch"}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-8 flex flex-col items-center justify-center gap-3 min-h-[280px] text-center">
                <Play className="w-8 h-8 text-slate-700" />
                <p className="text-sm text-slate-500">
                  Prediction appears here
                </p>
                <p className="text-xs text-slate-600">Press Start Monitoring</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
