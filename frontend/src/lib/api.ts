/*
  lib/api.ts
  Axios-based API client.

  All HTTP calls to the FastAPI backend go through this file.
  Using a shared axios instance with baseURL means we only need
  to change the URL in one place if the backend port changes.
*/

import axios from "axios";
import type { KpiValues, PredictionResult } from "@/types";

// Create a reusable axios instance pointed at the FastAPI backend.
// Next.js rewrites in next.config.js proxy /api/* to http://localhost:8000/api/*
// so we can use a relative base URL here.
// In Docker, NEXT_PUBLIC_BACKEND_URL points to the backend service.
// In local dev, Next.js rewrites /api/* → http://localhost:8000/api/* via next.config.js.
// Using a relative "/api" base here works for both cases.
const apiClient = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
  timeout: 10000, // 10 second timeout — increase if LSTM is slow
});

// --- Health check ---
export async function checkHealth(): Promise<{
  status: string;
  model_loaded: boolean;
  model_name: string;
  message: string;
}> {
  const { data } = await apiClient.get("/health");
  return data;
}

// --- Prediction ---
// Send 18 KPI values, receive risk label + probability + recommendation.
export async function predictRisk(
  kpi: KpiValues,
  model: "random_forest" | "xgboost" | "lstm" = "random_forest",
): Promise<PredictionResult> {
  const { data } = await apiClient.post<PredictionResult>(
    `/predict?model=${model}`,
    kpi,
  );
  return data;
}

// --- Live Stream ---
// GET /api/stream/next  — fetch the next row from the CSV + its prediction
export async function fetchNextRow(): Promise<StreamRow> {
  const { data } = await apiClient.get<StreamRow>("/stream/next");
  return data;
}

// GET /api/stream/reset — rewind the CSV cursor back to row 0
export async function resetStream(): Promise<void> {
  await apiClient.get("/stream/reset");
}

// Inline type for the stream row response
// (mirrors StreamRow in backend/app/api/stream.py)
export interface StreamRow {
  row_index: number;
  timestamp: string;
  total_rows: number;
  dl_mcs: number;
  dl_n_samples: number;
  dl_buffer_bytes: number;
  tx_brate_downlink_mbps: number;
  tx_pkts_downlink: number;
  dl_cqi: number;
  ul_mcs: number;
  ul_n_samples: number;
  ul_buffer_bytes: number;
  rx_brate_uplink_mbps: number;
  rx_pkts_uplink: number;
  rx_errors_uplink_pct: number;
  ul_sinr: number;
  phr: number;
  sum_requested_prbs: number;
  sum_granted_prbs: number;
  ul_turbo_iters: number;
  prb_grant_ratio: number;
  actual_risk: number;
  actual_future_risk?: number | null;
  predicted_risk: number;
  risk_label: "Normal" | "Degraded";
  probability: number;
  recommendation: string;
  forecast_horizon_seconds?: number;
  current_status?: "Normal" | "Degraded" | null;
  current_score?: number | null;
  early_warning?: boolean;
}

// --- Model comparison ---
export interface ModelMetrics {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export async function fetchComparison(): Promise<{ models: ModelMetrics[] }> {
  const { data } = await apiClient.get<{ models: ModelMetrics[] }>(
    "/comparison",
  );
  return data;
}
