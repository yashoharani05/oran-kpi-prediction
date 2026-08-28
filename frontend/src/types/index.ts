/*
  types/index.ts
  All TypeScript interfaces used across the frontend.
*/

// The 18 KPI fields sent to the backend prediction endpoint.
export interface KpiValues {
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
}

// Response from POST /api/predict
export interface PredictionResult {
  risk_label: "Normal" | "Degraded";
  risk_code: 0 | 1;
  probability: number;
  recommendation: string;
  model_used: string;
  // --- Added by the forecasting correction (additive, optional) ---
  // risk_label/risk_code/probability now describe the FORECAST state
  // ~forecast_horizon_seconds ahead, not the same-instant state.
  current_status?: "Normal" | "Degraded" | null;
  current_score?: number | null;
  forecast_horizon_seconds?: number;
  early_warning?: boolean;
}

// Simple status enum for async API calls
export type ApiStatus = "idle" | "loading" | "success" | "error";
