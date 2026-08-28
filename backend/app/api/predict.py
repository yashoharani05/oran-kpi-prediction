# =============================================================================
# app/api/predict.py
#
# Endpoints:
#   GET  /api/health      — status of all three models
#   POST /api/predict     — single prediction (?model=random_forest|xgboost|lstm)
#   GET  /api/comparison  — saved metrics JSON (naive baseline + all 3 models)
#
# METHODOLOGY NOTE (forecasting correction):
# The three ML models loaded here now forecast degradation
# FORECAST_HORIZON_SECONDS ahead of the submitted KPIs, not the same instant.
# 'current_status' in the response is computed separately by the
# deterministic rule (app/utils/degradation_rule.py), never by these models.
# See docs/FORECASTING_METHODOLOGY_UPDATE.md.
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, Query
from app.schemas import KpiInput, PredictionResponse, HealthResponse
from app.utils.recommender import get_recommendation
from app.utils.degradation_rule import current_status as compute_current_status
from app.ml.forecast_config import FORECAST_HORIZON_SECONDS, FORECAST_ALERT_THRESHOLD

router = APIRouter()

COMPARISON_PATH = os.path.join("models", "comparison_report.json")
WINDOW_SIZE     = 20   # must match train_lstm.py's LSTM_WINDOW_SIZE

FEATURE_COLS = [
    "dl_mcs", "dl_n_samples", "dl_buffer_bytes", "tx_brate_downlink_mbps",
    "tx_pkts_downlink", "dl_cqi", "ul_mcs", "ul_n_samples", "ul_buffer_bytes",
    "rx_brate_uplink_mbps", "rx_pkts_uplink", "rx_errors_uplink_pct",
    "ul_sinr", "phr", "sum_requested_prbs", "sum_granted_prbs",
    "ul_turbo_iters", "prb_grant_ratio",
]

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "lstm":          "LSTM",
}


def _get_models(request: Request):
    return getattr(request.app.state, "models", {})


# =============================================================================
# GET /api/health
# =============================================================================

@router.get("/health", response_model=HealthResponse, summary="Health check")
def health_check(request: Request):
    m = _get_models(request)
    rf_ok  = m.get("random_forest") is not None
    xgb_ok = m.get("xgboost")       is not None
    lstm_ok= m.get("lstm")           is not None

    loaded_names = [n for n, ok in [("RF", rf_ok), ("XGB", xgb_ok), ("LSTM", lstm_ok)] if ok]

    return HealthResponse(
        status       = "ok" if any([rf_ok, xgb_ok, lstm_ok]) else "error",
        model_loaded = any([rf_ok, xgb_ok, lstm_ok]),
        model_name   = " + ".join(loaded_names) if loaded_names else "None",
        message      = f"Loaded: RF={'yes' if rf_ok else 'no'}  XGB={'yes' if xgb_ok else 'no'}  LSTM={'yes' if lstm_ok else 'no'}",
    )


# =============================================================================
# POST /api/predict?model=...
# =============================================================================

@router.post("/predict", response_model=PredictionResponse, summary="Predict degradation risk")
def predict(
    kpi_input: KpiInput,
    request:   Request,
    model:     str = Query(default="random_forest",
                           description="Model: random_forest | xgboost | lstm"),
):
    if model not in MODEL_DISPLAY:
        raise HTTPException(400, detail=f"Unknown model. Choose: {list(MODEL_DISPLAY)}")

    models = _get_models(request)

    # ----------------------------------------------------------------
    # LSTM prediction path
    # LSTM needs a sequence of WINDOW_SIZE rows. Since the API receives
    # only ONE row, we replicate it WINDOW_SIZE times to fill the window.
    # This is a simplification suitable for a student project demo.
    # In production you would maintain a rolling buffer of recent rows.
    # ----------------------------------------------------------------
    if model == "lstm":
        lstm_model  = models.get("lstm")
        lstm_scaler = models.get("lstm_scaler")

        if lstm_model is None or lstm_scaler is None:
            raise HTTPException(503, detail="LSTM model not loaded. Run train_lstm.py first.")

        # Build a single-row DataFrame, scale it, replicate into a sequence
        row_dict = {col: getattr(kpi_input, col) for col in FEATURE_COLS}
        row_df   = pd.DataFrame([row_dict])
        row_scaled = lstm_scaler.transform(row_df)            # shape: (1, 18)
        # Replicate to create a (1, WINDOW_SIZE, 18) tensor
        sequence   = np.tile(row_scaled, (WINDOW_SIZE, 1))    # shape: (20, 18)
        X          = sequence[np.newaxis, :, :]               # shape: (1, 20, 18)

        try:
            probability = float(lstm_model.predict(X, verbose=0)[0][0])
            risk_code   = 1 if probability >= 0.5 else 0
        except Exception as e:
            raise HTTPException(500, detail=f"LSTM prediction failed: {e}")

    # ----------------------------------------------------------------
    # RF / XGBoost prediction path (both use the same sklearn API)
    # ----------------------------------------------------------------
    else:
        clf = models.get(model)
        if clf is None:
            raise HTTPException(503, detail=f"Model '{model}' not loaded.")

        X = pd.DataFrame([{col: getattr(kpi_input, col) for col in FEATURE_COLS}])
        try:
            risk_code   = int(clf.predict(X)[0])
            probability = float(clf.predict_proba(X)[0][1])
        except Exception as e:
            raise HTTPException(500, detail=f"Prediction failed: {e}")

    # ----------------------------------------------------------------
    # Current status: computed with the DETERMINISTIC rule (same logic as
    # label_dataset.py), independent of the ML forecast model — per section
    # 23 of the correction brief. This never uses the forecasting model.
    # ----------------------------------------------------------------
    kpi_dict = {col: getattr(kpi_input, col) for col in FEATURE_COLS}
    status_label, status_score = compute_current_status(kpi_dict)

    early_warning = (
        status_label == "Normal"
        and risk_code == 1
        and probability >= FORECAST_ALERT_THRESHOLD
    )

    return PredictionResponse(
        risk_label     = "Degraded" if risk_code == 1 else "Normal",
        risk_code      = risk_code,
        probability    = round(probability, 4),
        recommendation = get_recommendation(risk_code, probability),
        model_used     = MODEL_DISPLAY[model],
        current_status = status_label,
        current_score  = status_score,
        forecast_horizon_seconds = FORECAST_HORIZON_SECONDS,
        early_warning  = early_warning,
    )


# =============================================================================
# GET /api/comparison
# =============================================================================

@router.get("/comparison", summary="Three-way model comparison metrics")
def get_comparison():
    if not os.path.exists(COMPARISON_PATH):
        raise HTTPException(404, detail="Run train_xgboost.py and train_lstm.py first.")
    with open(COMPARISON_PATH) as f:
        return json.load(f)
