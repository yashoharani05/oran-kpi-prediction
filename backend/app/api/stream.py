# =============================================================================
# app/api/stream.py
#
# PURPOSE:
#   Simulate live O-RAN monitoring by streaming rows from the labeled CSV
#   one at a time, on demand. The frontend polls every 2 seconds.
#
# WHY "SIMULATE"?
#   We don't have a live O-RAN testbed running, so we replay the real
#   testbed CSV row by row. Each call to GET /api/stream/next returns
#   the next row in time order, wrapping back to row 0 when we reach
#   the end (loop forever).
#
# HOW IT WORKS:
#   1. At startup (in main.py) the CSV is loaded into a global StreamState.
#   2. GET /api/stream/next reads the current row, runs the ML model on it,
#      advances the cursor by 1, and returns the result.
#   3. The frontend calls this endpoint every 2 seconds.
#   4. GET /api/stream/status tells the frontend how many rows have been
#      processed and what the current index is.
#   5. GET /api/stream/reset rewinds the cursor to row 0.
#
# ENDPOINTS:
#   GET /api/stream/next    — return next row + prediction, advance cursor
#   GET /api/stream/status  — return cursor position and dataset size
#   GET /api/stream/reset   — rewind cursor to start
# =============================================================================

import pandas as pd
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.utils.recommender import get_recommendation

router = APIRouter()

# Path to the labeled dataset produced by label_dataset.py
DATASET_PATH = "data/processed/labeled_dataset.csv"

# The 18 feature columns the ML model expects (in training order)
FEATURE_COLS = [
    "dl_mcs", "dl_n_samples", "dl_buffer_bytes", "tx_brate_downlink_mbps",
    "tx_pkts_downlink", "dl_cqi", "ul_mcs", "ul_n_samples", "ul_buffer_bytes",
    "rx_brate_uplink_mbps", "rx_pkts_uplink", "rx_errors_uplink_pct",
    "ul_sinr", "phr", "sum_requested_prbs", "sum_granted_prbs",
    "ul_turbo_iters", "prb_grant_ratio",
]


# =============================================================================
# STREAM STATE
# A simple class that holds the loaded DataFrame and the current row cursor.
# One instance is created at startup and shared across all requests.
# =============================================================================

class StreamState:
    """
    Holds the dataset and tracks which row is next to be served.

    Attributes:
        df      — the full labeled DataFrame (2090 rows)
        cursor  — index of the next row to return (0-based)
        total   — total number of rows in the dataset
        loaded  — True if the CSV was loaded successfully
    """
    def __init__(self):
        self.df:     Optional[pd.DataFrame] = None
        self.cursor: int = 0
        self.total:  int = 0
        self.loaded: bool = False

    def load(self, path: str):
        """Load the labeled CSV from disk. Called once at startup."""
        self.df     = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        self.total  = len(self.df)
        self.loaded = True
        print(f"[StreamState] Loaded {self.total} rows from {path}")

    def next_row(self) -> pd.Series:
        """
        Return the current row and advance the cursor.
        Wraps back to 0 after the last row (infinite loop).
        """
        row = self.df.iloc[self.cursor]
        self.cursor = (self.cursor + 1) % self.total   # wrap around
        return row

    def reset(self):
        """Rewind the cursor to the beginning."""
        self.cursor = 0


# Global instance — created here, populated in main.py lifespan
stream_state = StreamState()


# =============================================================================
# RESPONSE SCHEMAS (simple, defined inline for clarity)
# =============================================================================

class StreamRow(BaseModel):
    """One streamed row: KPI values + the model's prediction."""

    # Metadata
    row_index:   int    # which row in the dataset (0-based)
    timestamp:   str    # original testbed timestamp from the CSV
    total_rows:  int    # total rows in dataset (useful for progress bar)

    # The 18 KPI feature values
    dl_mcs:                  float
    dl_n_samples:            int
    dl_buffer_bytes:         int
    tx_brate_downlink_mbps:  float
    tx_pkts_downlink:        int
    dl_cqi:                  float
    ul_mcs:                  float
    ul_n_samples:            int
    ul_buffer_bytes:         int
    rx_brate_uplink_mbps:    float
    rx_pkts_uplink:          int
    rx_errors_uplink_pct:    float
    ul_sinr:                 float
    phr:                     int
    sum_requested_prbs:      int
    sum_granted_prbs:        int
    ul_turbo_iters:          float
    prb_grant_ratio:         float

    # The actual label from the dataset (ground truth)
    actual_risk:   int    # 0 or 1 — what the label_dataset.py script assigned

    # ML model prediction
    predicted_risk:   int    # 0 or 1 — what the Random Forest predicts
    risk_label:       str    # "Normal" or "Degraded"
    probability:      float  # 0.0 – 1.0
    recommendation:   str


class StreamStatus(BaseModel):
    """Current position in the dataset stream."""
    cursor:     int    # next row to be served
    total_rows: int    # total rows in dataset
    loaded:     bool   # True if CSV is loaded
    progress_pct: float  # cursor / total * 100


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get(
    "/next",
    response_model=StreamRow,
    summary="Get the next simulated KPI reading",
    description=(
        "Returns the next row from the labeled CSV, runs the ML model on it, "
        "and advances the internal cursor. "
        "Call this every 2 seconds to simulate live monitoring."
    ),
)
def stream_next(request: Request):
    """
    Return the next row from the dataset and advance the cursor.

    Steps:
    1. Check the stream and model are loaded.
    2. Read the current row from the DataFrame.
    3. Advance the cursor (wraps at the end).
    4. Build a DataFrame with the 18 feature columns.
    5. Run model.predict() and model.predict_proba().
    6. Return all KPI values + prediction as a StreamRow.
    """
    # --- Guard: stream not loaded ---
    if not stream_state.loaded:
        raise HTTPException(
            status_code=503,
            detail="Stream dataset not loaded. Check server startup logs.",
        )

    # --- Guard: model not loaded ---
    models = getattr(request.app.state, "models", {})
    model  = models.get("random_forest") or models.get("xgboost")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No ML model loaded. Run the training scripts first.",
        )

    # --- Read the current row ---
    current_index = stream_state.cursor
    row = stream_state.next_row()   # also advances cursor

    # --- Build feature DataFrame (named columns avoid sklearn warning) ---
    X = pd.DataFrame([{col: row[col] for col in FEATURE_COLS}])

    # --- Run the ML model ---
    predicted_risk = int(model.predict(X)[0])
    probability    = float(model.predict_proba(X)[0][1])

    # --- Build recommendation ---
    recommendation = get_recommendation(predicted_risk, probability)
    risk_label     = "Degraded" if predicted_risk == 1 else "Normal"

    # --- Return the streamed row ---
    return StreamRow(
        row_index  = current_index,
        timestamp  = str(row.name),       # row.name is the datetime index
        total_rows = stream_state.total,

        # KPI values — cast to Python native types for JSON serialisation
        dl_mcs                 = float(row["dl_mcs"]),
        dl_n_samples           = int(row["dl_n_samples"]),
        dl_buffer_bytes        = int(row["dl_buffer_bytes"]),
        tx_brate_downlink_mbps = float(row["tx_brate_downlink_mbps"]),
        tx_pkts_downlink       = int(row["tx_pkts_downlink"]),
        dl_cqi                 = float(row["dl_cqi"]),
        ul_mcs                 = float(row["ul_mcs"]),
        ul_n_samples           = int(row["ul_n_samples"]),
        ul_buffer_bytes        = int(row["ul_buffer_bytes"]),
        rx_brate_uplink_mbps   = float(row["rx_brate_uplink_mbps"]),
        rx_pkts_uplink         = int(row["rx_pkts_uplink"]),
        rx_errors_uplink_pct   = float(row["rx_errors_uplink_pct"]),
        ul_sinr                = float(row["ul_sinr"]),
        phr                    = int(row["phr"]),
        sum_requested_prbs     = int(row["sum_requested_prbs"]),
        sum_granted_prbs       = int(row["sum_granted_prbs"]),
        ul_turbo_iters         = float(row["ul_turbo_iters"]),
        prb_grant_ratio        = float(row["prb_grant_ratio"]),

        # Ground truth label (from label_dataset.py)
        actual_risk    = int(row["degradation_risk"]),

        # Model prediction
        predicted_risk = predicted_risk,
        risk_label     = risk_label,
        probability    = round(probability, 4),
        recommendation = recommendation,
    )


@router.get(
    "/status",
    response_model=StreamStatus,
    summary="Get stream position",
    description="Returns the current cursor position and total rows in the dataset.",
)
def stream_status():
    """Returns the current position in the dataset stream."""
    return StreamStatus(
        cursor       = stream_state.cursor,
        total_rows   = stream_state.total,
        loaded       = stream_state.loaded,
        progress_pct = round(stream_state.cursor / stream_state.total * 100, 1)
                       if stream_state.total > 0 else 0.0,
    )


@router.get(
    "/reset",
    summary="Reset stream to start",
    description="Rewinds the dataset cursor back to row 0.",
)
def stream_reset():
    """Reset the cursor to the beginning of the dataset."""
    stream_state.reset()
    return {"message": "Stream reset to row 0.", "cursor": 0}
