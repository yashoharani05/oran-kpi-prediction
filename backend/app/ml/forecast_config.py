# =============================================================================
# forecast_config.py
#
# PURPOSE:
#   Single source of truth for the forecasting-correction constants used by
#   label_dataset.py, train_random_forest.py, train_xgboost.py, train_lstm.py,
#   and the backend prediction/stream API.
#
#   Added as part of the methodological correction described in
#   docs/FORECASTING_METHODOLOGY_UPDATE.md:
#
#     OLD: KPIs @ t  ->  degradation @ t            (same-instant classification)
#     NEW: KPIs @ t  ->  degradation @ t + horizon   (future forecasting)
#
#   Centralising these numbers means the horizon can be changed (e.g. to test
#   1s / 10s / 30s ahead) by editing this one file instead of hunting through
#   every training script.
# =============================================================================

# ---------------------------------------------------------------------------
# Sampling interval
#
# Verified from the raw testbed data: the README reports 2,090 rows spanning
# 8.7 minutes -> 8.7*60 / 2090 ≈ 0.2498s ≈ 250ms per row. This is an average,
# not a guarantee of a perfectly uniform clock — see
# `verify_sampling_interval()` below, which checks the *actual* timestamp
# deltas in a given dataset instead of blindly trusting this constant.
# ---------------------------------------------------------------------------
SAMPLE_INTERVAL_SECONDS = 0.25

# ---------------------------------------------------------------------------
# Forecast horizon — how far ahead the model must predict.
# ---------------------------------------------------------------------------
FORECAST_HORIZON_SECONDS = 5

FORECAST_HORIZON_ROWS = round(FORECAST_HORIZON_SECONDS / SAMPLE_INTERVAL_SECONDS)
# = 20 rows, i.e. row t predicts the degradation state at row t + 20.

# ---------------------------------------------------------------------------
# LSTM sequence length. Kept as a SEPARATE constant from
# FORECAST_HORIZON_ROWS on purpose (see docs/FORECASTING_METHODOLOGY_UPDATE.md
# section "LSTM alignment") — the two numbers happen to both be 20 today, but
# conflating them is exactly the bug described in the correction brief
# ("avoid an accidental 10-second forecast"). If you change one, do not
# assume the other should follow.
# ---------------------------------------------------------------------------
LSTM_WINDOW_SIZE = 20

# ---------------------------------------------------------------------------
# Early-warning alert threshold used by the API/stream layer.
# An early warning fires only when:
#     current_status == Normal AND forecast_status == Degraded
#     AND forecast_probability >= FORECAST_ALERT_THRESHOLD
# ---------------------------------------------------------------------------
FORECAST_ALERT_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Column names used across the pipeline. Centralised here so every script
# agrees on what to call things.
# ---------------------------------------------------------------------------
SESSION_ID_COL      = "session_id"        # which raw recording/session a row came from
CURRENT_LABEL_COL   = "degradation_risk"        # existing column — rule-based state AT time t ("label_now")
CURRENT_SCORE_COL   = "degradation_score"       # existing column — 0-7 rule-based score AT time t
FUTURE_LABEL_COL    = "degradation_risk_future"  # NEW — rule-based state AT time t + horizon ("label_future")

# Columns that must NEVER be fed to a model as an input feature.
NON_FEATURE_COLS = [
    SESSION_ID_COL,
    CURRENT_SCORE_COL,
    CURRENT_LABEL_COL,
    FUTURE_LABEL_COL,
]


def verify_sampling_interval(df, timestamp_col=None, tolerance_pct=20):
    """
    Inspect the actual time deltas between consecutive rows (per session, if a
    session column is present) and compare them against SAMPLE_INTERVAL_SECONDS.

    This exists because section 18 of the correction brief explicitly warns
    against "silently assuming row 20 always equals exactly 5 seconds" —
    we check it instead of assuming it.

    Args:
        df: DataFrame indexed by datetime timestamp (as produced by
            clean_dataset.py's set_time_index), optionally containing a
            SESSION_ID_COL column.
        timestamp_col: name of a timestamp column to use instead of the
            index, if the index is not already a DatetimeIndex.
        tolerance_pct: how far (in %) the median observed interval may drift
            from SAMPLE_INTERVAL_SECONDS before we print a warning.

    Returns:
        dict with the observed median/mean interval in seconds and whether
        it is within tolerance.
    """
    import numpy as np
    import pandas as pd

    if timestamp_col is not None:
        ts = pd.to_datetime(df[timestamp_col])
    else:
        ts = pd.to_datetime(df.index.to_series())

    if SESSION_ID_COL in df.columns:
        deltas = []
        for _, group in df.groupby(SESSION_ID_COL):
            g_ts = pd.to_datetime(group.index.to_series()) if timestamp_col is None else pd.to_datetime(group[timestamp_col])
            g_ts = g_ts.sort_values()
            deltas.append(g_ts.diff().dropna())
        deltas = pd.concat(deltas) if deltas else pd.Series(dtype="timedelta64[ns]")
    else:
        deltas = ts.sort_values().diff().dropna()

    if len(deltas) == 0:
        return {"median_seconds": None, "mean_seconds": None, "within_tolerance": None}

    median_s = deltas.median().total_seconds()
    mean_s   = deltas.mean().total_seconds()

    drift_pct = abs(median_s - SAMPLE_INTERVAL_SECONDS) / SAMPLE_INTERVAL_SECONDS * 100
    within_tolerance = drift_pct <= tolerance_pct

    print(f"\n  [Sampling interval check]")
    print(f"    Assumed interval : {SAMPLE_INTERVAL_SECONDS*1000:.0f} ms")
    print(f"    Observed median  : {median_s*1000:.1f} ms")
    print(f"    Observed mean    : {mean_s*1000:.1f} ms")
    if within_tolerance:
        print(f"    ✓ Within {tolerance_pct}% tolerance — FORECAST_HORIZON_ROWS={FORECAST_HORIZON_ROWS} "
              f"is a reasonable approximation of {FORECAST_HORIZON_SECONDS}s ahead.")
    else:
        print(f"    ⚠ Observed interval drifts {drift_pct:.1f}% from the assumed {SAMPLE_INTERVAL_SECONDS*1000:.0f} ms.")
        print(f"      FORECAST_HORIZON_ROWS={FORECAST_HORIZON_ROWS} may not correspond to exactly "
              f"{FORECAST_HORIZON_SECONDS}s. Consider a timestamp-based shift instead of a row-count shift.")

    return {
        "median_seconds": median_s,
        "mean_seconds": mean_s,
        "within_tolerance": within_tolerance,
    }
