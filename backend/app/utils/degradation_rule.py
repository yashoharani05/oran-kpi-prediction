# =============================================================================
# app/utils/degradation_rule.py
#
# PURPOSE:
#   Provide the CURRENT-instant network status ("Normal"/"Degraded") using
#   the SAME deterministic, rule-based scoring logic as
#   app/ml/label_dataset.py — but for a single incoming KPI row at request
#   time, using the quantile thresholds computed once during training and
#   persisted to models/degradation_thresholds.json.
#
# WHY THIS EXISTS (correction brief section 23):
#   The ML models (Random Forest / XGBoost / LSTM) now predict the FUTURE
#   state (~5 seconds ahead). They must NOT also be used to describe "what
#   is happening right now" — that would conflate two different questions.
#   "Current status" is intentionally the deterministic rule, exactly as
#   label_dataset.py defines it; "forecast status" is the ML model's output.
#
# This mirrors backend/app/ml/label_dataset.py's score_row()/create_labels()
# function almost verbatim — kept as a small, separate copy (rather than a
# shared import) because label_dataset.py is a batch/offline script with
# print-heavy instrumentation not suited to being called per-HTTP-request.
# =============================================================================

import json
import os
from typing import Optional

THRESHOLDS_PATH = os.path.join("models", "degradation_thresholds.json")

_cached_thresholds: Optional[dict] = None
_cached_score_threshold: Optional[int] = None


def _load_thresholds():
    """Load and cache the quantile thresholds saved by label_dataset.py."""
    global _cached_thresholds, _cached_score_threshold

    if _cached_thresholds is not None:
        return _cached_thresholds, _cached_score_threshold

    if not os.path.exists(THRESHOLDS_PATH):
        return None, None

    with open(THRESHOLDS_PATH) as f:
        payload = json.load(f)

    _cached_thresholds = payload["thresholds"]
    _cached_score_threshold = payload["degradation_score_threshold"]
    return _cached_thresholds, _cached_score_threshold


def score_row(kpis: dict, thresholds: dict) -> int:
    """
    Identical scoring logic to label_dataset.py's score_row() — kept in sync
    manually; if you change one, change the other. See label_dataset.py's
    SELECTED_FEATURES / score_row() for the full explanation of each rule.
    """
    score = 0

    if kpis["rx_errors_uplink_pct"] > thresholds["rx_errors_uplink_pct_high"]:
        score += 1
    if kpis["dl_cqi"] < thresholds["dl_cqi_low"]:
        score += 1
    if kpis["tx_brate_downlink_mbps"] < thresholds["tx_brate_downlink_mbps_low"]:
        score += 1
    if kpis["prb_grant_ratio"] < thresholds["prb_grant_ratio_low"]:
        score += 1
    if kpis["dl_mcs"] < thresholds["dl_mcs_low"]:
        score += 1
    if kpis["ul_sinr"] > 0 and kpis["ul_sinr"] < thresholds["ul_sinr_low"]:
        score += 1
    if kpis["ul_turbo_iters"] > 0 and kpis["ul_turbo_iters"] > thresholds["ul_turbo_iters_high"]:
        score += 1

    return score


def current_status(kpis: dict):
    """
    Return (status_label, score) for a single KPI row, e.g. ("Normal", 1) or
    ("Degraded", 3).

    If thresholds haven't been generated yet (label_dataset.py not run),
    returns (None, None) so callers can omit the field gracefully instead of
    crashing the prediction endpoint over a missing rule-based extra.
    """
    thresholds, score_threshold = _load_thresholds()
    if thresholds is None:
        return None, None

    score = score_row(kpis, thresholds)
    status = "Degraded" if score >= score_threshold else "Normal"
    return status, score
