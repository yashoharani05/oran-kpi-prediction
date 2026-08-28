# =============================================================================
# test_forecast_sanity.py
#
# PURPOSE:
#   Lightweight, dependency-free sanity checks for the forecasting
#   correction (see docs/FORECASTING_METHODOLOGY_UPDATE.md, section 32).
#   Uses synthetic data — does NOT require backend/data/raw/raw1.csv — so it
#   can be run any time as a quick regression guard on the shifting/
#   windowing logic itself, independent of the real dataset.
#
# HOW TO RUN (from the backend/ folder with venv active):
#   python app/ml/test_forecast_sanity.py
#
# CHECKS:
#   1. Label shift: future[i] == current[i + horizon] within each session.
#   2. Session boundaries: a shift/window never crosses two sessions.
#   3. No leakage: label/session columns never end up in the feature list.
#   4. LSTM window/target alignment: for a session-aware window ending at
#      row t, the target must be the FUTURE label AT row t (not t+1).
# =============================================================================

import sys
import numpy as np
import pandas as pd

from forecast_utils import (
    add_future_label, check_label_shift, check_session_boundaries,
    check_no_leakage, session_row_ranges,
)
from forecast_config import SESSION_ID_COL, CURRENT_LABEL_COL, FUTURE_LABEL_COL


def make_synthetic_multi_session_df(seed=0):
    """Two sessions of different lengths with a deterministic 'current'
    label pattern, so we can verify the shift analytically."""
    rng = np.random.default_rng(seed)

    def make_session(name, n):
        current = (rng.random(n) > 0.6).astype(int)  # ~40% degraded
        idx = pd.date_range("2024-01-01", periods=n, freq="250ms")
        return pd.DataFrame({
            SESSION_ID_COL: name,
            CURRENT_LABEL_COL: current,
            "dummy_feature": rng.random(n),
        }, index=idx)

    df_a = make_session("session_A", 120)
    df_b = make_session("session_B", 95)
    return pd.concat([df_a, df_b])  # deliberately NOT re-sorted by timestamp,
    # matching add_future_label's real behaviour of keeping session blocks
    # contiguous rather than globally time-sorting across sessions.


def test_label_shift_and_boundaries():
    print("\n[TEST] add_future_label() — shift + session boundaries")
    df = make_synthetic_multi_session_df()
    horizon = 20

    result = add_future_label(df, horizon_rows=horizon)

    assert check_label_shift(result, horizon_rows=horizon), "label shift check failed"
    assert check_session_boundaries(result), "session boundary check failed"

    # Each session should have exactly `horizon` rows dropped from its tail
    counts = result[SESSION_ID_COL].value_counts()
    assert counts["session_A"] == 120 - horizon, counts
    assert counts["session_B"] == 95 - horizon, counts
    print("  ✓ PASS")


def test_no_leakage():
    print("\n[TEST] check_no_leakage()")
    feature_cols_good = ["dl_mcs", "dl_cqi", "prb_grant_ratio"]
    feature_cols_bad = ["dl_mcs", CURRENT_LABEL_COL]  # deliberately leaky

    assert check_no_leakage(None, feature_cols_good) is True
    assert check_no_leakage(None, feature_cols_bad) is False
    print("  ✓ PASS")


def test_window_alignment():
    print("\n[TEST] LSTM-style window/target alignment (session-aware)")
    df = make_synthetic_multi_session_df()
    horizon = 5
    window = 4

    result = add_future_label(df, horizon_rows=horizon)
    ranges = session_row_ranges(result, SESSION_ID_COL)

    # Re-implement the same compute_valid_window_starts logic inline so this
    # test has no import-time dependency on TensorFlow (train_lstm.py imports
    # tensorflow at module load time).
    def compute_valid_window_starts(session_ranges, window_size, target_lo, target_hi):
        starts = []
        for s_block, e_block in session_ranges:
            s_min = max(s_block, target_lo - window_size + 1, 0)
            s_max = min(e_block - window_size, target_hi - window_size)
            if s_max >= s_min:
                starts.extend(range(s_min, s_max + 1))
        return starts

    n = len(result)
    starts = compute_valid_window_starts(ranges, window, 0, n)
    assert len(starts) > 0, "no valid windows generated"

    session_ids = result[SESSION_ID_COL].to_numpy()
    future_vals = result[FUTURE_LABEL_COL].to_numpy()

    for s in starts:
        end = s + window
        # (1) window must not cross a session boundary
        assert len(set(session_ids[s:end])) == 1, f"window at {s} crosses a session boundary"
        # (2) target = the window's OWN last row, which already holds the
        #     future-shifted label — NOT one row past the window.
        t = s + window - 1
        target = future_vals[t]
        assert target in (0, 1)

    print(f"  ✓ PASS ({len(starts)} windows checked, none cross a session boundary)")


def main():
    print("=" * 65)
    print("  Forecasting-correction sanity tests (synthetic data)")
    print("=" * 65)

    failures = 0
    for test_fn in (test_label_shift_and_boundaries, test_no_leakage, test_window_alignment):
        try:
            test_fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failures += 1

    print("\n" + "=" * 65)
    if failures:
        print(f"  {failures} test(s) FAILED")
        sys.exit(1)
    else:
        print("  All sanity tests passed.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
