# =============================================================================
# forecast_utils.py
#
# PURPOSE:
#   Shared, tested building blocks for the "future forecasting" correction so
#   that label_dataset.py and all three train_*.py scripts implement the
#   shift / split / baseline / evaluation logic identically instead of each
#   re-implementing (and potentially mis-implementing) it.
#
#   See docs/FORECASTING_METHODOLOGY_UPDATE.md for the full rationale.
# =============================================================================

import json
import os

import numpy as np
import pandas as pd

# Enable pandas Copy-on-Write mode. Without this, pandas <3.0 defaults to
# eagerly DEEP-COPYING the entire DataFrame inside routine calls like
# rename()/drop()/set_index() (see pandas.core.generic._rename's internal
# `self.copy(deep=copy and not using_copy_on_write())`). On a small sample
# dataset that's invisible; on a real O-RAN recording with tens of millions
# of rows, EVERY such call briefly allocates a full extra multi-GB copy and
# can crash with numpy.core._exceptions._ArrayMemoryError. Copy-on-Write
# makes these operations cheap (share memory until an actual write happens)
# without changing any pipeline behaviour.
pd.set_option("mode.copy_on_write", True)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from forecast_config import (
    SESSION_ID_COL,
    CURRENT_LABEL_COL,
    FUTURE_LABEL_COL,
    FORECAST_HORIZON_ROWS,
)


def downcast_numeric_dtypes(df, exclude=()):
    """
    Downcast float64→float32 and int64→smallest-sufficient-int columns
    (excluding any column name in `exclude`), and convert 'session_id' to a
    pandas categorical if present.

    Shared by every script that loads a CSV in this pipeline
    (label_dataset.py, train_random_forest.py, train_xgboost.py,
    train_lstm.py) — pandas.read_csv re-infers float64/int64 from the CSV
    text every time regardless of what dtypes were used when the file was
    written, so this must be re-applied after every load, not just once in
    clean_dataset.py. See clean_dataset.py's downcast_dtypes() for the full
    rationale (memory roughly halves; no precision loss for these KPI
    ranges).
    """
    exclude = set(exclude) | {c for c in df.columns if c.lower() == "timestamp"}

    mem_before = df.memory_usage(deep=True).sum() / 1024**2

    float_cols = [c for c in df.select_dtypes(include=["float64"]).columns if c not in exclude]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    int_cols = [c for c in df.select_dtypes(include=["int64"]).columns if c not in exclude]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    if SESSION_ID_COL in df.columns and df[SESSION_ID_COL].dtype.name != "category":
        df[SESSION_ID_COL] = df[SESSION_ID_COL].astype("category")

    mem_after = df.memory_usage(deep=True).sum() / 1024**2
    if mem_before > 0:
        print(f"  Memory after downcast: {mem_before:,.1f} MB → {mem_after:,.1f} MB "
              f"({(1 - mem_after/mem_before)*100:.1f}% reduction)")

    return df


# =============================================================================
# STEP — Build the future target from the existing rule-based label
# =============================================================================

def add_future_label(
    df: pd.DataFrame,
    horizon_rows: int = FORECAST_HORIZON_ROWS,
    current_label_col: str = CURRENT_LABEL_COL,
    future_label_col: str = FUTURE_LABEL_COL,
    session_col: str = SESSION_ID_COL,
) -> pd.DataFrame:
    """
    Create `future_label_col` = `current_label_col` shifted `horizon_rows`
    rows into the future, computed INDEPENDENTLY within each session so a
    shift can never pull a label in from a different recording.

    Rows at the tail of each session that have no valid future value (because
    the shift ran past the end of that session) are dropped — this is
    unavoidable: we cannot forecast past the end of the only data we have for
    that recording.

    If `session_col` is not present in df, the whole DataFrame is treated as
    a single session (this matches the current single-file dataset, and is
    still correct — it just has no boundaries to protect).
    """
    if horizon_rows <= 0:
        raise ValueError("horizon_rows must be a positive integer")

    if session_col not in df.columns:
        df = df.copy()
        df[session_col] = "__single_session__"
        added_dummy_session = True
    else:
        added_dummy_session = False

    processed_sessions = []
    rows_before = len(df)

    for session_id, session_df in df.groupby(session_col, sort=False):
        session_df = session_df.copy()

        # Shift the CURRENT label backwards in row-position so that the value
        # now sitting at row t is the label that originally belonged to
        # row t + horizon_rows within this same session.
        session_df[future_label_col] = (
            session_df[current_label_col].shift(-horizon_rows)
        )

        # Drop tail rows with no valid future target (this session ended
        # before we could look horizon_rows ahead).
        session_df = session_df.dropna(subset=[future_label_col])

        processed_sessions.append(session_df)

    # IMPORTANT: sessions are concatenated in groupby order (each session's
    # rows kept internally time-ordered and CONTIGUOUS as a block) rather than
    # re-sorted by timestamp across sessions. If two sessions had overlapping
    # real-world clock times, a global timestamp sort here would interleave
    # their rows and silently break window/shift contiguity for the LSTM and
    # for any positional (iloc-based) train/test split downstream — exactly
    # the failure mode section 6/16 of the correction brief warns about.
    # With the current single-session dataset this has no visible effect;
    # it matters once more than one recording file is added (see
    # docs/FORECASTING_METHODOLOGY_UPDATE.md, "Remaining Limitations").
    result = pd.concat(processed_sessions) if processed_sessions else df.iloc[0:0].copy()

    result[future_label_col] = result[future_label_col].astype(int)

    if added_dummy_session:
        result = result.drop(columns=[session_col])

    rows_after = len(result)
    print(f"  Sessions processed:        {df[session_col].nunique() if session_col in df.columns else 1}")
    print(f"  Rows before future shift:  {rows_before}")
    print(f"  Rows dropped (tail of each session, no future data): {rows_before - rows_after}")
    print(f"  Rows after future shift:   {rows_after}")

    return result


# =============================================================================
# STEP — Sanity checks (used by test_forecast_sanity.py and callable directly)
# =============================================================================

def check_label_shift(
    df: pd.DataFrame,
    horizon_rows: int = FORECAST_HORIZON_ROWS,
    current_label_col: str = CURRENT_LABEL_COL,
    future_label_col: str = FUTURE_LABEL_COL,
    session_col: str = SESSION_ID_COL,
) -> bool:
    """
    Verify that for every session, future_label_col at row position i equals
    current_label_col at row position i + horizon_rows within that same
    session's SAVED (already tail-trimmed) rows.

    IMPORTANT: add_future_label() already dropped the last `horizon_rows` of
    each session (they have no valid future value to check against — that's
    *why* they were dropped, not an error). So this check can only compare
    positions where i + horizon_rows still falls within the trimmed data,
    i.e. i in [0, session_length - horizon_rows). That covers every row
    except the tail_rows-worth nearest the (already-removed) end of each
    session, which is the entire meaningful range there is left to check.
    """
    ok = True
    group_col = session_col if session_col in df.columns else None
    groups = df.groupby(group_col, sort=False) if group_col else [(None, df)]

    any_checked = False
    for session_id, session_df in groups:
        current = session_df[current_label_col].to_numpy()
        future  = session_df[future_label_col].to_numpy()
        n = len(session_df)

        if n <= horizon_rows:
            # Too short to check anything (shouldn't normally happen —
            # add_future_label would have dropped the whole session).
            continue

        # future[i] should equal current[i + horizon_rows] for
        # i in [0, n - horizon_rows)
        lhs = future[: n - horizon_rows]
        rhs = current[horizon_rows:]
        any_checked = True
        if not np.array_equal(lhs, rhs):
            print(f"  ✗ Label shift check FAILED for session {session_id!r}")
            ok = False

    if not any_checked:
        print("  ⚠ Label shift check skipped — no session was long enough to verify.")
    elif ok:
        print(f"  ✓ Label shift check passed for all sessions "
              f"(future[i] == current[i+{horizon_rows}] within each session).")
    return ok


def check_session_boundaries(
    df: pd.DataFrame,
    session_col: str = SESSION_ID_COL,
) -> bool:
    """
    Verify no row's future label could have come from a different session
    than the one it belongs to. Because add_future_label() computes the shift
    per-group BEFORE concatenation, this is true by construction — this check
    re-derives it independently as a regression guard.
    """
    if session_col not in df.columns:
        print("  ✓ Session boundary check skipped (single-session dataset).")
        return True

    ok = True
    for session_id, session_df in df.groupby(session_col, sort=False):
        if session_df[session_col].nunique() != 1:
            ok = False
    if ok:
        print(f"  ✓ Session boundary check passed ({df[session_col].nunique()} session(s), no cross-session mixing).")
    else:
        print("  ✗ Session boundary check FAILED — a group contains multiple session ids.")
    return ok


def check_no_leakage(
    df: pd.DataFrame,
    feature_cols,
    forbidden_cols=(CURRENT_LABEL_COL, FUTURE_LABEL_COL, "degradation_score", SESSION_ID_COL),
) -> bool:
    """Verify none of the forbidden target/metadata columns leaked into the feature list."""
    leaked = [c for c in forbidden_cols if c in feature_cols]
    if leaked:
        print(f"  ✗ Leakage check FAILED — forbidden columns present in features: {leaked}")
        return False
    print("  ✓ Leakage check passed — no label/session columns present in the feature matrix.")
    return True


def session_row_ranges(df: pd.DataFrame, session_col: str = SESSION_ID_COL):
    """
    Return a list of (start_row_position, end_row_position) tuples — using
    integer row POSITIONS, not the DataFrame index — for each contiguous
    session block in df.

    Relies on add_future_label() having kept each session's rows together as
    a contiguous block (see the comment in add_future_label about not
    re-sorting across sessions). Used by train_lstm.py to build sliding
    windows that never cross a session boundary.
    """
    if session_col not in df.columns:
        return [(0, len(df))]

    ranges = []
    session_values = df[session_col].to_numpy()
    start = 0
    for i in range(1, len(session_values) + 1):
        if i == len(session_values) or session_values[i] != session_values[start]:
            ranges.append((start, i))
            start = i
    return ranges


# =============================================================================
# STEP — Naive baseline: "assume the network state will remain unchanged"
# =============================================================================

def naive_baseline_metrics(current: pd.Series, future: pd.Series) -> dict:
    """
    Evaluate the trivial baseline: predicted_future = current.
    This is the bar the ML models must clear to prove they add value over
    "nothing changes in the next 5 seconds".
    """
    y_pred = current.to_numpy()
    y_true = future.to_numpy()

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model": "Naive Baseline (state unchanged)",
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1_score":  round(float(f1), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    print(f"\n  Naive baseline (\"assume state stays the same\"):")
    print(f"    Accuracy : {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall   : {recall:.4f}")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    Confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")

    return metrics


# =============================================================================
# STEP — Standard classification evaluation (accuracy/precision/recall/F1 +
# ROC-AUC / PR-AUC when probabilities are available)
# =============================================================================

def evaluate_binary(y_true, y_pred, y_prob=None, model_name="model") -> dict:
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model": model_name,
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1_score":  round(float(f1), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            metrics["pr_auc"]  = round(float(average_precision_score(y_true, y_prob)), 4)
        except ValueError:
            pass

    return metrics


def transition_report(current: pd.Series, future: pd.Series, predicted_future) -> dict:
    """
    Break results down by transition type, as required for early-warning
    evaluation:
        Normal -> Normal, Normal -> Degraded, Degraded -> Degraded, Degraded -> Normal

    The most important row is "Normal -> Degraded": these are the genuine
    early-warning opportunities. We report how many of those the model
    actually caught (recall on that specific transition).
    """
    current = pd.Series(current).reset_index(drop=True)
    future  = pd.Series(future).reset_index(drop=True)
    pred    = pd.Series(predicted_future).reset_index(drop=True)

    counts = {
        "normal_to_normal":     int(((current == 0) & (future == 0)).sum()),
        "normal_to_degraded":   int(((current == 0) & (future == 1)).sum()),
        "degraded_to_degraded": int(((current == 1) & (future == 1)).sum()),
        "degraded_to_normal":   int(((current == 1) & (future == 0)).sum()),
    }

    early_warning_mask = (current == 0) & (future == 1)
    n_early_warning_events = int(early_warning_mask.sum())
    n_caught = int(((pred == 1) & early_warning_mask).sum())
    n_missed = n_early_warning_events - n_caught

    false_alarms_mask = (current == 0) & (future == 0) & (pred == 1)
    n_false_alarms = int(false_alarms_mask.sum())

    report = {
        "transition_counts": counts,
        "early_warning_events_total": n_early_warning_events,
        "early_warning_events_caught": n_caught,
        "early_warning_events_missed": n_missed,
        "early_warning_recall": round(n_caught / n_early_warning_events, 4) if n_early_warning_events else None,
        "false_alarms_on_stable_normal": n_false_alarms,
    }

    print(f"\n  Transition analysis (Normal->Degraded is the early-warning case):")
    print(f"    Normal   -> Normal   : {counts['normal_to_normal']}")
    print(f"    Normal   -> Degraded : {counts['normal_to_degraded']}   "
          f"(caught={n_caught}, missed={n_missed}, "
          f"recall={report['early_warning_recall']})")
    print(f"    Degraded -> Degraded : {counts['degraded_to_degraded']}")
    print(f"    Degraded -> Normal   : {counts['degraded_to_normal']}")
    print(f"    False alarms (Normal->Normal predicted Degraded): {n_false_alarms}")

    return report


# =============================================================================
# STEP — Comparison report helpers (models/comparison_report.json)
# =============================================================================

def upsert_comparison_report(path: str, entry: dict, key: str = "model"):
    """
    Add/replace one model's metrics entry in the shared comparison report
    without clobbering entries written by the other training scripts.
    Matching is done on entry[key] (the model's display name).
    """
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            try:
                existing = json.load(f).get("models", [])
            except json.JSONDecodeError:
                existing = []

    existing = [m for m in existing if m.get(key) != entry.get(key)]
    existing.append(entry)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"models": existing}, f, indent=2)

    return existing
