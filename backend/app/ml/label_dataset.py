# =============================================================================
# label_dataset.py
#
# PURPOSE:
#   Take the cleaned KPI dataset, select the most useful features for
#   predicting network degradation, and create a target label column
#   called 'degradation_risk' using a statistical (quantile-based) method.
#
# HOW TO RUN (from the backend/ folder with venv active):
#   python app/ml/label_dataset.py
#
# INPUT:
#   backend/data/processed/clean_kpi.csv
#
# OUTPUT:
#   backend/data/processed/labeled_dataset.csv
#
# KEY CONCEPT — Why quantiles instead of hardcoded thresholds?
#   Hardcoded thresholds (e.g. "if error > 30%") are brittle.
#   If the dataset changes, those numbers may be wrong.
#   Quantiles are RELATIVE to this dataset — they say
#   "the bottom 25% of CQI readings" rather than "CQI < 5".
#   This makes the labelling logic self-adapting and data-driven.
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# Using pathlib.Path so paths work on both Windows and Linux.
# =============================================================================

# Input: cleaned dataset produced by clean_dataset.py
# Updated from clean_kpi.csv -> cleaned_dataset.csv to match the
# new multi-file preprocessing pipeline.
CLEAN_DATA_PATH   = Path("data") / "processed" / "cleaned_dataset.csv"

# Output: labeled dataset used by all three training scripts
LABELED_DATA_PATH = Path("data") / "processed" / "labeled_dataset.csv"

# A row is labelled as DEGRADED (1) if it scores badly on >= this many indicators.
# Set to 2 so that a single bad reading does not trigger a false alarm.
# Requires two or more KPIs to be in a bad state simultaneously.
DEGRADATION_SCORE_THRESHOLD = 2


# =============================================================================
# HELPER — Section printer
# =============================================================================

def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
# STEP 1 — Load the cleaned dataset
# =============================================================================

def load_clean_data(path):
    """
    Load the cleaned CSV produced by clean_dataset.py.
    The timestamp column is the row index, so we tell pandas that here.
    """
    print_section("STEP 1 — Loading cleaned dataset")

    if not Path(path).exists():
        raise FileNotFoundError(
            f"\nERROR: Cannot find {path}\n"
            "Run clean_dataset.py first to generate the cleaned dataset."
        )

    # parse_dates=True converts the index back into datetime objects
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)

    print(f"  Loaded: {path}")
    print(f"  Rows:   {df.shape[0]}")
    print(f"  Cols:   {df.shape[1]}")
    print(f"  Time range: {df.index[0]}  →  {df.index[-1]}")
    return df


# =============================================================================
# STEP 2 — Select and explain the KPI features
# =============================================================================

# Each entry is: column_name → explanation of why it signals degradation.
# This dictionary is the single source of truth for which features we use.

SELECTED_FEATURES = {
    # --- UPLINK ERROR RATE ---
    "rx_errors_uplink_pct": (
        "Uplink packet error rate (%). "
        "HIGH VALUE = BAD. When the base station receives many corrupted packets "
        "from the UE, it must request retransmissions. This wastes bandwidth, "
        "increases latency, and is the strongest single indicator of radio link degradation."
    ),

    # --- CHANNEL QUALITY INDICATOR ---
    "dl_cqi": (
        "Downlink Channel Quality Indicator (0–15 scale, reported by the UE). "
        "LOW VALUE = BAD. The UE measures the received signal quality and reports "
        "this value to the base station. A low CQI forces the scheduler to use "
        "a more conservative (slower) modulation scheme to maintain reliability."
    ),

    # --- DOWNLINK THROUGHPUT ---
    "tx_brate_downlink_mbps": (
        "Actual downlink throughput in Megabits per second. "
        "LOW VALUE = BAD. This is the primary user-facing performance metric. "
        "Sustained low throughput directly means the user experience is poor, "
        "regardless of the cause (bad channel, congestion, or interference)."
    ),

    # --- PRB GRANT RATIO (derived feature) ---
    "prb_grant_ratio": (
        "Ratio of granted to requested Physical Resource Blocks "
        "(sum_granted_prbs / (sum_requested_prbs + 1)). "
        "LOW VALUE = BAD. A ratio well below 1.0 means the scheduler is denying "
        "the UE's bandwidth requests — a sign of congestion or deliberate limiting. "
        "This feature was engineered in Phase 1 and captures congestion better "
        "than either PRB column alone."
    ),

    # --- DOWNLINK MCS ---
    "dl_mcs": (
        "Downlink Modulation and Coding Scheme index (0–28). "
        "LOW VALUE = BAD. The base station chooses a higher MCS when the channel "
        "is good (faster, denser modulation like 64-QAM) and a lower MCS when the "
        "channel is poor (simpler, more robust modulation like QPSK). "
        "Consistently low MCS means the link is struggling."
    ),

    # --- UPLINK SINR ---
    "ul_sinr": (
        "Uplink Signal-to-Interference-plus-Noise Ratio (dB). "
        "LOW VALUE = BAD (when the UE is transmitting). "
        "SINR measures how much stronger the signal is compared to interference and noise. "
        "Note: ~73% of values are 0.0 because the UE was idle (not transmitting uplink). "
        "Zero means idle here, not degraded. The scoring logic handles this correctly "
        "by only flagging SINR as bad when the UE IS transmitting and SINR is still low."
    ),

    # --- TURBO DECODER ITERATIONS ---
    "ul_turbo_iters": (
        "Average turbo decoder iterations needed to decode each uplink packet. "
        "HIGH VALUE = BAD. The base station's decoder tries repeatedly to reconstruct "
        "a damaged packet. More iterations = weaker uplink signal quality. "
        "Like ul_sinr, this is zero when the UE is idle, which is normal."
    ),
}


def explain_features(df):
    """
    Print a formatted explanation of every selected feature.
    This output is useful for your dissertation and viva.
    """
    print_section("STEP 2 — Selected KPI features and why each matters")

    print(f"\n  {len(SELECTED_FEATURES)} features selected for labelling:\n")

    for i, (col, explanation) in enumerate(SELECTED_FEATURES.items(), start=1):
        # Show the feature name and its basic statistics
        col_data = df[col]
        print(f"  [{i}] {col}")
        print(f"      Min={col_data.min():.4f}  Mean={col_data.mean():.4f}  "
              f"Max={col_data.max():.4f}  Std={col_data.std():.4f}")

        # Wrap the explanation text at 70 chars for readability
        words = explanation.split()
        line = "      "
        for word in words:
            if len(line) + len(word) + 1 > 74:
                print(line)
                line = "      " + word + " "
            else:
                line += word + " "
        print(line)
        print()


# =============================================================================
# STEP 3 — Compute statistical thresholds using quantiles
# =============================================================================

def compute_thresholds(df):
    """
    Calculate data-driven thresholds from the dataset's own distribution.

    WHY QUANTILES?
    A quantile splits the data at a percentage point.
    - Q90 of rx_errors means: "the value that 90% of rows are BELOW"
      → Values above Q90 are in the worst 10% for error rate
    - Q25 of dl_cqi means: "the value that 25% of rows are BELOW"
      → Values below Q25 are in the worst 25% for channel quality

    This approach is self-calibrating: if you apply this script to a
    different dataset (more users, different radio conditions), the
    thresholds will automatically adjust to that data's distribution.
    """
    print_section("STEP 3 — Computing quantile-based thresholds")

    thresholds = {}

    # ----------------------------------------------------------------
    # rx_errors_uplink_pct — flag if in the TOP 10% (worst errors)
    # Using Q90 because Q75 = 0 (most rows have zero errors).
    # Q90 = 57.1% captures only genuinely high error events.
    # ----------------------------------------------------------------
    thresholds["rx_errors_uplink_pct_high"] = df["rx_errors_uplink_pct"].quantile(0.90)

    # ----------------------------------------------------------------
    # dl_cqi — flag if in the BOTTOM 25% (worst channel quality)
    # Q25 of CQI means the channel quality is below average.
    # ----------------------------------------------------------------
    thresholds["dl_cqi_low"] = df["dl_cqi"].quantile(0.25)

    # ----------------------------------------------------------------
    # tx_brate_downlink_mbps — flag if in the BOTTOM 10% (lowest throughput)
    # Using Q10 because near-zero throughput is common during idle periods.
    # Q10 filters out only the most sustained low-throughput readings.
    # ----------------------------------------------------------------
    thresholds["tx_brate_downlink_mbps_low"] = df["tx_brate_downlink_mbps"].quantile(0.10)

    # ----------------------------------------------------------------
    # prb_grant_ratio — flag if in the BOTTOM 25% (worst congestion)
    # A low grant ratio means the scheduler is heavily denying requests.
    # ----------------------------------------------------------------
    thresholds["prb_grant_ratio_low"] = df["prb_grant_ratio"].quantile(0.25)

    # ----------------------------------------------------------------
    # dl_mcs — flag if in the BOTTOM 25% (worst modulation quality)
    # Low MCS means the base station is forced to use simple/slow encoding.
    # ----------------------------------------------------------------
    thresholds["dl_mcs_low"] = df["dl_mcs"].quantile(0.25)

    # ----------------------------------------------------------------
    # ul_sinr — only flag when UE is actively transmitting (sinr > 0)
    # AND the signal quality is in the bottom 25% of active readings.
    # We compute Q25 only over non-zero rows to avoid the idle-zero effect.
    # ----------------------------------------------------------------
    sinr_active = df[df["ul_sinr"] > 0]["ul_sinr"]
    thresholds["ul_sinr_low"] = sinr_active.quantile(0.25) if len(sinr_active) > 0 else 0

    # ----------------------------------------------------------------
    # ul_turbo_iters — flag if in TOP 75% of active (non-zero) iterations
    # High turbo iterations = poor uplink signal quality.
    # Again, we exclude zeros (idle rows) from the calculation.
    # ----------------------------------------------------------------
    turbo_active = df[df["ul_turbo_iters"] > 0]["ul_turbo_iters"]
    thresholds["ul_turbo_iters_high"] = turbo_active.quantile(0.75) if len(turbo_active) > 0 else 0

    # --- Print the computed thresholds ---
    print("\n  Computed thresholds (all derived from data, no hardcoding):\n")
    print(f"  {'Threshold':<40} {'Value':>10}  {'Meaning'}")
    print(f"  {'-'*40} {'-'*10}  {'-'*30}")
    descriptions = {
        "rx_errors_uplink_pct_high": ("Q90", "flag if error rate above this"),
        "dl_cqi_low":                ("Q25", "flag if CQI below this"),
        "tx_brate_downlink_mbps_low":("Q10", "flag if throughput below this"),
        "prb_grant_ratio_low":       ("Q25", "flag if grant ratio below this"),
        "dl_mcs_low":                ("Q25", "flag if DL MCS below this"),
        "ul_sinr_low":               ("Q25*","flag if SINR below this (active only)"),
        "ul_turbo_iters_high":       ("Q75*","flag if turbo iters above this (active only)"),
    }
    for key, val in thresholds.items():
        q_label, meaning = descriptions[key]
        print(f"  {key:<40} {val:>10.4f}  ({q_label}) {meaning}")

    print("\n  * Computed over non-zero rows only (excludes UE idle periods)")

    return thresholds


# =============================================================================
# STEP 4 — Score each row and create the degradation_risk label
# =============================================================================

def score_row(row, thresholds):
    """
    Evaluate a single row against all thresholds and return a degradation score.

    Each bad condition contributes 1 point to the score.
    The final label is: score >= DEGRADATION_SCORE_THRESHOLD → DEGRADED (1)

    WHY A SCORE INSTEAD OF ANY SINGLE CONDITION?
    A single bad KPI can happen for innocent reasons:
      - Low throughput might mean the UE has nothing to send (idle)
      - Low CQI for one sample might be a measurement blip
    Requiring 2+ bad conditions simultaneously is much more reliable.
    It mirrors how network engineers actually diagnose problems: they look
    for a pattern of correlated bad signals, not one outlier.

    Returns:
        score (int): number of degradation indicators fired (0–7)
    """
    score = 0

    # Condition 1 — High uplink error rate
    if row["rx_errors_uplink_pct"] > thresholds["rx_errors_uplink_pct_high"]:
        score += 1

    # Condition 2 — Low channel quality
    if row["dl_cqi"] < thresholds["dl_cqi_low"]:
        score += 1

    # Condition 3 — Low downlink throughput
    if row["tx_brate_downlink_mbps"] < thresholds["tx_brate_downlink_mbps_low"]:
        score += 1

    # Condition 4 — Low PRB grant ratio (congestion)
    if row["prb_grant_ratio"] < thresholds["prb_grant_ratio_low"]:
        score += 1

    # Condition 5 — Low downlink MCS (poor modulation quality)
    if row["dl_mcs"] < thresholds["dl_mcs_low"]:
        score += 1

    # Condition 6 — Low uplink SINR while actively transmitting
    # Only score this if ul_sinr > 0 (i.e. UE is not idle)
    if row["ul_sinr"] > 0 and row["ul_sinr"] < thresholds["ul_sinr_low"]:
        score += 1

    # Condition 7 — High turbo decoder iterations while actively transmitting
    # Only score this if ul_turbo_iters > 0 (i.e. UE is not idle)
    if row["ul_turbo_iters"] > 0 and row["ul_turbo_iters"] > thresholds["ul_turbo_iters_high"]:
        score += 1

    return score


def create_labels(df, thresholds):
    """
    Apply score_row() to every row in the DataFrame to create:
      - 'degradation_score': how many bad conditions fired (0–7)
      - 'degradation_risk':  final binary label (0 = Normal, 1 = Degraded)
    """
    print_section("STEP 4 — Scoring rows and creating degradation_risk label")

    print(f"\n  Scoring each of {len(df)} rows against 7 KPI conditions...")
    print(f"  A row is labelled DEGRADED if score >= {DEGRADATION_SCORE_THRESHOLD}\n")

    # Apply score_row to every row — this is the core labelling step
    df["degradation_score"] = df.apply(
        lambda row: score_row(row, thresholds), axis=1
    )

    # Binary label: 1 if score meets threshold, 0 otherwise
    df["degradation_risk"] = (
        df["degradation_score"] >= DEGRADATION_SCORE_THRESHOLD
    ).astype(int)

    return df


# =============================================================================
# STEP 5 — Print the risk distribution
# =============================================================================

def print_risk_distribution(df):
    """
    Print how many rows were labelled Normal vs Degraded,
    and how the score breaks down across all rows.
    """
    print_section("STEP 5 — Risk distribution")

    total = len(df)
    n_degraded = df["degradation_risk"].sum()
    n_normal = total - n_degraded
    pct_degraded = (n_degraded / total) * 100
    pct_normal = (n_normal / total) * 100

    print(f"\n  {'Label':<12} {'Count':>8} {'Percentage':>12}  Bar")
    print(f"  {'-'*12} {'-'*8} {'-'*12}  ---")

    # Simple ASCII bar chart
    bar_normal   = "█" * int(pct_normal / 2)
    bar_degraded = "█" * int(pct_degraded / 2)
    print(f"  {'Normal (0)':<12} {n_normal:>8} {pct_normal:>10.1f}%  {bar_normal}")
    print(f"  {'Degraded (1)':<12} {n_degraded:>8} {pct_degraded:>10.1f}%  {bar_degraded}")
    print(f"  {'TOTAL':<12} {total:>8}")

    # --- Score breakdown ---
    print(f"\n  Score breakdown (how many KPI conditions fired per row):")
    print(f"  {'Score':<8} {'Rows':>8} {'%':>8}  Meaning")
    print(f"  {'-'*8} {'-'*8} {'-'*8}  -------")

    score_meanings = {
        0: "All KPIs healthy",
        1: "One KPI borderline — still Normal",
        2: f"Two KPIs bad — DEGRADED (threshold = {DEGRADATION_SCORE_THRESHOLD})",
        3: "Three KPIs bad — DEGRADED",
        4: "Four KPIs bad — DEGRADED",
        5: "Five KPIs bad — DEGRADED",
        6: "Six KPIs bad — severe degradation",
        7: "All seven KPIs bad — critical",
    }
    score_counts = df["degradation_score"].value_counts().sort_index()
    for score, count in score_counts.items():
        pct = (count / total) * 100
        label_str = f"[DEGRADED]" if score >= DEGRADATION_SCORE_THRESHOLD else "[Normal]  "
        meaning = score_meanings.get(int(score), "")
        print(f"  {int(score):<8} {count:>8} {pct:>7.1f}%  {label_str} {meaning}")

    # --- Class balance note ---
    print(f"\n  Class balance note:")
    ratio = n_normal / n_degraded if n_degraded > 0 else float("inf")
    print(f"  Normal : Degraded ≈ {ratio:.1f} : 1")

    if ratio > 5:
        print("  ⚠  Imbalanced dataset. Consider using class_weight='balanced'")
        print("     in Random Forest and XGBoost to compensate.")
    elif ratio < 1.5:
        print("  ⚠  Very balanced — check if threshold is too aggressive.")
    else:
        print("  ✓  Reasonable balance for binary classification.")


# =============================================================================
# STEP 6 — Print the final feature list
# =============================================================================

def print_feature_list(df):
    """
    Print the final list of feature columns and the label column.
    This is what will be fed into the ML models in Phase 2.
    """
    print_section("STEP 6 — Final feature list for ML training")

    # Feature columns = everything except the two new label columns
    feature_cols = [
        col for col in df.columns
        if col not in ("degradation_risk", "degradation_score")
    ]

    print(f"\n  Feature columns ({len(feature_cols)} total):")
    print(f"  {'#':<4} {'Column':<35} {'Min':>10} {'Mean':>10} {'Max':>10}")
    print(f"  {'-'*4} {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    for i, col in enumerate(feature_cols, start=1):
        col_min  = df[col].min()
        col_mean = df[col].mean()
        col_max  = df[col].max()
        print(f"  {i:<4} {col:<35} {col_min:>10.4f} {col_mean:>10.4f} {col_max:>10.4f}")

    print(f"\n  Label column:")
    print(f"       degradation_risk  →  0 = Normal, 1 = Degraded")
    print(f"       degradation_score →  0–7 (how many conditions fired, kept for reference)")


# =============================================================================
# STEP 7 — Save the labeled dataset
# =============================================================================

def save_labeled_data(df, path):
    """
    Save the labeled DataFrame to CSV.
    All original features + degradation_score + degradation_risk are included.
    The Phase 2 training script will load this file.
    """
    print_section("STEP 7 — Saving labeled dataset")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

    print(f"  ✓ Saved to: {path}")
    print(f"  Rows:    {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}  (18 features + degradation_score + degradation_risk)")
    print(f"  Size:    {Path(path).stat().st_size / 1024:.1f} KB")


# =============================================================================
# MAIN — Run all steps in order
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  Feature Engineering & Labelling — O-RAN KPI xApp")
    print("  FYP — ML-based KPI Prediction xApp")
    print("=" * 65)

    # Step 1 — Load the cleaned dataset
    df = load_clean_data(CLEAN_DATA_PATH)

    # Step 2 — Print feature explanations
    explain_features(df)

    # Step 3 — Compute data-driven quantile thresholds
    thresholds = compute_thresholds(df)

    # Step 4 — Score every row and create the label
    df = create_labels(df, thresholds)

    # Step 5 — Print risk distribution
    print_risk_distribution(df)

    # Step 6 — Print the final feature list
    print_feature_list(df)

    # Step 7 — Save the labeled dataset
    save_labeled_data(df, LABELED_DATA_PATH)

    print("\n" + "=" * 65)
    print("  ✓ Feature engineering complete!")
    print("  Next step: Phase 2 — Train Random Forest and XGBoost classifiers.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
