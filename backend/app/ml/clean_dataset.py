# =============================================================================
# clean_dataset.py
#
# PURPOSE:
#   Read ALL CSV files from backend/data/raw/, validate each one,
#   merge them into a single DataFrame, clean the combined data,
#   and save two output files:
#
#     combined_dataset.csv  — raw merged data (before cleaning)
#     cleaned_dataset.csv   — fully cleaned and ready for labelling
#
# HOW TO RUN (from the backend/ folder with venv active):
#   python app/ml/clean_dataset.py
#
# WHY MULTIPLE FILES?
#   Each CSV represents one recording session from the O-RAN testbed.
#   By merging them we get more training data, which generally leads to
#   better and more generalisable ML models.
#
# PIPELINE ORDER:
#   clean_dataset.py  →  label_dataset.py  →  train_*.py
# =============================================================================

from pathlib import Path
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
import numpy as np

# =============================================================================
# CONFIGURATION — change these paths if your folder structure is different
# Using pathlib.Path so this works on both Windows and Linux/Mac.
# On Windows, Path automatically uses backslashes; on Linux it uses forward slashes.
# =============================================================================

# Folder that contains all raw CSV files
RAW_DATA_DIR = Path("data") / "raw"

# Output folder for processed files
PROCESSED_DIR = Path("data") / "processed"

# Output 1: raw merged data (all files combined, before cleaning)
COMBINED_DATA_PATH = PROCESSED_DIR / "combined_dataset.csv"

# Output 2: cleaned and ready for labelling
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_dataset.csv"

# =============================================================================
# COLUMNS TO ALWAYS DROP — identity and experiment-config columns
#
# WHY A HARDCODED LIST?
# When merging many recording sessions, some columns like 'slice_id' or
# 'num_ues' may now have more than one unique value across different experiments.
# The zero-variance check no longer removes them automatically in that case.
#
# These columns should NEVER be given to the ML model because:
#   - They identify the subscriber or session, not the radio link quality
#   - They are set by the experimenter, not determined by network behaviour
#   - Including them causes the model to learn experiment metadata
#     instead of genuine radio performance patterns
# =============================================================================

COLUMNS_TO_ALWAYS_DROP = [
    # UE and subscriber identity
    "IMSI",               # subscriber ID — identifies one device, not performance
    "RNTI",               # temporary radio ID assigned to the UE

    # Network slice / experiment configuration
    "num_ues",            # number of connected devices (experiment setting)
    "slicing_enabled",    # whether network slicing was active
    "slice_id",           # which slice the UE belongs to
    "slice_prb",          # PRBs allocated to the slice
    "power_multiplier",   # TX power multiplier
    "scheduling_policy",  # which scheduler algorithm was active

    # Radio columns always zero in srsRAN exports
    "ul_rssi",            # uplink RSSI — not populated by srsRAN
    "dl_pmi",             # precoding matrix indicator — not used
    "dl_ri",              # rank indicator — not used
    "ul_n",               # uplink noise — not populated
    "tx_errors downlink (%)",    # always 0 in these recordings (raw name)
    "tx_errors_downlink_pct",    # always 0 in these recordings (renamed)
]


# =============================================================================
# HELPER — Section printer
# =============================================================================

def print_section(title):
    """Print a visible divider so console output is easy to scan."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
# STEP 1 — Discover and validate all CSV files in the raw folder
# =============================================================================

def load_and_validate_csv_files(raw_dir: Path):
    """
    Scan raw_dir for CSV files, validate each one, and return a list of
    DataFrames ready to merge.

    Validation rules:
      - Skip files that are empty (0 bytes or 0 rows)
      - Skip files that cannot be parsed as valid CSV
      - Skip files whose column set does not match the reference file
        (the first valid file sets the expected column list)

    Returns:
        valid_frames  — list of DataFrames (one per valid file)
        summary       — dict with counts for the final report
    """
    print_section("STEP 1 — Discovering and validating CSV files")

    # Find every .csv file in the raw folder (case-insensitive extension)
    all_csv_files = sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.CSV"))
    # Remove duplicates that might appear from both globs on case-sensitive filesystems
    seen = set()
    csv_files = []
    for f in all_csv_files:
        if f not in seen:
            seen.add(f)
            csv_files.append(f)

    print(f"\n  Raw data folder: {raw_dir.resolve()}")
    print(f"  CSV files found: {len(csv_files)}")

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"\nERROR: No CSV files found in {raw_dir.resolve()}\n"
            "Please place your raw KPI CSV files in that folder and try again."
        )

    valid_frames = []           # DataFrames that passed all checks
    skipped_files = []          # Files that were skipped and why
    reference_columns = None    # Columns from the first valid file (others must match)
    total_rows_before = 0       # Sum of rows across all valid files

    for csv_path in csv_files:
        print(f"\n  Checking: {csv_path.name}")

        # ----------------------------------------------------------------
        # Check 1: File must not be empty (0 bytes)
        # ----------------------------------------------------------------
        if csv_path.stat().st_size == 0:
            reason = "empty file (0 bytes)"
            print(f"    ⚠ SKIP — {reason}")
            skipped_files.append((csv_path.name, reason))
            continue

        # ----------------------------------------------------------------
        # Check 2: File must be parseable as valid CSV
        # ----------------------------------------------------------------
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            reason = f"could not parse CSV: {e}"
            print(f"    ⚠ SKIP — {reason}")
            skipped_files.append((csv_path.name, reason))
            continue

        # ----------------------------------------------------------------
        # Check 3: File must have at least one data row (not just a header)
        # ----------------------------------------------------------------
        if df.shape[0] == 0:
            reason = "CSV has 0 data rows (header only)"
            print(f"    ⚠ SKIP — {reason}")
            skipped_files.append((csv_path.name, reason))
            continue

        # ----------------------------------------------------------------
        # Check 4: Columns must match the reference file
        # The first valid file sets the reference column list.
        # Any subsequent file with different columns is skipped.
        # ----------------------------------------------------------------
        file_cols = set(df.columns.tolist())

        if reference_columns is None:
            # This is the first valid file — it sets the standard
            reference_columns = file_cols
            print(f"    ✓ OK — {df.shape[0]} rows, {df.shape[1]} cols "
                  f"(sets reference columns)")
        else:
            # Check that this file's columns match the reference
            extra_cols  = file_cols - reference_columns
            missing_cols = reference_columns - file_cols

            if extra_cols or missing_cols:
                reason = (
                    f"column mismatch — "
                    f"extra: {extra_cols if extra_cols else 'none'}, "
                    f"missing: {missing_cols if missing_cols else 'none'}"
                )
                print(f"    ⚠ SKIP — {reason}")
                skipped_files.append((csv_path.name, reason))
                continue
            else:
                print(f"    ✓ OK — {df.shape[0]} rows, {df.shape[1]} cols")

        # File passed all checks — add a source column so we can trace rows
        # back to their original file if needed
        df["_source_file"] = csv_path.name
        total_rows_before += df.shape[0]
        valid_frames.append(df)

    # ----------------------------------------------------------------
    # Print validation summary
    # ----------------------------------------------------------------
    print(f"\n  ── Validation Summary ──")
    print(f"  Files found:     {len(csv_files)}")
    print(f"  Files loaded:    {len(valid_frames)}")
    print(f"  Files skipped:   {len(skipped_files)}")

    if skipped_files:
        print(f"\n  Skipped files:")
        for name, reason in skipped_files:
            print(f"    - {name}: {reason}")

    if len(valid_frames) == 0:
        raise ValueError(
            "\nERROR: No valid CSV files could be loaded.\n"
            "Check the warnings above and fix the files in data/raw/."
        )

    summary = {
        "files_found":     len(csv_files),
        "files_processed": len(valid_frames),
        "files_skipped":   len(skipped_files),
        "rows_before_merge": total_rows_before,
    }

    return valid_frames, summary


# =============================================================================
# STEP 2 — Merge all valid DataFrames into one
# =============================================================================

def merge_dataframes(valid_frames):
    """
    Stack all valid DataFrames on top of each other (row-wise concatenation).

    pd.concat with ignore_index=True creates a fresh 0..N integer index
    for the combined DataFrame. We will replace this with the timestamp
    later in the pipeline.

    'sort=False' preserves the original column order from the first file.
    """
    print_section("STEP 2 — Merging all valid files")

    combined = pd.concat(valid_frames, ignore_index=True, sort=False)

    print(f"  Files merged:  {len(valid_frames)}")
    print(f"  Combined rows: {combined.shape[0]}")
    print(f"  Columns:       {combined.shape[1]}")

    return combined


# =============================================================================
# STEP 3 — Save the raw combined dataset (before cleaning)
# =============================================================================

def save_combined(df, path: Path):
    """
    Save the merged-but-not-yet-cleaned dataset.

    This gives you a checkpoint: if cleaning goes wrong, you can restart
    from here without re-reading all the raw files.
    """
    print_section("STEP 3 — Saving raw combined dataset")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"  ✓ Saved: {path}")
    print(f"  Size:    {path.stat().st_size / 1024:.1f} KB")
    print(f"  Rows:    {df.shape[0]}  Columns: {df.shape[1]}")


# =============================================================================
# STEP 4 — Remove empty columns (all-NaN columns)
# =============================================================================

def remove_empty_columns(df):
    """
    Drop columns where EVERY value is NaN.

    The O-RAN testbed CSV has unnamed separator columns (Unnamed: 4, etc.)
    that are completely empty — artefacts of the CSV export format.
    They carry no information and must be dropped before any cleaning.
    """
    print_section("STEP 4 — Removing completely empty columns")

    empty_cols = [col for col in df.columns if df[col].isnull().all()]

    if empty_cols:
        print(f"  Found {len(empty_cols)} empty column(s): {empty_cols}")
        df = df.drop(columns=empty_cols)
    else:
        print("  No completely empty columns found.")

    print(f"  Columns remaining: {df.shape[1]}")
    return df


# =============================================================================
# STEP 5 — Rename the internal source-tracking column to session_id
# =============================================================================

def drop_source_column(df):
    """
    Rename the _source_file column (added during validation) to 'session_id'.

    METHODOLOGY NOTE (forecasting correction):
    This column used to be DROPPED here. It is now KEPT and renamed, because
    label_dataset.py needs it to shift the degradation label into the future
    independently within each recording session — shifting across the
    boundary between two different CSV files would silently mix unrelated
    recordings and produce an invalid forecasting target.

    'session_id' is excluded from the ML feature matrix in every training
    script (see forecast_config.NON_FEATURE_COLS) and from the API payload,
    so it does not change what the models are allowed to learn from — it is
    metadata, not a feature.
    """
    if "_source_file" in df.columns:
        df = df.rename(columns={"_source_file": "session_id"})
        print("\n  Renamed internal '_source_file' column to 'session_id' "
              "(kept — needed for session-safe future-label shifting).")
    return df



# =============================================================================
# STEP 5b — Drop known identity and config columns explicitly
# =============================================================================

def drop_known_columns(df):
    """
    Explicitly drop identity and experiment-configuration columns that should
    never be used as ML features.

    This is a safety step that runs BEFORE the zero-variance check.
    With multiple recording sessions, columns like 'slice_id' or 'num_ues'
    may now have more than one unique value, so the zero-variance check
    would no longer catch them. This explicit list ensures they are always
    removed regardless of how many files were merged.
    """
    # Only drop columns that actually exist in this DataFrame
    # (some may already have been removed or may not exist in all recordings)
    to_drop = [col for col in COLUMNS_TO_ALWAYS_DROP if col in df.columns]

    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"  Dropped {len(to_drop)} identity/config column(s):")
        for col in to_drop:
            print(f"    - {col}")
    else:
        print("  No known identity/config columns found to drop.")

    print(f"  Columns remaining: {df.shape[1]}")
    return df


# =============================================================================
# STEP 6 — Remove zero-variance columns
# =============================================================================

def remove_zero_variance_columns(df):
    """
    Drop columns that contain only one unique value across the entire dataset.

    WHY: A column that never changes gives the ML model no information.
    For example, if all rows have IMSI = 1010123456002, the model cannot
    learn anything from that column.

    IMPORTANT for multi-file datasets:
    With multiple recording sessions, some columns that were constant in
    ONE session may vary across sessions (e.g. slice_id, num_ues).
    We only drop a column if it has <= 1 unique value across ALL rows.
    This is more conservative and correct than the single-file version.
    """
    print_section("STEP 6 — Removing zero-variance columns")

    # 'session_id' is deliberately exempt: with only one raw CSV file present
    # it will have exactly one unique value (nunique()==1) and would
    # otherwise be dropped here — but it must survive to label_dataset.py
    # regardless of how many sessions exist, so future-label shifting always
    # has a session boundary to respect.
    zero_var_cols = [
        col for col in df.columns
        if col != "session_id" and df[col].nunique() <= 1
    ]

    if zero_var_cols:
        print(f"  Found {len(zero_var_cols)} zero-variance column(s):")
        for col in zero_var_cols:
            val = df[col].dropna().unique()
            print(f"    - {col:<35} always = {val}")
        df = df.drop(columns=zero_var_cols)
    else:
        print("  No zero-variance columns found.")
        print("  (With multiple sessions, previously constant columns may now vary — good!)")

    print(f"\n  Columns remaining: {df.shape[1]}")
    return df


# =============================================================================
# STEP 7 — Remove duplicate rows
# =============================================================================

def downcast_dtypes(df):
    """
    Downcast numeric columns to the smallest dtype that can represent them
    without loss, and convert 'session_id' to a pandas categorical.

    WHY THIS MATTERS AT SCALE:
    pandas.read_csv defaults numeric columns to float64/int64 (8 bytes per
    value). For a 24-million-row, ~30-column real O-RAN recording that is
    several GB just for the numeric data — before accounting for the extra
    headroom every later operation needs (drop_duplicates() hashing,
    groupby(), sort_index(), model training, ...). Downcasting to float32/
    the smallest sufficient int type roughly HALVES memory use for the rest
    of the pipeline, with no change in the values themselves (KPI values
    here — MCS 0-28, percentages 0-100, small PRB counts — are nowhere near
    float32's ~7-significant-digit precision limit or int32's range).

    'session_id' is a repeated string (one value per source file, e.g.
    tens of thousands of rows sharing the same filename). Stored as plain
    'object'/string dtype, that's very expensive at this row count; as a
    pandas 'category' dtype, pandas stores each unique string ONCE and
    every row as a small integer code — often a >90% memory reduction for
    this specific column.

    SAFETY: the still-raw Unix-millisecond timestamp column (named
    'Timestamp' or 'timestamp' at this point in the pipeline, still a large
    int64 — e.g. 1617070531726) is deliberately EXCLUDED from int
    downcasting: that value is far larger than int32's ~2.1 billion range
    and would silently overflow/corrupt if downcast. It is converted to
    datetime64 by convert_data_types() later in the pipeline instead, which
    is unaffected by this step (datetime64 columns are a different dtype
    and are never selected here).
    """
    print_section("STEP 6b — Downcasting numeric dtypes (reduces memory ~50%)")

    mem_before = df.memory_usage(deep=True).sum() / 1024**2
    timestamp_cols = {c for c in df.columns if c.lower() == "timestamp"}

    float_cols = [c for c in df.select_dtypes(include=["float64"]).columns if c not in timestamp_cols]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    int_cols = [c for c in df.select_dtypes(include=["int64"]).columns if c not in timestamp_cols]
    for col in int_cols:
        # Signed downcast is safe for all our KPI columns (some, like
        # sum_requested_prbs, are always >= 0, but signed 'integer' still
        # downcasts to the smallest sufficient signed type — simpler and
        # avoids any edge case with a stray negative value from a sensor glitch).
        df[col] = pd.to_numeric(df[col], downcast="integer")

    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype("category")

    mem_after = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n  Downcast {len(float_cols)} float64 → float32 column(s)")
    print(f"  Downcast {len(int_cols)} int64 → smaller int column(s)")
    if "session_id" in df.columns:
        print(f"  Converted 'session_id' → category dtype")
    print(f"  Memory: {mem_before:,.1f} MB → {mem_after:,.1f} MB "
          f"({(1 - mem_after/mem_before)*100:.1f}% reduction)")

    return df


def remove_duplicates(df):
    """
    Remove rows that are exact duplicates of another row.

    WHY: When merging multiple files from the same testbed, the same
    measurement window might appear in more than one file if recordings
    overlap. Duplicate rows would give the ML model a biased view of
    those time periods and inflate training metrics.

    We keep the first occurrence and drop subsequent duplicates.

    NOTE: now that 'session_id' is retained (see drop_source_column), a
    duplicate is only detected when two rows from the SAME session are
    byte-identical. Two different sessions that happen to record identical
    KPI values are no longer treated as duplicates — which is correct: they
    are two genuinely different measurements from two different times, not
    an artefact of overlapping file exports.
    """
    print_section("STEP 7 — Removing duplicate rows")

    rows_before = len(df)
    df = df.drop_duplicates()
    rows_after  = len(df)
    removed     = rows_before - rows_after

    print(f"  Rows before deduplication: {rows_before}")
    print(f"  Duplicate rows removed:    {removed}")
    print(f"  Rows after deduplication:  {rows_after}")

    if removed == 0:
        print("  ✓ No duplicate rows found.")

    return df


# =============================================================================
# STEP 8 — Rename columns to clean Python-friendly names
# =============================================================================

def rename_columns(df):
    """
    Rename columns to lowercase snake_case without spaces, brackets, or %.

    WHY: Column names like 'tx_brate downlink [Mbps]' are awkward in Python
    code and can cause bugs. 'tx_brate_downlink_mbps' is safer and clearer.

    Only renames columns that exist in this DataFrame, so the function
    works correctly even if some columns were dropped earlier.
    """
    print_section("STEP 8 — Renaming columns to clean names")

    # Full rename map — covers all known O-RAN testbed column variants
    rename_map = {
        "Timestamp":                "timestamp",
        "dl_mcs":                   "dl_mcs",
        "dl_n_samples":             "dl_n_samples",
        "dl_buffer [bytes]":        "dl_buffer_bytes",
        "tx_brate downlink [Mbps]": "tx_brate_downlink_mbps",
        "tx_pkts downlink":         "tx_pkts_downlink",
        "dl_cqi":                   "dl_cqi",
        "ul_mcs":                   "ul_mcs",
        "ul_n_samples":             "ul_n_samples",
        "ul_buffer [bytes]":        "ul_buffer_bytes",
        "rx_brate uplink [Mbps]":   "rx_brate_uplink_mbps",
        "rx_pkts uplink":           "rx_pkts_uplink",
        "rx_errors uplink (%)":     "rx_errors_uplink_pct",
        "ul_sinr":                  "ul_sinr",
        "phr":                      "phr",
        "sum_requested_prbs":       "sum_requested_prbs",
        "sum_granted_prbs":         "sum_granted_prbs",
        "ul_turbo_iters":           "ul_turbo_iters",
    }

    # Only apply renames for columns that actually exist
    applicable = {old: new for old, new in rename_map.items() if old in df.columns}
    df = df.rename(columns=applicable)

    renamed_count = sum(1 for old, new in applicable.items() if old != new)
    print(f"  Renamed {renamed_count} column(s):")
    for old, new in applicable.items():
        if old != new:
            print(f"    '{old}'  →  '{new}'")

    print(f"\n  Final column list: {df.columns.tolist()}")
    return df


# =============================================================================
# STEP 9 — Convert data types
# =============================================================================

def convert_data_types(df):
    """
    Convert columns to their correct Python/pandas data types.

    The most important conversion is the timestamp:
      Raw value:  Unix milliseconds, e.g. 1617070531726 (a large integer)
      After:      Python datetime, e.g. 2021-03-30 02:15:31.726

    This is important because:
      - It lets us sort by time
      - It enables time-based operations (resampling, windowing)
      - The LSTM model needs rows in chronological order
    """
    print_section("STEP 9 — Converting data types")

    if "timestamp" in df.columns:
        # Handle both numeric (Unix ms) and already-parsed datetime strings
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            print("  'timestamp': converted from Unix milliseconds → datetime")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            print("  'timestamp': parsed from string → datetime")

        print(f"  Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    else:
        print("  Warning: no 'timestamp' column found — skipping datetime conversion.")

    print("\n  Data types after conversion:")
    print(df.dtypes.to_string())

    return df


# =============================================================================
# STEP 10 — Handle missing values
# =============================================================================

def handle_missing_values(df):
    """
    Find and fill any NaN values in the remaining columns.

    Strategy: fill missing numeric values with the column median.

    WHY MEDIAN (not mean)?
    The median is the middle value when sorted. It is not affected by
    extreme outliers. For example, if 99 rows have dl_mcs = 10 and
    one row has dl_mcs = 0, the mean would be pulled down, but the
    median would remain 10.

    WHY NOT DROP ROWS WITH MISSING VALUES?
    We keep all rows and fill missing values because:
      1. Dropping rows reduces the training data we have.
      2. In time-series data, each row represents a specific moment —
         dropping it creates a gap in the timeline.
    """
    print_section("STEP 10 — Handling missing values")

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    if missing_cols.empty:
        print("  ✓ No missing values found in any column.")
    else:
        print(f"  Found missing values in {len(missing_cols)} column(s):")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"\n    {col}: {count} missing ({pct:.1f}%)")

            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"      → Filled {count} values with median = {median_val:.4f}")
            else:
                # For non-numeric columns, fill with the most frequent value
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"      → Filled {count} values with mode = '{mode_val}'")

    return df


# =============================================================================
# STEP 11 — Detect invalid values
# =============================================================================

def detect_invalid_values(df):
    """
    Run sanity checks on the cleaned data and report anything suspicious.

    We do NOT remove invalid rows here — we report them so the student
    can decide whether to investigate further.

    Checks:
      1. Negative values in columns that should always be >= 0
      2. Error rate columns above 100% (physically impossible)
      3. CQI outside the 3GPP standard range of 0–15
      4. MCS outside the LTE/NR standard range of 0–28
    """
    print_section("STEP 11 — Detecting invalid values")

    issues_found = False

    # Columns that should never be negative
    non_negative_cols = [
        "dl_mcs", "dl_n_samples", "dl_buffer_bytes",
        "tx_brate_downlink_mbps", "tx_pkts_downlink", "dl_cqi",
        "ul_mcs", "ul_n_samples", "ul_buffer_bytes",
        "rx_brate_uplink_mbps", "rx_pkts_uplink",
        "rx_errors_uplink_pct", "sum_requested_prbs",
        "sum_granted_prbs", "ul_turbo_iters",
    ]

    print("\n  [Check 1] Columns that must not be negative:")
    for col in non_negative_cols:
        if col not in df.columns:
            continue
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"    ⚠ {col}: {neg_count} negative value(s)")
            issues_found = True
        else:
            print(f"    ✓ {col}: no negatives")

    print("\n  [Check 2] Error rate must be 0–100%:")
    for col in ["rx_errors_uplink_pct"]:
        if col not in df.columns:
            continue
        over_100 = (df[col] > 100).sum()
        if over_100 > 0:
            print(f"    ⚠ {col}: {over_100} rows exceed 100% (max = {df[col].max():.1f}%)")
            issues_found = True
        else:
            print(f"    ✓ {col}: max = {df[col].max():.2f}% — within range")

    if "dl_cqi" in df.columns:
        print("\n  [Check 3] dl_cqi must be 0–15:")
        out = ((df["dl_cqi"] < 0) | (df["dl_cqi"] > 15)).sum()
        if out > 0:
            print(f"    ⚠ {out} rows outside 0–15 range")
            issues_found = True
        else:
            print(f"    ✓ dl_cqi range: {df['dl_cqi'].min():.2f} – {df['dl_cqi'].max():.2f}")

    print("\n  [Check 4] MCS must be 0–28:")
    for col in ["dl_mcs", "ul_mcs"]:
        if col not in df.columns:
            continue
        out = ((df[col] < 0) | (df[col] > 28)).sum()
        if out > 0:
            print(f"    ⚠ {col}: {out} rows outside 0–28 range")
            issues_found = True
        else:
            print(f"    ✓ {col}: {df[col].min():.2f} – {df[col].max():.2f}")

    print()
    if issues_found:
        print("  ⚠ Some issues detected — review before training.")
    else:
        print("  ✓ All validity checks passed.")

    return df


# =============================================================================
# STEP 12 — Engineer the PRB grant ratio feature
# =============================================================================

def engineer_features(df):
    """
    Create one new feature derived from existing columns.

    prb_grant_ratio = sum_granted_prbs / (sum_requested_prbs + 1)

    WHY:
    Physical Resource Blocks (PRBs) are units of radio bandwidth.
    The UE requests how many it needs; the base station scheduler decides
    how many to grant.

      - Ratio close to 1.0 → scheduler meeting demand (no congestion)
      - Ratio much below 1.0 → scheduler denying requests (congestion)

    The +1 in the denominator prevents divide-by-zero when the UE is
    idle and requests 0 PRBs.

    This feature ranked 5th in Random Forest importance in single-file
    experiments — it captures congestion that neither PRB column alone reveals.
    """
    print_section("STEP 12 — Engineering derived features")

    if "sum_granted_prbs" in df.columns and "sum_requested_prbs" in df.columns:
        # +1 in denominator prevents divide-by-zero when UE requests 0 PRBs
        df["prb_grant_ratio"] = (
            df["sum_granted_prbs"] / (df["sum_requested_prbs"] + 1)
        )

        # Safety: clamp any infinity or extreme values that can appear when
        # sum_requested_prbs contains unexpected negative or zero values
        # across many recording sessions.
        # np.inf would cause sklearn to crash with "Input X contains infinity".
        df["prb_grant_ratio"] = df["prb_grant_ratio"].replace(
            [np.inf, -np.inf], np.nan
        )
        # Fill any NaN introduced by the clamp with the column median
        median_ratio = df["prb_grant_ratio"].median()
        df["prb_grant_ratio"] = df["prb_grant_ratio"].fillna(median_ratio)
        # Also clip to a sensible range — ratios above 2.0 are physically unrealistic
        df["prb_grant_ratio"] = df["prb_grant_ratio"].clip(lower=0.0, upper=2.0)

        print("  Created: prb_grant_ratio = sum_granted_prbs / (sum_requested_prbs + 1)")
        print(f"  Min: {df['prb_grant_ratio'].min():.4f}")
        print(f"  Max: {df['prb_grant_ratio'].max():.4f}")
        print(f"  Mean: {df['prb_grant_ratio'].mean():.4f}")
    else:
        print("  ⚠ PRB columns not found — skipping prb_grant_ratio engineering.")

    return df


# =============================================================================
# STEP 13 — Sort by timestamp and set as index
# =============================================================================

def set_time_index(df):
    """
    Sort the DataFrame by timestamp and use it as the row index.

    WHY:
    After merging multiple files, the rows may not be in chronological
    order. The LSTM model and time-based analysis need rows sorted by time.

    Setting the timestamp as the index also makes time-range slicing easy:
      df["2021-03-30":"2021-03-31"]
    """
    print_section("STEP 13 — Sorting by time and setting timestamp as index")

    if "timestamp" not in df.columns:
        print("  ⚠ No 'timestamp' column — skipping time index.")
        return df

    df = df.set_index("timestamp")
    df = df.sort_index()

    print(f"  Index set to 'timestamp'")
    print(f"  First reading: {df.index[0]}")
    print(f"  Last reading:  {df.index[-1]}")
    print(f"  Total rows:    {len(df)}")

    return df


# =============================================================================
# STEP 14 — Print preprocessing summary
# =============================================================================

def print_preprocessing_summary(summary, df_combined, df_cleaned):
    """
    Print a clear before/after summary table at the end of the pipeline.
    This is the table you can show in your viva or dissertation.
    """
    print_section("STEP 14 — Preprocessing Summary")

    rows_combined = df_combined.shape[0]
    rows_cleaned  = len(df_cleaned)
    dups_removed  = rows_combined - rows_cleaned  # approximate

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │              PREPROCESSING SUMMARY                  │
  ├───────────────────────────────┬─────────────────────┤
  │  Files found in data/raw/     │  {summary['files_found']:<19} │
  │  Files successfully loaded    │  {summary['files_processed']:<19} │
  │  Files skipped (with reason)  │  {summary['files_skipped']:<19} │
  ├───────────────────────────────┼─────────────────────┤
  │  Total rows before merge      │  {summary['rows_before_merge']:<19} │
  │  Rows in combined_dataset     │  {rows_combined:<19} │
  │  Duplicate rows removed       │  {max(0, rows_combined - rows_cleaned):<19} │
  │  Rows in cleaned_dataset      │  {rows_cleaned:<19} │
  ├───────────────────────────────┼─────────────────────┤
  │  Columns in cleaned dataset   │  {df_cleaned.shape[1]:<19} │
  │  Missing values remaining     │  {int(df_cleaned.isnull().sum().sum()):<19} │
  └───────────────────────────────┴─────────────────────┘
""")


# =============================================================================
# STEP 15 — Save the cleaned dataset
# =============================================================================

def save_cleaned(df, path: Path):
    """
    Save the final cleaned DataFrame to cleaned_dataset.csv.

    This file is the input to label_dataset.py in the next step.
    index=True keeps the timestamp as the first column of the CSV.
    """
    print_section("STEP 15 — Saving cleaned dataset")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

    print(f"  ✓ Saved: {path}")
    print(f"  Size:    {path.stat().st_size / 1024:.1f} KB")
    print(f"  Rows:    {len(df)}  Columns: {df.shape[1]}")


# =============================================================================
# MAIN — Run the full pipeline in order
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  O-RAN KPI Multi-File Dataset Cleaning Pipeline")
    print("  FYP — ML-based KPI Prediction xApp")
    print("=" * 65)

    # Step 1: Discover and validate all CSV files
    valid_frames, summary = load_and_validate_csv_files(RAW_DATA_DIR)

    # Step 2: Merge all valid files into one DataFrame
    df_combined = merge_dataframes(valid_frames)

    # Step 3: Save raw combined data as a checkpoint
    save_combined(df_combined, COMBINED_DATA_PATH)

    # --- Cleaning pipeline on the combined data ---

    # Step 4: Drop columns that are entirely empty (all NaN)
    # NOTE: no .copy() here — remove_empty_columns() already returns a new
    # DataFrame (via .drop()), so an extra .copy() would momentarily hold
    # TWO full copies of the combined dataset in memory at once. On a small
    # sample dataset that's harmless; on a large real recording (tens of
    # millions of rows) it can exhaust available RAM. Not needed either way.
    df = remove_empty_columns(df_combined)

    # Step 5: Remove the internal source-file tracking column
    df = drop_source_column(df)

    # Step 5b: Drop known identity and config columns explicitly
    # This handles cases where columns like 'slice_id' now vary across
    # multiple sessions and would not be caught by the zero-variance check
    print_section("STEP 5b — Dropping known identity/config columns")
    df = drop_known_columns(df)

    # Step 6: Drop any remaining columns with only one unique value
    df = remove_zero_variance_columns(df)

    # Step 6b: Downcast numeric dtypes BEFORE the expensive steps below —
    # halves memory use for remove_duplicates()'s row-hashing, rename,
    # missing-value handling, feature engineering, and sort_index().
    df = downcast_dtypes(df)

    # Step 7: Remove duplicate rows (can occur when sessions overlap)
    df = remove_duplicates(df)

    # Step 8: Rename columns to clean Python snake_case names
    df = rename_columns(df)

    # Step 9: Convert data types (especially the timestamp)
    df = convert_data_types(df)

    # Step 10: Fill any remaining missing values with column medians
    df = handle_missing_values(df)

    # Step 11: Run validity checks and report issues
    df = detect_invalid_values(df)

    # Step 12: Engineer the PRB grant ratio feature
    df = engineer_features(df)

    # Step 13: Sort by time and set timestamp as index
    df = set_time_index(df)

    # Step 14: Print the preprocessing summary
    print_preprocessing_summary(summary, df_combined, df)

    # Step 15: Save the cleaned dataset
    save_cleaned(df, CLEANED_DATA_PATH)

    print("\n" + "=" * 65)
    print("  ✓ Dataset cleaning complete!")
    print(f"  combined_dataset.csv → {COMBINED_DATA_PATH}")
    print(f"  cleaned_dataset.csv  → {CLEANED_DATA_PATH}")
    print("  Next step: python app/ml/label_dataset.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
