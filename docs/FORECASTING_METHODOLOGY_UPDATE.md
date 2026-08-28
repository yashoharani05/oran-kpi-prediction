# Forecasting Methodology Update

**Status:** Implemented in code and verified end-to-end against a synthetic
stand-in dataset (see "Evaluation Method" and "Tests Performed" below). The
project's own `data/raw/` was empty at the time of this change, so real
metrics on `raw1.csv` are not yet available — see "Commands to Run".

---

## Problem

The original pipeline trained all three models (Random Forest, XGBoost,
LSTM) to answer:

> "Is the network degraded **right now**, at this exact timestamp?"

Both the input features (`X`, 18 KPIs) and the target label (`y`,
`degradation_risk`) came from the **same row / same timestamp**. That is a
same-instant classification problem, not a forecast — despite the project
objective being "early prediction and notification of critical KPI
degradation".

## Why This Was Incorrect

A model that scores well at same-instant classification tells you nothing
about whether it can anticipate degradation *before* it happens. The FYP's
stated objective — a **proactive** xApp with **early warning** — requires
the target to describe a *future* moment relative to the input features:

```
classification (old):  KPIs @ t  ->  state @ t
forecasting   (new):   KPIs @ t  ->  state @ t + 5s
```

## Solution

```
degradation_risk          (existing, rule-based, unchanged — "label_now")
        │
        │  shift 20 rows into the future, INDEPENDENTLY per recording session
        ▼
degradation_risk_future   (new — "label_future", the actual ML target)
```

- The deterministic quantile/threshold degradation rule in
  `label_dataset.py` (`score_row()` / `compute_thresholds()`) is **unchanged**
  — it remains the ground-truth generator, exactly as the correction brief
  required. Internally we refer to its output as `label_now`.
- A new column, `degradation_risk_future`, is `degradation_risk` shifted
  20 rows (5 seconds, at the ~250ms sampling interval) into the future,
  computed **per session** so a shift can never pull a label in from a
  different recording. Rows at the tail of each session that have no valid
  future value are dropped (20 rows in the current single-session dataset).
- All three models now train on `X = 18 KPI features`, `y =
  degradation_risk_future`. `degradation_risk` (current state) and
  `session_id` are explicitly excluded from the feature matrix.

## Code Changes

| File | What changed | Why |
|---|---|---|
| `backend/app/ml/forecast_config.py` *(new)* | Central constants: horizon (5s / 20 rows), sampling interval (250ms), LSTM window size, alert threshold, column names | Single source of truth so the horizon can be changed in one place |
| `backend/app/ml/forecast_utils.py` *(new)* | `add_future_label()` (session-safe shift), sanity checks, naive-baseline metrics, transition analysis, comparison-report upsert | Shared logic reused identically by every training script instead of being re-implemented (and potentially re-broken) three times |
| `backend/app/ml/clean_dataset.py` | `_source_file` is renamed to `session_id` and **kept** instead of dropped; excluded from the zero-variance-column drop | `label_dataset.py` needs a session boundary to shift labels safely — this was previously discarded |
| `backend/app/ml/label_dataset.py` | Adds `degradation_risk_future` via `add_future_label()`; verifies the sampling-interval assumption against real timestamps; runs label-shift / session-boundary sanity checks; persists quantile thresholds to `models/degradation_thresholds.json` for the API | Creates the actual forecasting target and the artefact the API needs to compute "current status" independently of the ML model |
| `backend/app/ml/train_random_forest.py` | Target changed to `degradation_risk_future`; `degradation_risk`/`session_id` excluded from features; naive baseline computed on the same held-out split; model saved as `random_forest_forecast_5s.pkl` | Implements the corrected task for Random Forest |
| `backend/app/ml/train_xgboost.py` | Same changes as above for XGBoost; saved as `xgboost_forecast_5s.pkl` | Implements the corrected task for XGBoost |
| `backend/app/ml/train_lstm.py` | Sliding-window generator rewritten to (a) never build a window that crosses a session boundary and (b) target `y[window_end]` instead of `y[window_end + 1]` — since `y` is now already the future-shifted label, the window's *last* row is the correct prediction origin, not one row past it (see "LSTM Alignment" below); saved as `lstm_forecast_5s.keras` / `lstm_scaler_forecast_5s.pkl` | Implements the corrected task for LSTM without silently doubling the forecast horizon |
| `backend/app/utils/model_loader.py` | Paths updated to the new `_forecast_5s` model filenames | Old same-instant models are not silently loaded or overwritten |
| `backend/app/utils/degradation_rule.py` *(new)* | Re-implements the rule-based scoring against persisted thresholds, for a single request-time KPI row | "Current status" must come from the deterministic rule, not the forecasting model (see correction brief §23) |
| `backend/app/schemas.py` | `PredictionResponse` gets additive, optional fields: `current_status`, `current_score`, `forecast_horizon_seconds`, `early_warning` | Communicates the new future-vs-now distinction without breaking existing consumers of `risk_label`/`risk_code`/`probability` |
| `backend/app/api/predict.py` | Computes `current_status` via the rule, `early_warning` via threshold check | Current status and forecast are now visibly two different things |
| `backend/app/api/stream.py` | Adds `current_status`, `actual_future_risk` (ground truth for the forecast), `early_warning` with **de-duplication** (fires once per Normal→Degraded transition, not every 2s poll) | Avoids notification spam (§26) and lets the live demo show forecast accuracy honestly |
| `frontend/src/types/index.ts`, `frontend/src/lib/api.ts` | Additive optional fields mirroring the backend schema | Non-breaking |
| `frontend/src/components/AlertBox.tsx` | Shows "Current Status" and "N-second Forecast" side by side when the backend supplies both, plus an early-warning banner; falls back to the old single-box layout if `current_status` is absent | Makes the now-vs-future distinction visible in the UI |
| `frontend/src/app/live/page.tsx` | The "Ground Truth vs Prediction" panel now compares the model's forecast against `actual_future_risk` (the true future label), not `actual_risk` (the true *current* label), which was the wrong comparison once `predicted_risk` became a forecast | Otherwise the "accuracy" shown in the live demo would silently be measuring the wrong thing |

### Files deliberately left untouched
`main.py`, `app/api/predict.py`'s route signatures, `app/utils/recommender.py`,
`docker-compose.yml`, all Dockerfiles, `KpiInput`/`HealthResponse` schemas,
and all other frontend components (`KpiCard`, `KpiChart`, `ModelComparison`,
`ProbabilityGauge`, `StreamProgress`, `Header`) — none of these needed to
change for the correction and were left exactly as they were.

## Forecast Configuration

```
Sampling interval : 250ms  (verified against real timestamps at runtime,
                             not assumed blindly — see label_dataset.py's
                             sampling-interval check)
Forecast horizon   : 5 seconds = 20 rows
LSTM window size   : 20 rows   (a SEPARATE constant from the horizon —
                             see "LSTM Alignment" below)
Early-warning alert threshold: 0.70 probability
```

All of the above live in `backend/app/ml/forecast_config.py`.

## LSTM Alignment (the "accidental double horizon" bug this avoids)

The original LSTM built a window of 20 past rows `[i : i+20)` and targeted
`y[i+20]` — i.e. **one row past the end of the window**. Once `y` is
replaced with the future-shifted label, using `y[i+20]` instead of the
correct `y[i+19]` (the window's own last row) would have meant:

```
window ends at row t (= i+19)
y[i+20] already means "state at (i+20) + 20 rows ahead" = state at t + 21 rows
                                                          ≈ 5.25s ahead, not 5s
```

Small in this case, but the failure mode generalizes badly (e.g. with a
30-row window and a 20-row horizon it would silently become a 50-row / 12.5s
forecast). The corrected generator targets `y[window_start + window_size - 1]`
— the window's own last row — which is exactly `t`, so the already-future-
shifted label at `t` correctly means "5 seconds after `t`". A worked example
is printed at the start of every LSTM training run
(`print_window_alignment_example()` in `train_lstm.py`).

## Evaluation Method

- **Chronological split** (unchanged from before): `shuffle=False`,
  first 80% of time-ordered rows train, last 20% test — for Random Forest
  and XGBoost.
- **Session-boundary-aware windowing** (new, LSTM only): windows are never
  allowed to span two different sessions, and a window's prediction origin
  must fall inside the train or test partition it's assigned to (a small
  amount of pre-split context is allowed into a window so the first test
  sequence isn't artificially starved of history — this is context, not
  label leakage).
- **Session holdout** (§12 of the brief) is *not* implemented as a second
  evaluation mode: the current dataset is a single recording session, so
  there is nothing to hold out. `forecast_utils.session_row_ranges()` and
  the per-session `add_future_label()` logic are written to generalize to
  multiple sessions if/when more recordings are added — session holdout
  would then be a straightforward addition (train on sessions 1-4, test on
  session 5) using the same building blocks.

## Naive Baseline Comparison

Implemented in `forecast_utils.naive_baseline_metrics()`: "assume the
network state 5 seconds from now is whatever it is right now"
(`predicted_future = current`). Computed on the exact same held-out test
rows/sequences as each model, and written into
`models/comparison_report.json` as its own entry
(`"Naive Baseline (state unchanged)"`) alongside RF/XGBoost/LSTM, so the API
and frontend's `ModelComparison` component show it automatically without
further wiring.

**On the synthetic verification dataset** (random KPI values — see "Tests
Performed"), none of the three models beat the naive baseline's F1 score.
This is the *expected, correct* outcome for data with no real temporal
signal, and the framework surfaces it plainly rather than hiding it — which
is exactly what §10 of the correction brief asked for. **This is not a
result on your real testbed data** — see "Commands to Run" for how to get
real numbers.

## API / UI Changes

- `POST /api/predict` and `GET /api/stream/next` responses gained:
  `current_status`, `current_score` (rule-based, "now"), `early_warning`
  (bool), `forecast_horizon_seconds` (int). All additive/optional —
  existing `risk_label`/`risk_code`/`probability`/`recommendation`/
  `model_used` fields are unchanged in shape, only in *meaning* (they now
  describe the forecast, not the same instant).
- `GET /api/stream/next` additionally returns `actual_future_risk` — the
  true future label from the dataset — so the live demo can show forecast
  accuracy against ground truth rather than against the wrong (current-
  instant) label.
- Early-warning alerts in the stream endpoint are de-duplicated: `fire_alert`
  is only `True` on the Normal→Degraded transition edge, not on every 2-second
  poll while the condition persists (§26 — avoid notification spam).
- `AlertBox.tsx` now shows Current Status and the N-second Forecast side by
  side (with an early-warning banner) when the backend supplies
  `current_status`; it degrades gracefully to the old single-box layout
  against an older backend response.

## Tests Performed

Since `backend/data/raw/` contained no CSV at the time of this change, a
synthetic 2,090-row dataset matching the real schema (18 raw KPI columns,
250ms spacing, one session) was generated and run through the **entire**
pipeline to validate the code, not just review it:

1. `clean_dataset.py` → `session_id` correctly retained (19 columns out).
2. `label_dataset.py` → sampling-interval check passed (250ms observed vs
   250ms assumed); label-shift sanity check passed; session-boundary check
   passed; 20 tail rows correctly dropped (2090 → 2070); future-target
   class distribution and Normal→Degraded transition counts printed
   correctly.
3. `train_random_forest.py` → trained, evaluated, naive baseline computed
   on the same split, model saved as `random_forest_forecast_5s.pkl`.
4. `train_xgboost.py` → trained, evaluated, comparison table printed
   (baseline + RF + XGB all present and not clobbering each other in
   `comparison_report.json`).
5. `train_lstm.py` → session-aware window generator produced the correct
   sequence counts, trained for 8 epochs (early stopping), evaluated,
   naive baseline + transition report computed on the LSTM's own test
   sequences, all four entries (baseline, RF, XGB, LSTM) present in the
   final comparison report.
6. **Unit-level correctness check** (standalone, outside the app): a
   synthetic two-session, non-uniform-length dataset was used to verify
   that (a) no generated LSTM window ever spans two sessions, and (b) every
   window's target equals the correct "state at `t + horizon`" value — both
   checks passed.
7. **Full API test** via FastAPI's `TestClient` (in-process, no network):
   `/api/health`, `/api/predict?model=random_forest`,
   `/api/predict?model=lstm`, `/api/stream/next`, `/api/comparison` were
   all called against the trained synthetic models and returned the
   expected shapes, including `current_status` (independently correct
   from the ML forecast in the observed response), `actual_future_risk`,
   and `early_warning`.

**What was not executed:** training/evaluation against your real
`raw1.csv` testbed recording, and no frontend build/render check (no
Node.js toolchain available in this environment) — the TypeScript changes
are type-consistent additions and were reviewed manually, not compiled.

## Commands to Run (on your real dataset)

From `backend/` with your virtualenv active and `raw1.csv` placed in
`backend/data/raw/`:

```bash
python app/ml/clean_dataset.py
python app/ml/label_dataset.py
python app/ml/train_random_forest.py
python app/ml/train_xgboost.py
python app/ml/train_lstm.py
uvicorn main:app --reload
```

(Same as before — no new commands were introduced; `run_training.sh` and
`docker compose exec backend sh run_training.sh` still work unchanged.)

Frontend, unchanged:
```bash
cd frontend
npm install
npm run dev
```

## Documentation Added/Updated

- `docs/FORECASTING_METHODOLOGY_UPDATE.md` (this file) — new.
- `README.md` — model table and "what this project does" updated to
  describe forecasting instead of same-instant classification (old
  near-perfect F1 numbers removed — they described the wrong task).

**Not updated** (flagged here rather than silently left stale):
`docs/ARCHITECTURE.md`, `docs/API_DOCS.md`, `docs/USER_GUIDE.md`,
`docs/DEVELOPER_GUIDE.md`, `docs/TESTING_GUIDE.md`, `docs/LIMITATIONS.md`,
`docs/FUTURE_IMPROVEMENTS.md`, `docs/PRESENTATION_NOTES.md`,
`docs/VIVA_QA.md` still describe the pre-correction same-instant design in
places (e.g. old model filenames, old F1 numbers, no mention of
`current_status`/forecasting). They should be reviewed before submission;
this file is the authoritative source for what actually changed.

## Remaining Limitations

- **5-second horizon is a design choice**, not a derived optimum — changing
  it only requires editing `FORECAST_HORIZON_SECONDS` in
  `forecast_config.py`, but no sweep over 1s/10s/30s horizons has been run.
- **Sampling-interval assumption**: verified to be ~250ms on this dataset,
  but this is checked at label-generation time, not guaranteed for future
  recordings — `verify_sampling_interval()` will print a warning if a new
  recording drifts more than 20% from 250ms, at which point a timestamp-
  based (not row-count-based) shift should be used instead.
- **Single-session dataset**: the session-safety code (session-aware
  shifting, session-aware LSTM windowing, `session_row_ranges()`) is
  written to generalize to multiple recording files, but has only been
  exercised against a synthetic 2-session dataset in isolation, not against
  a real multi-session O-RAN recording. With multiple sessions that overlap
  in real-world clock time, the *global* chronological 80/20 split used by
  RF/XGBoost degenerates to "whichever session-blocks land in the last 20%
  by row position" rather than a true wall-clock split — session holdout
  (§12) would be the more defensible evaluation in that scenario.
- **No hyperparameter re-tuning**: RF/XGBoost/LSTM hyperparameters were
  deliberately kept as in the original same-instant version (per the
  correction brief's "preserve existing hyperparameters initially"
  instruction), so any accuracy drop reflects the harder task, not a
  regressed model.
- **Naive-baseline outcome above is from synthetic random data**, not your
  real testbed recording — re-run the pipeline on `raw1.csv` to get results
  that mean something.
- **Colosseum/simulation generalization**: as previously noted in
  `docs/LIMITATIONS.md`, results are specific to this testbed configuration
  and dataset size; this was true before the correction and remains true
  after it.
