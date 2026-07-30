# User Guide

How to use the two dashboard pages once the backend and frontend are running.

---

## Dashboard 1 — Manual Prediction (http://localhost:3000)

### Header bar

Sticky at the top of every page. Shows:
- Project title on the left
- **Model status pill** — green "Model Ready" when all three models are loaded, red if backend is offline
- A live clock updating every second

---

### Model selector

Three toggle buttons choose which ML model makes the prediction:
- **Random Forest** (sky blue) — F1 ~97%, most explainable
- **XGBoost** (violet) — F1 ~99%, highest accuracy
- **LSTM** (emerald) — F1 ~32%, experimental (low due to short window)

Switching model clears the current prediction. The active model button is highlighted.

---

### KPI Input Form

18 number fields grouped into three sections:

**Downlink KPIs** — what the base station sends to the phone:
`dl_mcs`, `dl_n_samples`, `dl_buffer_bytes`, `tx_brate_downlink_mbps`, `tx_pkts_downlink`, `dl_cqi`

**Uplink KPIs** — what the phone sends to the base station:
`ul_mcs`, `ul_n_samples`, `ul_buffer_bytes`, `rx_brate_uplink_mbps`, `rx_pkts_uplink`, `rx_errors_uplink_pct`

**Signal and Resources:**
`ul_sinr`, `phr`, `sum_requested_prbs`, `sum_granted_prbs`, `ul_turbo_iters`, `prb_grant_ratio`

**Preset buttons** at the top of the form:
- **Load Healthy Sample** — fills typical normal-operating values from the dataset
- **Load Degraded Sample** — fills values including 100% error rate, low CQI, low MCS

---

### KPI Summary Cards

Eight colour-coded tiles update instantly as you type:
- **Sky blue** — value is in the healthy range
- **Amber** — borderline value
- **Red** — value in the degraded range

The thresholds used for colouring match the quantile thresholds from `label_dataset.py`.

---

### KPI Health Bar Chart

Six most important KPIs normalised to 0–100% so they can share one axis. For metrics where low is bad (throughput, CQI), the bar is inverted — a tall bar always means healthy. The dashed amber line marks the degradation threshold. Bars crossing into the bad zone turn red.

---

### Predict Button

Click **Predict Network Risk** to send the 18 values to the backend.

- Loading spinner appears during the request
- On success: gauge, risk label, and alert box update
- On error: red error box shows the problem (usually "backend not running")

---

### Probability Gauge

SVG arc fills from green → amber → red based on degradation probability:
- **0–35%** — green — low risk
- **35–65%** — amber — borderline
- **65–100%** — red — high risk

---

### Risk Label

Large text below the gauge: either **NORMAL** (green) or **DEGRADED** (red).

---

### Alert Box

Colour-coded panel showing:
- The verdict with an icon (check mark or warning triangle)
- The confidence percentage
- Recommendation text adapted to the probability level:
  - Very confident Normal → "No action required"
  - Borderline Normal → "Continue monitoring"
  - Low confidence Degraded → "LOW RISK: Marginal degradation"
  - Moderate Degraded → "MODERATE RISK: Review KPI trends"
  - High confidence Degraded → "HIGH RISK: Immediate investigation"

For Degraded predictions, the box has a red pulsing border.

---

### Model Comparison Panel (bottom of page)

Loaded automatically from `GET /api/comparison` when the page opens. Shows:
- Metrics table: Accuracy, Precision, Recall, F1 for all three models
- False negative summary: how many degradation events each model missed
- Grouped bar chart comparing all four metrics visually

---

### How It Differs Panel

Three cards explaining the conceptual difference between RF, XGBoost, and LSTM in plain English, plus an honest note about why LSTM scores lower.

---

## Dashboard 2 — Live Monitoring (http://localhost:3000/live)

Polls `GET /api/stream/next` every 2 seconds, replaying the labeled CSV row by row.

### Controls

| Button | What it does |
|---|---|
| **Start Monitoring** | Begins the 2-second polling loop |
| **Pause** | Stops the loop without resetting history |
| **Reset** | Rewinds CSV cursor to row 0, clears all history and stats |
| **← Manual Predict** | Returns to the home page |

---

### Session Stats

Three counters at the top (reset on each page load or Reset click):
- **Rows Seen** — total rows processed this session
- **Normal** — count of Normal predictions (green)
- **Degraded** — count and percentage of Degraded predictions (red)

---

### Dataset Progress Bar

A thin bar showing current position in the dataset. Label: "Row X / 19053569 — N%". Wraps back to 0 when the last row is reached.

---

### Live KPI Cards

Eight metric tiles update every 2 seconds with the current row's values. Colour coding is the same as the manual dashboard. The timestamp in the top-right shows the original testbed recording time of the current row.

---

### Live Chart

Scrolling Recharts LineChart showing the last 30 readings:
- **Sky blue line** — downlink throughput in Mbps (left axis)
- **Red line** — uplink error rate in % (right axis)

Animation is disabled so new points appear without a redraw flicker.

---

### Probability Gauge

Same SVG arc as the manual dashboard. Animates to each new prediction's probability every 2 seconds.

---

### Risk Label and Alert Box

Both update every 2 seconds. During degraded periods (multiple consecutive rows with high error rates), the red pulsing box remains visible for several updates — this is the most dramatic moment in a live demo.

---

### Ground Truth vs Prediction Panel

Shows for the current row:
- **Actual label** — from `label_dataset.py` (0 or 1)
- **Predicted label** — from the model right now
- **Match indicator** — ✓ Correct or ✗ Mismatch

Mismatches are the model's errors visible in real time. With XGBoost, these are rare.

---

## Tips for Viva Demo

1. Load healthy sample → predict → show green result (10 seconds)
2. Load degraded sample → predict → show HIGH RISK red result (10 seconds)
3. Switch to XGBoost → predict → same result, different model name (5 seconds)
4. Navigate to /live → Start → let 5–10 rows stream through (20 seconds)
5. Point to ground truth panel when a degraded row appears
6. Scroll back to home → point to model comparison table
7. Total demo time: under 2 minutes
