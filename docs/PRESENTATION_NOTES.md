# Presentation Notes

Suggested structure and talking points for a 10–15 minute viva or demo.

---

## Suggested Slide Order (7 slides)

### Slide 1 — Title

**ML-based KPI Prediction xApp for O-RAN Network Monitoring**
Your name, student number, supervisor, date.

Opening line: "5G networks produce dozens of performance measurements every 250 milliseconds. This project builds an ML system that reads those measurements and predicts network degradation before users notice it."

---

### Slide 2 — Problem and Motivation

- 5G base stations produce 18+ KPI measurements every 250ms
- Network engineers cannot watch all of them manually
- Degradation often begins gradually and is detectable before users notice
- O-RAN's Near-RT RIC is designed to host exactly this kind of intelligence

What this project does:
- Collects multiple recording sessions from a real srsRAN testbed (19M+ rows)
- Trains three ML models to classify each moment as Normal or Degraded
- Builds a live web dashboard showing predictions every 2 seconds

---

### Slide 3 — Dataset and Pipeline

**Dataset:** Multiple CSV files from an O-RAN srsRAN testbed, merged to 19M+ rows

**Five-step pipeline:**
1. `clean_dataset.py` — validates and merges all CSVs, drops identity columns, removes duplicates, engineers prb_grant_ratio
2. `label_dataset.py` — scores each row against 7 quantile-based KPI conditions, labels as Normal/Degraded
3. `train_random_forest.py` — 100 parallel trees, majority vote
4. `train_xgboost.py` — 100 sequential trees, each corrects the last
5. `train_lstm.py` — 20-step sliding window, memory-efficient generator (fixed the 27GB RAM crash)

Label distribution: approximately 71% Normal, 29% Degraded

---

### Slide 4 — Model Results

| Model | F1 Score | Precision | Recall | Missed Events |
|---|---|---|---|---|
| Random Forest | ~97% | 100% | ~95% | Few |
| XGBoost | ~99% | 100% | ~98% | Fewest |
| LSTM (w=20) | ~32% | 100% | ~19% | Many |

Key points to make:
- Precision is 100% for all three — zero false alarms
- XGBoost is the strongest model on this dataset
- LSTM's low recall is expected — 5-second window is not enough temporal context
- LSTM recall will improve with a larger window size (future work)

> Note: Your actual numbers will differ from the table above because you have more data.

---

### Slide 5 — System Architecture

Three layers:
1. **Data pipeline** — five Python scripts run once to produce trained model files
2. **FastAPI backend** — loads models at startup, serves 6 REST endpoints
3. **Next.js frontend** — two pages: manual prediction (/) and live monitoring (/live)

Key technical decisions to mention:
- Models loaded into RAM at startup — predictions are instant
- Live stream uses a DataFrame + cursor counter — no database needed
- LSTM training uses a Keras generator — 750 KB per batch instead of 27 GB

---

### Slide 6 — Live Demo

Suggested script (about 3 minutes):

1. Open **http://localhost:3000**
2. "This is the manual prediction dashboard. I can choose which model to use."
3. Click "Load Healthy Sample" → Click "Predict Network Risk"
4. "Green result, zero probability. The network is healthy."
5. Click "Load Degraded Sample" → Click "Predict Network Risk"
6. "Red result, high probability. The system says HIGH RISK."
7. Switch to XGBoost → Predict again → "Same verdict, different model."
8. Open **http://localhost:3000/live** → Click "Start Monitoring"
9. "This replays the CSV dataset one row every 2 seconds. Everything updates automatically."
10. Point to the ground truth panel: "This shows actual label versus predicted. A ✗ means the model made a mistake."
11. Scroll back to home → point to comparison table → "XGBoost wins on F1 and recall."

---

### Slide 7 — Limitations and Future Work

**Honest limitations:**
- Simulated live stream, not a real E2 interface connection
- LSTM receives a replicated single row, not genuine temporal context
- Labels are student-defined using quantile thresholds
- Models trained once, no concept drift handling

**Future work:**
- Real E2 interface via Near-RT RIC
- Rolling buffer so LSTM sees genuine 20-row history
- SHAP explanations per prediction (which KPI caused the alert)
- Larger LSTM window size (200+ rows = 50 seconds of history)

**Closing statement:** "XGBoost achieves around 99% F1 on this dataset. The LSTM result is lower but honest — it demonstrates that temporal models need longer sequences than 5 seconds to show their advantage. With a rolling buffer and a larger window, LSTM would likely surpass the tree models."

---

## Key Numbers to Have Ready

| Fact | Value |
|---|---|
| Total rows after merging | ~19 million |
| Original columns per CSV | 36 |
| Columns after cleaning | 18 + 1 derived = 19 |
| Identity/config columns always dropped | 13 |
| Normal / Degraded split | ~71% / 29% |
| Train/test split | 80/20, no shuffle |
| Random Forest trees | 100 |
| XGBoost max_depth | 4 |
| LSTM window size | 20 rows = 5 seconds |
| LSTM batch size | 512 (generator, not 27 GB array) |
| LSTM total parameters | 7,073 |
| API endpoints | 6 |
| Live stream interval | 2 seconds |
| Feature importance #1 | dl_mcs (22.97%) |

---

## Words to Use and Avoid

**Use these:**
- "I chose this approach because..."
- "This is a known limitation — specifically..."
- "The result is expected because..."
- "If I had more time, I would..."
- "This is a trade-off between X and Y"

**Avoid these:**
- "It's just a simple project"
- "I didn't have time to..."
- "The model is bad because..."
- Apologising for LSTM's low recall — explain it instead
