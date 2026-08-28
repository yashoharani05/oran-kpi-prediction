# ML-based KPI Prediction xApp for O-RAN Network Monitoring

**Final Year Project — BSc Software Engineering**
**Dataset:** Real O-RAN testbed measurements (srsRAN) — 2,090 rows, 8.7 minutes

---

## What This Project Does

Forecasts 5G radio network degradation **~5 seconds ahead** from Key
Performance Indicators (KPIs), using three machine learning models trained
on real testbed data. The model sees KPIs at time `t` and predicts the
network's degradation state at `t + 5s` — not the same instant — so the
xApp can raise an early warning before conditions actually degrade.

> See `docs/FORECASTING_METHODOLOGY_UPDATE.md` for the full rationale: this
> project originally trained on same-instant labels (predicting "now" from
> "now", which is classification, not forecasting) and was corrected to
> predict 5 seconds ahead, evaluated against a naive "nothing changes"
> baseline. Re-run the training pipeline below to get current F1/accuracy
> numbers for the forecasting task — the old same-instant numbers no longer
> apply and have been removed from this table.

| Model | Task |
|---|---|
| Random Forest | Forecast degradation ~5s ahead |
| XGBoost | Forecast degradation ~5s ahead |
| LSTM | Forecast degradation ~5s ahead (20-row sliding window) |
| Naive Baseline | "Assume state stays the same" — the bar the ML models must beat |

---

## Running With Docker (Recommended)

Docker runs everything without installing Python or Node.js on your PC.

**Requires:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (free)

### Step 1 — Place the dataset
```
backend\data\raw\raw1.csv
```

### Step 2 — Build and start
```
docker compose up --build
```

### Step 3 — Train the ML models (first time only)
```
docker compose exec backend sh run_training.sh
docker compose restart backend
```

### Step 4 — Open the dashboard
- **http://localhost:3000** — prediction dashboard
- **http://localhost:3000/live** — live monitoring
- **http://localhost:8000/docs** — API docs

Full Docker guide: `docs/DOCKER_GUIDE.md`

---

## Running Without Docker (Manual Setup)

**Requires:** Python 3.10+, Node.js 18+

**Terminal 1 — Backend:**
```
cd backend
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app\ml\clean_dataset.py
python app\ml\label_dataset.py
python app\ml\train_random_forest.py
python app\ml\train_xgboost.py
python app\ml\train_lstm.py
uvicorn main:app --reload
```

**Terminal 2 — Frontend:**
```
cd frontend
npm install
npm run dev
```

Full installation guide: `docs/INSTALLATION_GUIDE.md`

---

## Project Structure

```
fyp-oran-kpi-prediction/
├── docker-compose.yml        ← Docker: one command to start everything
├── README.md                 ← This file
│
├── docs/                     ← All submission documentation
│   ├── DOCKER_GUIDE.md
│   ├── INSTALLATION_GUIDE.md
│   ├── ARCHITECTURE.md
│   ├── API_DOCS.md
│   ├── USER_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   ├── TESTING_GUIDE.md
│   ├── LIMITATIONS.md
│   ├── FUTURE_IMPROVEMENTS.md
│   ├── PRESENTATION_NOTES.md
│   └── VIVA_QA.md
│
├── backend/                  ← Python + FastAPI + ML
│   ├── Dockerfile
│   ├── run_training.sh
│   ├── main.py
│   ├── requirements.txt
│   ├── app/
│   │   ├── api/              ← predict.py, stream.py
│   │   ├── ml/               ← 5 training scripts
│   │   ├── schemas.py
│   │   └── utils/
│   ├── models/               ← Saved .pkl and .keras files
│   └── data/
│
└── frontend/                 ← Next.js + TypeScript + Tailwind
    ├── Dockerfile
    ├── next.config.js
    └── src/
        ├── app/              ← page.tsx, live/page.tsx
        └── components/       ← 9 React components
```

---

## Pages

| URL | Purpose |
|---|---|
| http://localhost:3000 | Manual prediction dashboard |
| http://localhost:3000/live | Live streaming monitor (2 s refresh) |
| http://localhost:8000/docs | Interactive API documentation |

## Documentation

| File | What it covers |
|---|---|
| `docs/DOCKER_GUIDE.md` | Docker setup, commands, troubleshooting |
| `docs/INSTALLATION_GUIDE.md` | Manual Python + Node.js setup |
| `docs/ARCHITECTURE.md` | System, data flow, and model diagrams |
| `docs/API_DOCS.md` | All API endpoints with examples |
| `docs/USER_GUIDE.md` | How to use both dashboards |
| `docs/DEVELOPER_GUIDE.md` | Code structure, how to extend |
| `docs/TESTING_GUIDE.md` | Manual test checklists |
| `docs/LIMITATIONS.md` | Honest limitations with severity |
| `docs/FUTURE_IMPROVEMENTS.md` | 12 concrete next steps |
| `docs/PRESENTATION_NOTES.md` | 7-slide structure + talking points |
| `docs/VIVA_QA.md` | 25 viva questions with full answers |
| `docs/FORECASTING_METHODOLOGY_UPDATE.md` | **Read this first** — the same-instant → forecasting correction: what changed, why, and what still needs re-running |

> Note: `ARCHITECTURE.md`, `API_DOCS.md`, `USER_GUIDE.md`, `DEVELOPER_GUIDE.md`,
> `TESTING_GUIDE.md`, `LIMITATIONS.md`, `FUTURE_IMPROVEMENTS.md`,
> `PRESENTATION_NOTES.md`, and `VIVA_QA.md` were written for the original
> same-instant version of this project and have not yet been updated for the
> forecasting correction (old model filenames, old F1 numbers, no mention of
> `current_status`/forecasting). `FORECASTING_METHODOLOGY_UPDATE.md` is the
> current source of truth until they're refreshed.
