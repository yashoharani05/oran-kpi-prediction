# Developer Guide

How the code is structured and how to extend it.

---

## Backend Structure

```
backend/
├── main.py                      ← App entry point. Loads models, registers routers, CORS.
├── requirements.txt
├── run_training.sh              ← Runs all 5 ML scripts inside Docker
│
├── app/
│   ├── schemas.py               ← Pydantic request/response shapes (KpiInput, PredictionResponse, etc.)
│   ├── api/
│   │   ├── predict.py           ← GET /health, POST /predict, GET /comparison
│   │   └── stream.py            ← GET /stream/next, /status, /reset
│   ├── ml/
│   │   ├── clean_dataset.py     ← Multi-file cleaning pipeline (15 steps)
│   │   ├── label_dataset.py     ← Quantile-based labelling
│   │   ├── train_random_forest.py
│   │   ├── train_xgboost.py
│   │   └── train_lstm.py        ← Memory-efficient generator version
│   └── utils/
│       ├── model_loader.py      ← load_all_models() → dict of RF, XGB, LSTM, scaler
│       └── recommender.py       ← get_recommendation(risk_code, probability) → string
│
├── models/                      ← Saved .pkl and .keras files
└── data/
    ├── raw/                     ← Place all raw CSV files here
    └── processed/               ← combined_dataset.csv, cleaned_dataset.csv, labeled_dataset.csv
```

---

## How Models Are Shared Across Requests

At startup, `main.py` calls `load_all_models()` which returns a dict stored in `app.state`:

```python
app.state.models = {
    "random_forest": <RandomForestClassifier>,
    "xgboost":       <XGBClassifier>,
    "lstm":          <Keras Sequential model>,
    "lstm_scaler":   <MinMaxScaler>,
}
```

Route handlers access it with:
```python
models = getattr(request.app.state, "models", {})
model  = models.get("xgboost")
```

This means models are loaded once at startup and stay in RAM — no disk access per request.

---

## How the Stream Works

`stream.py` defines a `StreamState` class:

```python
class StreamState:
    df:     pd.DataFrame   # entire labeled_dataset.csv in RAM
    cursor: int            # index of next row to serve
    total:  int            # len(df)

    def next_row(self):
        row = self.df.iloc[self.cursor]
        self.cursor = (self.cursor + 1) % self.total  # wrap at end
        return row
```

The DataFrame is loaded at startup and never re-read from disk. Each `/stream/next` call is a single pandas iloc lookup — very fast.

---

## How the LSTM Generator Works

The `WindowGenerator` class in `train_lstm.py` inherits from `keras.utils.Sequence`:

```python
class WindowGenerator(Sequence):
    def __len__(self):
        return ceil(self.n_sequences / self.batch_size)

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, self.n_sequences)
        X_batch = np.stack([self.X[i:i+window] for i in range(start, end)])
        y_batch = self.y[start+window : end+window]
        return X_batch, y_batch
```

Keras calls `__getitem__(batch_idx)` automatically during training. Memory stays at ~750 KB per batch regardless of dataset size.

---

## Adding a New API Endpoint

1. Add the route to `app/api/predict.py` (or create a new router file)
2. Register it in `main.py` with `app.include_router()`
3. Add a Pydantic response model to `app/schemas.py` if needed

Example:
```python
@router.get("/feature-importance")
def feature_importance(request: Request):
    model = request.app.state.models.get("random_forest")
    if model is None:
        raise HTTPException(503, "RF not loaded")
    return {
        "features": model.feature_names_in_.tolist(),
        "scores":   model.feature_importances_.tolist()
    }
```

---

## Adding a New ML Model

1. Create `app/ml/train_yourmodel.py` following the same pattern
2. Save the model with `joblib.dump()` or `model.save()`
3. Add loading logic in `app/utils/model_loader.py`
4. Add the key to `MODEL_DISPLAY` in `app/api/predict.py`
5. Add the prediction branch in the `predict()` route handler

---

## Frontend Structure

```
frontend/src/
├── app/
│   ├── globals.css          ← JetBrains Mono font + Tailwind + CSS animations
│   ├── layout.tsx           ← Root HTML shell
│   ├── page.tsx             ← Home page — all state lives here
│   └── live/page.tsx        ← Live monitoring — setInterval polling
│
├── components/
│   ├── AlertBox.tsx         ← Green/red result panel
│   ├── Header.tsx           ← Sticky bar with health indicator + clock
│   ├── KpiCard.tsx          ← Single metric tile
│   ├── KpiChart.tsx         ← Recharts bar chart (6 normalised KPIs)
│   ├── KpiInputForm.tsx     ← 18 grouped inputs + preset buttons
│   ├── LiveChart.tsx        ← Dual-axis scrolling line chart
│   ├── ModelComparison.tsx  ← Metrics table + bar chart + differences panel
│   ├── ProbabilityGauge.tsx ← SVG arc gauge (green → amber → red)
│   └── StreamProgress.tsx   ← Dataset progress bar
│
├── lib/api.ts               ← All Axios calls (checkHealth, predictRisk, fetchNextRow, etc.)
└── types/index.ts           ← TypeScript interfaces
```

---

## State Management

No Redux. All state lives in the page component using `useState`:

`page.tsx` owns: `kpiValues`, `prediction`, `status`, `selectedModel`, `comparisonData`, `modelLoaded`, `errorMsg`

`live/page.tsx` owns: `currentRow`, `history`, `isRunning`, `stats`, `modelLoaded`, `errorMsg`

The `setInterval` in `live/page.tsx` is stored in a `useRef` so it survives re-renders without being recreated. The cleanup function clears it on component unmount.

---

## Adding a New Frontend Page

1. Create `src/app/mypage/page.tsx`
2. The route `/mypage` is automatically available — no router config needed
3. Add a `Link href="/mypage"` somewhere to navigate to it

---

## Configuration

**Backend paths** — set at the top of each script as `pathlib.Path` constants:
```python
RAW_DATA_DIR       = Path("data") / "raw"
CLEANED_DATA_PATH  = Path("data") / "processed" / "cleaned_dataset.csv"
LABELED_DATA_PATH  = Path("data") / "processed" / "labeled_dataset.csv"
```

**Frontend API URL** — configured in two places:
- `next.config.js` — rewrites `/api/*` to `${NEXT_PUBLIC_BACKEND_URL}/api/*`
- `src/lib/api.ts` — `baseURL: "/api"` (uses the rewrite)

In Docker, `NEXT_PUBLIC_BACKEND_URL=http://backend:8000`. Locally it falls back to `http://localhost:8000`.

---

## Useful Commands

```bash
# Backend
uvicorn main:app --reload              # start dev server
uvicorn main:app --reload --port 8001  # different port

# Frontend
npm run dev                             # start dev server
npm run build                           # build for production
npm run lint                            # TypeScript + ESLint check

# ML pipeline (from backend/)
python app\ml\clean_dataset.py
python app\ml\label_dataset.py
python app\ml\train_random_forest.py
python app\ml\train_xgboost.py
python app\ml\train_lstm.py

# Docker
docker compose up --build
docker compose exec backend sh run_training.sh
docker compose restart backend
docker compose down
```
