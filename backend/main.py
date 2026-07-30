# =============================================================================
# main.py  —  FastAPI entry point
#
# HOW TO RUN:
#   uvicorn main:app --reload
# =============================================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
from app.api.stream  import router as stream_router, stream_state
from app.utils.model_loader import load_all_models

DATASET_PATH = "data/processed/labeled_dataset.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[Startup] O-RAN KPI Prediction xApp API starting...")

    # Load both ML models into app.state.models dict
    app.state.models = load_all_models()

    # Load dataset for the live stream
    try:
        stream_state.load(DATASET_PATH)
        print(f"[Startup] Stream dataset loaded ({stream_state.total} rows).")
    except Exception as e:
        print(f"[Startup] Stream dataset error: {e}")

    print("[Startup] Server ready.\n")
    yield
    print("\n[Shutdown] Shutting down.")
    app.state.models = {}


app = FastAPI(
    title="O-RAN KPI Prediction xApp API",
    description=(
        "Endpoints:\n"
        "- GET  /api/health           — server + model status\n"
        "- POST /api/predict          — single prediction (?model=random_forest|xgboost|lstm)\n"
        "- GET  /api/comparison       — side-by-side metrics JSON\n"
        "- GET  /api/stream/next      — next CSV row + prediction\n"
        "- GET  /api/stream/status    — cursor position\n"
        "- GET  /api/stream/reset     — rewind to row 0"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api",        tags=["Prediction"])
app.include_router(stream_router,  prefix="/api/stream", tags=["Live Stream"])


@app.get("/", tags=["Root"])
def root():
    return {
        "project":  "ML-based KPI Prediction xApp for O-RAN",
        "status":   "running",
        "docs":     "http://localhost:8000/docs",
    }
