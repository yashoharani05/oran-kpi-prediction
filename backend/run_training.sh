#!/bin/sh
# =============================================================================
# run_training.sh
#
# PURPOSE:
#   Run the full ML training pipeline inside the Docker backend container.
#   This script is run ONCE after first starting the containers,
#   before the backend can serve predictions.
#
# HOW TO USE (from the project root folder):
#   docker compose exec backend sh run_training.sh
#
# WHY IS THIS NEEDED?
#   The trained model files (.pkl, .keras) are too large to bundle in Git.
#   Instead, you train them once inside the container.
#   The models are saved to /app/models/ which is mounted to ./backend/models/
#   on your PC, so they persist even after the container stops.
#
# TOTAL TIME: approximately 3-5 minutes (TF loads slowly on first run).
# =============================================================================

echo ""
echo "=================================================="
echo "  O-RAN KPI Prediction — ML Training Pipeline"
echo "=================================================="
echo ""

echo "[1/5] Cleaning raw dataset..."
python app/ml/clean_dataset.py
echo ""

echo "[2/5] Engineering features and creating labels..."
python app/ml/label_dataset.py
echo ""

echo "[3/5] Training Random Forest..."
python app/ml/train_random_forest.py
echo ""

echo "[4/5] Training XGBoost..."
python app/ml/train_xgboost.py
echo ""

echo "[5/5] Training LSTM..."
python app/ml/train_lstm.py
echo ""

echo "=================================================="
echo "  Training complete!"
echo "  Models saved to /app/models/"
echo "  Restart the backend to load the new models:"
echo "    docker compose restart backend"
echo "=================================================="
echo ""
