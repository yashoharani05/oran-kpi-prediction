# =============================================================================
# backend/Dockerfile
#
# Builds the FastAPI backend into a Docker image.
#
# WHAT IS THIS FILE?
#   A Dockerfile is a recipe. Each line is an instruction that Docker follows
#   to build an image — a self-contained package with Python, all dependencies,
#   and the app code already installed.
#
# HOW TO BUILD MANUALLY (not needed with docker-compose):
#   docker build -t oran-backend .
#
# HOW TO RUN MANUALLY (not needed with docker-compose):
#   docker run -p 8000:8000 oran-backend
# =============================================================================

# -----------------------------------------------------------------------
# Stage 1: base image
# -----------------------------------------------------------------------
# We start from the official Python 3.11 slim image.
# "slim" means it has only what Python needs — no GUI apps, no compilers.
# This keeps the image small (~150 MB base vs ~900 MB full).
FROM python:3.11-slim

# -----------------------------------------------------------------------
# Set the working directory inside the container.
# All subsequent commands run from /app.
# -----------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------
# System dependencies needed by TensorFlow and Pandas.
# libgomp1: OpenMP runtime (parallel computing, used by XGBoost/sklearn)
# We install these before copying code so Docker can cache this layer.
# If only app code changes, Docker reuses this cached layer → faster rebuilds.
# -----------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# Copy requirements.txt first (before the rest of the code).
# WHY: Docker caches each layer. If requirements.txt hasn't changed,
# Docker skips pip install and reuses the cached layer.
# This makes rebuilds much faster when you only change Python code.
# -----------------------------------------------------------------------
COPY requirements.txt .

# -----------------------------------------------------------------------
# Install Python dependencies.
# --no-cache-dir: don't store pip's download cache (saves image space).
# -----------------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------
# Copy the rest of the application code into the container.
# -----------------------------------------------------------------------
COPY . .

# -----------------------------------------------------------------------
# Expose port 8000 so Docker knows this container listens there.
# This is documentation only — the actual port mapping is in docker-compose.yml.
# -----------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------
# Set environment variable to suppress TensorFlow startup messages.
# -----------------------------------------------------------------------
ENV TF_CPP_MIN_LOG_LEVEL=3

# -----------------------------------------------------------------------
# The command to run when the container starts.
# --host 0.0.0.0 : listen on all network interfaces (required inside Docker)
# --port 8000    : the port uvicorn listens on
# (no --reload because we don't need live reload in Docker)
# -----------------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
