# Docker Setup Guide

Docker lets you run the entire project without installing Python or Node.js on your PC. One command starts everything.

---

## What is Docker?

Docker packages software into isolated "containers". Instead of installing Python 3.11, pip, Node.js 18, and dozens of packages, you install Docker and it handles everything inside containers. Your PC stays clean.

**Image** — a snapshot of an environment (the blueprint).
**Container** — a running instance of an image (the house built from blueprints).
**docker compose** — a tool that starts multiple containers together with one command.

---

## Step 1 — Install Docker Desktop (Windows)

1. Download from: **https://www.docker.com/products/docker-desktop/**
2. Run the installer (keep default settings)
3. When asked about WSL 2 backend, click Install
4. Restart your PC when prompted
5. Open Docker Desktop — wait for the green "Engine running" status

Verify:
```
docker --version
docker compose version
```

---

## Step 2 — Place Your CSV Files

Copy all your raw KPI CSV files into:
```
backend\data\raw\
```

---

## Step 3 — Build and Start

Open Command Prompt in the project root (where `docker-compose.yml` is):

```
docker compose up --build
```

What happens:
- Docker builds the Python backend image from `backend/Dockerfile`
- Docker builds the Next.js frontend image from `frontend/Dockerfile`
- Both containers start on a private network (`oran-network`)
- The frontend waits for the backend healthcheck to pass before starting

First run downloads base images and installs packages — takes 5–15 minutes depending on connection speed.

---

## Step 4 — Run ML Training (first time only)

Open a second Command Prompt and run:

```
docker compose exec backend sh run_training.sh
```

This runs all five ML scripts inside the backend container. Progress is printed as each script completes. Training 19M+ rows takes time — expect 30–60 minutes for all five scripts.

Model files are saved to `backend/models/` on your PC via the volume mount, so they survive `docker compose down`.

---

## Step 5 — Restart Backend to Load Models

```
docker compose restart backend
```

After restart, the backend loads the newly trained models from the volume.

---

## Step 6 — Open the Dashboard

Wait ~30 seconds for models to load, then open:
- **http://localhost:3000** — main dashboard
- **http://localhost:3000/live** — live monitoring
- **http://localhost:8000/docs** — API documentation

---

## Daily Workflow (after first setup)

Start:
```
docker compose up
```

Stop:
```
docker compose down
```

Or press Ctrl+C in the terminal running compose.

---

## All Commands

| Command | What it does |
|---|---|
| `docker compose up --build` | Build images and start both containers |
| `docker compose up` | Start using existing images (faster) |
| `docker compose down` | Stop and remove containers |
| `docker compose restart backend` | Restart only the backend |
| `docker compose restart frontend` | Restart only the frontend |
| `docker compose logs backend` | See backend logs |
| `docker compose logs -f` | Watch live logs from both |
| `docker compose exec backend sh` | Open a shell inside the backend |
| `docker compose exec backend sh run_training.sh` | Run ML training |
| `docker compose ps` | Show running containers |

---

## How the Containers Communicate

Docker Compose creates a private network called `oran-network`. Services find each other by name, not `localhost`:

```
Your browser (port 3000)
      │
      ▼
 oran-frontend container (Next.js)
      │
      │  /api/* rewrites to http://backend:8000/api/*
      │  ("backend" = Docker's DNS for the backend container)
      ▼
 oran-backend container (FastAPI, port 8000)
      │
      ▼
 ML models in RAM + labeled_dataset.csv
```

This is why `next.config.js` uses `NEXT_PUBLIC_BACKEND_URL=http://backend:8000` in Docker, but falls back to `http://localhost:8000` for local development.

---

## Volumes — Shared Files

Three folders are shared between your PC and the backend container:

| Your PC | Container path | Mode |
|---|---|---|
| `./backend/data/raw/` | `/app/data/raw/` | Read-only (you add CSVs, container reads them) |
| `./backend/data/processed/` | `/app/data/processed/` | Read-write (container writes cleaned CSVs) |
| `./backend/models/` | `/app/models/` | Read-write (container writes trained models) |

Files in these folders survive `docker compose down` — you only need to train once.

---

## Troubleshooting

**"Docker Engine not running"**
Open Docker Desktop and wait for the green status dot before running compose commands.

**Port 3000 or 8000 already in use**
Change in `docker-compose.yml`:
```yaml
ports:
  - "3001:3000"  # use 3001 on your PC
```

**"Model is not loaded" after starting**
Training hasn't been run yet. Run:
```
docker compose exec backend sh run_training.sh
docker compose restart backend
```

**Frontend shows "Could not reach the API"**
Backend may still be loading models (especially LSTM). Wait 60 seconds after `docker compose up` and refresh. Check status with `docker compose ps` — both should show `running (healthy)`.

**"No space left on device"**
Docker images use disk space. Clean up unused images:
```
docker system prune -a
```

**After changing source code**
```
docker compose up --build
```
Docker only rebuilds layers that changed — usually fast for code changes.

---

## File Structure Added by Docker

```
fyp-oran-kpi-prediction/
├── docker-compose.yml          ← Start everything: docker compose up --build
├── .dockerignore               ← Root-level exclusions
│
├── backend/
│   ├── Dockerfile              ← python:3.11-slim + requirements.txt
│   ├── .dockerignore           ← Excludes venv/, __pycache__/
│   └── run_training.sh         ← Runs all 5 ML scripts in sequence
│
└── frontend/
    ├── Dockerfile              ← Two-stage: node:18-alpine builder + runner
    ├── .dockerignore           ← Excludes node_modules/, .next/
    └── next.config.js          ← output: "standalone" + env-based API URL
```
