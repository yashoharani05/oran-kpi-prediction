# Frontend — O-RAN KPI Prediction xApp Dashboard

Next.js + TypeScript + Tailwind CSS. Two pages: manual prediction and live monitoring.

---

## Folder Structure

```
frontend/
├── package.json               ← Dependencies: Next.js, Tailwind, Axios, Recharts
├── tsconfig.json              ← TypeScript config (@/ path alias → src/)
├── next.config.js             ← Standalone output + API proxy to backend
├── tailwind.config.js         ← JetBrains Mono + Inter fonts, dark palette
├── postcss.config.js          ← Required by Tailwind
├── Dockerfile                 ← Two-stage Docker build (builder + runner)
│
└── src/
    ├── app/
    │   ├── globals.css        ← Google Fonts import + Tailwind directives + animations
    │   ├── layout.tsx         ← Root HTML shell
    │   ├── page.tsx           ← / — Manual prediction dashboard
    │   └── live/
    │       └── page.tsx       ← /live — Live monitoring (2 s auto-refresh)
    │
    ├── components/
    │   ├── AlertBox.tsx       ← Green/red result panel + recommendation
    │   ├── Header.tsx         ← Sticky top bar with health indicator + clock
    │   ├── KpiCard.tsx        ← Single metric tile (label, value, colour coding)
    │   ├── KpiChart.tsx       ← Recharts bar chart — 6 KPI health scores
    │   ├── KpiInputForm.tsx   ← 18 grouped inputs + preset buttons
    │   ├── LiveChart.tsx      ← Scrolling dual-axis line chart
    │   ├── ModelComparison.tsx← Metrics table + bar chart + differences panel
    │   ├── ProbabilityGauge.tsx ← SVG arc gauge (green → amber → red)
    │   └── StreamProgress.tsx ← Dataset position progress bar
    │
    ├── lib/
    │   └── api.ts             ← All Axios calls: health, predict, stream, comparison
    │
    └── types/
        └── index.ts           ← TypeScript interfaces (KpiValues, PredictionResult, etc.)
```

---

## Pages

| URL                        | What it shows                                               |
| -------------------------- | ----------------------------------------------------------- |
| http://localhost:3000      | Manual prediction: enter KPIs, choose model, run prediction |
| http://localhost:3000/live | Live monitoring: auto-streams CSV rows every 2 seconds      |

---

## Setup and Run (Windows, without Docker)

### Requirements

- Node.js 18 or higher — https://nodejs.org/en (choose LTS)

### Step 1 — Install dependencies

```
npm install
```

### Step 2 — Start dev server

```
npm run dev
```

Dashboard available at: **http://localhost:3000**

The backend must be running at `http://localhost:8000` for predictions to work.

---

## Setup and Run (Docker)

```
docker compose up --build
```

The frontend container starts automatically after the backend is healthy.

---

## How API Calls Work

All API calls go through `src/lib/api.ts` using an Axios instance with `baseURL: "/api"`.

`next.config.js` rewrites `/api/*` to the backend:

- Local dev: `http://localhost:8000/api/*`
- Docker: `http://backend:8000/api/*` (via `NEXT_PUBLIC_BACKEND_URL` env variable)

This means no hardcoded ports anywhere in component code.

---

## Available Scripts

| Command         | What it does                     |
| --------------- | -------------------------------- |
| `npm run dev`   | Start dev server with hot reload |
| `npm run build` | Build for production             |
| `npm run start` | Serve the production build       |
| `npm run lint`  | TypeScript + ESLint check        |

---

## Common Errors

| Error                     | Cause                       | Fix                                           |
| ------------------------- | --------------------------- | --------------------------------------------- |
| `Could not reach the API` | Backend not running         | Start `uvicorn main:app --reload` in backend/ |
| `Model Offline` in header | `/api/health` returns false | Run training scripts, restart backend         |
| Port 3000 in use          | Another process             | `npm run dev -- --port 3001`                  |
| Styles missing            | Tailwind not set up         | Delete `node_modules/`, run `npm install`     |
