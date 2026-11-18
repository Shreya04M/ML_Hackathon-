# PathSnap Backend

The FastAPI backend manages trip discovery, cross-laptop REST API communication, Folium route visualisation, live ML + HMM inference, and isolated server-side evaluation.

## Implemented Features

- Backend health monitoring (`GET /health`)
- Dynamic discovery of approved evaluation trips (`GET /trips`)
- Live 20-feature extraction, tuned `HistGradientBoosting` emission scoring, Distance HMM/Viterbi decoding, and constrained recovery (`POST /predict`)
- Ground-truth-free Folium route map rendering (`GET /map/{trip_id}` and `POST /predict` `map_html`)
- Strict server-side accuracy & route overlap scoring (`.route` never exposed to model or UI)
- Structured Laptop 1 terminal telemetry (`[GET /trips]`, `[MODEL LOAD]`, `[POST /predict] START/COMPLETE`, `[MAP RENDER]`)
- Open CORS policy (`allow_origins=["*"]`) for reliable two-laptop LAN/Hotspot evaluation

## Installation

From the project root:

```powershell
python -m pip install -r requirements.txt
```

## Dataset Configuration

[backend/services.py](./services.py) resolves datasets in priority order:
1. `PATHSNAP_TRIP_DIR` environment variable (if set)
2. `./evaluator_test` (when present on Laptop 1 — intentionally git-ignored)
3. `./Student folder/student_test` (development fallback)

```powershell
$env:PATHSNAP_TRIP_DIR = ".\evaluator_test"
```

## Start the Backend (Laptop 1)

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Backend health status |
| `GET` | `/trips` | Available trip IDs |
| `POST` | `/predict` | Live route reconstruction request |
| `GET` | `/map/{trip_id}` | Interactive raw Folium GPS map |
| `GET` | `/docs` | Swagger API documentation |
