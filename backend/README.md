
# PathSnap Backend

Implemented by Member 4.

## Overview

The FastAPI backend manages trip discovery, API communication,
route visualisation and server-side prediction integration.

## Implemented Features

- Backend health monitoring
- Dynamic discovery of available trips
- Prediction request and response schemas
- Raw GPS trajectory visualisation
- Road network visualisation
- Predicted route overlay support
- API error handling
- Configurable dataset directory
- CORS configuration for the evaluator UI

## Installation

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Dataset Configuration

`backend/services.py` resolves datasets in priority order:
1. `PATHSNAP_TRIP_DIR` environment variable (if set)
2. `./evaluator_test` (when present on Laptop 1 — intentionally git-ignored)
3. `./Student folder/student_test` (development fallback)

To point explicitly to `evaluator_test` in Windows PowerShell:

```powershell
$env:PATHSNAP_TRIP_DIR = ".\evaluator_test"
```

## Start the Backend (Laptop 1)

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Keep the Laptop 1 terminal visible during evaluation. Each API call emits structured inference logs alongside Uvicorn HTTP access logs:
- `[GET /trips] data_dir=... | discovered=10 | trips=[...]`
- `[MODEL LOAD] Loaded tuned classifier artifact: ...selected_best_model_tuned.joblib`
- `[POST /predict] START data_dir=... | trip_id=... | route=None (ground truth isolated from model)`
- `[MAP RENDER] trip_id=... | gps_points=... | road_arcs=... | predicted_arcs=... | ground_truth_exposed=False`
- `[POST /predict] COMPLETE ... | acc=... | route_f1=... | precision=... | recall=... | jaccard=... | road_overlap_f1=... | latency=...s`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Backend health status |
| GET | /trips | Available trip IDs |
| POST | /predict | Live route reconstruction request |
| GET | /map/{trip_id} | Interactive raw Folium GPS map |
| GET | /docs | Swagger API documentation |

## Two-Laptop Deployment & CORS

Laptop 1 hosts the backend, trained model (`artifacts/models/selected_best_model_tuned.joblib`), road-network loader, Distance HMM/Viterbi decoder, and server-side evaluation data (`evaluator_test/`).

`backend/main.py` configures `CORSMiddleware` with `allow_origins=["*"]` and `allow_methods=["*"]` so Laptop 2 can connect over Wi-Fi / Mobile Hotspot from `http://localhost:5173`, `127.0.0.1`, or a LAN IP without CORS errors. Ensure Windows Firewall permits inbound TCP port `8000` on Laptop 1.

## Data Privacy

Reference `.route` routes remain strictly server-side on Laptop 1 (`route=None` passed to feature extraction and prediction, and ground truth is omitted from Folium HTML and JSON responses). Only predictions, interactive maps without ground truth, latency, and computed evaluation metrics are returned to Laptop 2.

## Implementation Status

- Live end-to-end pipeline integrated ([backend/services.py](./services.py)): feature extraction (`preprocessing.spatial_pipeline`), Bayesian-tuned `HistGradientBoosting` emission scoring (`artifacts/models/selected_best_model_tuned.joblib`), distance-based HMM/Viterbi + constrained recovery (`route_reconstruction`), structured Laptop 1 terminal telemetry, and server-side evaluation.
- Verified with `TestClient(app)` and across all 10 `evaluator_test` and 10 `student_test` trips.
