
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

For Windows PowerShell:

```powershell
$env:PATHSNAP_TRIP_DIR="./Student folder/student_test"
```

Use the actual approved dataset directory on the server laptop.

## Start the Backend

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Backend health status |
| GET | /trips | Available trip IDs |
| POST | /predict | Route reconstruction request |
| GET | /map/{trip_id} | Interactive route map |
| GET | /docs | Swagger API documentation |

## Two-Laptop Deployment

Laptop 1 hosts the backend, trained model, road network,
HMM/Viterbi decoder and evaluation data.

Laptop 2 sends requests using the backend laptop's IPv4
address through the React evaluator interface.

Both laptops must have network connectivity to each other.

## Data Privacy

Reference routes must remain on the server.
Only predictions, visualisations and computed evaluation
metrics are returned to the evaluator interface.

## Implementation Status

- Live end-to-end pipeline integrated (`backend/services.py`): feature extraction (`preprocessing.spatial_pipeline`), Bayesian-tuned `HistGradientBoosting` emission scoring (`artifacts/models/selected_best_model_tuned.joblib`), distance-based HMM/Viterbi + constrained recovery (`route_reconstruction`), and server-side evaluation.
- Verified across all 10 `evaluator_test` and 10 `student_test` trips with terminal inference logs on Laptop 1.
