
# PathSnap Evaluator UI

Implemented by Member 4 using React and Vite.

## Overview

The evaluator interface connects to the FastAPI backend
to select trips, request route reconstruction and display
route visualisations and evaluation results.

## Implemented Features

- Backend connection indicator
- Dynamic trip selection
- Route reconstruction request button
- Loading and error handling
- Interactive Folium maps
- Raw GPS trajectory display
- Road network overlay
- Predicted route display support
- Evaluation metrics panel

## Evaluation Metrics

The interface supports displaying:

- Accuracy
- Route Precision
- Route Recall
- Route F1
- Jaccard Similarity
- Execution Time

Metrics are computed on the backend, not the frontend.

## Installation

From the project root:

```powershell
cd frontend
npm install
```

## Local Development

```powershell
npm run dev
```

Open `http://localhost:5173`.

## Two-Laptop Deployment (Laptop 2 — UI Client)

Connect Laptop 1 (FastAPI server) and Laptop 2 (React UI) to the same Mobile Hotspot or Wi-Fi/LAN.

On Laptop 1, obtain its IPv4 address with `ipconfig` and ensure FastAPI is listening on `--host 0.0.0.0 --port 8000`.

On Laptop 2, verify Laptop 1 is reachable and launch Vite pointing to Laptop 1:

```powershell
cd frontend
Invoke-RestMethod http://<LAPTOP_1_IP>:8000/health
$env:VITE_API_BASE_URL = "http://<LAPTOP_1_IP>:8000"
npm run dev -- --host 0.0.0.0 --port 5173
```

*(Both `VITE_API_BASE_URL` and `VITE_API_URL` are read by [src/services/api.js](./src/services/api.js), or can be placed inside `frontend/.env`.)*

Open `http://localhost:5173` on Laptop 2.

## Production Build

```powershell
npm run build
```

## Implementation Status

- Full UI integrated with the live ML + Distance HMM/Viterbi backend across all 3 views (`Live Evaluation`, `Model Benchmarks`, and `Pipeline Architecture`).
- Verified with `npm run build` (`vite build` production bundle) and live FastAPI endpoints (`GET /health`, `GET /trips`, `POST /predict`).
