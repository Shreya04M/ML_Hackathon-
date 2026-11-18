# PathSnap Evaluator UI

React + Vite evaluator interface designed for Laptop 2 in the two-laptop evaluation architecture.

## Implemented Features

- Live backend connection monitor (`GET /health`)
- Trip selector with all 10 approved evaluation trips (`TripSelector.jsx`)
- Ground-truth-free Folium interactive route map viewer (`RouteMap.jsx`)
- Server-computed KPI dashboard, predicted arc sequence viewer & copy action, and session history (`Results.jsx`, `Home.jsx`)
- Built-in Model Benchmarks and Pipeline Architecture reference tabs (`BenchmarksView.jsx`, `ArchitectureView.jsx`)

## Two-Laptop Deployment (Laptop 2 — UI Client)

```powershell
cd frontend
npm install
Invoke-RestMethod http://<LAPTOP_1_IP>:8000/health
$env:VITE_API_BASE_URL = "http://<LAPTOP_1_IP>:8000"
npm run dev -- --host 0.0.0.0 --port 5173
```

Open **`http://localhost:5173`** on Laptop 2. Both `VITE_API_BASE_URL` and `VITE_API_URL` are supported by [src/services/api.js](src/services/api.js).

## Production Build

```powershell
npm run build
```
