# PathSnap – Distributed Map-Matching & Route Reconstruction

End-to-end machine learning and Hidden Markov Model (HMM) map-matching system (`UE24CS352A: Machine Learning`, PES University) that reconstructs continuous vehicle routes from noisy GPS trajectories and serves real-time predictions across a two-laptop FastAPI + React architecture.

## System Pipeline & Architecture

```text
Raw (.track, .nodes, .arcs)
  │
  ▼  preprocessing/ (Member 1)
Metric local-meter projection, gap/zero-length guards, adaptive cKDTree (K=6), 20 leakage-free features
  │
  ▼  models/ & artifacts/ (Member 2)
Bayesian-tuned HistGradientBoostingClassifier (GP-EI) → Calibrated candidate emission probabilities
  │
  ▼  route_reconstruction/ (Member 3)
Distance-based HMM + Viterbi decoding + physically constrained intermediate road recovery
  │
  ▼  evaluation/ & backend/ & frontend/ (Member 4)
Server-side route evaluation (ground truth isolated from model/UI) → FastAPI (Laptop 1) ↔ React UI (Laptop 2)
```

## Key Verified Results

| Stage | Key Outcome | Evidence |
| :--- | :--- | :--- |
| **Candidate Generation (`K=6`)** | **`99.32%` coverage** (`0.68%` miss rate on `student_test`) | `preprocessing/spatial_pipeline.py` |
| **5-Model Comparison** | Selected **`HistGradientBoosting`** (`87.90%` Top-1, `0.8771` Seq F1, `0.65 MB`) over LR, KNN, DT, RF | `models/artifacts/results/section_3_1_model_comparison.csv` |
| **GP-EI Bayesian Tuning** | Improved CV Macro F1 (`0.8608 → 0.8659`), Test Macro F1 (`0.8928 → 0.8998`), Log Loss (`0.1672 → 0.1583`) | `models/bayesian_tuning.py`, `tuned_vs_untuned_comparison.csv` |
| **Distance HMM + Recovery** | Boosted `student_test` accuracy **`87.90% → 90.79%`**, continuity **`92.12% → 97.85%`**, Road-Overlap F1 **`0.9455`** | `route_reconstruction/experiments/phase2_improved/overall_metrics.json` |

## Repository Structure

- `preprocessing/` — Data loading, cleaning, road-network analysis, metric `cKDTree` candidate generation, and 20-feature extraction.
- `visualisation/` — Interactive 4-layer Folium maps (`visualisation/maps/`).
- `models/` — 5 ML model families (+ MLP), `GroupKFold` CV, 12 evaluation metrics, GP-EI tuning, plots, and Phase 1/Story reports.
- `optimisation/` — Optuna TPE Bayesian optimisation framework.
- `route_reconstruction/` — Distance-based HMM, Viterbi decoder, Dijkstra shortest paths, and constrained intermediate road recovery.
- `evaluation/` — Shared classification, CV, and route evaluation metrics.
- `artifacts/` — Production-ready tuned `.joblib` classifier and optimised HMM parameter JSON.
- `backend/` & `frontend/` — FastAPI backend (Laptop 1) and React + Vite evaluator UI (Laptop 2).
- `report/` — Consolidated final technical report ([report/FINAL_TECHNICAL_REPORT.md](report/FINAL_TECHNICAL_REPORT.md)).
- `tests/` — Automated unit and API integration tests (`pytest`).

## Two-Laptop Live Evaluation Setup & Architecture

Per the PathSnap specifications, the final demonstration separates the ML server from the evaluator UI across two laptops connected to the **same Mobile Hotspot or Wi-Fi/LAN**:

```text
┌──────────────────────────────────────────────────────────┐        HTTP REST (LAN IPv4 :8000)        ┌──────────────────────────────────────────────┐
│  LAPTOP 1 — Model, Pipeline & API Server (Backend)       │ ◄────────────────────────────────────────► │  LAPTOP 2 — Evaluator UI (Frontend)          │
│                                                          │   GET  /health                             │                                              │
│  1. FastAPI (`uvicorn --host 0.0.0.0 --port 8000`)       │   GET  /trips   (10 evaluator_test IDs)    │  • React + Vite (`npm run dev -- --port 5173`)│
│  2. Active Dataset: `evaluator_test/` (git-ignored)      │   POST /predict { "trip_id": "..." }       │  • `VITE_API_BASE_URL=http://<LAPTOP_1_IP>:8000`
│  3. Inference runs strictly with `route=None`:           │   Returns: predicted_route, metrics,       │  • Displays 10 reserved trip options,        │
│     • `extract_trip_candidates_and_features(..., K=6)`   │            execution_time, map_html        │    interactive Folium map (NO ground truth), │
│     • `selected_best_model_tuned.joblib` (HistGB GP-EI)  │                                            │    server-computed metrics & run history     │
│     • Distance HMM + Viterbi + Constrained Recovery      │                                            └──────────────────────────────────────────────┘
│  4. Server-side evaluation reads `.route` post-predict   │
│  5. Prints live `[GET /trips]` & `[POST /predict]` logs  │
└──────────────────────────────────────────────────────────┘
```

### 1. Laptop 1 Setup (Your Laptop — Backend, Model, HMM & Server-Side Evaluator)
1. Clone/pull `main`, create/activate a virtual environment, and verify tests:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m pytest -v
   ```
2. Ensure the reserved `evaluator_test/` directory (containing all 10 trips `00000015`..`00000094`) is copied to the repository root on Laptop 1 (`evaluator_test/` is intentionally git-ignored so `.route` labels are never pushed to GitHub).  
   `backend/services.py` resolves datasets in priority order: `PATHSNAP_TRIP_DIR` env var $\rightarrow$ `./evaluator_test` $\rightarrow$ `./Student folder/student_test`.
3. Find Laptop 1's IPv4 address (`ipconfig` $\rightarrow$ look for Wi-Fi / Hotspot IPv4, e.g. `192.168.137.1` or `192.168.x.x`). If Windows Firewall blocks inbound LAN requests on a new network, allow TCP port `8000` in an elevated PowerShell:
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
4. Start the FastAPI server on all network interfaces and keep this CMD/PowerShell terminal visible for the evaluators:
   ```powershell
   $env:PATHSNAP_TRIP_DIR = ".\evaluator_test"
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
5. **What to show the evaluator on Laptop 1's CMD window**: every request from Laptop 2 prints live timestamps, Uvicorn HTTP status (`GET /health`, `GET /trips`, `POST /predict`), model artifact load (`[MODEL LOAD]`), non-leakage confirmation (`route=None (ground truth isolated from model)`), Folium render stats (`ground_truth_exposed=False`), and live server-side evaluation metrics (`acc`, `route_f1`, `precision`, `recall`, `jaccard`, `road_overlap_f1`, `latency`).

### 2. Laptop 2 Setup (Teammate Laptop — React Evaluator Interface)
1. Connect Laptop 2 to the **same hotspot / Wi-Fi network** as Laptop 1 and verify reachability in PowerShell before launching the UI:
   ```powershell
   Invoke-RestMethod http://<LAPTOP_1_IP>:8000/health
   # Expected output: @{status=healthy}
   ```
2. Configure and start the React + Vite frontend (`CORS` on Laptop 1 allows all origins cleanly):
   ```powershell
   cd frontend
   npm install
   $env:VITE_API_BASE_URL = "http://<LAPTOP_1_IP>:8000"
   npm run dev -- --host 0.0.0.0 --port 5173
   ```
   *(Alternatively, place `VITE_API_BASE_URL=http://<LAPTOP_1_IP>:8000` inside `frontend/.env` before running `npm run dev`.)*
3. Open **`http://localhost:5173`** on Laptop 2:
   - Confirm the top status pill shows **Backend Connected** and all **10 evaluation trips** appear.
   - Let the evaluator choose **any trip**, click **Reconstruct Route**, watch Laptop 1's terminal execute the live pipeline (`~1–10 s` depending on trip length), and inspect the interactive Folium map and KPI cards on Laptop 2.

### 3. Reproducing Training, Tuning & Phase 2 HMM Evaluation
```powershell
python -m models.train
python -m models.bayesian_tuning
python -m route_reconstruction.run_phase2_evaluation
```
