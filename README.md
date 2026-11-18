# PathSnap — Distributed Map-Matching & Route Reconstruction

End-to-end machine learning and Hidden Markov Model (HMM) map-matching system (`UE24CS352A: Machine Learning`, PES University) that reconstructs continuous vehicle routes from noisy GPS trajectories and serves real-time predictions across a two-laptop FastAPI + React architecture.

## System Pipeline & Architecture

```text
Raw (.track, .nodes, .arcs)
  │
  ▼  preprocessing/
Metric local-meter tangent-plane projection, gap/zero-length guards, adaptive cKDTree (K=6), 20 leakage-free features
  │
  ▼  models/ & artifacts/models/
Bayesian-tuned HistGradientBoostingClassifier (GP-EI) → Calibrated candidate emission probabilities
  │
  ▼  route_reconstruction/ & artifacts/hmm/
Distance-based HMM + Viterbi decoding + physically constrained intermediate road recovery
  │
  ▼  evaluation/ & backend/ & frontend/
Server-side route evaluation (ground truth isolated from model & UI) → FastAPI (Laptop 1) ↔ React UI (Laptop 2)
```

## Key Verified Results

| Stage | Key Outcome | Evidence |
| :--- | :--- | :--- |
| **Candidate Generation (`K=6`)** | **`99.32%` coverage** (`0.68%` miss rate on `student_test`) | [preprocessing/spatial_pipeline.py](preprocessing/spatial_pipeline.py) |
| **5-Model Comparison** | Selected **`HistGradientBoosting`** (`87.90%` Top-1, `0.8771` Seq F1, `0.65 MB`) over LR, KNN, DT, RF | [experiments/results/section_3_1_model_comparison.csv](experiments/results/section_3_1_model_comparison.csv) |
| **GP-EI Bayesian Tuning** | Improved CV Macro F1 (`0.8608 → 0.8659`), Test Macro F1 (`0.8928 → 0.8998`), Log Loss (`0.1672 → 0.1583`) | [experiments/results/tuned_vs_untuned_comparison.csv](experiments/results/tuned_vs_untuned_comparison.csv) |
| **Distance HMM + Recovery** | Boosted `student_test` accuracy **`87.90% → 90.79%`**, continuity **`92.12% → 97.85%`**, Road-Overlap F1 **`0.9455`** | [experiments/results/phase2_overall_metrics.json](experiments/results/phase2_overall_metrics.json) |

## Repository Structure

- [preprocessing/](preprocessing/) — Data loading, cleaning diagnostics, road-network analysis, metric `cKDTree` candidate generation, and 20-feature extraction.
- [models/](models/) — 5 ML model families (+ optional MLP), `GroupKFold` CV, 12 metrics, GP-EI Bayesian optimisation, plots, and training orchestrator.
- [optimisation/](optimisation/) — Optuna TPE Bayesian optimisation framework and experiment tracking.
- [route_reconstruction/](route_reconstruction/) — Distance-based HMM, Viterbi decoder, Dijkstra shortest paths, and constrained intermediate road recovery.
- [evaluation/](evaluation/) — Shared classification, cross-validation, model comparison, visualisation, and route overlap metrics.
- [visualisation/](visualisation/) — Multi-layer interactive Folium map generator and exported HTML route comparisons ([visualisation/maps/](visualisation/maps/)).
- [artifacts/](artifacts/) — Persisted production `.joblib` classifiers ([artifacts/models/](artifacts/models/)) and Optuna-tuned HMM parameters ([artifacts/hmm/](artifacts/hmm/)).
- [experiments/](experiments/) — Consolidated CSV/JSON metrics, Bayesian trial logs, exploration summary, and all 16 result PNGs ([experiments/results/](experiments/results/)).
- [backend/](backend/) & [frontend/](frontend/) — FastAPI inference & evaluation server (Laptop 1) and React + Vite evaluator UI (Laptop 2).
- [notebooks/](notebooks/) — Exploratory Data Analysis notebook ([01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb)).
- [report/](report/) — Consolidated team technical report ([report/FINAL_TECHNICAL_REPORT.md](report/FINAL_TECHNICAL_REPORT.md)).
- [tests/](tests/) — Automated unit and API integration tests (`pytest`).

## Two-Laptop Live Evaluation Setup & Architecture

Per the PathSnap deployment specification, live evaluation separates the ML backend from the UI across two laptops on the **same Mobile Hotspot or Wi-Fi/LAN**:

```text
┌──────────────────────────────────────────────────────────┐        HTTP REST (LAN IPv4 :8000)        ┌──────────────────────────────────────────────┐
│  LAPTOP 1 — Model, Pipeline & API Server (Backend)       │ ◄────────────────────────────────────────► │  LAPTOP 2 — Evaluator UI (Frontend)          │
│                                                          │   GET  /health                             │                                              │
│  1. FastAPI (`uvicorn --host 0.0.0.0 --port 8000`)       │   GET  /trips   (10 evaluator_test IDs)    │  • React + Vite (`npm run dev -- --port 5173`)│
│  2. Active Dataset: `evaluator_test/` (git-ignored)      │   POST /predict { "trip_id": "..." }       │  • `VITE_API_BASE_URL=http://<LAPTOP_1_IP>:8000`
│  3. Inference runs strictly with `route=None`:           │   Returns: predicted_route, metrics,       │  • Displays 10 reserved trip options,        │
│     • `extract_trip_candidates_and_features(..., K=6)`   │            execution_time, map_html        │    interactive Folium map (NO ground truth), │
│     • `artifacts/models/selected_best_model_tuned.joblib`│                                            │    server-computed metrics & run history     │
│     • Distance HMM + Viterbi + Constrained Recovery      │                                            └──────────────────────────────────────────────┘
│  4. Server-side evaluation reads `.route` post-predict   │
│  5. Prints live `[GET /trips]` & `[POST /predict]` logs  │
└──────────────────────────────────────────────────────────┘
```

### 1. Laptop 1 Setup (Backend, Model, HMM & Server-Side Evaluator)
1. Clone/pull `main`, create/activate a virtual environment, install dependencies, and run tests:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m pytest -v
   ```
2. Place `evaluator_test/` (all 10 reserved trips `00000015`..`00000094`) in the repository root on Laptop 1 (`evaluator_test/` is git-ignored so `.route` labels are never pushed). `backend/services.py` resolves datasets in priority order: `PATHSNAP_TRIP_DIR` $\rightarrow$ `./evaluator_test` $\rightarrow$ `./Student folder/student_test`.
3. Obtain Laptop 1's IPv4 address (`ipconfig`) and allow TCP port `8000` through Windows Firewall if running on a Public/Hotspot profile:
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
   ```
4. Start FastAPI on all interfaces and keep the terminal visible to display live logs:
   ```powershell
   $env:PATHSNAP_TRIP_DIR = ".\evaluator_test"
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

### 2. Laptop 2 Setup (React Evaluator Interface)
1. Connect Laptop 2 to the same Mobile Hotspot / Wi-Fi network as Laptop 1 and verify reachability:
   ```powershell
   Invoke-RestMethod http://<LAPTOP_1_IP>:8000/health
   # Expected output: @{status=healthy}
   ```
2. Install UI dependencies and start Vite pointing to Laptop 1:
   ```powershell
   cd frontend
   npm install
   $env:VITE_API_BASE_URL = "http://<LAPTOP_1_IP>:8000"
   npm run dev -- --host 0.0.0.0 --port 5173
   ```
3. Open **`http://localhost:5173`** on Laptop 2, select any evaluation trip, click **Reconstruct Route**, and observe live logs on Laptop 1 alongside the interactive Folium map and metrics on Laptop 2.

### 3. Reproducing Training, Tuning & Phase 2 HMM Evaluation
```powershell
python -m models.train
python -m models.bayesian_tuning
python -m route_reconstruction.run_phase2_evaluation
```
