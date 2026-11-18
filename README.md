# PathSnap — Distributed Map-Matching & Route Reconstruction

**PathSnap** (`UE24CS352A: Machine Learning`, PES University) is an end-to-end GPS map-matching and sequential route reconstruction system. Given a noisy `~1 Hz` vehicle GPS trajectory (`.track`) and a directed road network (`.nodes`, `.arcs`), PathSnap projects coordinates into local tangent-plane meters, ranks candidate road segments using a **Bayesian-tuned `HistGradientBoostingClassifier`**, decodes globally coherent routes with a **distance-based Hidden Markov Model (HMM) + Viterbi algorithm + physically constrained road recovery**, and serves live predictions and interactive Folium visualisations across a **two-laptop FastAPI + React architecture**.

---

## 1. End-to-End System Pipeline

```text
Raw Trip Inputs (.track, .nodes, .arcs)
  │
  ▼  1. Spatial Preprocessing & Feature Engineering (preprocessing/)
  │     • Equirectangular local-meter projection centered on trip mean latitude
  │     • Guards for timestamp dropouts (>10s), stationary jitter (<0.5m), zero-length arcs
  │     • Adaptive cKDTree candidate generation (K=6) → 99.32% coverage (0.68% miss rate)
  │     • 20 leakage-free features (spatial, intra-obs ranking, kinematics, topology)
  │
  ▼  2. ML Candidate Scoring & GP-EI Tuning (models/, artifacts/models/)
  │     • 5 algorithm families compared via 5-fold trip-grouped GroupKFold CV (80 train trips)
  │     • Selected HistGradientBoostingClassifier tuned via 12-trial GP-EI Bayesian optimisation
  │     • Outputs calibrated candidate emission probabilities P(match=1 | x)
  │
  ▼  3. Sequential HMM Decoding & Route Recovery (route_reconstruction/, artifacts/hmm/)
  │     • Optuna-TPE optimised Distance HMM (beta=30.0m, transition_weight=1.9486)
  │     • Directed shortest-path Dijkstra transition scoring + log-space Viterbi decoding
  │     • Constrained intermediate arc insertion (max 60 m/s · dt + 30 m tolerance)
  │
  ▼  4. Server-Side Evaluation & Two-Laptop Serving (evaluation/, backend/, frontend/)
        • FastAPI (Laptop 1) executes live inference with route=None, reads .route strictly
          post-prediction for server-side scoring, and logs telemetry to the host terminal
        • React + Vite UI (Laptop 2) renders the 10 trips, interactive Folium map, & KPI cards
```

### Strict Partition & Anti-Leakage Design
| Partition | Path | Trips | Usage |
| :--- | :--- | :---: | :--- |
| **`train`** | `Student folder/train/` | `80` | Model fitting, 5-fold `GroupKFold` CV, GP-EI tuning, Optuna HMM optimisation |
| **`student_test`** | `Student folder/student_test/` | `10` | Development benchmarking across all 5 models & Phase 2 HMM recovery |
| **`evaluator_test`** | `evaluator_test/` *(git-ignored)* | `10` | Reserved unseen live evaluation; `.route` never enters features, models, or UI |

---

## 2. Verified Key Benchmarks

| Stage | Key Measured Outcome | Artifact / Report |
| :--- | :--- | :--- |
| **Candidate Generation (`K=6`)** | **`99.32%` coverage** (`0.68%` miss rate across `19,160` `student_test` obs) | [preprocessing/spatial_pipeline.py](preprocessing/spatial_pipeline.py) |
| **5-Model Comparison** | **`HistGradientBoosting`** selected over LR, KNN, DT, RF (`87.90%` Top-1, `0.8771` Seq F1, `0.65 MB`) | [experiments/results/section_3_1_model_comparison.csv](experiments/results/section_3_1_model_comparison.csv) |
| **GP-EI Bayesian Tuning** | CV Macro F1 `0.8608 → 0.8659`, Test F1 `0.8928 → 0.8998`, Log Loss `0.1672 → 0.1583` (`-12.4%` latency) | [experiments/results/tuned_vs_untuned_comparison.csv](experiments/results/tuned_vs_untuned_comparison.csv) |
| **Distance HMM + Recovery** | Accuracy **`87.90% → 90.79%`**, Continuity **`92.12% → 97.85%`**, Road-Overlap F1 **`0.9455`** | [experiments/results/phase2_overall_metrics.json](experiments/results/phase2_overall_metrics.json) |
| **Live `evaluator_test` (10 trips)** | Mean Pointwise Accuracy **`92.10%`**, Route F1 **`90.21%`**, Road-Overlap F1 **`92.39%`** (`4.63 s` avg) | [report/FINAL_TECHNICAL_REPORT.md](report/FINAL_TECHNICAL_REPORT.md) |

---

## 3. Repository Structure

| Directory / File | Purpose |
| :--- | :--- |
| [preprocessing/](preprocessing/) | Raw loaders, track/network sanity checks, equirectangular metric projection, `cKDTree` candidate generation, and 20-feature extraction |
| [models/](models/) | Five classifier families (+ optional MLP), `GroupKFold` CV, 12 metrics, GP-EI Bayesian tuning, plotting, and training orchestrator |
| [optimisation/](optimisation/) | Optuna TPE Bayesian optimisation framework and experiment tracking |
| [route_reconstruction/](route_reconstruction/) | Road graph Dijkstra solver, distance transition scorer, log-space Viterbi decoder, and constrained route recovery |
| [evaluation/](evaluation/) | Shared classification, cross-validation, model ranking, and route overlap metrics |
| [visualisation/](visualisation/) | 4-layer interactive Folium map generator and exported HTML route comparisons ([visualisation/maps/](visualisation/maps/)) |
| [artifacts/](artifacts/) | Production `.joblib` models ([artifacts/models/](artifacts/models/)) and Optuna-tuned HMM parameters ([artifacts/hmm/](artifacts/hmm/)) |
| [experiments/](experiments/) | Consolidated CSV/JSON metrics, Bayesian trial logs, exploration report, and all 16 PNG figures ([experiments/results/](experiments/results/)) |
| [backend/](backend/) | FastAPI server (`GET /health`, `GET /trips`, `POST /predict`, `GET /map/{trip_id}`), CORS config, and live server-side evaluation |
| [frontend/](frontend/) | React + Vite evaluator UI with live map rendering, KPI cards, run history, Model Benchmarks, and Architecture tabs |
| [notebooks/](notebooks/) | Exploratory data analysis notebook ([01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb)) |
| [report/](report/) | Consolidated technical report ([FINAL_TECHNICAL_REPORT.md](report/FINAL_TECHNICAL_REPORT.md)) matching `Deliverables.pdf` Sections `2.2`–`8` |
| [tests/](tests/) | Automated unit and API integration tests (`pytest`) |

---

## 4. Quickstart: Local Single-Laptop Setup

### Prerequisites
- **Python `3.10+`** (verified on Python `3.12.7`)
- **Node.js `18+`** and **`npm`**

### Step 1 — Install Python & Frontend Dependencies
```powershell
git clone https://github.com/Shreya04M/ML_Hackathon-.git
cd ML_Hackathon-

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

### Step 2 — Run Backend & Frontend Locally
Open **Terminal 1** (Backend API on port `8000`):
```powershell
.\venv\Scripts\Activate.ps1
# Defaults to .\evaluator_test if present, otherwise .\Student folder\student_test
$env:PATHSNAP_TRIP_DIR = ".\Student folder\student_test"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **Terminal 2** (React UI on port `5173`):
```powershell
cd frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev
```

Open **`http://localhost:5173`** in your browser (or explore Swagger API docs at `http://127.0.0.1:8000/docs`).

---

## 5. Two-Laptop Live Evaluation Setup

Per the hackathon deployment specification, evaluation separates the ML backend (**Laptop 1**) from the evaluator UI (**Laptop 2**) over a shared Wi-Fi / Mobile Hotspot:

```text
Laptop 1 (0.0.0.0:8000)                           Laptop 2 (localhost:5173)
FastAPI + Tuned HistGB + HMM/Viterbi    ◄──────►  React + Vite UI
evaluator_test/ (git-ignored, .route on server)   VITE_API_BASE_URL=http://<LAPTOP_1_IP>:8000
Prints live [GET /trips] & [POST /predict] logs   Displays 10 trips, Folium map & KPI cards
```

### Laptop 1 (Backend, Models & Server-Side Evaluator)
1. Ensure `evaluator_test/` (10 reserved trips `00000015`..`00000094`) is present at the repository root.
2. Find Laptop 1's LAN/Hotspot IPv4 address via `ipconfig` and allow TCP port `8000` in an elevated PowerShell if on a Public/Hotspot network:
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
   ```
3. Launch FastAPI and keep the terminal visible to show live inference logs to the evaluator:
   ```powershell
   .\venv\Scripts\Activate.ps1
   $env:PATHSNAP_TRIP_DIR = ".\evaluator_test"
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

### Laptop 2 (Evaluator UI)
1. Connect to the same hotspot/LAN as Laptop 1 and verify connectivity:
   ```powershell
   Invoke-RestMethod http://<LAPTOP_1_IP>:8000/health
   # Expected: @{status=healthy}
   ```
2. Launch the UI pointing to Laptop 1 (`CORS` in [backend/main.py](backend/main.py) permits cross-origin requests):
   ```powershell
   cd frontend
   npm install
   $env:VITE_API_BASE_URL = "http://<LAPTOP_1_IP>:8000"
   npm run dev -- --host 0.0.0.0 --port 5173
   ```
3. Open **`http://localhost:5173`** on Laptop 2, select any evaluation trip, and click **Reconstruct Route**.

---

## 6. Testing & Reproducing Experiments

### Automated Test Suite (`pytest`) & Frontend Build
```powershell
# Run full automated test suite (data loader, spatial pipeline, ML models, Optuna, HMM, FastAPI)
python -m pytest -v

# Run targeted test modules
python -m pytest -v tests/test_preprocessing.py tests/test_models.py
python -m pytest -v tests/test_route_reconstruction.py
python -m pytest -v tests/test_api.py

# Verify frontend production bundle
cd frontend
npm run build
```

### Reproducing Offline Training, Tuning & Phase 2 Evaluation
```powershell
# 1. Train & compare all 5 baseline models (5-fold GroupKFold CV + student_test + 12 plots)
python -m models.train

# 2. Run 12-trial GP-EI Bayesian hyperparameter optimisation on HistGradientBoosting
python -m models.bayesian_tuning

# 3. Run Phase 2 Distance HMM + constrained recovery benchmark & generate comparison plots
python -m route_reconstruction.run_phase2_evaluation
python -m route_reconstruction.plot_phase2_results
```

---

## 7. Contributing

1. **Branching**: Create a descriptive branch from `main` (e.g., `git checkout -b feature/<topic>` or `fix/<issue>`).
2. **Engineering & Anti-Leakage Rules**:
   - Never pass `.route` ground-truth data into feature extraction, candidate generation, model inference, or frontend payloads (`route=None` must be enforced during prediction).
   - Keep cross-validation strictly grouped by `trip_id` (`GroupKFold`).
   - Use strictly Bayesian hyperparameter search strategies (no `GridSearchCV` or `RandomizedSearchCV`).
   - Never commit `evaluator_test/`, `.env` secrets, or large `.parquet` caches (`.gitignore` excludes them automatically).
3. **Pre-PR Verification**: Ensure `python -m pytest -v` passes with zero failures, `cd frontend && npm run build` succeeds, and relevant README/report tables stay synchronized before opening a pull request against `main`.
