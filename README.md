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

## Setup & Execution

### 1. Python Environment (Laptop 1 — Backend & ML Pipeline)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

Point `PATHSNAP_TRIP_DIR` to the evaluation trip directory (`evaluator_test` or `Student folder\student_test`) and start FastAPI on Laptop 1:
```powershell
$env:PATHSNAP_TRIP_DIR = ".\evaluator_test"
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend UI (Laptop 2 — Evaluator Interface)
```powershell
cd frontend
npm install
# Set VITE_API_BASE_URL=http://<LAPTOP_1_IP>:8000 in frontend/.env
npm run dev -- --host 0.0.0.0 --port 5173
```

### 3. Reproducing Training, Tuning & Phase 2 HMM Evaluation
```powershell
python -m models.train
python -m models.bayesian_tuning
python -m route_reconstruction.run_phase2_evaluation
```
