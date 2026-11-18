# PathSnap – Distributed Map-Matching & Route Reconstruction

Initial repository skeleton for reconstructing road routes from GPS trajectories. The ML pipeline, Bayesian optimisation, HMM/Viterbi reconstruction, and production prediction flow are **NOT IMPLEMENTED**.

## Objectives and architecture

Raw `.track`, `.nodes`, `.arcs`, and `.route` files → preprocessing/features → candidate classification → HMM/Viterbi reconstruction → evaluation → FastAPI/React presentation. Ground truth must remain server-side and must never be exposed by the prediction API.

## Team branches

| Member | Branch | Responsibility |
|---|---|---|
| 1 | `feature/data-processing` | Loading, cleaning, road network, candidates, features, Folium |
| 2 | `feature/ml-models` | Six models, group-aware evaluation, comparisons |
| 3 | `feature/route-engine` | Optimisation, HMM probabilities, Viterbi, route evaluation |
| 4 | `feature/api-frontend` | FastAPI, React/Vite, maps, deployment |

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
uvicorn backend.main:app --reload
cd frontend
npm install
npm run dev
```

## Git workflow

Do not commit directly to `main`. After reviewing the skeleton:

```powershell
git add .
git commit -m "Create PathSnap project skeleton"
git push origin main
git switch -c feature/data-processing main
git switch -c feature/ml-models main
git switch -c feature/route-engine main
git switch -c feature/api-frontend main
```

## Status and open contracts

This is an initial skeleton only: no data is parsed, models are trained, predictions are generated, or results fabricated. Agree the parsed trip schema, candidate/feature columns, target labels, artifact format, and train/testing partition names after inspecting the supplied dataset.
