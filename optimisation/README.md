# Bayesian Hyperparameter Optimisation

Optuna TPE Bayesian hyperparameter optimisation and search-space definitions compatible with all five PathSnap classifier families (`logistic_regression`, `knn`, `decision_tree`, `random_forest`, `hist_gradient_boosting`).

## Module Structure

| File | Description |
| :--- | :--- |
| [search_space.py](search_space.py) | Hyperparameter bounds and distributions per model family |
| [experiment_tracker.py](experiment_tracker.py) | Optimisation trial and best-parameter CSV/JSON logging |
| [bayesian_optimization.py](bayesian_optimization.py) | Trip-grouped `GroupKFold` Optuna TPE optimisation engine |
| [run_tuning.py](run_tuning.py) | CLI runner supporting external CSV feature tables |

*(For the canonical `K=6` `HistGradientBoosting` GP-EI tuning run reported in the final pipeline, see [models/bayesian_tuning.py](../models/bayesian_tuning.py) and [experiments/results/bayesian_optimisation_trials.csv](../experiments/results/bayesian_optimisation_trials.csv).)*

## Verification

```powershell
python -m pytest tests/test_all_models.py -s -v
```
