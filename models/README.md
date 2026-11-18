# Models — ML Experimentation, Selection & Bayesian Tuning

Core machine learning experimentation, trip-grouped cross-validation, selection diagnostics, and Gaussian-Process Expected Improvement (`GP-EI`) Bayesian tuning over all **80 training trips (`1,250,502` rows)** and **10 `student_test` trips (`114,960` rows, `19,160` observations)**.

For complete compulsory per-model empirical analysis (Q1–Q9), confusion matrices, fold breakdowns, and visualisations, see [report/FINAL_TECHNICAL_REPORT.md](../report/FINAL_TECHNICAL_REPORT.md).

## Module Architecture

| File | Responsibility |
| :--- | :--- |
| [schema_adapter.py](schema_adapter.py) | Re-exports canonical `preprocessing.spatial_pipeline` schema/builders and provides `load_feature_dataset()` |
| [logistic_regression.py](logistic_regression.py) | **Model 1** — Linear & Probabilistic (`SimpleImputer` + `StandardScaler` + `LogisticRegression`) |
| [knn.py](knn.py) | **Model 2** — Instance-based Distance-Weighted KNN (`StratifiedSubsampledKNN` + `StandardScaler` + `KDTree`) |
| [decision_tree.py](decision_tree.py) | **Model 3** — Tree-based regularised `DecisionTreeClassifier` |
| [random_forest.py](random_forest.py) | **Model 4** — Bagging Ensemble `RandomForestClassifier` |
| [gradient_boosting.py](gradient_boosting.py) | **Model 5** — Histogram Boosting `HistGradientBoostingClassifier` (**Selected Best Model**) |
| [mlp.py](mlp.py) | Optional `MLPClassifier` implementation |
| [model_selection.py](model_selection.py) | 5-fold `GroupKFold` CV (`trip_id`), 12 evaluation metrics, ranking, and Bayesian CV objective helper |
| [plots.py](plots.py) | Generates all Section 3.3 diagnostic visualisations from measured outputs into [experiments/results/graphs/](../experiments/results/graphs/) |
| [train.py](train.py) | End-to-end training orchestrator exporting models to [artifacts/models/](../artifacts/models/) and metrics to [experiments/results/](../experiments/results/) |
| [bayesian_tuning.py](bayesian_tuning.py) | Strictly Bayesian `GP-Matern(nu=2.5) + Expected Improvement` 12-trial optimiser over `HistGradientBoosting` |

## Reproduction

```powershell
# Train all 5 baseline models, run 5-fold GroupKFold CV, evaluate student_test, and export plots/artifacts
python -m models.train

# Monitor live progress
Get-Content experiments/results/training_progress.log -Wait

# Run 12-trial GP-EI Bayesian hyperparameter optimisation (no GridSearch/RandomSearch)
python -m models.bayesian_tuning
```

## Consolidated Summary (`student_test`)

| Model | CV Macro F1 | Top-1 Accuracy | Continuity | Test Macro F1 | Log Loss | Inference (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Untuned)** | `0.8608` | **`0.8790`** | `0.9212` | `0.8928` | `0.1672` | `0.3112` |
| **HistGradientBoosting (GP-EI Tuned)** ★ | **`0.8659`** | `0.8777` | — | **`0.8998`** | **`0.1583`** | **`0.2726`** |
| **Random Forest** | `0.8732` | `0.8783` | **`0.9219`** | `0.8999` | **`0.1556`** | `0.1325` |
| **KNN (Distance-Weighted)** | **`0.8918`** | `0.8662` | `0.9145` | **`0.9135`** | `0.3444` | `22.3724` |
| **Decision Tree** | `0.8389` | `0.8727` | `0.9149` | `0.8850` | `0.2059` | **`0.0182`** |
| **Logistic Regression** | `0.8459` | `0.8714` | `0.9167` | `0.8701` | `0.2148` | `0.0288` |
