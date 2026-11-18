# Artifacts

Production inference assets exported for Laptop 1 FastAPI serving ([backend/services.py](../backend/services.py)):

- **`models/selected_best_model_tuned.joblib`** (`1.16 MB`): Final Bayesian-tuned (`GP-EI` Trial 6) `HistGradientBoostingClassifier` pipeline.
- **`models/selected_best_model.joblib`** (`0.65 MB`): Untuned `HistGradientBoostingClassifier` baseline selected from the 5-model comparison.
- **`models/logistic_regression.joblib`**, **`knn.joblib`**, **`decision_tree.joblib`**, **`random_forest.joblib`**, **`hist_gradient_boosting.joblib`**: All five fitted baseline candidate classifiers evaluated across `student_test`.
- **`models/selected_model_metadata.json`**: Multi-criteria ranking scores, ordered 20-feature schema (`FEATURE_COLUMNS`), and benchmark summary.
- **`hmm/best_hmm_parameters.json`**: Optimised distance-based HMM emission/transition parameters (`beta=30.0`, `transition_weight=1.948614369534382`, `backward_tolerance_m=30.0`) tuned via Optuna TPE across the 80 training trips.
