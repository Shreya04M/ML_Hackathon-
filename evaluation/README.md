# Evaluation Framework

Shared metrics, trip-grouped cross-validation, model ranking tables, visualisation helpers, and route overlap scoring modules.

## Modules

| File | Description |
| :--- | :--- |
| [metrics.py](metrics.py) | Accuracy, macro precision, macro recall, Macro F1, log loss, and confusion matrix |
| [cross_validation.py](cross_validation.py) | Trip-grouped `GroupKFold` cross-validation and execution timing |
| [model_comparison.py](model_comparison.py) | Model ranking, CSV comparison tables, and JSON experiment logging |
| [visualisations.py](visualisations.py) | Performance bar charts, confusion matrices, execution time, and tuned-vs-untuned charts |
| [route_evaluation.py](route_evaluation.py) | Server-side route arc overlap (`route_precision`, `route_recall`, `route_f1`, `jaccard_similarity`, `exact_route_match`) |
| [student_evaluation.py](student_evaluation.py) | Standalone evaluation helpers for `student_test` feature tables |

## Integration Status

- All six modules are implemented and integrated into `backend/services.py` (`evaluate_route`) and the end-to-end test suite (`tests/`).
- Route metrics strictly operate server-side on Laptop 1; `.route` reference labels never leave the server.
