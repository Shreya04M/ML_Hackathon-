# Evaluation Framework

The evaluation module provides a common framework for measuring, comparing and visualising machine learning model performance in PathSnap.

## Implemented Modules

| File | Description |
|---|---|
| `metrics.py` | Accuracy, macro precision, macro recall, Macro F1, log loss and confusion matrix. |
| `cross_validation.py` | Trip-based group-aware cross-validation and execution time measurement. |
| `model_comparison.py` | Model ranking, CSV comparison tables and JSON experiment records. |
| `visualisations.py` | Performance graphs, confusion matrices, execution times and tuned versus untuned comparisons. |
| `route_evaluation.py` | Route arc overlap, precision, recall, F1 and Jaccard similarity. |
| `student_evaluation.py` | Evaluation of fitted models on separate student-testing data. |

## Cross-Validation

The framework uses GroupKFold with five folds by default. Trip IDs are used as groups to prevent observations from the same trip appearing in both training and validation sets.

Example:

```python
from evaluation.cross_validation import run_group_aware_cv

result = run_group_aware_cv(
    estimator=model,
    X=X_train,
    y=y_train,
    groups=trip_ids,
    n_splits=5
)

print(result["summary"])
```

## Model Comparison

```python
from evaluation.model_comparison import (
    build_model_comparison,
    save_comparison,
    save_experiment
)

comparison = build_model_comparison(results)
save_comparison(comparison)

for result in results:
    save_experiment(result, result["model"])
```

Models are provisionally ranked using mean Macro F1, followed by mean accuracy.

## Student Testing

```python
from evaluation.student_evaluation import (
    evaluate_student_test
)

result = evaluate_student_test(
    model,
    X_test,
    y_test
)
```

The student-testing dataset must remain separate from model training and cross-validation.

## Visualisations

```python
from evaluation.visualisations import (
    plot_model_comparison,
    plot_execution_time,
    plot_confusion_matrix,
    plot_tuned_vs_untuned
)

plot_model_comparison(comparison)
plot_execution_time(comparison)
```

Additional functions support confusion matrices and tuned versus untuned model comparisons.

## Output Directories

- `experiments/logs/` — JSON experiment records.
- `experiments/results/` — CSV comparison tables.
- `experiments/results/graphs/` — Generated performance graphs.

## Evaluation Conventions

- Binary candidate labels: 0 = incorrect candidate, 1 = correct candidate.
- Probability column order: [0, 1].
- Precision, recall and F1 use macro averaging.
- Execution times are measured in seconds.
- Inference time measures model prediction, excluding route reconstruction.
- Route overlap metrics operate on unique arc IDs and do not measure connectivity.

## Implementation Status

All six evaluation modules have passed initial synthetic functionality tests.

Pending integration:

- Evaluation of all five trained models.
- Actual student-testing results.
- Observation-level accuracy and candidate miss rate.
- Bayesian optimisation comparison.
- Final model selection analysis.

Synthetic test results are not considered actual experimental performance.