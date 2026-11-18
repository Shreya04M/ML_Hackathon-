
"""Student-testing evaluation for PathSnap."""

from typing import Any
from time import perf_counter

from evaluation.metrics import calculate_metrics


def evaluate_student_test(
    estimator: Any,
    X_test: Any,
    y_test: Any,
    model_name: str | None = None,
    variant: str = "untuned",
) -> dict[str, Any]:
    """Evaluate a fitted model on separate student-test data."""

    # Measure prediction time.
    start = perf_counter()
    y_pred = estimator.predict(X_test)
    inference_time = perf_counter() - start

    # Obtain prediction probabilities if available.
    y_proba = None
    probability_time = None

    if hasattr(estimator, "predict_proba"):
        start = perf_counter()
        y_proba = estimator.predict_proba(X_test)
        probability_time = perf_counter() - start

    # Calculate reusable classification metrics.
    metrics = calculate_metrics(
        y_test, y_pred, y_proba
    )

    return {
        "model": model_name or type(estimator).__name__,
        "variant": variant,
        "dataset": "student_test",
        "test_samples": len(y_test),
        "inference_time": inference_time,
        "probability_time": probability_time,
        "metrics": metrics,
    }