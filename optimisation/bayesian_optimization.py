"""
PathSnap - Bayesian Hyperparameter Optimisation
Member 3 - Model Optimisation
"""

import numpy as np
import optuna
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline

from optimisation.search_space import (
    define_search_space,
    suggest_parameters,
)
from optimisation.experiment_tracker import ExperimentTracker


RANDOM_STATE = 42


def evaluate_model(model, X, y):
    """Calculate model performance metrics."""

    predictions = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        metrics["log_loss"] = log_loss(
            y, probabilities, labels=model.classes_
        )

    return metrics


def configure_model(model, parameters):
    """Clone the model and apply suggested hyperparameters."""

    new_model = clone(model)

    if isinstance(new_model, Pipeline):
        final_step = new_model.steps[-1][0]

        parameters = {
            f"{final_step}__{key}": value
            for key, value in parameters.items()
        }

    new_model.set_params(**parameters)

    return new_model


def optimise_hyperparameters(
    model,
    X,
    y,
    search_space,
    *,
    groups,
    n_trials=30,
    n_splits=5,
):
    """Perform Bayesian optimisation using group-aware CV."""

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y and groups must have equal lengths")

    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough unique trips for cross-validation")

    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups))

    def objective(trial):

        parameters = suggest_parameters(
            trial, search_space
        )

        trial_model = configure_model(
            model, parameters
        )

        scores = cross_val_score(
            trial_model,
            X,
            y,
            cv=splits,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
        )

        trial.set_user_attr(
            "fold_scores", scores.tolist()
        )

        trial.set_user_attr(
            "cv_std", float(np.std(scores))
        )

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    # Include baseline hyperparameters when valid

    estimator = model

    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]

    baseline_params = estimator.get_params()

    initial_params = {}

    for name, spec in search_space.items():

        if name not in baseline_params:
            continue

        value = baseline_params[name]
        parameter_type = spec[0]

        if parameter_type == "categorical":

            if value in spec[1]:
                initial_params[name] = value

        elif parameter_type in ("int", "float"):

            if value is not None:
                if spec[1] <= value <= spec[2]:
                    initial_params[name] = value

    if len(initial_params) == len(search_space):
        study.enqueue_trial(initial_params)

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    return study


def run_optimisation(
    model_name,
    baseline_model,
    X_train,
    y_train,
    train_groups,
    X_test,
    y_test,
    n_trials=30,
    n_splits=5,
    output_dir="experiments",
    artifacts_dir="artifacts",
):
    """Run optimisation, retraining and final evaluation."""

    tracker = ExperimentTracker(output_dir)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    train_groups = np.asarray(train_groups)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    search_space = define_search_space(model_name)

    print("\nStarting Bayesian optimisation...")
    print(f"Selected model: {model_name}")
    print(f"Number of trials: {n_trials}")

    # Step 1: Evaluate untuned model using training CV

    cv = GroupKFold(n_splits=n_splits)

    baseline_scores = cross_val_score(
        clone(baseline_model),
        X_train,
        y_train,
        groups=train_groups,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        error_score="raise",
    )

    # Step 2: Perform Bayesian optimisation

    study = optimise_hyperparameters(
        model=baseline_model,
        X=X_train,
        y=y_train,
        search_space=search_space,
        groups=train_groups,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    # Step 3: Save optimisation results

    tracker.save_trials(study, model_name)
    tracker.save_best_parameters(study, model_name)

    # Step 4: Retrain baseline model

    untuned_model = clone(baseline_model)

    untuned_model.fit(X_train, y_train)

    # Step 5: Retrain tuned model

    tuned_model = configure_model(
        baseline_model,
        study.best_params
    )

    tuned_model.fit(X_train, y_train)

    # Step 6: Evaluate both models on student-testing data

    baseline_metrics = evaluate_model(
        untuned_model, X_test, y_test
    )

    tuned_metrics = evaluate_model(
        tuned_model, X_test, y_test
    )

    baseline_metrics["cv_macro_f1"] = float(
        np.mean(baseline_scores)
    )

    tuned_metrics["cv_macro_f1"] = study.best_value

    # Step 7: Save performance comparison

    comparison = tracker.save_comparison(
        model_name,
        baseline_metrics,
        tuned_metrics,
    )

    # Step 8: Save trained models

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        untuned_model,
        artifacts_dir / f"{model_name}_baseline.pkl",
    )

    joblib.dump(
        tuned_model,
        artifacts_dir / f"{model_name}_tuned.pkl",
    )
    # Retain the model with the better CV Macro F1

    baseline_cv = float(np.mean(baseline_scores))

    if study.best_value > baseline_cv:
        final_model = tuned_model
        selected_version = "tuned"
        final_cv_score = study.best_value
    else:
        final_model = untuned_model
        selected_version = "baseline"
        final_cv_score = baseline_cv

    joblib.dump(
        final_model,
        artifacts_dir / f"{model_name}_final_model.pkl",
    )

    print("\n========== FINAL MODEL SELECTION ==========")
    print("Selected version:", selected_version)
    print(f"Final CV Macro F1: {final_cv_score:.4f}")

    # Step 9: Display results

    print("\n========== OPTIMISATION RESULTS ==========")

    print("\nBest hyperparameters:")
    print(study.best_params)

    print(f"\nBaseline CV Macro F1: {np.mean(baseline_scores):.4f}")
    print(f"Best tuning CV Macro F1: {study.best_value:.4f}")

    print("\nTuned vs Untuned:")
    print(comparison.to_string(index=False))

    print("\nOptimisation completed successfully!")

    return {
        "study": study,
        "model": final_model,
        "tuned_model": tuned_model,
        "selected_version": selected_version,
        "final_cv_score": final_cv_score,
        "best_parameters": study.best_params,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "comparison": comparison,
    }
