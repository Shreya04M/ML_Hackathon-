# PathSnap - Bayesian Hyperparameter Optimisation

## Member 3 - Phase 1

This module performs Bayesian hyperparameter optimisation
of the selected machine learning model using Optuna.

## Supported Models

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- HistGradientBoosting

## Module Structure

| File | Description |
|---|---|
| search_space.py | Hyperparameter search spaces |
| experiment_tracker.py | Experiment and result logging |
| bayesian_optimization.py | Core optimisation framework |
| run_tuning.py | Actual dataset integration |

## Methodology

- Optimisation library: Optuna
- Sampling algorithm: TPE
- Objective function: Macro F1 Score
- Cross-validation: GroupKFold
- Default folds: 5
- Default trials: 30
- Random seed: 42

Trip IDs are used as groups to prevent observations
from the same trip appearing in both training and
validation folds.

The framework retains the baseline model if the
optimised model does not improve CV Macro F1.

## Installation

Run from the project root:

    python -m pip install -r requirements.txt

## Compatibility Testing

    python -m pytest tests/test_all_models.py -s -v

All five supported models passed compatibility testing
using synthetic data.

## Demonstration

    python -m tests.test_member3_demo

The demonstration uses synthetic data and does not
represent actual PathSnap model performance.

## Actual Model Tuning

    python -m optimisation.run_tuning --help

Required inputs:

- Processed training dataset
- Processed student-testing dataset
- Selected baseline model
- Ordered feature column names
- Trip IDs and labels

The final input schema and preprocessing pipeline
must be confirmed with Members 1 and 2.

## Generated Outputs

- Optimisation trial records (CSV)
- Best hyperparameters (JSON)
- Tuned vs untuned comparison (CSV)
- Baseline model (PKL)
- Tuned model (PKL)
- Final selected model (PKL)

## Current Status

- Optimisation framework implemented
- Experiment tracking implemented
- All five models tested using synthetic data
- Automatic best-model retention implemented
- Actual dataset integration script implemented
- Real dataset tuning pending teammate handoff

Evaluator-testing data must not be used for
model selection or hyperparameter optimisation.