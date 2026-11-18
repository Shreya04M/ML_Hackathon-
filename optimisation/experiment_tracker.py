"""
PathSnap - Experiment Tracker
Member 3 - Bayesian Hyperparameter Optimisation
"""

import json
from pathlib import Path

import pandas as pd


class ExperimentTracker:
    """Record optimisation trials and model performance."""

    def __init__(self, output_dir="experiments"):

        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_trials(self, study, model_name):
        """Save all Optuna optimisation trials."""

        trials_df = study.trials_dataframe()
        trials_df.insert(0, "model_name", model_name)

        file_path = (
            self.logs_dir / f"{model_name}_optimisation_trials.csv"
        )

        trials_df.to_csv(file_path, index=False)

        print(f"Optimisation trials saved: {file_path}")

        return file_path

    def save_best_parameters(self, study, model_name):
        """Save the best hyperparameters and CV score."""

        results = {
            "model_name": model_name,
            "best_trial": study.best_trial.number,
            "best_macro_f1": study.best_value,
            "best_parameters": study.best_params,
            "total_trials": len(study.trials),
        }

        file_path = (
            self.results_dir / f"{model_name}_best_parameters.json"
        )

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print(f"Best parameters saved: {file_path}")

        return file_path

    def save_comparison(
        self, model_name, baseline_metrics, tuned_metrics
    ):
        """Save tuned versus untuned model performance."""

        records = []

        for metric in sorted(
            baseline_metrics.keys() & tuned_metrics.keys()
        ):

            baseline = float(baseline_metrics[metric])
            tuned = float(tuned_metrics[metric])

            records.append({
                "model_name": model_name,
                "metric": metric,
                "untuned": baseline,
                "tuned": tuned,
                "difference": tuned - baseline,
            })

        comparison_df = pd.DataFrame(records)

        file_path = (
            self.results_dir / f"{model_name}_tuned_vs_untuned.csv"
        )

        comparison_df.to_csv(file_path, index=False)

        print(f"Model comparison saved: {file_path}")

        return comparison_df
