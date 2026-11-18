# Experiments

Central directory storing measured experimental tables, JSON records, trial logs, dataset exploration findings, and all generated plots:

- [results/dataset_exploration_report.md](results/dataset_exploration_report.md) — Dataset inventory & anomaly audit across `80 train` and `10 student_test` trips.
- [results/section_3_1_model_comparison.csv](results/section_3_1_model_comparison.csv) & [.md](results/section_3_1_model_comparison.md) — 5-model `GroupKFold` CV and `student_test` comparison.
- [results/detailed_experiment_results.json](results/detailed_experiment_results.json) — Complete per-fold CV, confusion matrices, and per-model configuration metadata.
- [results/per_trip_student_test_breakdown.csv](results/per_trip_student_test_breakdown.csv) — Per-trip pointwise & continuity breakdown across all 10 `student_test` trips.
- [results/bayesian_optimisation_trials.csv](results/bayesian_optimisation_trials.csv), [results/tuned_vs_untuned_comparison.csv](results/tuned_vs_untuned_comparison.csv), & [.json](results/tuned_vs_untuned_comparison.json) — 12-trial GP-EI Bayesian optimisation logs and tuned-vs-untuned comparison.
- [results/student_test_candidate_probabilities.csv.gz](results/student_test_candidate_probabilities.csv.gz) — Selected model candidate probabilities on `student_test`.
- [results/hmm_optimisation_trials.csv](results/hmm_optimisation_trials.csv) & [results/hmm_cache_metadata.json](results/hmm_cache_metadata.json) — 25-trial Optuna TPE HMM parameter optimisation logs over the 80 training trips.
- [results/phase2_overall_metrics.json](results/phase2_overall_metrics.json), [results/phase2_per_trip_comparison.csv](results/phase2_per_trip_comparison.csv), [results/phase2_recovered_routes.json](results/phase2_recovered_routes.json), & [results/phase2_runtime_comparison.csv](results/phase2_runtime_comparison.csv) — Phase 2 Distance HMM + constrained route recovery metrics.
- [results/training_progress.log](results/training_progress.log) & [results/training_status.json](results/training_status.json) — Orchestrator execution logs.
- `results/graphs/` — All 12 Phase 1 ML charts and 4 Phase 2 HMM/recovery comparison charts.
