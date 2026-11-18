# Experiments

Central directory storing measured experimental tables, JSON logs, dataset exploration findings, and visualisations:

- `results/`:
  - `dataset_exploration_report.md` — Dataset inventory & anomaly audit.
  - `section_3_1_model_comparison.csv` — 5-model cross-validation & student-test comparison.
  - `bayesian_optimisation_trials.csv` & `tuned_vs_untuned_comparison.csv` — 12-trial GP-EI Bayesian optimisation logs.
  - `per_trip_student_test_breakdown.csv` — Per-trip pointwise & continuity metrics across the 10 `student_test` trips.
  - `phase2_per_trip_comparison.csv` & `phase2_overall_metrics.json` — Distance-based HMM + constrained route recovery metrics.
  - `evaluator_test_verification.csv` — Live API/pipeline execution benchmark across all 10 unseen `evaluator_test` trips.
  - `graphs/` — All Phase 1 (12 charts) and Phase 2 (4 charts) PNG visualisations.
