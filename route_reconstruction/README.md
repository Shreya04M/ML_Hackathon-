# Route Reconstruction (Phase 2 — HMM, Viterbi & Constrained Recovery)

Hidden Markov Model (HMM), log-space Viterbi decoding, and physically constrained intermediate directed road-segment recovery for PathSnap.

## Architecture

| Module | Responsibility |
| :--- | :--- |
| [road_graph.py](road_graph.py) | Directed `.arcs` graph, node Haversine distances, and shortest-path Dijkstra utilities without reverse-edge fabrication |
| [emission_probability.py](emission_probability.py) | Normalizes calibrated ML candidate emission probabilities and converts to $\log \hat{p}_{t,k}$ |
| [distance_transition.py](distance_transition.py) | Projects consecutive GPS observations onto candidate arcs, computes directed road discrepancy vs. GPS Haversine step with `backward_tolerance_m = 30.0` |
| [transition_probability.py](transition_probability.py) | Connectivity-only and distance-based transition log matrices (`-inf` for unreachable transitions) |
| [viterbi.py](viterbi.py) | Log-space Viterbi trellis decoding with automatic section restarts across unreachable or non-increasing-timestamp intervals |
| [constrained_recovery.py](constrained_recovery.py) | Intermediate road insertion bounded by `max_speed_mps * dt + gps_tolerance_m` (`60 m/s`, `30 m`) |
| [route_recovery.py](route_recovery.py) | Unified route recovery interface used by both offline evaluation and live `backend/services.py` inference |
| [route_metrics.py](route_metrics.py) | Sequence LCS F1, length-weighted road-overlap F1, and `.route` reference loaders |
| [prepare_hmm_cache.py](prepare_hmm_cache.py) | Precomputes 80-trip directed transition distance cache under `data/processed/hmm_validation_cache/` |
| [optimise_hmm.py](optimise_hmm.py) | 25-trial Optuna TPE optimisation on the 80 training trips exporting [artifacts/hmm/best_hmm_parameters.json](../artifacts/hmm/best_hmm_parameters.json) |
| [run_phase2_evaluation.py](run_phase2_evaluation.py) | Full 10-trip `student_test` HMM & constrained recovery benchmark outputting metrics to [experiments/results/](../experiments/results/) |
| [plot_phase2_results.py](plot_phase2_results.py) | Generates Phase 2 comparison figures in [experiments/results/graphs/](../experiments/results/graphs/) |

## Measured Phase 2 Gains (`10 student_test` trips, `19,160` observations)

| Method | Observation Accuracy | Direct Continuity | Sequence LCS F1 | Road-Overlap F1 |
| :--- | :---: | :---: | :---: | :---: |
| Classifier-Only (`HistGradientBoosting`) | `87.90%` | `92.12%` | — | — |
| Connectivity-Only HMM | `85.38%` | **`98.70%`** | — | — |
| Distance HMM (Unoptimised) | `86.54%` | `96.95%` | — | — |
| **Distance HMM (Optuna-Tuned & Constrained Recovery)** ★ | **`90.79%`** | `97.85%` | **`0.8889`** | **`0.9455`** |

Selected parameters from Trial 21/25 (80 training trips / 208,417 obs): `beta = 30.0`, `transition_weight = 1.948614369534382`, `backward_tolerance_m = 30.0`.
