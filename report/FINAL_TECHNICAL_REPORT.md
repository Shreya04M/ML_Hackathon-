# PathSnap: Distributed Map-Matching & Route Reconstruction — Consolidated Technical Report

> **Course**: `UE24CS352A: Machine Learning` — PES University, Department of Computer Science and Engineering (RR Campus)  
> **Architecture**: End-to-End Raw GPS $\rightarrow$ Spatial Preprocessing & 20-Feature Pipeline $\rightarrow$ 5-Model Experimentation & Selection (`HistGradientBoosting`) $\rightarrow$ GP-EI Bayesian Optimisation $\rightarrow$ Distance-Based HMM + Viterbi & Constrained Route Recovery $\rightarrow$ Two-Laptop FastAPI & React Deployment  
> **Verification Guarantee**: Every table, metric, plot, configuration, and log in this consolidated report is reproducible directly from the repository codebase and committed experiment outputs in [experiments/results/](../experiments/results/). Reference routes for `evaluator_test` never entered feature extraction, model training, model selection, hyperparameter tuning, HMM parameter optimisation, or UI responses.

---

## 2.2 Dataset Understanding and Exploratory Data Analysis

### 1. Partitions, Schema & Strict Leakage Separation
| Partition | Source Path | Trips | Raw GPS Obs | Candidate Rows (`K=6`) | Positive Rate | Permitted Usage |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`train`** | `Student folder/train/` | **80** | `208,417` | `1,250,502` | `15.03%` | Model fitting, 5-fold trip-grouped `GroupKFold` CV, GP-EI Bayesian tuning, Optuna HMM optimisation |
| **`student_test`** | `Student folder/student_test/` | **10** | `19,160` | `114,960` | `16.67%` | Held-out Phase 1 model comparison & Phase 2 HMM/Viterbi evaluation |
| **`evaluator_test`** | `evaluator_test/` *(git-ignored)* | **10** | Reserved | Unseen | Unseen | Final live two-laptop server-side evaluation only (`route=None` during prediction) |

- **10 `student_test` Trip IDs**: `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`.
- **10 Reserved `evaluator_test` Trip IDs**: `00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094`.

Each trip directory contains four headerless, whitespace-separated files:
- **`.track`**: `longitude, latitude, timestamp` (`~1 Hz` noisy GPS points).
- **`.nodes`**: `longitude, latitude` indexed by 0-based `node_id`.
- **`.arcs`**: `from_node_id, to_node_id` indexed by 0-based `arc_id` (two-way streets are antiparallel directed pairs).
- **`.route`**: chronological sequence of ground-truth `arc_id`s. Crucially, `.track` rows do **not** align 1-to-1 with `.route` rows (e.g., `student_test` trip `00000000` has `2,503` GPS observations across `179` traversed road arcs).

### 2. Data Exploration Audit & Preprocessing Pipeline
Every anomaly across all `80 train` and `10 student_test` trips was audited in [notebooks/01_dataset_exploration.ipynb](../notebooks/01_dataset_exploration.ipynb) and handled inside [preprocessing/spatial_pipeline.py](../preprocessing/spatial_pipeline.py) *(canonical production pipeline; [preprocessing/data_cleaning.py](../preprocessing/data_cleaning.py) provides additional standalone track/network sanity diagnostics)*:

| Property / Anomaly Audited | `train` (`80` Trips) | `student_test` (`10` Trips) | Preprocessing Action Implemented |
| :--- | :---: | :---: | :--- |
| **GPS Observations / Trip** | `361` – `19,313` *(med `1,529.0`)* | `262` – `3,958` *(med `1,420.0`)* | Strict `trip_id`-grouped CV (`GroupKFold`) preventing autocorrelation leakage across folds |
| **Road Graph Scale** | `9,608` nodes / `19,526.5` arcs *(max `307,392`)* | `6,596.5` nodes / `13,145.0` arcs *(max `56,027`)* | Adaptive `scipy.spatial.cKDTree` indexing with up to `35` samples per long arc (`60 m` spacing) |
| **Geographic Latitude Spread** | `34.7°N` to `61.7°N` | `42.6°N` to `61.7°N`, `-93.4°W` | Equirectangular local tangent-plane meter projection ($\Delta x = R\cos\bar{\phi}\,\Delta\lambda$, $\Delta y = R\,\Delta\phi$, $R = 6{,}371{,}000\text{ m}$) |
| **Missing Values (`NaN`) & Timestamps** | `0` NaNs / `0` non-monotonic | `0` NaNs / `0` non-monotonic | `SimpleImputer(strategy="median")` inside `sklearn.pipeline.Pipeline` fit strictly on training folds |
| **GPS Sampling Intervals & Dropouts** | `96.50%` exactly `1.0 s`; `28` trips `>30 s` gap *(max `153,064 s`)* | `98.62%` exactly `1.0 s`; `2` trips `>30 s` gap *(max `210 s` in trip `00000065`)* | Guard `dt = max(dt, 1e-3)`, clip speed to `[0, 60] m/s`, disable central-diff heading across gaps `> 10 s` |
| **Stationary Jitter & Zero-Length Arcs** | `6` stationary steps; `100` zero-length arcs | `0` stationary; `6` zero-length arcs | Safe denominator `v_len2 > 1e-12`; neutral `90°` heading (`cos=0`) when displacement `< 0.5 m` or speed `< 0.25 m/s` |
| **Road-Network & Route Connectivity** | `79/80` multi-component networks (`0` isolated nodes); `70/80` connected routes (`13/31,256` breaks) | `10/10` multi-component networks (`0` isolated nodes); `9/10` connected routes (`1/3,467` break) | Downstream Viterbi & constrained recovery restart cleanly across unreachable graph gaps instead of fabricating arcs |
| **High-Offset Outlier Trips** | `00000033` (`122.4 m` med offset), `00000054` (`138.5 m` med offset) | None (`0.68%` miss rate overall) | Explains Fold 1 CV Top-1 drop (`00000054` has `35.42%` candidate miss rate) |

### 3. Candidate Generation & 20 Engineered Features
Using adaptive `cKDTree` arc sampling (`5` base samples + intermediate points every `60 m`, max `35`/arc, top `pool_k = 60` re-ranked by exact perpendicular distance), selecting **`K = 6` unique candidates** per observation achieves **`99.32%` coverage (`0.68%` Candidate Miss Rate)** on `student_test` (`19,029 / 19,160` observations) with zero `NaN`/`inf` values.

| # | Feature Name | Physical Group & Mathematical Definition | Train Mean ± Std *(Median)* | Test Mean *(Median)* |
| :-: | :--- | :--- | :---: | :---: |
| 1 | `ortho_dist_m` | Orthogonal distance (m) from GPS point to clamped segment projection | `47.40 ± 76.95` *(`19.25`)* | `46.67` *(`15.64`)* |
| 2 | `dist_to_start_m` | Euclidean distance (m) from GPS point to arc start node | `106.85 ± 174.25` *(`55.30`)* | `127.59` *(`45.24`)* |
| 3 | `dist_to_end_m` | Euclidean distance (m) from GPS point to arc end node | `106.65 ± 172.93` *(`55.36`)* | `127.72` *(`45.42`)* |
| 4 | `proj_fraction` | Clamped projection fraction $t_{\text{proj}} \in [0, 1]$ along arc | `0.500 ± 0.426` *(`0.498`)* | `0.500` *(`0.500`)* |
| 5 | `is_endpoint_proj` | `1` if clamped at `0` or `1`, `0` if interior (separates true road from cross-streets) | `0.568 ± 0.495` *(`1.000`)* | `0.523` *(`1.000`)* |
| 6 | `segment_length_m` | Road arc length in meters | `130.29 ± 242.55` *(`62.36`)* | `174.47` *(`60.92`)* |
| 7 | `log_segment_length_m` | $\ln(1 + \text{segment\_length\_m})$ compressing highway arc tails | `4.129 ± 1.188` *(`4.149`)* | `4.236` *(`4.126`)* |
| 8 | `dist_rank` | Ordinal proximity rank `0..5` within observation | `2.500 ± 1.708` *(`2.500`)* | `2.500` *(`2.500`)* |
| 9 | `dist_diff_from_min_m` | Excess orthogonal distance $d_{\perp,k} - \min_j d_{\perp,j}$ over closest candidate | `34.66 ± 74.76` *(`9.62`)* | `40.63` *(`8.36`)* |
| 10 | `dist_ratio_to_min` | Relative proximity ratio $(d_{\perp,k}+0.5)/(\min_j d_{\perp,j}+0.5)$ | `11.95 ± 36.82` *(`2.24`)* | `13.70` *(`2.53`)* |
| 11 | `dist_zscore_in_obs` | Within-observation standardized orthogonal distance z-score | `0.000 ± 0.988` *(`0.086`)* | `0.000` *(`0.196`)* |
| 12 | `gps_speed_mps` | Clipped step speed $\in [0, 60]\text{ m/s}$ | `13.67 ± 9.87` *(`11.07`)* | `15.39` *(`14.12`)* |
| 13 | `dt_prev_s` | Elapsed timestamp step $\Delta t$ (s) capturing dropouts | `2.181 ± 354.78` *(`1.000`)* | `1.058` *(`1.000`)* |
| 14 | `heading_diff_deg` | Unsigned bearing difference $|\theta_{\text{GPS}} - \theta_{\text{arc}}| \in [0^\circ, 180^\circ]$ | `87.45 ± 64.98` *(`85.38`)* | `85.96` *(`82.09`)* |
| 15 | `heading_cos` | $\cos(\Delta\theta) \in [-1, +1]$ separating forward (`+1`) vs antiparallel (`-1`) | `0.033 ± 0.825` *(`0.080`)* | `0.051` *(`0.138`)* |
| 16 | `heading_sin` | $\sin(|\Delta\theta|) \in [0, 1]$ peaking for perpendicular cross-streets | `0.448 ± 0.344` *(`0.363`)* | `0.393` *(`0.264`)* |
| 17 | `speed_weighted_heading_cos` | $\min(v_t/5, 1)\cdot\cos(\Delta\theta)$ suppressing low-speed GPS bearing noise | `0.031 ± 0.795` *(`0.063`)* | `0.050` *(`0.117`)* |
| 18 | `prev_cand_connected` | `1` if same arc or topological edge from any $t-1$ candidate (no label leakage) | `0.927 ± 0.260` *(`1.000`)* | `0.922` *(`1.000`)* |
| 19 | `min_dist_from_prev_cand_m` | Minimum node gap (m) from any $t-1$ candidate `to_node` to `from_node` | `4.41 ± 30.10` *(`0.00`)* | `4.76` *(`0.00`)* |
| 20 | `node_in_out_degree_sum` | $\text{in\_degree}(\text{from}) + \text{out\_degree}(\text{to})$ capturing junction branching | `4.104 ± 1.261` *(`4.000`)* | `4.281` *(`4.000`)* |

*(8 non-feature metadata columns — `trip_id`, `obs_id`, `timestamp`, `candidate_arc_id`, `from_node_id`, `to_node_id`, `ref_arc_id`, `is_ref_in_candidates` — plus target `label` are strictly isolated from model inputs.)*

**Empirical Feature Contribution Check** ([feature_importance_analysis.png](../experiments/results/graphs/feature_importance_analysis.png)):
- **HistGradientBoosting** *(Permutation Importance on `student_test`)*: `dist_diff_from_min_m` (`38.5%`), `heading_diff_deg` (`30.3%`), `is_endpoint_proj` (`12.7%`), `speed_weighted_heading_cos` (`8.7%`).
- **Random Forest** *(Gini Share)*: `dist_diff_from_min_m` (`14.16%`), `dist_rank` (`13.26%`), `is_endpoint_proj` (`11.01%`), `dist_zscore_in_obs` (`10.83%`).

![Stage 2 Data Audit](../experiments/results/graphs/stage2_data_audit_anomalies.png)
![Stage 3 Feature Separation](../experiments/results/graphs/stage3_feature_separation.png)

- **Representative Trajectory & Network Folium Map**: [visualisation/maps/trip_00000000_comparison.html](../visualisation/maps/trip_00000000_comparison.html) *(and see Section 6 for interactive 4-layer Folium visualisations).*

---

## 3. Model Experimentation and Comparison

All 5 core algorithm families (`models/logistic_regression.py`, `knn.py`, `decision_tree.py`, `random_forest.py`, `gradient_boosting.py`; plus optional `mlp.py` smoke-tested in `tests/`) were evaluated via **5-Fold Trip-Grouped `GroupKFold`** on all 80 training trips (`1,250,502` rows) and tested on all 10 `student_test` trips (`114,960` candidate rows, `19,160` observations).

### 3.1 Overall Model Comparison

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** ★ | Boosting | `lr=0.08, max_iter=200, max_leaf_nodes=63, l2=1.0, class_weight=balanced, early_stopping=True` | `0.8608` *(±0.0492)* | **`0.8790`** | `0.9212` | `0.8928` | `0.1672` | `0.3112` |
| **Random Forest** | Ensemble (Bagging) | `n_estimators=120, max_depth=16, min_samples_leaf=10, max_samples=0.5, class_weight=balanced_subsample` | `0.8732` *(±0.0409)* | `0.8783` | **`0.9219`** | `0.8999` | **`0.1556`** | `0.1325` |
| **KNN (Distance-Weighted)** | Instance-based | `k=15, weights=distance, algorithm=kd_tree, p=2, StandardScaler, max_index=25000` | **`0.8918`** *(±0.0221)* | `0.8662` | `0.9145` | **`0.9135`** | `0.3444` | `22.3724` |
| **Decision Tree** | Tree-based | `criterion=gini, max_depth=12, min_samples_leaf=20, class_weight=balanced` | `0.8389` *(±0.0683)* | `0.8727` | `0.9149` | `0.8850` | `0.2059` | **`0.0182`** |
| **Logistic Regression** | Linear and probabilistic | `C=1.0, L2 penalty, solver=lbfgs, class_weight=balanced, StandardScaler` | `0.8459` *(±0.0349)* | `0.8714` | `0.9167` | `0.8701` | `0.2148` | `0.0288` |

#### Fold-by-Fold `GroupKFold` CV Breakdown & Confusion Matrix Summary (`student_test`: `114,960` rows)
| Model | Fold Macro F1 (`[F1, F2, F3, F4, F5]`) | `TN` | `FP` | `FN` | `TP` | Pos Recall | Sequence F1 | Fit Time (s) | Artifact Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** ★ | `[0.7636, 0.8734, 0.8846, 0.8849, 0.8977]` | `89,246` | `6,549` | **`1,057`** | **`18,108`** | **`94.5%`** | **`0.8771`** | `12.36 s` | `0.65 MB` |
| **Random Forest** | `[0.7934, 0.8776, 0.8912, 0.9005, 0.9035]` | `90,027` | `5,768` | `1,215` | `17,950` | `93.7%` | `0.8757` | `49.20 s` | `25.03 MB` |
| **KNN (Distance-Weighted)** | `[0.8517, 0.8904, 0.8944, 0.9167, 0.9059]` | **`93,116`** | **`2,679`** | `2,830` | `16,335` | `85.2%` | `0.8608` | `0.05 s` *(index)* | `2.92 MB` |
| **Decision Tree** | `[0.7034, 0.8573, 0.8791, 0.8765, 0.8784]` | `88,657` | `7,138` | `1,096` | `18,069` | `94.3%` | `0.8685` | `8.72 s` *(diag)* | `0.11 MB` |
| **Logistic Regression** | `[0.7781, 0.8476, 0.8637, 0.8716, 0.8684]` | `87,288` | `8,507` | `1,025` | `18,140` | `94.7%` | `0.8640` | `6.32 s` | `0.002 MB` |

*(Fold 1 contains outlier trip `00000054` with a `35.42%` candidate miss rate; Folds 2, 3, and 5 have `0.46%–2.45%` miss rates.)*

#### Full-Dataset vs. Diagnostic Stride-Sample Verification
While core training, 5-fold `GroupKFold` CV, and testing ran on **100% of `1,250,502` train rows** and **100% of `114,960` test rows** (except KNN's `25,000`-row internal KD-tree reference bank queried across 100% of CV/test rows), an 80-trip stride slice (`113,904` rows, every 11th complete observation across all 80 trips) was verified to match full-dataset statistics within **`±0.0004` (`0.04%`)** Macro F1:
- Mean `ortho_dist_m`: `47.396 m` (full) vs `47.439 m` (stride, `+0.043 m`); mean `gps_speed_mps`: `13.667` vs `13.679 m/s`.
- Full vs. stride Train Macro F1: LR `0.8512` vs `0.8509`; DT `0.8911` vs `0.8908`; RF `0.9074` vs `0.9071`; HistGB `0.9028` vs `0.9029`; Tuned HistGB `0.9197` vs `0.9200`.

### 3.2 Compulsory Empirical Analysis for Every Model

#### 3.2.1 Logistic Regression (`models/logistic_regression.py`)
- **Q1 (Suitability)**: Classical probabilistic map-matching models orthogonal GPS error and heading discrepancy as additive log-odds terms; Logistic Regression directly learns a linear hyperplane $\log\frac{p}{1-p}=\mathbf{w}^\top\mathbf{x}+b$ over `StandardScaler`-normalized features.
- **Q2 (Strengths & Limitations)**: Fastest parametric fit (`6.32 s`) and near-instant inference (`0.0288 s` / `2.88 ms/trip`), smallest artifact (`0.002 MB`), and minimal train-CV gap (`+0.0050`). Limited because linear boundaries cannot apply conditional interaction rules (e.g., ignoring noisy GPS bearing at speed `< 1 m/s` while enforcing heading alignment at `25 m/s`).
- **Q5–Q6 (Config & Objective)**: `C=1.0`, `penalty='l2'`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=500`, `random_state=42`; minimizes L2-regularised Binary Log Loss.
- **Q7–Q9 (Results & Fit Diagnosis)**: On `student_test`: Accuracy `0.9171`, Macro Precision `0.8346`, Macro Recall `0.9289`, **Macro F1 `0.8701`**, **Top-1 Accuracy `0.8714`**, **Log Loss `0.2148`**, Continuity `0.9167` (`8,507 FP`, `1,025 FN`). Diagnosed as **underfitting (high bias)** — lowest CV (`0.8459`) and Test (`0.8701`) Macro F1 despite almost zero variance (`+0.0050` train-vs-CV gap).

#### 3.2.2 Distance-Weighted KNN (`models/knn.py`)
- **Q1 (Suitability)**: Tests whether non-parametric inverse-distance neighborhood voting in standardized 20D feature space cleanly separates intersection geometry without global parametric constraints.
- **Q2 (Strengths & Limitations)**: Highest candidate-level Precision (`0.9148`) and lowest CV fold variance (`±0.0221`). Severely unsuitable for deployment because KD-tree query cost causes **`22.3724 s` latency (`72×` slower than HistGB)** and neighborhood truncation drives **Log Loss to `0.3444` (`>2×` worse calibration)**.
- **Q5–Q6 (Config & Objective)**: `n_neighbors=15`, `weights='distance'`, `algorithm='kd_tree'`, `leaf_size=40`, `p=2`, `max_train_samples=25000`, `StandardScaler`; instance-based weighted voting (non-parametric, no iterative loss objective).
- **Q7–Q9 (Results & Fit Diagnosis)**: On `student_test`: Accuracy `0.9521`, **Macro F1 `0.9135`**, yet **lowest Top-1 Accuracy (`0.8662`)**, **worst Log Loss (`0.3444`)**, and **lowest Sequence F1 (`0.8608`)** because it drops `2,830` true segments (`85.2%` positive recall vs `94.5%` for HistGB). Small train-CV gap (`0.9009` vs `0.8918`), but calibration distortion and `2,830` False Negatives degrade route decoding.

#### 3.2.3 Decision Tree (`models/decision_tree.py`)
- **Q1 (Suitability)**: Learns hierarchical conditional splits (distance excess $\rightarrow$ speed-gated heading alignment $\rightarrow$ endpoint clamp) natively without scaling.
- **Q2 (Strengths & Limitations)**: Fastest inference overall (`0.0182 s` / `1.82 ms/trip`). Suffers from unsmoothed axis-aligned staircase splits that overfit training city layouts.
- **Q5–Q6 (Config & Objective)**: `criterion='gini'`, `max_depth=12`, `min_samples_split=40`, `min_samples_leaf=20`, `class_weight='balanced'`, `random_state=42`; recursive weighted Gini Impurity minimization.
- **Q7–Q9 (Results & Fit Diagnosis)**: On `student_test`: Accuracy `0.9284`, **Macro F1 `0.8850`**, **Top-1 Accuracy `0.8727`**, **Log Loss `0.2059`**, Continuity `0.9149` (`7,138 FP`, `1,096 FN`). Exhibited the **largest overfitting gap (`Train 0.8908 - CV 0.8389 = +0.0519`)** and highest cross-fold variance (`±0.0683`, collapsing to `0.7034` on noisy Fold 1).

#### 3.2.4 Random Forest (`models/random_forest.py`)
- **Q1 (Suitability)**: Bootstrap-aggregates `120` trees with `sqrt(d)` feature subsampling to cure single-tree variance while retaining non-linear interaction power.
- **Q2 (Strengths & Limitations)**: Strong generalisation (`CV Macro F1 0.8732 ± 0.0409`, lowest untuned Log Loss `0.1556`, highest classifier-only continuity `0.9219`). Requires **`4×` slower fit (`49.20 s`)** and a **`38×` larger artifact (`25.03 MB`)** than `HistGradientBoosting`.
- **Q5–Q6 (Config & Objective)**: `n_estimators=120`, `max_depth=16`, `min_samples_leaf=10`, `max_features='sqrt'`, `max_samples=0.5`, `class_weight='balanced_subsample'`, `random_state=42`; ensemble Gini Impurity reduction.
- **Q7–Q9 (Results & Fit Diagnosis)**: On `student_test`: Accuracy `0.9393`, **Macro F1 `0.8999`**, **Top-1 Accuracy `0.8783`**, **Log Loss `0.1556`**, Sequence F1 `0.8757` (`5,768 FP`, `1,215 FN`). Reduces train-vs-CV gap to `+0.0338` relative to Decision Tree (`+0.0519`), with `158` more missed true segments (`1,215` vs `1,057 FN`) than HistGradientBoosting.

#### 3.2.5 HistGradientBoosting — **Selected Model** (`models/gradient_boosting.py`)
- **Q1 (Suitability)**: Sequentially corrects residual classification errors directly under L2-regularised Binary Log Loss, sharpening discrimination between true carriageways and ambiguous antiparallel/intersecting candidates.
- **Q2 (Strengths & Limitations)**: Achieves **#1 Observation Top-1 Accuracy (`0.8790`, `0.8851` given candidate hit)**, **#1 Sequence F1 (`0.8771`)**, **`94.5%` positive recall (`18,108 TP`, only `1,057 FN`)**, `12.36 s` training fit, and compact `0.65 MB` artifact size. Requires tuned regularization/leaf depth (`learning_rate`, `max_leaf_nodes`, `l2_regularization`) to curb false positives — directly addressed via GP-EI Bayesian optimisation.
- **Q5–Q6 (Config & Objective)**: `loss='log_loss'`, `learning_rate=0.08`, `max_iter=200`, `max_leaf_nodes=63`, `max_depth=12`, `min_samples_leaf=30`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=True`, `random_state=42`; minimizes regularized Binary Log Loss.
- **Q7–Q9 (Results & Fit Diagnosis)**: On `student_test`: Accuracy `0.9338`, **Macro F1 `0.8928`**, **Top-1 Accuracy `0.8790`**, **Log Loss `0.1672`**, Continuity `0.9212`, Sequence F1 `0.8771` (`89,246 TN`, `6,549 FP`, `1,057 FN`, `18,108 TP`). Iterative boosting log loss converges smoothly over 200 iterations (`0.6931 -> 0.1283` train; `0.6931 -> 0.1381` val) with moderate train-vs-CV gap (`+0.0420`), and GP-EI tuning lowers test Log Loss to `0.1583` while eliminating `724` false alarms.

### 3.3 Model Visualisations (Generated From Actual Measured Outputs)
*(Iterative loss plots are included exclusively for iterative `HistGradientBoosting`; no synthetic training curves were fabricated for non-iterative models.)*

- [CV Macro F1 Comparison](../experiments/results/graphs/cv_macro_f1_comparison.png) · [Student-Test Metrics](../experiments/results/graphs/student_test_metrics_comparison.png) · [Log Loss Calibration](../experiments/results/graphs/log_loss_comparison.png) · [Confusion Matrices](../experiments/results/graphs/confusion_matrices_grid.png) · [Overfitting Diagnostics](../experiments/results/graphs/overfitting_diagnostics.png) · [Computational Efficiency](../experiments/results/graphs/computational_efficiency.png) · [Feature Importance](../experiments/results/graphs/feature_importance_analysis.png) · [Per-Trip Top-1 Heatmap](../experiments/results/graphs/stage7_per_trip_top1_heatmap.png) · [HistGradientBoosting Iterative Loss Curve](../experiments/results/graphs/iterative_training_validation_loss.png)

---

## 4. Bayesian Hyperparameter Optimisation (`HistGradientBoosting`)

Implemented strictly as a **Gaussian Process (`ConstantKernel * Matern(nu=2.5) + WhiteKernel`) + Expected Improvement (GP-EI)** sequential strategy in [models/bayesian_tuning.py](../models/bayesian_tuning.py) (zero `GridSearchCV` / `RandomizedSearchCV`; zero `evaluator_test` involvement).

1. **Search Space & Ranges (`5D` + fixed `max_iter=250, class_weight="balanced", early_stopping=True`)**:
   - `learning_rate ∈ [0.03, 0.18]` (`log10`)
   - `max_leaf_nodes ∈ [31, 127]` (`int`)
   - `max_depth ∈ [8, 18]` (`int`)
   - `min_samples_leaf ∈ [15, 80]` (`int`)
   - `l2_regularization ∈ [0.1, 10.0]` (`log10`)
2. **Optimisation Objective & Trials**: Maximised 5-fold trip-grouped `GroupKFold` CV Macro F1 across 80 training trips over **12 sequential trials** (`3` initial space-filling anchors + `9` GP-EI acquisition evaluations).

| Trial | Phase | Surrogate CV Macro F1 | Trial Time | `learning_rate` | `max_leaf_nodes` | `max_depth` | `min_samples_leaf` | `l2_regularization` |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Anchor *(Baseline)* | `0.86988` | `5.73 s` | `0.08000` | `63` | `12` | `30` | `1.00000` |
| 2 | Anchor | `0.87450` | `3.57 s` | `0.11430` | `74` | `16` | `59` | `0.18600` |
| 3 | Anchor | `0.87911` | `3.34 s` | `0.15823` | `102` | `16` | `26` | `0.81413` |
| **6** | **GP-EI Best ★** | **`0.88017`** | **`2.95 s`** | **`0.16973`** | **`120`** | **`15`** | **`33`** | **`0.38900`** |
| 12 | GP-EI *(worst)* | `0.85775` | `5.90 s` | `0.03337` | `36` | `8` | `29` | `0.11423` |

*(Full 12-trial CSV logged in [experiments/results/bayesian_optimisation_trials.csv](../experiments/results/bayesian_optimisation_trials.csv).)*

3. **Full-Dataset Verification (`1,250,502` train / `114,960` `student_test` rows)**:

| Variant | Full 80-Trip CV Macro F1 | Student-Test Top-1 Acc | Student-Test Macro F1 | Student-Test Log Loss | Sequence F1 | Fit / Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned HistGradientBoosting** | `0.8608 (±0.0492)` | **`0.8790`** | `0.8928` | `0.1672` | `0.8771` | `12.36 s` / `0.3112 s` |
| **Bayesian-Tuned (Trial 6 GP-EI)** ★ | **`0.8659 (±0.0475)`** | `0.8777` | **`0.8998`** | **`0.1583`** *(-5.3%)* | **`0.8799`** | **`12.25 s` / `0.2726 s`** *(-12.4%)* |

![Bayesian Optimisation Progression](../experiments/results/graphs/stage9_bayesian_tuning_progression.png)

---

## 5. HMM and Viterbi Route Reconstruction

Implemented in [route_reconstruction/](../route_reconstruction/) with decoder parameters tuned via **25 Optuna TPE trials strictly on the 80 training trips** (`208,417` OOF observations; [artifacts/hmm/best_hmm_parameters.json](../artifacts/hmm/best_hmm_parameters.json)).

1. **Hidden States**: Candidate road arcs (`candidate_arc_id`) per GPS observation $t$, preserving exact candidate-to-arc alignment and deduplicating duplicate entries by keeping max emission probability.
2. **Emission Probabilities ([emission_probability.py](../route_reconstruction/emission_probability.py))**: Normalize positive candidate probabilities $\hat{p}_{t,k}$ per observation and convert to log-emissions $\log \hat{p}_{t,k}$.
3. **Transition Probabilities ([distance_transition.py](../route_reconstruction/distance_transition.py), [road_graph.py](../route_reconstruction/road_graph.py))**: Directed shortest-path Dijkstra distances over `.arcs` without fabricating reverse edges; exponential decay (`beta = 30.0 m`, `transition_weight = 1.948614369534382`) on the discrepancy between GPS Haversine step distance and candidate-to-candidate road distance. Same-arc backward jitter within `backward_tolerance_m = 30.0 m` is treated as zero progress instead of forcing a city-block loop.
4. **Missing Candidates / GPS Gaps & Disconnected Segments ([viterbi.py](../route_reconstruction/viterbi.py), [constrained_recovery.py](../route_reconstruction/constrained_recovery.py))**:
   - Unreachable transitions or non-increasing timestamps receive $-\infty$ transition log-score rather than collapsing long detours onto a finite floor; if all state transitions across an interval are unreachable, Viterbi cleanly restarts a new route section.
   - Constrained intermediate recovery connects consecutive arcs along directed shortest paths only when road distance obeys $d_{\text{road}} \le 60\text{ m/s} \cdot \Delta t + 30\text{ m}$; implausible jumps (e.g., `8,826 m` in `1 s`) start separate route sections rather than fabricating paths.
5. **Classifier-Only vs. Distance HMM + Constrained Recovery (`10 student_test` trips, `19,160` obs)**:

| Method | Observation Accuracy | Direct Route Continuity | Sequence LCS F1 | Road-Overlap F1 | Operational Summary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Classifier-Only (`HistGradientBoosting`)** | `87.90%` | `92.12%` | — | — | `1,509` pointwise continuity breaks |
| **Connectivity-Only HMM** | `85.38%` | **`98.70%`** | — | — | Over-smooths dense junctions (`-2.52%` acc vs classifier) |
| **Distance HMM (Unoptimised)** | `86.54%` | `96.95%` | — | — | Resolves connectivity-only over-smoothing |
| **Distance HMM (Optimised & Corrected)** ★ | **`90.79%`** | `97.85%` | **`0.8889`** | **`0.9455`** | **`+2.89%` accuracy**, **`+5.73%` continuity**, `540` recovered arcs, `39` rejected jumps |

- **Visualisations**: [Observation Metrics](../experiments/results/graphs/observation_metrics.png) · [Route Quality](../experiments/results/graphs/route_quality.png) · [Recovery Comparison](../experiments/results/graphs/recovery_comparison.png) · [Runtime Comparison](../experiments/results/graphs/runtime_comparison.png)

---

## 6. Geographical Visualisation Using Folium

Generated from actual pipeline predictions using [visualisation/folium_maps.py](../visualisation/folium_maps.py) with distinguishable colors, markers, legends, bounding-box zoom, and `LayerControl` (`Road Network`, `Raw GPS Observations`, `Reference Route (Green)`, `Predicted Route (Red)`):
- [visualisation/maps/trip_00000000_comparison.html](../visualisation/maps/trip_00000000_comparison.html) (`2,503` GPS observations across dense urban intersections).
- [visualisation/maps/trip_00000052_comparison.html](../visualisation/maps/trip_00000052_comparison.html) (`262` GPS observations, `0.00%` candidate miss rate, `98.47%` HMM accuracy).

*(Insert exported screenshot of an interactive Folium comparison map here when converting this report to PDF.)*

---

## 7. API, UI and Two-Laptop Deployment

- **Laptop 1 — Backend, Model & Server-Side Evaluation ([backend/](../backend/))**:
  - FastAPI exposes `GET /health`, `GET /trips`, `POST /predict`, and `GET /map/{trip_id}` with CORS enabled (`allow_origins=["*"]`).
  - On `POST /predict`, [backend/services.py](../backend/services.py) loads `artifacts/models/selected_best_model_tuned.joblib` and `artifacts/hmm/best_hmm_parameters.json`, executes `extract_trip_candidates_and_features(..., route=None)` $\rightarrow$ `predict_candidate_probabilities()` $\rightarrow$ distance `reconstruct_route()` $\rightarrow$ `recover_continuous_route()`, evaluates `.route` strictly server-side after inference, and prints structured telemetry (`[GET /trips]`, `[MODEL LOAD]`, `[POST /predict] START`, `[MAP RENDER]`, `[POST /predict] COMPLETE`) to the host terminal.
- **Laptop 2 — React Evaluator UI ([frontend/](../frontend/))**:
  - Connects via `VITE_API_BASE_URL=http://<LAPTOP_1_IP>:8000` (`src/services/api.js`), displays connection status and all 10 evaluation trip options (`TripSelector.jsx`), renders the ground-truth-free Folium route map (`RouteMap.jsx`), displays 6 KPI metrics + run history (`Results.jsx`, `Home.jsx`), and provides dedicated Model Benchmarks and Pipeline Architecture views (`BenchmarksView.jsx`, `ArchitectureView.jsx`).
- **Live Verification Summary**:
  - Executed end-to-end across all 10 unseen `evaluator_test` trips (`00000015`..`00000094`): **`92.10%` mean pointwise accuracy**, **`90.21%` mean Route F1**, **`92.39%` mean Road-Overlap F1** (`4.632 s` average latency per trip).
  - *(Insert live screenshots of Laptop 1 FastAPI terminal logs and Laptop 2 React UI here when exporting this report to PDF.)*

---

## 8. Academic Integrity & Reproducibility

- All code, trained `.joblib` pipelines, GP-EI and Optuna trial logs, comparison tables, PNG figures, and interactive Folium maps are committed in the repository and reproducible from [models/train.py](../models/train.py), [models/bayesian_tuning.py](../models/bayesian_tuning.py), and [route_reconstruction/run_phase2_evaluation.py](../route_reconstruction/run_phase2_evaluation.py).
- Reference `.route` files for `evaluator_test` remain isolated on Laptop 1 (`.gitignore`) and are never exposed to feature engineering, model inference, HMM decoding, or frontend payloads.
- Automated development tools were used strictly for modular software engineering, refactoring, and formatting; all modeling decisions, experiments, metrics, and architectural choices were executed and verified directly by the team.
