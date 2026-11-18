# PathSnap — Phase 1 Technical Deliverables & Experimental Report

> **Project**: PathSnap: Distributed Map-Matching & Route Reconstruction (`UE24CS352A: Machine Learning`, PES University RR Campus)  
> **Scope**: Problem Understanding · Data Preprocessing & Quality Audit · Candidate Generation & Feature Selection · 5-Model Training & Group-Aware CV · Best Model Selection · Bayesian Hyperparameter Optimisation  
> **Branch**: `feature/ml-models` (Strictly bounded to `models/`)

---

## 1. Problem Understanding

### 1.1 Challenge Context
Real-world GPS trajectories suffer from urban canyon multipath reflections, satellite signal interference, temporary dropouts, and irregular sampling intervals. As a result, raw GPS coordinates drift onto parallel side-streets, jump across building footprints, or appear misaligned at complex junctions. **PathSnap** formulates map-matching as a two-stage machine learning and sequential decoding pipeline:
1. **Candidate Classification (Phase 1 Core)**: For every GPS observation $t$, retrieve the top $K$ geometrically closest candidate road segments (`arc_id`s) and predict calibrated match probabilities $\hat{p}_{t,k} = P(\text{true road segment} \mid \mathbf{x}_{t,k})$.
2. **Sequential Route Reconstruction (Phase 2)**: Feed those calibrated classifier probabilities as emission likelihoods into a **Hidden Markov Model (HMM) + Viterbi decoder** over road-network connectivity to recover continuous vehicle paths, deployed across two laptops (Laptop 1: FastAPI + trained model; Laptop 2: React UI).

### 1.2 Raw Dataset Structure & Partitioning
Each trip directory contains up to four headerless whitespace-separated files:
- **`.track`**: `[longitude, latitude, timestamp_s]` — noisy GPS trajectory (`~1 Hz`).
- **`.nodes`**: `[longitude, latitude]` — road-network intersection vertices (`0-based node_id`).
- **`.arcs`**: `[from_node_id, to_node_id]` — directed road segments (`0-based arc_id`).
- **`.route`**: `[arc_id]` — chronological reference sequence of road segments traversed (ground truth).

**Crucial structural finding**: Rows in `.track` (observations per second) do **not** correspond 1-to-1 with rows in `.route` (road segments traversed). For example, trip `00000000` has `2,503` GPS observations in `.track` but `179` traversed arcs in `.route`.

| Partition | Trips | GPS Observations | Candidate Rows (`K=6`) | Strict Role & Leakage Policy |
| :--- | :---: | :---: | :---: | :--- |
| **`train`** | **80** | `208,417` | `1,250,502` (`15.03%` positive) | Model fitting, 5-fold trip-grouped `GroupKFold` CV, Bayesian tuning |
| **`student_test`** | **10** | `19,160` | `114,960` (`16.67%` positive) | Development evaluation, 5-model comparison, tuned vs. untuned verification |
| **`evaluator_test`** | **10** | *Reserved* | *Unseen* | Final live evaluation only; zero training/selection/threshold leakage |

*(The 10 held-out `student_test` trip IDs are `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`; the 10 reserved unseen evaluation IDs are `00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094`.)*

---

## 2. Data Preprocessing & Empirical Quality Audit

Programmatic inspection of all 90 available trips in `Student folder/Student folder` uncovered critical real-world anomalies that directly shaped our pipeline ([models/schema_adapter.py](file:///c:/Shashank_documents/B-Tech/SEM-5/mini-projects/ml-hackathon/models/schema_adapter.py)):

1. **Angular vs. Physical Metric Coordinates**:
   - Raw coordinates are WGS84 `(longitude, latitude)` in degrees. Ordinary Euclidean distance in degrees distorts East–West spacing by $\cos(\text{latitude})$ across cities spanning latitudes $34.7^\circ\text{N}$ to $61.7^\circ\text{N}$.
   - **Solution**: Per-trip local tangent-plane equirectangular projection into **meters** centered on each trip's mean latitude ($\Delta x = R \cos\bar{\phi}\,\Delta\lambda$, $\Delta y = R\,\Delta\phi$, $R = 6{,}371{,}000\text{ m}$).
2. **Sampling Irregularities & GPS Dropouts**:
   - Zero `NaN` values and zero non-monotonic timestamps ($\Delta t \le 0$) exist across all 90 trips.
   - In `train`: `96.50%` (`201,053 / 208,337`) of consecutive steps are `1.0 s` apart; `7,284` steps (`3.50%`) exceed `1.0 s`, with `28 / 80` trips having gaps `> 30 s` (max gap `153,064 s`). In `student_test`: `98.62%` (`18,886 / 19,150`) are `1.0 s`, with `264` steps `> 1.0 s` (max gap `210.0 s` in trip `00000065`).
   - **Solution**: Guard velocity divisions (`dt >= 1e-3`, speed clipped to `[0, 60] m/s`) and use central differences only over compact gaps (`<= 10 s`).
3. **Degenerate Geometry & Stationary Jitter**:
   - Audited `100` zero-length arcs in `train` and `6` in `student_test` (`from_node == to_node` coordinates), plus `6` zero-displacement GPS steps in `train`.
   - **Solution**: Safe length denominators (`v_len2 > 1e-12`) and neutral heading defaults ($90^\circ$, $\cos=0$) whenever displacement `< 0.5 m` or speed `< 0.25 m/s` to avoid dividing by GPS jitter.
4. **Reference Route Noise & Multi-Component Networks**:
   - `79 / 80` training road graphs and `10 / 10` student-test graphs have multiple weakly connected components (though `0` isolated vertices exist).
   - `70 / 80` training `.route` sequences and `9 / 10` student-test `.route` sequences are 100% topologically contiguous (`13 / 31,256` disconnected transitions in train; `1 / 3,467` in student_test).
   - Two training trips (`00000033` with `5,844` obs at `122.4 m` median distance to `.route`, and `00000054` with `19,313` obs at `138.5 m` median distance) exhibit severe tunnel/offset noise (`~12%` of training rows), explaining why Fold 1 CV Top-1 Accuracy drops when those trips fall into the validation fold.
5. **Leakage-Free Preprocessing Pipelines**:
   - Every learned preprocessor (`SimpleImputer(strategy="median")` and `StandardScaler()` for scale-sensitive models) is encapsulated inside a scikit-learn `Pipeline`, ensuring fit happens strictly inside each training fold during CV and strictly on the 80 training trips prior to `student_test` evaluation.

---

## 3. Candidate Generation & Feature Selection / Engineering

### 3.1 Adaptive Spatial Candidate Generation ($K=6$)
- Each directed arc is adaptively sampled (5 base points `[0.0, 0.25, 0.5, 0.75, 1.0]` plus extra points every `60 m` for long highway arcs up to 35 points) and indexed via `scipy.spatial.cKDTree`.
- For each GPS point, the nearest `pool_k = 60` sample vertices are retrieved, exact perpendicular point-to-segment distances are computed, and the top **`K = 6` unique `arc_id`s** are retained.
- **Measured Candidate Coverage**:
  - **`student_test` (10 trips, 19,160 obs)**: **`99.32%` coverage (`0.68%` Candidate Miss Rate)** (`131` missed observations; `18,995` single-positive + `34` edge-padded obs = `19,165` positive candidate rows).
  - **`train` (80 trips, 208,417 obs)**: **`89.98%` coverage (`10.02%` Candidate Miss Rate)** (driven primarily by the two high-offset outlier trips `00000033` and `00000054`; across the remaining 78 trips coverage exceeds `99%`).

### 3.2 Selected 20 Leakage-Free Features (`FEATURE_COLUMNS`)
Every feature is computed solely from raw `.track`, `.nodes`, and `.arcs` geometry—zero reference route labels ever enter features:

| Feature Group | Exact Feature Names | Physical & Map-Matching Rationale |
| :--- | :--- | :--- |
| **Absolute Spatial Proximity** (6) | `ortho_dist_m`, `dist_to_start_m`, `dist_to_end_m`, `proj_fraction`, `is_endpoint_proj`, `segment_length_m`, `log_segment_length_m` | Captures Gaussian GPS orthogonal error relative to segment endpoints and distinguishes true road segments from short cross-street stubs clamped at `t=0` or `t=1`. |
| **Intra-Observation Relative Ranking** (4) | `dist_rank`, `dist_diff_from_min_m`, `dist_ratio_to_min`, `dist_zscore_in_obs` | Normalizes absolute GPS noise levels across cities/sensors by comparing each candidate against its `K-1` competing candidates at timestamp $t$. |
| **GPS Kinematics & Directional Alignment** (6) | `gps_speed_mps`, `dt_prev_s`, `heading_diff_deg`, `heading_cos`, `heading_sin`, `speed_weighted_heading_cos` | Essential to disambiguate antiparallel two-way road arcs (`heading_cos ≈ +1` vs `-1`) and suppress noisy GPS heading during urban stops (`speed < 5 m/s`). |
| **Unsupervised Sequential & Graph Context** (3) | `prev_cand_connected`, `min_dist_from_prev_cand_m`, `node_in_out_degree_sum` | Measures whether a candidate is topologically reachable from the previous observation's candidate pool and captures junction complexity without using labels. |

*Top Empirical Predictors* (verified in [feature_importance_analysis.png](file:///c:/Shashank_documents/B-Tech/SEM-5/mini-projects/ml-hackathon/models/artifacts/plots/feature_importance_analysis.png)): `dist_diff_from_min_m` (`38.5%` permutation share in `HistGradientBoosting`, `59.8%` Gini share in `Decision Tree`), `heading_diff_deg` (`30.3%` in `HistGradientBoosting`), `is_endpoint_proj` (`12.7%`), and `speed_weighted_heading_cos` (`8.7%`).

---

## 4. Five-Model Training & Group-Aware Cross-Validation

All 5 models span **5 distinct algorithm families** and were evaluated identically using:
- **Training & CV**: **5-Fold Trip-Grouped `GroupKFold(n_splits=5)`** across the 80 training trips (`1,250,502` rows), asserting zero shared trips across train/validation splits.
- **Unbiased Train-Fit Diagnostics**: Evaluated across all 80 training trips (`113,904` stride-sampled rows) to measure overfitting (`Train Macro F1 - CV Macro F1`).
- **Held-Out Evaluation**: All 10 `student_test` trips (`114,960` candidate rows, `19,160` observations).

### 4.1 Consolidated Model Comparison Table (PDF Section 3.1 Format)

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** ★ | Boosting | `lr=0.08, max_iter=200, max_leaf_nodes=63, l2=1.0, class_weight=balanced, early_stopping=True` | `0.8608` *(±0.0492)* | **`0.8790`** | `0.9212` | `0.8928` | `0.1672` | `0.3112` *(31.12 ms/trip)* |
| **Random Forest** | Ensemble (Bagging) | `n_estimators=120, max_depth=16, min_samples_leaf=10, max_samples=0.5, class_weight=balanced_subsample` | `0.8732` *(±0.0409)* | `0.8783` | **`0.9219`** | `0.8999` | **`0.1556`** | `0.1325` *(13.25 ms/trip)* |
| **KNN (Distance-Weighted)** | Instance-based | `k=15, weights=distance, algorithm=kd_tree, p=2, StandardScaler, max_index=25000` | **`0.8918`** *(±0.0221)* | `0.8662` | `0.9145` | **`0.9135`** | `0.3444` | `22.3724` *(2,237.24 ms/trip)* |
| **Decision Tree** | Tree-based | `criterion=gini, max_depth=12, min_samples_leaf=20, class_weight=balanced` | `0.8389` *(±0.0683)* | `0.8727` | `0.9149` | `0.8850` | `0.2059` | **`0.0182`** *(1.82 ms/trip)* |
| **Logistic Regression** | Linear and probabilistic | `C=1.0, L2 penalty, solver=lbfgs, class_weight=balanced, StandardScaler` | `0.8459` *(±0.0349)* | `0.8714` | `0.9167` | `0.8701` | `0.2148` | `0.0288` *(2.88 ms/trip)* |

*(★ = Selected model for Phase 1 & Member 3 Bayesian Tuning / HMM integration.)*

### 4.2 Extended Overfitting & Sequence Metrics Summary

| Model | Train Macro F1 | CV Macro F1 | Overfit Gap (Train - CV) | Student-Test Acc | Precision (Macro) | Recall (Macro) | Sequence F1 | Train Time (s) | Disk Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | `0.9029` | `0.8608` | `+0.0420` | `0.9338` | `0.8613` | **`0.9382`** | **`0.8771`** | `12.36 s` | **`0.65 MB`** |
| **Random Forest** | `0.9071` | `0.8732` | `+0.0338` | `0.9393` | `0.8717` | **`0.9382`** | `0.8757` | `49.20 s` | `25.03 MB` |
| **KNN (Distance-Weighted)** | `0.9009` | `0.8918` | `+0.0091` | `0.9521` | `0.9148` | `0.9122` | `0.8608` | `2.08 s` | `2.92 MB` |
| **Decision Tree** | `0.8908` | `0.8389` | `+0.0519` | `0.9284` | `0.8523` | `0.9341` | `0.8685` | `29.78 s` | `0.11 MB` |
| **Logistic Regression** | `0.8509` | `0.8459` | `+0.0050` | `0.9171` | `0.8346` | `0.9289` | `0.8640` | `6.32 s` | `<0.01 MB` |

---

## 5. Best Model Selection & Multi-Criteria Justification

We selected **`HistGradientBoosting`** (`composite_score = 0.87857`, ahead of Random Forest `0.85083`, Logistic Regression `0.85024`, Decision Tree `0.83062`, and KNN `0.77938`) based on six concrete engineering and empirical criteria:

1. **Highest End-to-End Map-Matching Accuracy**: Achieves **#1 Observation-level Top-1 Accuracy (`87.90%`, or `88.51%` when reference arc is in candidates)** and **#1 Route Sequence F1 (`0.8771`)** on `student_test`, outperforming KNN despite KNN's higher raw candidate Macro F1 (`0.9135` vs `0.8928`). Confusion matrix analysis proves why: KNN drops to `85.2%` positive-class recall (`2,830` False Negatives), whereas HistGradientBoosting attains **`94.5%` positive-class recall (`18,108 TP`, only `1,057 FN`)**, ensuring true road segments rank #1 per GPS point.
2. **Calibrated HMM Emission Probabilities (`Log Loss = 0.1672`)**: KNN suffers severe boundary probability truncation (`Log Loss = 0.3444`, >2× worse), creating zero-probability spikes that break Viterbi paths. HistGradientBoosting directly optimizes regularized binary log-loss over sequential residuals.
3. **4× Faster Training for Bayesian Tuning (`12.36 s` vs `49.20 s`)**: Enables rapid multi-trial 5-fold GroupKFold Bayesian hyperparameter optimisation compared to Random Forest.
4. **38× Smaller Production Artifact (`0.65 MB` vs `25.03 MB`) & Fast FastAPI Serving (`31.12 ms/trip`)**: Compact shallow trees deploy cleanly to Laptop 1 (vs KNN's `2,237 ms/trip`, which is **72× slower**).
5. **Genuine Iterative Learning Convergence**: Binary log loss decays smoothly from `0.6931` $\to$ `0.1283` (train) and `0.6931` $\to$ `0.1381` (10% internal early-stopping validation split) with zero divergence.

---

## 6. Bayesian Hyperparameter Optimisation (Strictly GP-EI, Zero Grid/Random Search)

Implemented in [models/bayesian_tuning.py](file:///c:/Shashank_documents/B-Tech/SEM-5/mini-projects/ml-hackathon/models/bayesian_tuning.py) using **strictly Bayesian Hyperparameter Optimisation** (`GaussianProcessRegressor` with `ConstantKernel * Matern(nu=2.5) + WhiteKernel` surrogate and **Expected Improvement** acquisition function — **zero `GridSearchCV` or `RandomizedSearchCV`**).

- **Search Space (5D)**: `learning_rate ∈ [0.03, 0.18]` (log10), `max_leaf_nodes ∈ [31, 127]`, `max_depth ∈ [8, 18]`, `min_samples_leaf ∈ [15, 80]`, `l2_regularization ∈ [0.1, 10.0]` (log10).
- **Optimisation Objective**: 5-Fold trip-grouped `GroupKFold` CV Macro F1-score on the 80 training trips (`12` trials: `3` initial anchors + `9` GP-EI sequential acquisitions; total wall time `130.85 s` including full 80-trip verification & refit).
- **Optimal Configuration Found (Trial 6 — GP-EI)**:
  - `learning_rate = 0.16973`, `max_leaf_nodes = 120`, `max_depth = 15`, `min_samples_leaf = 33`, `l2_regularization = 0.389`, `max_iter = 250`.

### Untuned vs. Bayesian-Tuned Performance Comparison

| Variant | Full 80-Trip CV Macro F1 | Student-Test Top-1 Accuracy | Student-Test Macro F1 | Student-Test Log Loss | Sequence F1 | Route Continuity | Inference Time (s) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned HistGradientBoosting** | `0.8608 (±0.0492)` | **`0.8790`** | `0.8928` | `0.1672` | `0.8771` | **`0.9212`** | `0.3112 s` | `12.36 s` |
| **Bayesian-Tuned HistGradientBoosting (GP-EI)** | **`0.8659 (±0.0475)`** | `0.8777` | **`0.8998`** | **`0.1583`** | **`0.8799`** | `0.9209` | **`0.2726 s`** | **`12.25 s`** |
| **Net Improvement (Δ)** | **`+0.0051`** *(lower std)* | `-0.0013` | **`+0.0070`** | **`-0.0089` (-5.3%)** | **`+0.0028`** | `-0.0003` | **`-12.4%` faster** | `-0.11 s` |

**Conclusion**: Bayesian optimisation simultaneously improved **80-trip CV Macro F1 (`+0.0051`)**, reduced fold-to-fold variance (`0.0492` $\to$ `0.0475`), lifted **Student-Test Macro F1 (`+0.0070`)** and **Sequence F1 (`0.8799`)**, cut **Log Loss by `5.3%` (`0.1672` $\to$ `0.1583`)** for cleaner HMM emission probabilities, and accelerated inference latency by `12.4%` (`27.26 ms/trip`).

---

## 7. Quick-Reference Index of Artifacts & Charts

- **Trained Models (`models/artifacts/models/`)**: `selected_best_model.joblib` (`0.65 MB`), `selected_best_model_tuned.joblib` (`1.16 MB`), plus all 5 individual `.joblib` models and `selected_model_metadata.json`.
- **CSV / JSON Results (`models/artifacts/results/`)**: `section_3_1_model_comparison.csv`, `bayesian_optimisation_trials.csv`, `tuned_vs_untuned_comparison.csv`, `per_trip_student_test_breakdown.csv`, `detailed_experiment_results.json`, `m3_handoff_student_test_probabilities.csv.gz`.
- **Visualisations (`models/artifacts/plots/`)**: `cv_macro_f1_comparison.png`, `student_test_metrics_comparison.png`, `log_loss_comparison.png`, `confusion_matrices_grid.png`, `overfitting_diagnostics.png`, `computational_efficiency.png`, `iterative_training_validation_loss.png`, `feature_importance_analysis.png`.
- **Live Monitoring Files**: `models/artifacts/results/training_status.json` & `models/artifacts/results/training_progress.log`.
