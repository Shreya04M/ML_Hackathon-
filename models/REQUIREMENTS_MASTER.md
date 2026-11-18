# PathSnap: End-to-End Requirements, Evaluation Rubric & Member 2 Architecture Specification

> **Document Status**: Authoritative Requirements & Traceability Master Document  
> **Project**: PathSnap — Distributed Map-Matching & Route Reconstruction (`UE24CS352A: Machine Learning`, PES University RR Campus)  
> **Scope Owner**: **Member 2 — ML Model Development & Selection (`models/`, branch `feature/ml-models`)**  
> **Strict Isolation Guarantee**: All Member 2 code, artifacts, visualisations, logs, and documentation reside exclusively inside `models/`. Zero edits leak into Member 1 (`preprocessing/`, `visualisation/`), Member 3 (`optimisation/`, `route_reconstruction/`), or Member 4 (`evaluation/`, `backend/`, `frontend/`) modules.

---

## 1. End-to-End Project Specification (Extracted Verbatim & Structurally From All 5 Official PDFs)

### 1.1 `Main Guidelines.pdf` — Hackathon Guide
1. **Challenge Overview (Section 1)**:
   - GPS tracking systems produce inaccurate location measurements due to urban buildings, signal interference, multipath effects, GPS dropouts, and low sampling frequencies.
   - Vehicles travelling along a road may appear to move onto a parallel street, jump across buildings, or deviate from their actual route.
   - Objective: Develop **PathSnap**, a distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to a road network.
   - Reference map semantics: Blue dots represent raw GPS points; the green line represents the actual vehicle path (ground truth).
2. **Pre-Implementation Mandate (Section 2)**:
   - Thoroughly read all supporting documents before beginning implementation or using AI tools.
   - Make informed technical decisions grounded in the provided dataset and guidelines rather than generic AI suggestions.
3. **Two-Laptop Deployment Architecture (Section 3)**:
   - **Laptop 1 (Model and API)**: Hosts the trained classifier, HMM/Viterbi implementation, road-network information, and server-side evaluation process.
   - **Laptop 2 (User Interface)**: Displays the ten evaluation-trip options, sends the selected trip to the API, and displays the returned route and results.
4. **Expected 12-Stage Work Process (Section 4)**:
   1. Understand the problem.
   2. Explore and understand the raw dataset.
   3. Develop an independent data-processing and feature-engineering approach.
   4. Train and compare multiple machine learning models.
   5. Select the best-performing model based on experimental evidence.
   6. Perform Bayesian hyperparameter optimisation.
   7. Implement HMM and Viterbi-based route reconstruction.
   8. Export the trained model and required artifacts.
   9. Develop the FastAPI backend.
   10. Develop the evaluator-facing UI.
   11. Integrate and test the complete system.
   12. Prepare the final technical report.

---

### 1.2 `Dataset Exploration.pdf` — Dataset Description & Exploration
1. **Raw File Schemas (Section 3)**:
   | File Extension | General Meaning | Exact Column Schema (No Headers, Whitespace-Separated) |
   | :--- | :--- | :--- |
   | `.track` | GPS observations associated with a trajectory | 3 columns: `longitude`, `latitude`, `timestamp` (seconds). Nominally 1 Hz (~97% of steps are 1 s apart). |
   | `.nodes` | Geographical nodes in the road network | 2 columns: `longitude`, `latitude` (0-based row index serves as `node_id`). |
   | `.arcs` | Directed road-network segments connecting nodes | 2 integer columns: `from_node_id`, `to_node_id` (0-based row index serves as `arc_id`). |
   | `.route` | Reference route information, where supplied | 1 integer column: chronological sequence of ground-truth `arc_id` values traversed by the vehicle. |
   - **Critical Warning**: *"Do not assume that a row in a GPS file corresponds directly to a row in a road-network file."*
2. **Initial Dataset Exploration Checklist (Section 4)**:
   - Number of trips in each partition; number of GPS observations per trip; file availability and file sizes; data types and column names; missing or invalid GPS observations; timestamp ordering and irregular time gaps; geographical extent of each trajectory; number of road-network nodes and segments; road-network connectivity; isolated nodes and disconnected components; trips with unusually large GPS gaps or other anomalies; availability and structure of reference route information.
3. **Geographical Data Considerations (Section 5)**:
   - Longitude and latitude are angular coordinates; ordinary Euclidean distance in degrees is NOT an appropriate measure of physical distance.
   - Must choose and justify an appropriate projected coordinate system or geodesic distance calculation, verify coordinate order (`lon, lat`), inspect coordinate plausibility, and handle GPS noise, low sampling rates, gaps, and irregular timestamps.
4. **Reference Route & Data Limitations (Section 6)**:
   - Treat `.route` as reference information; do not assume error-free physical ground truth.
   - GPS observations may not lie exactly on the road network, and supplied `.route` files may contain occasional discontinuities or limitations that must be documented.
5. **Dataset Partitions & 8 Mandatory Leakage Prevention Rules (Section 7)**:
   | Partition | Trips | Permitted Use |
   | :--- | :---: | :--- |
   | `Training` | **80** | Training, development, cross-validation, and model optimisation |
   | `Student testing` | **10** | Development-stage evaluation and model comparison |
   | `Evaluator testing` | **10** | Final evaluation only (reserved/unseen) |
   - **Rules**:
     1. No observations from the same trip may be distributed across different partitions (strict trip-level `GroupKFold` during CV).
     2. Evaluator-testing data must not be used for model training.
     3. Evaluator-testing labels must not be used for feature selection.
     4. Evaluator-testing results must not be used to choose algorithms or hyperparameters.
     5. Do not manually alter predictions based on known evaluation routes.
     6. Do not hardcode expected answers for evaluation trips.
     7. Fit learned preprocessing transformations (scalers, imputers) using permitted training data only (inside each CV fold / on the 80 training trips only).
     8. Keep evaluator ground truth separate from the prediction pipeline.

---

### 1.3 `Models and Training Guidelines.pdf` — Primary Member 2 Specification
1. **Independent Model Selection & Experimentation (Section 2)**:
   - Explore different ML algorithms across multiple algorithm families.
   - Objective is not merely high accuracy, but demonstrating deep technical understanding of **why** a particular model suits PathSnap.
   - **Minimum 5 distinct models** must be trained, evaluated, and compared with experimental evidence before selecting the final model.
2. **Algorithm Families (Section 3)**:
   - Linear & probabilistic (`LogisticRegression`, etc.), Instance-based (`KNN`, Distance-Weighted KNN), Tree-based (`DecisionTree`), Ensemble (`RandomForest`), Boosting (`HistGradientBoosting`), Kernel-based (`SVM`), Neural networks (`MLP`), Unsupervised/clustering.
   - Must explain how candidate probabilities are produced for downstream HMM/Viterbi emissions.
3. **Mandatory 8-Point Experimental Design (Section 4)**:
   1. Train at least five distinct models.
   2. Record the configuration of each model.
   3. Use the 80 training trips for model fitting and cross-validation.
   4. Use the 10 student-testing trips for development-stage model comparisons.
   5. Apply preprocessing consistently.
   6. Avoid selecting models using training accuracy alone.
   7. Record training time and inference time where practical.
   8. Retain experiment logs, random seeds, configurations, and results.
4. **Mandatory 12 Evaluation Metrics (Section 5)**:
   - `Accuracy`, `Precision`, `Recall`, `F1-score` (Macro F1), `Confusion Matrix`, `Log Loss`, `Observation-level Top-1 Accuracy`, `Candidate Miss Rate`, `Route Continuity`, `Sequence-level Performance`, `Training Time`, `Inference Time`.
   - **Observation-level Top-1 Accuracy**: Select the candidate with the highest predicted score for each GPS observation and compare against the reference road segment.
   - **Candidate Miss Rate**: Fraction of GPS observations where the reference road segment is absent from the generated candidate set.
   - Must explain the **primary metric** used to select the final model and why it is appropriate.
5. **Model Comparison & Selection Criteria (Section 6)**:
   - Compare generalisation performance, stability across validation folds, probability quality (Log Loss), inference speed, model complexity, and compatibility with the sequential decoder (HMM/Viterbi).
6. **Bayesian Optimisation & HMM/Viterbi Context (Sections 7–9)**:
   - Member 2 supplies the selected untuned model, loading code, and candidate probability predictions so Member 3 can run Bayesian hyperparameter tuning (using group-aware CV on the 80 training trips) and HMM/Viterbi decoding.
7. **Final Evaluation & Independent Implementation (Sections 5, 6, 11)**:
   - Ground-truth labels stay server-side; API logs on Laptop 1 are inspected by evaluators; all code, configurations, weights, and results must be independent and verifiable on GitHub.

---

### 1.4 `Deliverables.pdf` — Technical Report & Submission Guidelines
- **Phase 1 Deadline Note**: *"There will be first phase evaluation before lunch break."*
- **Deliverable 1 (Public GitHub Repository)**: Complete source code for data processing, feature engineering, model training (every model experimented with), model evaluation, optimisation, HMM/Viterbi, API, UI, deployment, visualisations, supporting CSV/JSON logs, model weights, and `README.md`.
- **Deliverable 2 (Consolidated PDF Technical Report)** — **Section 3 (Member 2 Core)**:
  - **Section 3.1 Overall Model Comparison Table** (Exact columns):
    `Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time`
    *(Note: Use actual measured results; if a metric is N/A, state N/A and explain why.)*
  - **Section 3.2 Compulsory Questions for EVERY Model** (Separate subsection per model):
    - *A. Model Understanding*:
      - Q1: Why did you consider this algorithm suitable for the PathSnap problem?
      - Q2: What are the major strengths and limitations of this algorithm for this task?
    - *B. Training and Configuration*:
      - Q5: What were the important hyperparameters and their initial values?
      - Q6: What loss function or optimisation objective was used?
    - *C. Experimental Results*:
      - Q7: What was the model's performance on the student-testing dataset?
      - Q8: What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss (where applicable) of the model on the student-testing dataset?
      - Q9: Did you observe overfitting or underfitting? What evidence supports your conclusion?
    - *(Mandatory Constraint: Connect every explanation to actual experiments, observations, and measured results—no generic textbook prose.)*
  - **Section 3.3 Model Visualisations (COMPULSORY FOR EVERY MODEL)**:
    - Cross-validation Macro F1-score comparison across all models.
    - Student-testing Top-1 Accuracy and Macro F1-score comparison.
    - Log Loss comparison, wherever applicable.
    - Confusion matrices as diagnostic visualisations.
    - **Training and validation loss plots ONLY for models that produce epoch/iteration-wise training histories** (do NOT fabricate artificial curves for algorithms that lack training histories).
    - Plots comparing performance across models to justify model selection.

---

### 1.5 `Assessment and Plagiarism Policy.pdf` & 25-Mark Evaluation Rubric
1. **Evaluation Weightage (25 Marks Total)**:
   | Evaluation Component | Marks | Member Alignment |
   | :--- | :---: | :--- |
   | Data Preprocessing and Feature Extraction | 5 | Member 1 |
   | **Model selection and training** | **5** | **Member 2 (Direct Ownership)** |
   | Hyperparameter tuning | 5 | Member 3 (+ M2 handoff) |
   | **ML Ops** (Pipeline reproducibility, artifacts, logging, API serving) | **5** | **Members 2, 3, 4** |
   | **Demo and Deliverables** (Report tables, charts, live verified execution) | **5** | **All Members** |
2. **Evaluator Instructions**:
   - Check Laptop 1 API logs (mandatory).
   - Verify actual trained model execution (predictions must come from saved trained pipelines, never hardcoded outputs).
   - Viva readiness: Every member must explain preprocessing, feature engineering, model selection, overfitting/underfitting evidence, and HMM compatibility.

---

## 2. Empirical Dataset Audit (`Student folder/Student folder`)

Programmatic inspection of all 90 available trips (`80 train` + `10 student_test`) yielded these exact ground-truth properties:

| Metric | Training Partition (`train/`, 80 trips) | Student Testing (`student_test/`, 10 trips) |
| :--- | :---: | :---: |
| **Total GPS Observations (`.track`)** | `208,417` | `19,160` |
| **Observations per Trip (min / median / max)** | `361` / `1,529.0` / `19,313` | `262` / `1,420.0` / `3,958` |
| **Road Nodes per Trip (min / median / max)** | `299` / `9,608.0` / `155,003` | `1,770` / `6,596.5` / `26,635` |
| **Road Arcs per Trip (min / median / max)** | `649` / `19,526.5` / `307,392` | `3,577` / `13,145.0` / `56,027` |
| **Reference Route Length (arcs: min / med / max)** | `59` / `280.0` / `2,266` | `95` / `250.0` / `1,323` |
| **Timestamp Steps Exactly `1.0 s`** | `201,053 / 208,337` (`96.50%`) | `18,886 / 19,150` (`98.62%`) |
| **Timestamp Steps `> 1.0 s` / Max Gap** | `7,284` (`3.50%`) / `153,064.0 s` | `264` (`1.38%`) / `210.0 s` |
| **Non-monotonic Timestamps ($\Delta t \le 0$) & NaNs** | `0` / `0` | `0` / `0` |
| **Trips with GPS Gap `> 30 s`** | `28 / 80` trips | `2 / 10` trips (`00000065`, `00000076`) |
| **Stationary GPS Steps ($\Delta \text{pos} = 0$)** | `6` steps | `0` steps |
| **Degenerate Zero-Length Arcs (`from == to` coords)** | `100` arcs | `6` arcs |
| **100% Connected Reference Routes** | `70 / 80` trips (`13 / 31,256` disc. transitions) | `9 / 10` trips (`1 / 3,467` disc. transitions) |
| **Multi-Component Road Networks** | `79 / 80` trips | `10 / 10` trips |

- **Student Testing Trip IDs**: `['00000000', '00000007', '00000009', '00000040', '00000052', '00000056', '00000065', '00000076', '00000077', '00000091']`
- **Reserved Evaluator Testing IDs (Unseen)**: `['00000015', '00000017', '00000023', '00000030', '00000049', '00000059', '00000067', '00000073', '00000087', '00000094']`

---

## 3. Member 2 Architectural Traceability Matrix

| Official Requirement | Implementation File in `models/` | Verification Artifact |
| :--- | :--- | :--- |
| **Phase 1 Core Models (5) + Skeleton MLP (6)** | `logistic_regression.py`, `knn.py`, `decision_tree.py`, `random_forest.py`, `gradient_boosting.py`, `mlp.py` | Saved `.joblib` models + JSON metadata in `models/artifacts/models/` |
| **Leakage-Free Spatial Feature Ingestion** | `schema_adapter.py` | Supports M1 processed feature tables + standalone 20-feature equirectangular spatial builder (`K=5`, `1.32%` miss rate on `student_test`) |
| **Group-Aware 5-Fold CV (`trip_id` grouping)** | `model_selection.py` (`run_grouped_cross_validation`) | `GroupKFold(n_splits=5)` fold logs & out-of-fold metrics JSON |
| **All 12 Evaluation Metrics** | `model_selection.py` (`evaluate_model`) | `detailed_experiment_results.json` & `section_3_1_model_comparison.csv` |
| **Section 3.3 Compulsory Plots (No Fabricated Curves)** | `plots.py` | `models/artifacts/plots/*.png` (CV F1, Test Top-1 & Macro F1, Log Loss, Confusion Matrices, Latency Pareto, HistGB/MLP real loss history, Feature Importances) |
| **M3 / M4 Reproducible Handoff** | `train.py` & `__init__.py` | `selected_best_model.joblib`, `m3_handoff_student_test_probabilities.csv.gz`, `load_final_model()` |
| **Section 3.2 Q1–Q9 Experimental Report** | `README.md` | Empirical answers tied to measured CSV/JSON outputs |
