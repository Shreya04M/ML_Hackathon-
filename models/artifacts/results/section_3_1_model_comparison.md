| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KNN (Distance-Weighted) | Instance-based | k=15, weights=distance, algorithm=kd_tree, p=2, StandardScaler, max_index=25000 | 0.8918 | 0.8662 | 0.9145 | 0.9135 | 0.3444 | 22.3724 |
| Random Forest | Ensemble (Bagging) | n_estimators=120, max_depth=16, min_samples_leaf=10, max_samples=0.5, class_weight=balanced_subsample | 0.8732 | 0.8783 | 0.9219 | 0.8999 | 0.1556 | 0.1325 |
| HistGradientBoosting | Boosting | lr=0.08, max_iter=200, max_leaf_nodes=63, l2=1.0, class_weight=balanced, early_stopping=True | 0.8608 | 0.879 | 0.9212 | 0.8928 | 0.1672 | 0.3112 |
| Decision Tree | Tree-based | criterion=gini, max_depth=12, min_samples_leaf=20, class_weight=balanced | 0.8389 | 0.8727 | 0.9149 | 0.885 | 0.2059 | 0.0182 |
| Logistic Regression | Linear and probabilistic | C=1.0, L2 penalty, solver=lbfgs, class_weight=balanced, StandardScaler | 0.8459 | 0.8714 | 0.9167 | 0.8701 | 0.2148 | 0.0288 |
