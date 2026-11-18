import React from "react";
import { BarChartIcon, CheckCircleIcon } from "./Icons";

const MODEL_COMPARISON = [
  {
    name: "HistGradientBoosting",
    family: "Boosting",
    selected: true,
    cvF1: "0.8608 ± 0.049",
    top1: "87.90%",
    continuity: "92.12%",
    testF1: "0.8928",
    logLoss: "0.1672",
    seqF1: "0.8771",
    inferMs: "31.12 ms",
  },
  {
    name: "Random Forest",
    family: "Ensemble (Bagging)",
    selected: false,
    cvF1: "0.8732 ± 0.041",
    top1: "87.83%",
    continuity: "92.19%",
    testF1: "0.8999",
    logLoss: "0.1556",
    seqF1: "0.8757",
    inferMs: "13.25 ms",
  },
  {
    name: "Logistic Regression",
    family: "Linear Probabilistic",
    selected: false,
    cvF1: "0.8459 ± 0.035",
    top1: "87.14%",
    continuity: "91.67%",
    testF1: "0.8701",
    logLoss: "0.2148",
    seqF1: "0.8640",
    inferMs: "2.88 ms",
  },
  {
    name: "Decision Tree",
    family: "Tree-based",
    selected: false,
    cvF1: "0.8389 ± 0.068",
    top1: "87.27%",
    continuity: "91.49%",
    testF1: "0.8850",
    logLoss: "0.2059",
    seqF1: "0.8685",
    inferMs: "1.82 ms",
  },
  {
    name: "KNN (Distance-Weighted)",
    family: "Instance-based",
    selected: false,
    cvF1: "0.8918 ± 0.022",
    top1: "86.62%",
    continuity: "91.45%",
    testF1: "0.9135",
    logLoss: "0.3444",
    seqF1: "0.8608",
    inferMs: "2237.24 ms",
  },
];

const PIPELINE_PROGRESSION = [
  {
    stage: "Classifier Only (HistGB Baseline)",
    accuracy: 87.90,
    continuity: 92.12,
    f1: 87.71,
  },
  {
    stage: "Distance HMM + Viterbi + Constrained Recovery",
    accuracy: 90.79,
    continuity: 97.85,
    f1: 94.55,
  },
];

export function BenchmarksView() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Hero Summary Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 14,
        }}
      >
        <div className="metric-tile">
          <div className="metric-label">Candidate Coverage (K=6)</div>
          <div className="metric-value">99.32%</div>
          <div className="progress-track">
            <div className="progress-fill high" style={{ width: "99.32%" }} />
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            0.68% miss rate across 19,160 test obs
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-label">HMM Pointwise Accuracy</div>
          <div className="metric-value">90.79%</div>
          <div className="progress-track">
            <div className="progress-fill high" style={{ width: "90.79%" }} />
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            +2.89% lift over raw classifier (87.90%)
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-label">Route Continuity</div>
          <div className="metric-value">97.85%</div>
          <div className="progress-track">
            <div className="progress-fill high" style={{ width: "97.85%" }} />
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            18,739 / 19,150 connected transitions
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-label">Road-Overlap F1</div>
          <div className="metric-value">0.9455</div>
          <div className="progress-track">
            <div className="progress-fill high" style={{ width: "94.55%" }} />
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            Precision 90.90% · Recall 99.18%
          </div>
        </div>
      </div>

      {/* 5-Model Comparison Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <BarChartIcon size={17} />
            5-Model Family Evaluation (GroupKFold CV &amp; Student Test)
          </h2>
          <span className="badge badge-info">
            114,960 Candidate Pairs · 10 Held-Out Trips
          </span>
        </div>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Family</th>
                <th>CV Macro F1</th>
                <th>Top-1 Acc</th>
                <th>Continuity</th>
                <th>Test Macro F1</th>
                <th>Log Loss ↓</th>
                <th>Seq F1</th>
                <th>Latency / Trip</th>
              </tr>
            </thead>
            <tbody>
              {MODEL_COMPARISON.map((row) => (
                <tr
                  key={row.name}
                  className={row.selected ? "row-highlight" : ""}
                >
                  <td style={{ fontWeight: 600 }}>
                    <div
                      style={{ display: "flex", alignItems: "center", gap: 8 }}
                    >
                      {row.name}
                      {row.selected && (
                        <span className="badge badge-success">
                          <CheckCircleIcon size={12} /> Selected
                        </span>
                      )}
                    </div>
                  </td>
                  <td style={{ color: "var(--text-secondary)" }}>
                    {row.family}
                  </td>
                  <td className="mono">{row.cvF1}</td>
                  <td className="mono" style={{ fontWeight: 600 }}>
                    {row.top1}
                  </td>
                  <td className="mono">{row.continuity}</td>
                  <td className="mono">{row.testF1}</td>
                  <td className="mono">{row.logLoss}</td>
                  <td className="mono">{row.seqF1}</td>
                  <td className="mono">{row.inferMs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Two-Column: GP-EI Bayesian Tuning & Phase 2 HMM Progression */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))",
          gap: 20,
        }}
      >
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              GP-EI Bayesian Hyperparameter Tuning
            </h2>
            <span className="badge badge-success">No Grid / Random Search</span>
          </div>
          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Untuned Baseline</th>
                  <th>GP-EI Tuned</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>CV Macro F1</td>
                  <td className="mono">0.8608</td>
                  <td className="mono" style={{ fontWeight: 600 }}>
                    0.8659
                  </td>
                  <td>
                    <span className="badge badge-success">+0.0051</span>
                  </td>
                </tr>
                <tr>
                  <td>Test Macro F1</td>
                  <td className="mono">0.8928</td>
                  <td className="mono" style={{ fontWeight: 600 }}>
                    0.8998
                  </td>
                  <td>
                    <span className="badge badge-success">+0.0070</span>
                  </td>
                </tr>
                <tr>
                  <td>Log Loss (Calibration) ↓</td>
                  <td className="mono">0.1672</td>
                  <td className="mono" style={{ fontWeight: 600 }}>
                    0.1583
                  </td>
                  <td>
                    <span className="badge badge-success">-0.0089</span>
                  </td>
                </tr>
                <tr>
                  <td>Sequence F1</td>
                  <td className="mono">0.8771</td>
                  <td className="mono" style={{ fontWeight: 600 }}>
                    0.8799
                  </td>
                  <td>
                    <span className="badge badge-success">+0.0028</span>
                  </td>
                </tr>
                <tr>
                  <td>Inference Time (10 trips)</td>
                  <td className="mono">0.3112 s</td>
                  <td className="mono" style={{ fontWeight: 600 }}>
                    0.2726 s
                  </td>
                  <td>
                    <span className="badge badge-success">-12.4%</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              Distance HMM + Constrained Recovery Lift
            </h2>
            <span className="badge badge-info">
              β=30.0m · w=1.95
            </span>
          </div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            {PIPELINE_PROGRESSION.map((item) => (
              <div
                key={item.stage}
                style={{
                  padding: 14,
                  borderRadius: 10,
                  background: "var(--bg-subtle)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    marginBottom: 10,
                    color: "var(--text-primary)",
                  }}
                >
                  {item.stage}
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 10,
                  }}
                >
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                      Accuracy
                    </div>
                    <div className="mono" style={{ fontSize: 16, fontWeight: 700 }}>
                      {item.accuracy.toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                      Continuity
                    </div>
                    <div className="mono" style={{ fontSize: 16, fontWeight: 700 }}>
                      {item.continuity.toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                      Route / Seq F1
                    </div>
                    <div className="mono" style={{ fontSize: 16, fontWeight: 700 }}>
                      {item.f1.toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}

            <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
              Optuna-tuned HMM parameters (25 trials, 80 training trips, 208,417
              observations):{" "}
              <code className="mono">beta = 30.0m</code>,{" "}
              <code className="mono">transition_weight = 1.9486</code>,{" "}
              <code className="mono">backward_tolerance_m = 30.0m</code>.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

