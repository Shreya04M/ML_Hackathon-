
import React from "react";

export function Results({ result }) {
  const metrics = result?.metrics || {};

  function percentage(value) {
    return typeof value === "number"
      ? `${(value * 100).toFixed(2)}%`
      : "—";
  }

  const items = [
    ["Accuracy", percentage(metrics.accuracy)],
    ["Route Precision", percentage(metrics.route_precision)],
    ["Route Recall", percentage(metrics.route_recall)],
    ["Route F1", percentage(metrics.route_f1)],
    ["Jaccard Similarity", percentage(metrics.jaccard_similarity)],
    [
      "Execution Time",
      typeof result?.execution_time === "number"
        ? `${result.execution_time.toFixed(3)} s`
        : "—",
    ],
  ];

  return (
    <section style={{ marginTop: "24px" }}>
      <h2>Evaluation Results</h2>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "16px",
      }}>
        {items.map(([label, value]) => (
          <div key={label} style={{
            padding: "20px",
            border: "1px solid #e2e8f0",
            borderRadius: "12px",
            background: "#f8fafc",
          }}>
            <p style={{ color: "#64748b" }}>{label}</p>
            <h2>{value}</h2>
          </div>
        ))}
      </div>

      {!result && (
        <p style={{ color: "#64748b" }}>
          Awaiting model prediction and evaluation.
        </p>
      )}
    </section>
  );
}