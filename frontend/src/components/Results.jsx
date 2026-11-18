import React, { useState } from "react";
import { CheckCircleIcon, CheckIcon, CopyIcon, ShieldCheckIcon } from "./Icons";

export function Results({ result }) {
  const [copied, setCopied] = useState(false);

  if (!result) {
    return (
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Server-Side Evaluation Metrics</h2>
          <span className="badge badge-neutral">
            <ShieldCheckIcon /> Ground Truth Isolated
          </span>
        </div>
        <div
          className="card-body"
          style={{
            color: "var(--text-muted)",
            fontSize: 13.5,
            textAlign: "center",
            padding: "28px 20px",
          }}
        >
          Run route reconstruction on a trip to view server-computed accuracy,
          overlap F1, Jaccard similarity, and the recovered road-arc sequence.
        </div>
      </div>
    );
  }

  const metrics = result.metrics || {};
  const arcSequence = Array.isArray(result.predicted_route)
    ? result.predicted_route
    : [];

  function formatPct(value) {
    return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
  }

  function pctValue(value) {
    return typeof value === "number"
      ? Math.max(0, Math.min(100, value * 100))
      : 0;
  }

  function barTone(value) {
    if (typeof value !== "number") return "";
    if (value >= 0.9) return "high";
    if (value >= 0.75) return "";
    return "mid";
  }

  const kpis = [
    {
      label: "Observation Accuracy",
      raw: metrics.accuracy,
      display: formatPct(metrics.accuracy),
      hint: "Pointwise HMM arc match",
    },
    {
      label: "Route F1 Score",
      raw: metrics.route_f1,
      display: formatPct(metrics.route_f1),
      hint: "Harmonic mean (P / R)",
    },
    {
      label: "Route Precision",
      raw: metrics.route_precision,
      display: formatPct(metrics.route_precision),
      hint: "Predicted arcs in truth",
    },
    {
      label: "Route Recall",
      raw: metrics.route_recall,
      display: formatPct(metrics.route_recall),
      hint: "Truth arcs recovered",
    },
    {
      label: "Jaccard Similarity",
      raw: metrics.jaccard_similarity,
      display: formatPct(metrics.jaccard_similarity),
      hint: "|Pred ∩ True| / |Pred ∪ True|",
    },
    {
      label: "Inference Latency",
      raw: null,
      display:
        typeof result.execution_time === "number"
          ? `${result.execution_time.toFixed(3)} s`
          : "—",
      hint: "End-to-end backend time",
      noBar: true,
    },
  ];

  async function handleCopyRoute() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(arcSequence));
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard API unavailable in non-HTTPS context
    }
  }

  return (
    <div className="card">
      <div className="card-header" style={{ flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <h2 className="card-title">
            Server-Side Evaluation · Trip {result.trip_id}
          </h2>
          {metrics.exact_route_match ? (
            <span className="badge badge-success">
              <CheckCircleIcon size={13} /> Exact Route Match
            </span>
          ) : (
            <span className="badge badge-info">
              {arcSequence.length} Recovered Arcs
            </span>
          )}
        </div>

        <span className="badge badge-neutral">
          <ShieldCheckIcon /> Ground Truth Isolated
        </span>
      </div>

      <div className="card-body">
        <div className="metrics-grid">
          {kpis.map((item) => (
            <div key={item.label} className="metric-tile">
              <div className="metric-label">
                <span>{item.label}</span>
              </div>
              <div className="metric-value">{item.display}</div>
              {!item.noBar && (
                <div className="progress-track">
                  <div
                    className={`progress-fill ${barTone(item.raw)}`}
                    style={{ width: `${pctValue(item.raw)}%` }}
                  />
                </div>
              )}
              <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                {item.hint}
              </div>
            </div>
          ))}
        </div>

        {arcSequence.length > 0 && (
          <div style={{ marginTop: 18 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 8,
              }}
            >
              <span
                style={{
                  fontSize: 12.5,
                  fontWeight: 600,
                  color: "var(--text-secondary)",
                }}
              >
                Reconstructed Road Arc Sequence ({arcSequence.length} arcs)
              </span>
              <button
                type="button"
                className="btn-ghost"
                onClick={handleCopyRoute}
              >
                {copied ? <CheckIcon /> : <CopyIcon />}
                {copied ? "Copied JSON" : "Copy Sequence"}
              </button>
            </div>

            <div className="arc-sequence-box">
              {arcSequence.map((arcId, idx) => (
                <span key={`${idx}-${arcId}`} className="arc-pill">
                  {arcId}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
