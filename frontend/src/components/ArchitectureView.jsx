import React from "react";
import { LayersIcon, ShieldCheckIcon } from "./Icons";

const STAGES = [
  {
    num: "STAGE 01 · PREPROCESSING",
    title: "Spatial Projection & Candidate Generation",
    desc: "Projects lon/lat into local metric coordinates (meters), guards against zero-length arcs & GPS timestamp gaps, and builds an adaptive cKDTree (K=6) achieving 99.32% candidate coverage.",
    tags: ["cKDTree (K=6)", "20 Leakage-Free Features", "Local-Meter Projection"],
  },
  {
    num: "STAGE 02 · ML EMISSIONS",
    title: "Bayesian-Tuned HistGradientBoosting",
    desc: "Scores each (observation, candidate arc) pair with calibrated emission probabilities. Selected over 5 model families via GroupKFold CV and tuned with GP-EI Bayesian Optimisation.",
    tags: ["HistGradientBoosting", "GP-EI Tuned", "Calibrated Log-Loss 0.1583"],
  },
  {
    num: "STAGE 03 · ROUTE DECODING",
    title: "Distance HMM + Viterbi & Constrained Recovery",
    desc: "Combines ML emission log-probabilities with road-graph shortest-path transition costs (β=30m, w=1.9486), followed by physically constrained intermediate arc recovery (max 60 m/s).",
    tags: ["Viterbi Decoding", "Dijkstra Shortest Path", "97.85% Continuity"],
  },
  {
    num: "STAGE 04 · DISTRIBUTED SERVING",
    title: "Ground-Truth-Isolated FastAPI ↔ React",
    desc: "Laptop 1 loads only .track, .nodes, .arcs for inference (route=None). Ground-truth .route is read strictly server-side post-prediction to compute overlap F1, accuracy, and Folium HTML.",
    tags: ["FastAPI Backend", "React + Vite UI", "Zero Label Leakage"],
  },
];

const FEATURE_GROUPS = [
  {
    group: "Spatial & Projection (7)",
    items: [
      "ortho_dist_m",
      "dist_to_start_m",
      "dist_to_end_m",
      "proj_fraction",
      "is_endpoint_proj",
      "segment_length_m",
      "log_segment_length_m",
    ],
  },
  {
    group: "Within-Observation Competition (4)",
    items: [
      "dist_rank",
      "dist_diff_from_min_m",
      "dist_ratio_to_min",
      "dist_zscore_in_obs",
    ],
  },
  {
    group: "Kinematic & Heading Alignment (6)",
    items: [
      "gps_speed_mps",
      "dt_prev_s",
      "heading_diff_deg",
      "heading_cos",
      "heading_sin",
      "speed_weighted_heading_cos",
    ],
  },
  {
    group: "Road Graph Topology (3)",
    items: [
      "prev_cand_connected",
      "min_dist_from_prev_cand_m",
      "node_in_out_degree_sum",
    ],
  },
];

export function ArchitectureView() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <LayersIcon size={17} />
            End-to-End Map-Matching Pipeline
          </h2>
          <span className="badge badge-success">
            <ShieldCheckIcon /> Zero Data Leakage Verified
          </span>
        </div>
        <div className="card-body">
          <div className="arch-grid">
            {STAGES.map((s) => (
              <div key={s.num} className="stage-card">
                <div className="stage-num">{s.num}</div>
                <div
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    color: "var(--text-primary)",
                    lineHeight: 1.3,
                  }}
                >
                  {s.title}
                </div>
                <div
                  style={{
                    fontSize: 13,
                    color: "var(--text-secondary)",
                    flex: 1,
                  }}
                >
                  {s.desc}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {s.tags.map((tag) => (
                    <span key={tag} className="badge badge-neutral">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">20 Leakage-Free Engineered Features</h2>
          <span className="badge badge-info">
            100% Observable from (.track, .nodes, .arcs)
          </span>
        </div>
        <div className="card-body">
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
              gap: 16,
            }}
          >
            {FEATURE_GROUPS.map((fg) => (
              <div
                key={fg.group}
                style={{
                  padding: 16,
                  borderRadius: 10,
                  background: "var(--bg-subtle)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: "var(--text-primary)",
                    marginBottom: 10,
                  }}
                >
                  {fg.group}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {fg.items.map((feat) => (
                    <span key={feat} className="arc-pill">
                      {feat}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

