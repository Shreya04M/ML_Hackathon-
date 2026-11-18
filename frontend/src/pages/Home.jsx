import React, { useCallback, useEffect, useState } from "react";
import { ArchitectureView } from "../components/ArchitectureView";
import { BenchmarksView } from "../components/BenchmarksView";
import {
  AlertCircleIcon,
  BarChartIcon,
  CheckCircleIcon,
  LayersIcon,
  MapIcon,
  PlayIcon,
  RefreshIcon,
  RouteIcon,
} from "../components/Icons";
import { Results } from "../components/Results";
import { RouteMap } from "../components/RouteMap";
import { TripSelector } from "../components/TripSelector";
import { API_URL, checkHealth, listTrips, predictTrip } from "../services/api";

const PIPELINE_STEPS = [
  "Load .track, .nodes, .arcs (route=None)",
  "Extract 20 spatial & kinematic features (K=6)",
  "HistGradientBoosting (GP-EI) emission scores",
  "Viterbi Distance HMM + constrained recovery",
];

export function Home() {
  const [activeTab, setActiveTab] = useState("live");
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [connected, setConnected] = useState(false);
  const [checkingHealth, setCheckingHealth] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const initialise = useCallback(async () => {
    setCheckingHealth(true);
    setError("");
    try {
      const health = await checkHealth();
      const isHealthy = health.status === "healthy";
      setConnected(isHealthy);

      if (isHealthy) {
        const availableTrips = await listTrips();
        setTrips(availableTrips);
        setSelectedTrip((prev) =>
          prev && availableTrips.includes(prev) ? prev : availableTrips[0] || ""
        );
      }
    } catch (err) {
      setConnected(false);
      setError(err.message);
    } finally {
      setCheckingHealth(false);
    }
  }, []);

  useEffect(() => {
    initialise();
  }, [initialise]);

  async function handlePrediction(tripId = selectedTrip) {
    if (!tripId || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await predictTrip(tripId);
      setResult(response);
      setHistory((prev) => {
        const filtered = prev.filter((item) => item.trip_id !== response.trip_id);
        return [response, ...filtered].slice(0, 10);
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleSelectTrip(trip) {
    setSelectedTrip(trip);
    setError("");
    const cached = history.find((h) => h.trip_id === trip);
    if (cached) {
      setResult(cached);
    } else {
      setResult(null);
    }
  }

  function formatPct(val) {
    return typeof val === "number" ? `${(val * 100).toFixed(2)}%` : "—";
  }

  return (
    <div>
      {/* Sticky Top Header */}
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <div className="brand-icon">
              <RouteIcon size={20} />
            </div>
            <div>
              <div className="brand-title">PathSnap</div>
              <div className="brand-subtitle">
                Distributed Map-Matching &amp; Route Reconstruction · UE24CS352A
              </div>
            </div>
          </div>

          <nav className="nav-tabs" aria-label="Primary sections">
            <button
              type="button"
              className={`nav-tab ${activeTab === "live" ? "active" : ""}`}
              onClick={() => setActiveTab("live")}
            >
              <MapIcon size={15} />
              Live Reconstruction
            </button>
            <button
              type="button"
              className={`nav-tab ${activeTab === "benchmarks" ? "active" : ""}`}
              onClick={() => setActiveTab("benchmarks")}
            >
              <BarChartIcon size={15} />
              Model &amp; HMM Benchmarks
            </button>
            <button
              type="button"
              className={`nav-tab ${activeTab === "architecture" ? "active" : ""}`}
              onClick={() => setActiveTab("architecture")}
            >
              <LayersIcon size={15} />
              Pipeline Architecture
            </button>
          </nav>

          <div className="status-group">
            <div
              className={`health-pill ${connected ? "online" : "offline"}`}
              title={`Backend endpoint: ${API_URL}`}
            >
              <span className="pulse-dot" />
              {checkingHealth
                ? "Connecting..."
                : connected
                ? "FastAPI Connected"
                : "Backend Offline"}
            </div>
            <button
              type="button"
              className="btn-ghost"
              onClick={initialise}
              disabled={checkingHealth}
              title="Re-check backend connection and reload trips"
            >
              <RefreshIcon size={14} className={checkingHealth ? "spin" : ""} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="page-container">
        {activeTab === "live" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div className="workspace-grid">
              {/* Left Control Rail */}
              <aside style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div className="card">
                  <div className="card-header">
                    <h2 className="card-title">Trip Control &amp; Inference</h2>
                  </div>
                  <div className="card-body">
                    <TripSelector
                      trips={trips}
                      selectedTrip={selectedTrip}
                      disabled={loading}
                      onChange={handleSelectTrip}
                    />

                    <button
                      type="button"
                      className="btn-primary"
                      style={{ marginTop: 16 }}
                      onClick={() => handlePrediction(selectedTrip)}
                      disabled={!selectedTrip || loading || !connected}
                    >
                      {loading ? (
                        <>
                          <RefreshIcon size={16} className="spin" />
                          Reconstructing Route...
                        </>
                      ) : (
                        <>
                          <PlayIcon size={15} />
                          Reconstruct Route
                        </>
                      )}
                    </button>

                    {error && (
                      <div className="alert-error" role="alert">
                        <AlertCircleIcon size={16} />
                        <div>
                          <div style={{ fontWeight: 600 }}>Request Failed</div>
                          <div>{error}</div>
                        </div>
                      </div>
                    )}

                    {/* Execution Stepper */}
                    <div className="stepper">
                      <div
                        style={{
                          fontSize: 11.5,
                          fontWeight: 600,
                          textTransform: "uppercase",
                          letterSpacing: "0.04em",
                          color: "var(--text-muted)",
                          marginBottom: 2,
                        }}
                      >
                        Inference Pipeline
                      </div>
                      {PIPELINE_STEPS.map((label, idx) => {
                        const isDone =
                          !loading && result?.trip_id === selectedTrip;
                        const isActive = loading;
                        return (
                          <div
                            key={label}
                            className={`step-row ${
                              isDone ? "done" : isActive ? "active" : ""
                            }`}
                          >
                            <span className="step-badge">
                              {isDone ? "✓" : idx + 1}
                            </span>
                            <span>{label}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Active Model Artifact Card */}
                <div className="card">
                  <div className="card-header">
                    <h2 className="card-title">Active Production Artifacts</h2>
                  </div>
                  <div
                    className="card-body"
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: 10,
                      fontSize: 12.5,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 8,
                      }}
                    >
                      <span style={{ color: "var(--text-muted)" }}>
                        Classifier
                      </span>
                      <span className="mono" style={{ fontWeight: 600 }}>
                        HistGB (GP-EI)
                      </span>
                    </div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 8,
                      }}
                    >
                      <span style={{ color: "var(--text-muted)" }}>
                        Candidate Tree
                      </span>
                      <span className="mono" style={{ fontWeight: 600 }}>
                        cKDTree (K=6)
                      </span>
                    </div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 8,
                      }}
                    >
                      <span style={{ color: "var(--text-muted)" }}>
                        HMM Params
                      </span>
                      <span className="mono" style={{ fontWeight: 600 }}>
                        β=30m · w=1.95
                      </span>
                    </div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 8,
                      }}
                    >
                      <span style={{ color: "var(--text-muted)" }}>
                        Backend API
                      </span>
                      <span
                        className="mono"
                        style={{
                          fontSize: 11.5,
                          color: "var(--text-secondary)",
                        }}
                      >
                        {API_URL}
                      </span>
                    </div>
                  </div>
                </div>
              </aside>

              {/* Right Main Viewport */}
              <section
                style={{ display: "flex", flexDirection: "column", gap: 20 }}
              >
                <RouteMap
                  tripId={selectedTrip}
                  result={result}
                  loading={loading}
                />
                <Results result={result} />
              </section>
            </div>

            {/* Session Run History */}
            {history.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">Session Evaluation History</h2>
                  <span className="badge badge-neutral">
                    {history.length}{" "}
                    {history.length === 1 ? "trip evaluated" : "trips evaluated"}
                  </span>
                </div>
                <div className="data-table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Trip ID</th>
                        <th>Obs Accuracy</th>
                        <th>Route F1</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Jaccard</th>
                        <th>Arcs</th>
                        <th>Latency</th>
                        <th>Match</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map((item) => {
                        const m = item.metrics || {};
                        const isCurrent = item.trip_id === selectedTrip;
                        return (
                          <tr
                            key={item.trip_id}
                            className={isCurrent ? "row-highlight" : ""}
                            style={{ cursor: "pointer" }}
                            onClick={() => handleSelectTrip(item.trip_id)}
                          >
                            <td className="mono" style={{ fontWeight: 600 }}>
                              {item.trip_id}
                            </td>
                            <td className="mono">{formatPct(m.accuracy)}</td>
                            <td className="mono" style={{ fontWeight: 600 }}>
                              {formatPct(m.route_f1)}
                            </td>
                            <td className="mono">
                              {formatPct(m.route_precision)}
                            </td>
                            <td className="mono">{formatPct(m.route_recall)}</td>
                            <td className="mono">
                              {formatPct(m.jaccard_similarity)}
                            </td>
                            <td className="mono">
                              {item.predicted_route?.length ?? 0}
                            </td>
                            <td className="mono">
                              {typeof item.execution_time === "number"
                                ? `${item.execution_time.toFixed(3)}s`
                                : "—"}
                            </td>
                            <td>
                              {m.exact_route_match ? (
                                <span className="badge badge-success">
                                  <CheckCircleIcon size={12} /> Exact
                                </span>
                              ) : (
                                <span className="badge badge-neutral">
                                  Partial
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "benchmarks" && <BenchmarksView />}
        {activeTab === "architecture" && <ArchitectureView />}
      </main>
    </div>
  );
}
