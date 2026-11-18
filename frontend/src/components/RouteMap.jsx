import React from "react";
import { getTripMapUrl } from "../services/api";
import { ExternalLinkIcon, MapIcon, RefreshIcon } from "./Icons";

export function RouteMap({ tripId, result, loading }) {
  const predictedMap =
    result?.trip_id === tripId && result?.map_html ? result.map_html : null;

  return (
    <div className="card" style={{ overflow: "hidden" }}>
      <div className="card-header">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <h2 className="card-title">
            <MapIcon size={17} />
            Interactive Geospatial View
          </h2>
          {tripId && (
            <span
              className={`badge ${
                predictedMap ? "badge-success" : "badge-info"
              }`}
            >
              {predictedMap
                ? `Reconstructed Route · Trip ${tripId}`
                : `Raw GPS & Road Graph · Trip ${tripId}`}
            </span>
          )}
        </div>

        {tripId && (
          <a
            href={getTripMapUrl(tripId)}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-ghost"
            title="Open Folium map in a new tab"
          >
            Open Map <ExternalLinkIcon />
          </a>
        )}
      </div>

      <div className="map-frame-wrap">
        {!tripId ? (
          <div
            style={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: 32,
              textAlign: "center",
              color: "var(--text-muted)",
              gap: 10,
            }}
          >
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: 12,
                background: "var(--bg-surface)",
                border: "1px solid var(--border-subtle)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--accent)",
              }}
            >
              <MapIcon size={22} />
            </div>
            <div
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "var(--text-primary)",
              }}
            >
              No Trip Selected
            </div>
            <div style={{ fontSize: 13, maxWidth: 380 }}>
              Select an evaluation trip from the left panel to inspect its raw
              GPS trajectory and run live HistGradientBoosting + HMM route
              reconstruction.
            </div>
          </div>
        ) : (
          <>
            {loading && (
              <div className="map-overlay">
                <RefreshIcon size={26} className="spin" />
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 600,
                    color: "var(--text-primary)",
                  }}
                >
                  Reconstructing Route for Trip {tripId}...
                </div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  Extracting 20 features (K=6) → HistGradientBoosting → Viterbi
                  HMM + Recovery
                </div>
              </div>
            )}

            <iframe
              key={`${tripId}-${predictedMap ? "pred" : "raw"}`}
              src={predictedMap ? undefined : getTripMapUrl(tripId)}
              srcDoc={predictedMap || undefined}
              title={`Route map for ${tripId}`}
              className="map-iframe"
            />
          </>
        )}
      </div>

      <div className="map-legend">
        <div className="legend-items">
          <span className="legend-item">
            <span
              className="legend-swatch"
              style={{ background: "#2563eb" }}
            />
            Raw GPS Trajectory
          </span>
          <span className="legend-item">
            <span
              className="legend-swatch"
              style={{ background: "#16a34a" }}
            />
            Start Node
          </span>
          <span className="legend-item">
            <span
              className="legend-swatch"
              style={{ background: "#dc2626" }}
            />
            End Node
          </span>
          <span className="legend-item">
            <span
              className="legend-line"
              style={{ background: "#64748b" }}
            />
            Road Network
          </span>
          <span className="legend-item">
            <span
              className="legend-line"
              style={{ background: "#dc2626" }}
            />
            Reconstructed Route
          </span>
        </div>

        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          Folium Interactive Layer Control
        </span>
      </div>
    </div>
  );
}
