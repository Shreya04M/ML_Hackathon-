
import React from "react";
import { getTripMapUrl } from "../services/api";

export function RouteMap({ tripId, result }) {
  if (!tripId) {
    return (
      <p>Select a trip to view its GPS trajectory.</p>
    );
  }

  const predictedMap =
    result?.trip_id === tripId ? result.map_html : null;

  return (
    <div>
      <p style={{ color: "#64748b" }}>
        {predictedMap
          ? "Reconstructed route"
          : "GPS trajectory"} for Trip {tripId}
      </p>

      <iframe
        key={tripId}
        src={predictedMap ? undefined : getTripMapUrl(tripId)}
        srcDoc={predictedMap || undefined}
        title={`Route map for ${tripId}`}
        style={{
          width: "100%",
          height: "500px",
          border: "1px solid #cbd5e1",
          borderRadius: "10px",
        }}
      />

      <p style={{ fontSize: "14px" }}>
        <span style={{ color: "blue" }}>
          &#9679; Raw GPS trajectory
        </span>
        &nbsp;&nbsp;
        <span style={{ color: "green" }}>
          &#9679; Start
        </span>
        &nbsp;&nbsp;
        <span style={{ color: "red" }}>
          &#9679; End
        </span>
        &nbsp;&nbsp;
        <span style={{ color: "gray" }}>
          &#9679; Road Network
        </span>
        &nbsp;&nbsp;
        <span style={{ color: "#dc2626" }}>
          &#9473; Predicted Route
        </span>
      </p>
    </div>
  );
}
