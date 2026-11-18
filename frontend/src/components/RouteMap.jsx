
import React from "react";
import { getTripMapUrl } from "../services/api";

export function RouteMap({ tripId }) {
  if (!tripId) {
    return (
      <p>Select a trip to view its GPS trajectory.</p>
    );
  }

  return (
    <div>
      <p style={{ color: "#64748b" }}>
        GPS trajectory for Trip {tripId}
      </p>

      <iframe
        key={tripId}
        src={getTripMapUrl(tripId)}
        title={`Route map for ${tripId}`}
        style={{
          width: "100%",
          height: "500px",
          border: "1px solid #cbd5e1",
          borderRadius: "10px",
        }}
      />

      <p style={{ fontSize: "14px" }}>
        🔵 Raw GPS trajectory
        &nbsp;&nbsp; 🟢 Start
        &nbsp;&nbsp; 🔴 End
        &nbsp;&nbsp; ⚪ Road Network
      </p>
    </div>
  );
}