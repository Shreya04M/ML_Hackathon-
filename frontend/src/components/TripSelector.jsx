
import React from "react";

export function TripSelector({
  trips,
  selectedTrip,
  onChange,
  disabled,
}) {
  return (
    <div>
      <label htmlFor="trip">
        Select Evaluation Trip
      </label>

      <select
        id="trip"
        value={selectedTrip}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        style={{
          display: "block",
          width: "100%",
          padding: "12px",
          marginTop: "10px",
          borderRadius: "8px",
          border: "1px solid #cbd5e1",
          fontSize: "16px",
        }}
      >
        <option value="">Select a trip</option>

        {trips.map((trip) => (
          <option key={trip} value={trip}>
            Trip {trip}
          </option>
        ))}
      </select>
    </div>
  );
}