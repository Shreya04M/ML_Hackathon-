import React from "react";

export function TripSelector({ trips, selectedTrip, onChange, disabled }) {
  const quickTrips = trips.slice(0, 8);

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 8,
        }}
      >
        <label
          htmlFor="trip"
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: "var(--text-secondary)",
          }}
        >
          Evaluation Trip ID
        </label>
        <span className="badge badge-neutral mono">
          {trips.length} {trips.length === 1 ? "trip" : "trips"}
        </span>
      </div>

      <select
        id="trip"
        className="select-input"
        value={selectedTrip}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled || trips.length === 0}
      >
        <option value="">
          {trips.length === 0
            ? "No evaluation trips loaded"
            : "Choose an evaluation trip..."}
        </option>
        {trips.map((trip) => (
          <option key={trip} value={trip}>
            Trip {trip}
          </option>
        ))}
      </select>

      {quickTrips.length > 0 && (
        <div className="trip-chips" aria-label="Quick trip selection">
          {quickTrips.map((trip) => (
            <button
              key={trip}
              type="button"
              disabled={disabled}
              className={`trip-chip ${selectedTrip === trip ? "active" : ""}`}
              onClick={() => onChange(trip)}
              title={`Select and reconstruct Trip ${trip}`}
            >
              #{trip.replace(/^0+/, "") || "0"}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
