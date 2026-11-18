
import React, { useEffect, useState } from "react";
import { Results } from "../components/Results";
import {
  checkHealth,
  listTrips,
  predictTrip,
} from "../services/api";

import { TripSelector } from "../components/TripSelector";
import { RouteMap } from "../components/RouteMap";
export function Home() {
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    async function initialise() {
      try {
        const health = await checkHealth();

        setConnected(health.status === "healthy");

        const availableTrips = await listTrips();
        setTrips(availableTrips);
      } catch (err) {
        setError(err.message);
      }
    }

    initialise();
  }, []);

  async function handlePrediction(tripId = selectedTrip) {
    if (!tripId) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await predictTrip(tripId);
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main
      style={{
        maxWidth: "1000px",
        margin: "40px auto",
        fontFamily: "Arial, sans-serif",
        padding: "24px",
        color: "#0f172a",
      }}
    >
      <h1 style={{ fontSize: "38px" }}>
        PathSnap
      </h1>

      <p style={{ color: "#64748b" }}>
        Distributed Map-Matching & Route Reconstruction
      </p>

      <p style={{
        color: connected ? "#15803d" : "#b91c1c"
      }}>
        {connected
          ? "● Backend Connected"
          : "● Backend Not Connected"}
      </p>

      <div style={{
        padding: "24px",
        border: "1px solid #e2e8f0",
        borderRadius: "12px",
        marginTop: "24px",
      }}>
        <h2>Trip Selection</h2>

        <p>{trips.length} evaluation trips available</p>

        <TripSelector
          trips={trips}
          selectedTrip={selectedTrip}
          disabled={loading}
          onChange={(trip) => {
            setSelectedTrip(trip);
            setResult(null);
            setError("");
            handlePrediction(trip);
          }}
        />

        <button
          onClick={handlePrediction}
          disabled={!selectedTrip || loading}
          style={{
            marginTop: "20px",
            padding: "14px 24px",
            background: "#2563eb",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: !selectedTrip || loading
              ? "not-allowed"
              : "pointer",
            opacity: !selectedTrip || loading ? 0.5 : 1,
          }}
        >
          {loading
            ? "Reconstructing..."
            : "Reconstruct Route"}
        </button>

        {error && (
          <p role="alert" style={{ color: "#b91c1c" }}>
            {error}
          </p>
        )}
      </div>

      <div style={{
        marginTop: "24px",
        border: "1px solid #e2e8f0",
        borderRadius: "12px",
        padding: "24px",
        minHeight: "250px",
        background: "#f8fafc",
      }}>
        <h2>Route Visualisation</h2>
        <RouteMap
  tripId={selectedTrip}
  result={result}
/>
      </div>

      <Results result={result} />
    </main>
  );
}