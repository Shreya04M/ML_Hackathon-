
/** PathSnap backend API client. */

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(endpoint, options = {}) {
  const response = await fetch(
    `${API_URL}${endpoint}`,
    options
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));

    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : `API error: ${response.status}`
    );
  }

  return response.json();
}

export async function checkHealth() {
  return request("/health");
}

export async function listTrips() {
  return request("/trips");
}

export async function predictTrip(tripId) {
  return request("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      trip_id: tripId,
    }),
  });
}
export function getTripMapUrl(tripId) {
  return `${API_URL}/map/${encodeURIComponent(tripId)}`;
}