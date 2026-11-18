
# PathSnap Evaluator UI

Implemented by Member 4 using React and Vite.

## Overview

The evaluator interface connects to the FastAPI backend
to select trips, request route reconstruction and display
route visualisations and evaluation results.

## Implemented Features

- Backend connection indicator
- Dynamic trip selection
- Route reconstruction request button
- Loading and error handling
- Interactive Folium maps
- Raw GPS trajectory display
- Road network overlay
- Predicted route display support
- Evaluation metrics panel

## Evaluation Metrics

The interface supports displaying:

- Accuracy
- Route Precision
- Route Recall
- Route F1
- Jaccard Similarity
- Execution Time

Metrics are computed on the backend, not the frontend.

## Installation

From the project root:

```bash
cd frontend
npm ci
```

## Local Development

```bash
npm run dev
```

Open http://localhost:5173

## Two-Laptop Deployment

Connect both laptops to the same accessible network.

On Laptop 1, start the FastAPI backend and obtain
its IPv4 address using ipconfig.

On Laptop 2, open PowerShell in the frontend directory:

```powershell
$env:VITE_API_URL="http://BACKEND_IP:8000"
npm run dev
```

Replace BACKEND_IP with Laptop 1's actual IPv4 address.

Open http://localhost:5173 on Laptop 2.

## Production Build

```bash
npm run build
```

## Implementation Status

The frontend has passed browser testing and production
build verification.

Backend connectivity through the server's network IP
has been tested on the same laptop.

Actual cross-device testing and end-to-end reconstructed
route display remain pending model and HMM integration.
