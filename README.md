# HEATLINE — Hyperlocal Heat-Risk Predictor

100% local, offline Python application. No external APIs, no internet
connection required at runtime. Everything — the microclimate model, the
machine-learning surrogate, and the dashboard — runs on your own machine.

## What it does

- Simulates a 400-segment street-level grid for a city district, each
  segment with its own tree cover, building shadow, surface material
  (asphalt/concrete/grass/pavers/bare soil), traffic density, and
  water proximity.
- Computes a physically-motivated **street-level heat risk index**
  (0–100, WBGT-inspired) that reacts to time of day, ambient
  temperature, shading, and surface heat retention.
- Trains a local **RandomForest surrogate model** on the simulated data
  so risk can be predicted instantly for new conditions.
- Produces **safe work windows** (Safe / Caution / Unsafe hours) for
  outdoor workers across a 24-hour forecast.
- Plans a **safer walking/working route** between two street segments,
  favoring cooler segments over the most direct path.
- Renders all of this in a colorful, professional Dash dashboard.

## Setup

```bash
pip install dash plotly numpy pandas scikit-learn
```

(If you hit a "break system packages" error on Linux:
`pip install dash plotly numpy pandas scikit-learn --break-system-packages`)

## Run

```bash
cd src
python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

## Files

- `src/heat_model.py` — core simulation, risk model, RandomForest
  surrogate, safe-window logic, and route planner. Runnable standalone
  (`python heat_model.py`) to print sample output to the console.
- `src/app.py` — the Dash dashboard application (all UI + callbacks).

## Using the dashboard

1. **Conditions panel (left)** — drag the Air Temperature and Hour of
   Day sliders to see the whole district recompute in real time.
2. **Thermal Scan** — the main scatter map colors every street segment
   by current risk (green = low, red = extreme). Hover any point for
   details (surface type, tree cover, local temperature).
3. **24-Hour Risk Forecast** — bar chart of average district risk by
   hour, with a dashed line marking the currently selected hour.
4. **Safe Work Windows** — reads off Safe / Caution / Unsafe hour
   ranges and gives a one-line scheduling recommendation.
5. **Route Planner** — pick a start and end segment, click **Find
   Safest Route**, and the app draws a path that favors cooler streets
   over the shortest path, with average route risk shown above the map.

All data is synthetically generated at startup with a fixed random
seed, so results are reproducible across runs — no sensors, no live
weather feed, no external calls of any kind.
