"""
Hyperlocal Heat-Risk Predictor - Core Model
100% local, offline, no external API calls.

Simulates a city grid of street segments, each with microclimate features
(tree cover, building shadow, surface material, traffic density, water proximity).
Uses a physically-motivated formula + RandomForest to estimate a
Wet-Bulb-Globe-Temperature-like heat risk index at fine spatial resolution,
then derives safe work windows and safer walking routes.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

SURFACE_TYPES = ["asphalt", "concrete", "grass", "pavers", "bare_soil"]
SURFACE_ALBEDO = {"asphalt": 0.08, "concrete": 0.35, "grass": 0.25, "pavers": 0.30, "bare_soil": 0.20}
SURFACE_HEAT_RETENTION = {"asphalt": 1.35, "concrete": 1.10, "grass": 0.55, "pavers": 1.00, "bare_soil": 0.85}


def generate_city_grid(n_points=400, grid_size=20, seed=42):
    """Create a synthetic street-level grid representing a city district."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, grid_size, int(np.sqrt(n_points))),
                          np.linspace(0, grid_size, int(np.sqrt(n_points))))
    xs, ys = xs.flatten(), ys.flatten()
    n = len(xs)

    tree_cover = np.clip(rng.beta(2, 3, n) * (1 - 0.3 * np.sin(xs / 3)), 0, 1)
    building_shadow = np.clip(rng.beta(2, 4, n), 0, 1)
    surface = rng.choice(SURFACE_TYPES, n, p=[0.35, 0.25, 0.2, 0.1, 0.1])
    traffic_density = np.clip(rng.gamma(2, 0.25, n), 0, 1)
    water_proximity = np.clip(1 - rng.exponential(0.4, n), 0, 1)  # closer=1
    elevation = rng.normal(0, 5, n)
    humidity_base = np.clip(45 + water_proximity * 20 + rng.normal(0, 5, n), 20, 95)

    df = pd.DataFrame({
        "segment_id": [f"SEG-{i:04d}" for i in range(n)],
        "x": xs, "y": ys,
        "tree_cover": tree_cover,
        "building_shadow": building_shadow,
        "surface": surface,
        "traffic_density": traffic_density,
        "water_proximity": water_proximity,
        "elevation": elevation,
        "humidity_base": humidity_base,
    })
    return df


def compute_heat_index(df, air_temp_c, hour, humidity_override=None):
    """
    Physically-motivated street-level heat index.
    Combines: ambient temp, surface heat retention/albedo, shading, traffic exhaust heat,
    humidity, and time-of-day solar loading -> a WBGT-like risk score (0-100+).
    """
    solar_factor = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, 1) if 6 <= hour <= 18 else 0.05
    albedo = df["surface"].map(SURFACE_ALBEDO).values
    retention = df["surface"].map(SURFACE_HEAT_RETENTION).values

    shade_factor = 1 - (0.6 * df["tree_cover"].values + 0.4 * df["building_shadow"].values)
    shade_factor = np.clip(shade_factor, 0.15, 1.0)

    surface_heat = solar_factor * retention * (1 - albedo) * 4.0
    traffic_heat = df["traffic_density"].values * 1.2
    elevation_cool = -df["elevation"].values * 0.03

    humidity = (humidity_override if humidity_override is not None
                else df["humidity_base"].values)

    local_temp = air_temp_c + surface_heat * shade_factor + traffic_heat + elevation_cool

    # Simplified WBGT approximation (Australian BOM style), scaled down
    wbgt = local_temp + 0.15 * (humidity / 100 * 30)
    wbgt = wbgt - (df["tree_cover"].values * 1.2)  # tree evapotranspiration cooling

    risk_score = np.clip((wbgt - 24) * 3.0, 0, 100)

    out = df.copy()
    out["local_temp_c"] = local_temp.round(1)
    out["wbgt_c"] = wbgt.round(1)
    out["risk_score"] = risk_score.round(1)
    out["risk_level"] = pd.cut(risk_score, bins=[-1, 25, 50, 75, 200],
                                labels=["Low", "Moderate", "High", "Extreme"])
    return out


def train_surrogate_model(df):
    """Train a fast RandomForest surrogate so risk can be predicted for new conditions instantly."""
    features = ["tree_cover", "building_shadow", "traffic_density",
                "water_proximity", "elevation", "humidity_base"]
    surface_dummies = pd.get_dummies(df["surface"], prefix="surf")
    X = pd.concat([df[features], surface_dummies], axis=1)
    y = df["risk_score"]
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()


def safe_work_windows(hourly_risk_by_hour):
    """Given risk score per hour (0-23), return list of (start,end,level) safe/unsafe windows."""
    windows = []
    cur_level = None
    start_h = 0
    for h in range(24):
        r = hourly_risk_by_hour[h]
        level = "Safe" if r < 40 else ("Caution" if r < 65 else "Unsafe")
        if level != cur_level:
            if cur_level is not None:
                windows.append((start_h, h, cur_level))
            cur_level = level
            start_h = h
    windows.append((start_h, 24, cur_level))
    return windows


def daily_hourly_forecast(df, base_temp=34, seed=1):
    """Simulate a diurnal temperature curve and compute average city risk per hour."""
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    # diurnal curve peaking ~15:00
    diurnal = base_temp + 6 * np.sin(np.pi * (hours - 6) / 14) - 2
    diurnal = np.clip(diurnal, base_temp - 8, base_temp + 8)
    diurnal += rng.normal(0, 0.4, 24)

    avg_risk = []
    for h, t in zip(hours, diurnal):
        r = compute_heat_index(df, air_temp_c=t, hour=h)
        avg_risk.append(r["risk_score"].mean())
    return pd.DataFrame({"hour": hours, "air_temp_c": diurnal.round(1), "avg_risk": np.round(avg_risk, 1)})


def safest_route(df_risk, start_id, end_id):
    """
    Greedy nearest-neighbor pathfinding across the grid that prefers low-risk segments.
    Returns ordered list of segment rows forming a route.
    """
    coords = df_risk[["x", "y"]].values
    risk = df_risk["risk_score"].values
    ids = df_risk["segment_id"].values
    idx = {sid: i for i, sid in enumerate(ids)}

    start_i, end_i = idx[start_id], idx[end_id]
    visited = set([start_i])
    path = [start_i]
    current = start_i
    max_steps = 60

    for _ in range(max_steps):
        if current == end_i:
            break
        cx, cy = coords[current]
        ex, ey = coords[end_i]
        dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
        dist_to_end = np.sqrt((coords[:, 0] - ex) ** 2 + (coords[:, 1] - ey) ** 2)
        candidates = np.where((dists > 0) & (dists < 2.2))[0]
        candidates = [c for c in candidates if c not in visited]
        if not candidates:
            candidates = [c for c in np.argsort(dists) if c not in visited][:8]
        if len(candidates) == 0:
            break
        # score: progress toward end (lower dist_to_end) + low risk penalty
        scores = [dist_to_end[c] + risk[c] * 0.04 for c in candidates]
        nxt = candidates[int(np.argmin(scores))]
        visited.add(nxt)
        path.append(nxt)
        current = nxt

    return df_risk.iloc[path].reset_index(drop=True)


if __name__ == "__main__":
    grid = generate_city_grid()
    risk = compute_heat_index(grid, air_temp_c=36, hour=14)
    print(risk[["segment_id", "risk_score", "risk_level"]].head(10))
    model, cols = train_surrogate_model(risk)
    print("Model trained. Features:", cols)
    hourly = daily_hourly_forecast(risk)
    print(hourly)
