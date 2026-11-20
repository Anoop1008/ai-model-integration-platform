"""Basic drift indicator using the Population Stability Index (PSI).

PSI compares how recent request values are spread across the training bins:
    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)
0 means identical distributions; larger means the live inputs have shifted away
from what the model was trained on. Rule of thumb: <0.1 stable, 0.1-0.25 moderate,
>0.25 significant.

This is a basic indicator, not full monitoring: numeric features only, a small
in-memory window, no alerting, and it resets when the process restarts.
"""

import numpy as np

MIN_SAMPLES = 50
EPS = 1e-4  # avoids log(0) / divide-by-zero for empty bins


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value <= 0.25:
        return "moderate"
    return "significant"


def compute_drift(rows: list[dict], training_stats: dict) -> dict:
    n = len(rows)
    if n < MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "n_samples": n,
            "min_samples": MIN_SAMPLES,
            "note": "basic PSI indicator, not full monitoring",
        }

    features = {}
    for name, ref in training_stats["features"].items():
        values = np.array([r[name] for r in rows], dtype=float)
        edges = np.array(ref["edges"])
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
        value = psi(np.array(ref["proportions"]), counts / n)
        features[name] = {"psi": round(value, 4), "status": psi_status(value)}

    worst = max(f["psi"] for f in features.values())
    return {
        "status": psi_status(worst),
        "n_samples": n,
        "features": features,
        "note": "basic PSI indicator, not full monitoring",
    }
