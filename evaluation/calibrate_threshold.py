"""Pick the churn decision threshold on independent data and save it with the model.

Why: on the IBM Telco benchmark the default 0.5 cut-off flags more customers as churners
than actually leave. The ranking (ROC-AUC) is fine, so only the cut-off needs moving.

How: split the IBM data 50/50 (stratified). Choose the threshold that maximises accuracy
on the "tune" half, then report before/after on the untouched "holdout" half, so the
reported gain is not measured on the same rows used to pick it.

Writes decision_threshold and a bumped model_version into artifacts/training_stats.json.
train.py rewrites that file, so rerun this script after retraining.

Run: python evaluation/calibrate_threshold.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

from app.pipeline import ARTIFACTS_DIR, FEATURES, InferencePipeline
from evaluation.evaluate_real import load_real, scores

MODEL_VERSION = "churn-tf-v1.1"
# shortcut: support_tickets is missing in the IBM data, filled with 1 (typical in training); recalibrate on real ticket data when available.
TICKETS_FILL = 1
OUT = Path(__file__).parent / "results" / "threshold_calibration.json"


def main() -> None:
    df, y = load_real()
    pipeline = InferencePipeline.load()
    x = pipeline.preprocessor.transform(df.assign(support_tickets=TICKETS_FILL)[FEATURES]).astype("float32")
    proba = pipeline.model.predict(x, verbose=0, batch_size=1024).ravel()

    idx_tune, idx_hold = train_test_split(np.arange(len(y)), test_size=0.5, random_state=42, stratify=y)
    grid = np.round(np.arange(0.30, 0.81, 0.01), 2)
    tune_acc = [float(((proba[idx_tune] >= t) == y[idx_tune]).mean()) for t in grid]
    threshold = float(grid[int(np.argmax(tune_acc))])

    result = {
        "chosen_threshold": threshold,
        "objective": "max accuracy on tune half",
        "support_tickets_fill": TICKETS_FILL,
        "holdout_rows": int(len(idx_hold)),
        "holdout_actual_churn_rate": round(float(y[idx_hold].mean()), 4),
        "holdout_at_0.5": scores(y[idx_hold], proba[idx_hold], 0.5),
        f"holdout_at_{threshold}": scores(y[idx_hold], proba[idx_hold], threshold),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    stats_path = ARTIFACTS_DIR / "training_stats.json"
    stats = json.loads(stats_path.read_text())
    stats["decision_threshold"] = threshold
    stats["model_version"] = MODEL_VERSION
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"saved decision_threshold={threshold}, model_version={MODEL_VERSION} to {stats_path}")


if __name__ == "__main__":
    main()
