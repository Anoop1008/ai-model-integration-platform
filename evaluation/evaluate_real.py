"""Evaluate the deployed (synthetic-trained) model on independent, realistic data.

Dataset: IBM "Telco Customer Churn" (7,043 customers), the standard public churn benchmark,
https://github.com/IBM/telco-customer-churn-on-icp4d. IBM describes it as a fictional telco
company, so it is realistic sample data, not verified real customers. It was made
independently of train.py's generator, which is what makes it a fair out-of-sample test.
Downloaded on demand to data/ (gitignored).

Questions answered:
1. How well does the deployed model rank and classify real customers?
2. How far is real data from the synthetic training data (PSI)?
3. What could a model trained on the real data reach with the same features (baseline)?

Limitation: the real data has no support_tickets column, so it is filled with a constant;
results are shown for 0, 1 and 2 tickets to expose how much that choice matters.

Run: python evaluation/evaluate_real.py
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.drift import compute_drift
from app.pipeline import FEATURES, InferencePipeline

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
CSV = ROOT / "data" / "telco.csv"
OUT = Path(__file__).parent / "results" / "real_world.json"
CONTRACTS = {"Month-to-month": "month-to-month", "One year": "one-year", "Two year": "two-year"}
REAL_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "contract_type"]


def load_real() -> tuple[pd.DataFrame, np.ndarray]:
    if not CSV.exists():
        CSV.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, CSV)
    raw = pd.read_csv(CSV)
    df = pd.DataFrame(
        {
            "tenure_months": raw["tenure"].astype(int),
            "monthly_charges": raw["MonthlyCharges"].astype(float),
            # Blank for brand-new customers (tenure 0): nothing billed yet.
            "total_charges": pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0),
            "contract_type": raw["Contract"].map(CONTRACTS),
        }
    )
    assert df["contract_type"].notna().all(), "unexpected Contract value"
    return df, (raw["Churn"] == "Yes").astype(int).to_numpy()


def scores(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba >= 0.5
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "precision_churn": round(precision_score(y, pred, zero_division=0), 4),
        "recall_churn": round(recall_score(y, pred), 4),
        "predicted_churn_rate": round(float(pred.mean()), 4),
    }


def main() -> None:
    df, y = load_real()
    pipeline = InferencePipeline.load()
    result = {
        "dataset": "IBM Telco Customer Churn",
        "rows": int(len(df)),
        "actual_churn_rate": round(float(y.mean()), 4),
        "majority_class_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
        "deployed_model": {},
    }

    for tickets in (0, 1, 2):
        x = pipeline.preprocessor.transform(df.assign(support_tickets=tickets)[FEATURES]).astype("float32")
        proba = pipeline.model.predict(x, verbose=0, batch_size=1024).ravel()
        result["deployed_model"][f"support_tickets={tickets}"] = scores(y, proba)

    # Same PSI the API uses, real data vs synthetic training data (support_tickets left out: it is filled in).
    drift = compute_drift(df.assign(support_tickets=1).to_dict("records"), pipeline.training_stats)
    result["psi_real_vs_training"] = {k: v for k, v in drift["features"].items() if k != "support_tickets"}

    # Reference: logistic regression trained on the real data, same 4 real features, 5-fold CV.
    baseline = make_pipeline(
        ColumnTransformer(
            [
                ("num", StandardScaler(), REAL_FEATURES[:3]),
                ("cat", OneHotEncoder(), ["contract_type"]),
            ]
        ),
        LogisticRegression(max_iter=1000),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_proba = cross_val_predict(baseline, df[REAL_FEATURES], y, cv=cv, method="predict_proba")[:, 1]
    result["baseline_trained_on_real_cv5"] = scores(y, cv_proba)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
