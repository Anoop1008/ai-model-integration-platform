"""Train the churn classifier and save every artifact the API needs.

Outputs (all in artifacts/):
- preprocessor.joblib   fitted ColumnTransformer; the API reuses this exact object so
                        serving-time preprocessing is identical to training-time.
- churn_model.keras     TensorFlow classifier.
- training_stats.json   per-feature bin edges + proportions (reference for PSI drift),
                        mean churn probability, metrics, model version.

The data is SYNTHETIC: generated below with a fixed seed, not real customers.
Run: python train.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
N_ROWS = 2000
MODEL_VERSION = "churn-tf-v1"
ARTIFACTS = Path(__file__).parent / "artifacts"

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]
CATEGORICAL = ["contract_type"]
CONTRACTS = ["month-to-month", "one-year", "two-year"]
N_BINS = 10


def make_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    # Total charges ~ tenure * monthly, with some noise (discounts, price changes).
    total = (tenure * monthly * rng.uniform(0.9, 1.1, n)).round(2)
    tickets = np.clip(rng.poisson(1.5, n), 0, 10)
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])

    # Noisy logistic churn: short tenure, high charges, many tickets and
    # month-to-month contracts all raise the churn probability.
    contract_effect = pd.Series(contract).map(
        {"month-to-month": 1.0, "one-year": -0.5, "two-year": -1.5}
    ).to_numpy()
    z = (
        0.3
        - 0.06 * tenure
        + 0.025 * (monthly - 70)
        + 0.35 * tickets
        + contract_effect
        + rng.normal(0, 0.5, n)
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-z)))

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_tickets": tickets,
            "contract_type": contract,
            "churn": churn,
        }
    )


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(categories=[CONTRACTS], handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def build_model(n_features: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def bin_stats(values: np.ndarray) -> dict:
    """Quantile bin edges and the share of training rows in each bin.

    Only the inner edges are stored; at serving time the outer bins are open-ended
    (-inf, inf) so out-of-range requests still land in a bin.
    """
    inner = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
    return {"edges": inner.tolist(), "proportions": (counts / counts.sum()).tolist()}


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)  # seeds Python, NumPy and TF together
    rng = np.random.default_rng(SEED)

    df = make_data(N_ROWS, rng)
    X, y = df[NUMERIC + CATEGORICAL], df["churn"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Fit on the train split only, so no test information leaks into scaling.
    pre = build_preprocessor().fit(X_train)
    Xt_train = pre.transform(X_train).astype("float32")
    Xt_test = pre.transform(X_test).astype("float32")

    model = build_model(Xt_train.shape[1])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        Xt_train,
        y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )

    test_proba = model.predict(Xt_test, verbose=0).ravel()
    train_proba = model.predict(Xt_train, verbose=0).ravel()
    accuracy = accuracy_score(y_test, test_proba >= 0.5)
    roc_auc = roc_auc_score(y_test, test_proba)

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(pre, ARTIFACTS / "preprocessor.joblib")
    model.save(ARTIFACTS / "churn_model.keras")

    stats = {
        "model_version": MODEL_VERSION,
        "n_train_rows": int(len(X_train)),
        "churn_rate_train": float(y_train.mean()),
        "mean_churn_probability": float(train_proba.mean()),
        "test_accuracy": round(float(accuracy), 4),
        "test_roc_auc": round(float(roc_auc), 4),
        "features": {c: bin_stats(X_train[c].to_numpy(dtype=float)) for c in NUMERIC},
    }
    (ARTIFACTS / "training_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"rows={N_ROWS} train={len(X_train)} test={len(X_test)} churn_rate={y.mean():.3f}")
    print(f"test_accuracy={accuracy:.4f} test_roc_auc={roc_auc:.4f}")
    print(f"saved artifacts to {ARTIFACTS}")


if __name__ == "__main__":
    main()
