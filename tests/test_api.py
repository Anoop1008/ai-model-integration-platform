from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.drift import psi
from app.main import app
from app.pipeline import FEATURES
from app.schemas import ChurnRequest
from tests.conftest import VALID

RESPONSE_FIELDS = {
    "request_id",
    "prediction",
    "confidence_score",
    "churn_probability",
    "risk_level",
    "inference_latency_ms",
    "model_version",
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "churn-tf-v1"
    assert body["uptime_seconds"] >= 0
    # Linux only; None on Windows dev machines. Validated in Docker.
    mem = body["memory"]
    assert mem is None or set(mem) == {"cgroup_mb", "cgroup_peak_mb", "rss_anon_mb"}


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "/predict/churn" in r.text


def test_valid_prediction(client):
    r = client.post("/predict/churn", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["prediction"] in {"Churn", "Retain"}
    assert 0 <= body["churn_probability"] <= 1
    assert 0.5 <= body["confidence_score"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["inference_latency_ms"] > 0
    # Prediction must agree with the probability.
    assert (body["prediction"] == "Churn") == (body["churn_probability"] >= 0.5)


@pytest.mark.parametrize(
    "payload, bad_field",
    [
        ({**VALID, "tenure_months": -1}, "tenure_months"),
        ({**VALID, "contract_type": "weekly"}, "contract_type"),
        ({k: v for k, v in VALID.items() if k != "support_tickets"}, "support_tickets"),
    ],
    ids=["negative_tenure", "bad_contract_type", "missing_field"],
)
def test_validation_errors_return_422(client, payload, bad_field):
    r = client.post("/predict/churn", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert any(d["loc"][-1] == bad_field for d in body["details"])


def test_preprocessing_is_deterministic(client):
    pipeline = client.app.state.pipeline
    req = ChurnRequest(**VALID)
    a, b = pipeline.preprocess(req), pipeline.preprocess(req)
    np.testing.assert_array_equal(a, b)
    # Must equal the fitted training transformer applied directly.
    direct = pipeline.preprocessor.transform(pd.DataFrame([VALID], columns=FEATURES))
    np.testing.assert_allclose(a, direct.astype("float32"))
    # 4 scaled numeric + 3 one-hot contract columns.
    assert a.shape == (1, 7)
    assert pipeline.infer(a) == pipeline.infer(b)


def test_metrics_recent(client):
    for _ in range(12):
        client.post("/predict/churn", json=VALID)
    body = client.get("/metrics/recent").json()
    assert body["count"] == 10  # ring buffer keeps only the last 10
    assert body["predictions"][0]["input"] == VALID
    assert set(body["predictions"][0]["output"]) == RESPONSE_FIELDS


def test_drift_insufficient_then_real_status(client):
    for _ in range(10):
        client.post("/predict/churn", json=VALID)
    body = client.get("/metrics/drift").json()
    assert body["status"] == "insufficient_data"
    assert body["n_samples"] == 10

    # 50 identical, very long-tenure customers: far from the training spread.
    shifted = {**VALID, "tenure_months": 110, "total_charges": 10000.0}
    for _ in range(50):
        client.post("/predict/churn", json=shifted)
    body = client.get("/metrics/drift").json()
    assert body["n_samples"] == 60
    assert body["status"] == "significant"
    assert set(body["features"]) == {"tenure_months", "monthly_charges", "total_charges", "support_tickets"}
    assert body["features"]["tenure_months"]["status"] == "significant"


def test_psi_math():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(p, p) == 0.0
    assert psi(p, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_20_concurrent_requests(client):
    with ThreadPoolExecutor(max_workers=20) as pool:
        responses = list(pool.map(lambda _: client.post("/predict/churn", json=VALID), range(20)))
    assert [r.status_code for r in responses] == [200] * 20
    assert len({r.json()["request_id"] for r in responses}) == 20


def test_500_returns_consistent_json(client, monkeypatch):
    def boom(_req):
        raise RuntimeError("secret internal detail")

    # raise_server_exceptions=False so we see the response a real client would get.
    with TestClient(app, raise_server_exceptions=False) as c:
        monkeypatch.setattr(c.app.state.pipeline, "predict", boom)
        r = c.post("/predict/churn", json=VALID)
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "message": "Internal server error", "details": None}
