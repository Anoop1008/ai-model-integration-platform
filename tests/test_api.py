import time
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
    assert body["model_version"] == "churn-tf-v1.1"
    assert body["decision_threshold"] == 0.71
    assert body["uptime_seconds"] >= 0
    # Process memory is reported on Linux and Windows; cgroup numbers only inside a Linux container.
    mem = body["memory"]
    assert set(mem) == {"cgroup_mb", "cgroup_peak_mb", "rss_anon_mb", "process_mb", "process_peak_mb"}
    assert 0 < mem["process_mb"] <= mem["process_peak_mb"]


@pytest.mark.parametrize("page", ["/", "/predict", "/recent", "/drift", "/performance", "/quality"])
def test_page_routes_serve_the_app(client, page):
    r = client.get(page)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert f'href="{page}"' in r.text  # each route has its sidebar link


def test_metrics_evaluation(client):
    body = client.get("/metrics/evaluation").json()
    assert body["training"]["test_roc_auc"] == 0.8554
    # Every result file the pages chart must be present (and copied into the Docker image).
    for name in ("benchmark_baseline", "benchmark_optimized", "benchmark_render", "benchmark_render_high_load",
                 "threshold_calibration", "real_world"):
        assert body[name] is not None, name
    assert [r["concurrency"] for r in body["benchmark_optimized"]["results"]] == [1, 10, 25]
    # The cut-off the page explains must be the one the API actually uses.
    assert body["threshold_calibration"]["chosen_threshold"] == client.app.state.pipeline.threshold


def test_frontend_served(client):
    r = client.get("/")
    assert '<script src="/static/app.js"></script>' in r.text
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "/predict/churn" in js.text
    # Fonts are self-hosted, not fetched from Google.
    assert "fonts.googleapis.com" not in r.text
    font = client.get("/static/fonts/nunito-sans-latin.woff2")
    assert font.status_code == 200 and font.content[:4] == b"wOF2"
    assert font.headers["content-type"] == "font/woff2"


def test_valid_prediction(client):
    r = client.post("/predict/churn", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["prediction"] in {"Churn", "Retain"}
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["inference_latency_ms"] > 0
    # Prediction must agree with the probability and the calibrated cut-off.
    threshold = client.app.state.pipeline.threshold
    assert threshold == 0.71
    churn = body["prediction"] == "Churn"
    assert churn == (body["churn_probability"] >= threshold)
    # Confidence is the probability of the predicted class.
    expected = body["churn_probability"] if churn else 1 - body["churn_probability"]
    assert abs(body["confidence_score"] - expected) < 1e-3


def test_threshold_between_05_and_cutoff(client):
    # A customer scoring between 0.5 and 0.71 was "Churn" before calibration and is "Retain" now.
    pipeline = client.app.state.pipeline
    r = pipeline.postprocess(0.6, latency_ms=1.0)
    assert (r.prediction, r.risk_level, r.confidence_score) == ("Retain", "Medium", 0.4)
    high = pipeline.postprocess(0.71, latency_ms=1.0)
    assert (high.prediction, high.risk_level) == ("Churn", "High")


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

    monkeypatch.setattr(client.app.state.pipeline, "predict", boom)
    # raise_server_exceptions=False so we see the response a real client would get. No `with`:
    # reuse the session's running app instead of starting (and later closing) a second one.
    r = TestClient(app, raise_server_exceptions=False).post("/predict/churn", json=VALID)
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "message": "Internal server error", "details": None}


LOYAL = {"tenure_months": 60, "monthly_charges": 45.0, "total_charges": 2700.0, "support_tickets": 0, "contract_type": "two-year"}


def test_predictions_are_stored_in_sqlite(client):
    assert client.get("/metrics/summary").json()["total"] == 0
    ids = [client.post("/predict/churn", json=VALID).json()["request_id"] for _ in range(3)]
    client.post("/predict/churn", json=LOYAL)
    rows = client.app.state.store._query("SELECT request_id FROM predictions ORDER BY id")
    assert [r["request_id"] for r in rows][:3] == ids
    s = client.get("/metrics/summary").json()
    assert (s["total"], s["churn"], s["churn_rate"]) == (4, 3, 0.75)
    assert s["by_risk"] == {"High": 3, "Low": 1}
    assert [(c["contract_type"], c["n"], c["churn_rate"]) for c in s["by_contract"]] == [("month-to-month", 3, 1.0), ("two-year", 1, 0.0)]


def test_api_key_required_when_configured(client):
    client.app.state.api_key = "s3cret"
    try:
        assert client.post("/predict/churn", json=VALID).status_code == 401
        wrong = client.get("/metrics/recent", headers={"X-API-Key": "wrong"})
        assert wrong.status_code == 401 and wrong.json()["error"] == "unauthorized"
        for path in ("/metrics/drift", "/metrics/summary"):
            assert client.get(path).status_code == 401
        assert client.post("/predict/churn", json=VALID, headers={"X-API-Key": "s3cret"}).status_code == 200
        # Platform health checks, the page and the published results stay public.
        for path in ("/health", "/", "/metrics/evaluation"):
            assert client.get(path).status_code == 200
    finally:
        client.app.state.api_key = None


def test_rate_limit_returns_429(client):
    from app.security import RateLimiter

    original, client.app.state.limiter = client.app.state.limiter, RateLimiter(3)
    try:
        codes = [client.post("/predict/churn", json=VALID).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        r = client.post("/predict/churn", json=VALID)
        assert r.json()["error"] == "rate_limited" and int(r.headers["retry-after"]) >= 1
    finally:
        client.app.state.limiter = original


def test_rate_limit_ignores_faked_forwarded_for(client):
    from app.security import RateLimiter

    original, client.app.state.limiter = client.app.state.limiter, RateLimiter(2)
    try:
        # A new fake X-Forwarded-For on every request must not buy a fresh budget.
        codes = [client.post("/predict/churn", json=VALID, headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code for i in range(3)]
        assert codes == [200, 200, 429]
    finally:
        client.app.state.limiter = original


def test_client_ip_header_when_configured(client):
    from app.security import RateLimiter

    original, client.app.state.limiter = client.app.state.limiter, RateLimiter(1)
    client.app.state.client_ip_header = "cf-connecting-ip"
    try:
        post = lambda ip: client.post("/predict/churn", json=VALID, headers={"CF-Connecting-IP": ip}).status_code  # noqa: E731
        assert [post("1.1.1.1"), post("1.1.1.1"), post("2.2.2.2")] == [200, 429, 200]  # one budget per visitor
    finally:
        client.app.state.limiter, client.app.state.client_ip_header = original, None


def test_rate_limiter_window_and_per_client():
    from app.security import RateLimiter

    limiter = RateLimiter(1, window=0.2)
    assert limiter.check("a") == 0
    assert limiter.check("a") >= 1  # second request inside the window is refused
    assert limiter.check("b") == 0  # other clients have their own budget
    time.sleep(0.25)
    assert limiter.check("a") == 0  # window passed


def test_security_headers(client):
    page = client.get("/")
    csp = page.headers["content-security-policy"]
    script_src = csp.split("script-src")[1].split(";")[0]
    assert script_src.strip() == "'self'"  # no inline scripts allowed
    assert "frame-ancestors 'none'" in csp
    assert client.get("/health").headers["x-content-type-options"] == "nosniff"


def test_http_errors_use_the_same_shape(client):
    r = client.get("/no-such-page")
    assert r.status_code == 404
    assert r.json() == {"error": "not_found", "message": "Not Found", "details": None}
