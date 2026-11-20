# Production-Ready AI Model Integration Platform

A TensorFlow churn classifier served by a modular FastAPI inference pipeline.

> **The data is synthetic.** `train.py` generates 2,000 telecom customer rows with a fixed seed. Churn is a noisy logistic function of the features. No real customer data is used.

## Architecture and request flow

```
                         ┌──────────────── process startup (lifespan, once) ─────────────────┐
                         │ load churn_model.keras + preprocessor.joblib + training_stats.json │
                         │ compile tf.function graph, run one warm-up inference               │
                         └────────────────────────────────────────────────────────────────────┘

 POST /predict/churn
        │
        ▼
 ┌─────────────────┐  invalid  ┌──────────────────────────┐
 │ Pydantic v2      │─────────▶│ 422 {"error", "message",  │
 │ ChurnRequest     │          │      "details"}           │
 └────────┬────────┘          └──────────────────────────┘
          │ valid
          ▼          ── run_in_threadpool (keeps the event loop free) ──
 ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────────────┐
 │ preprocess       │──▶│ infer            │──▶│ postprocess               │
 │ fitted sklearn   │   │ TF graph,        │   │ prediction, confidence,   │
 │ ColumnTransformer│   │ training=False   │   │ risk level, latency, ver. │
 └─────────────────┘   └─────────────────┘   └────────────┬─────────────┘
                                                           ▼
                                   ring buffers: last 10 predictions (/metrics/recent)
                                                 last 1000 inputs  (/metrics/drift, PSI)
                                                           ▼
                                                  200 ChurnResponse JSON
```

| File | Role |
|---|---|
| `train.py` | Generates data, fits preprocessor (train split only) and model, writes `artifacts/` |
| `app/schemas.py` | Pydantic request/response contracts |
| `app/pipeline.py` | `InferencePipeline`: `preprocess` → `infer` → `postprocess`, warm-up |
| `app/drift.py` | PSI drift indicator |
| `app/main.py` | FastAPI app, lifespan, endpoints, error handlers |
| `tests/` | pytest suite |
| `benchmarks/latency_check.py` | asyncio + httpx latency check |

## Model (real run of `python train.py`)

| Metric | Value |
|---|---|
| Rows (train / test) | 2,000 (1,600 / 400) |
| Churn rate | 0.321 |
| Test accuracy | 0.7825 |
| Test ROC-AUC | 0.8554 |

## Run locally

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py                 # optional: artifacts/ is already committed
uvicorn app.main:app --port 8000
```

Swagger UI: http://localhost:8000/docs

### Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## curl examples

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict/churn -H "Content-Type: application/json" \
  -d '{"tenure_months":5,"monthly_charges":95.5,"total_charges":477.5,"support_tickets":4,"contract_type":"month-to-month"}'
# {"request_id":"...","prediction":"Churn","confidence_score":0.8624,"churn_probability":0.8624,
#  "risk_level":"High","inference_latency_ms":6.569,"model_version":"churn-tf-v1"}

curl http://localhost:8000/metrics/recent
curl http://localhost:8000/metrics/drift
```

Invalid input returns 422 with a consistent shape: `{"error": "validation_error", "message": ..., "details": [...]}`. Unhandled errors return 500 `{"error": "internal_error", ...}` without leaking internals.

## Tests

```bash
pytest -v
```

11 tests: health, valid prediction, 422 for negative tenure / bad contract type / missing field, preprocessing determinism, `/metrics/recent`, drift `insufficient_data` then a real status, PSI math, 20 concurrent requests, and the 500 error shape.

## Latency before / after

`benchmarks/latency_check.py` sends 50 `/predict/churn` requests per concurrency level and polls `/health` during the load. Single uvicorn worker, local Windows machine, Python 3.11. Numbers are copied from `benchmarks/results/*.json`.

- **baseline-naive** (git tag `baseline-naive`): model loaded inside the request handler, inference run directly in the async handler, no warm-up.
- **optimized** (git tag `optimized`): model loaded once in the lifespan, warm-up at startup, inference offloaded with `run_in_threadpool`, forward pass compiled with `tf.function`.

| Concurrency | Version | predict p50 (ms) | predict p95 (ms) | errors | /health p50 (ms) | /health p95 (ms) | req/s |
|---|---|---|---|---|---|---|---|
| 1 | baseline-naive | 49.05 | 69.37 | 0 | 4.51 | 22.77 | 17.62 |
| 1 | optimized | 6.48 | 10.58 | 0 | 5.17 | 9.14 | 78.21 |
| 10 | baseline-naive | 443.19 | 852.14 | 0 | 399.80 | 1174.69 | 21.90 |
| 10 | optimized | 6.57 | 303.92 | 0 | 6.29 | 16.67 | 144.40 |
| 25 | baseline-naive | 1098.01 | 1382.46 | 0 | 528.76 | 1226.93 | 20.10 |
| 25 | optimized | 84.77 | 232.18 | 0 | 24.59 | 67.71 | 183.37 |

What this shows: in the naive version `/health` does no work yet waits ~0.4–1.2 s under load, because inference blocks the event loop. In the optimized version it stays at tens of ms.

An intermediate run (`optimized-eager-keras.json`: lifespan + threadpool + warm-up, but eager Keras calls) reached only 61.62 req/s at concurrency 25 with predict p50 352.83 ms. Eager Keras calls spend most of their time in Python while holding the GIL, so worker threads contend. Compiling the forward pass with `tf.function` cut a single call from 1.98 ms to 0.31 ms in a local measurement and fixed this.

Caveats: 50 requests per level is a small sample, so p95 values are noisy. The occasional p95 spike at concurrency 10 is likely thread-pool start-up; not investigated further.

## Drift indicator

`GET /metrics/drift` computes the **Population Stability Index (PSI)** for each numeric feature, comparing the last 1,000 request inputs against the training distribution saved in `training_stats.json` (10 quantile bins per feature):

```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```

| PSI | Status |
|---|---|
| < 0.1 | stable |
| 0.1 – 0.25 | moderate |
| > 0.25 | significant |

Under 50 requests it returns `insufficient_data`. The overall status is the worst feature's status.

This is a **basic indicator, not full monitoring**: numeric features only, in-memory window lost on restart, no alerting, no label/accuracy tracking.

## Design decisions and limitations

- **Synthetic data.** The model only demonstrates the pipeline; its metrics say nothing about real churn.
- **No authentication** or rate limiting on any endpoint.
- **Single worker.** Each extra worker would load its own model copy, multiplying memory. Concurrency comes from the thread pool.
- **Basic drift only.** See above.
- **No database.** Recent predictions and the drift window live in memory and reset on restart.
- **Not deployed on AWS.** Target is Render (Docker web service).
- **Pinned dependencies.** `preprocessor.joblib` must be loaded by the same scikit-learn version that saved it.
- **Memory.** The local (Windows, non-container) server used 367 MB working set after startup + one request. Render's free instance has 512 MB.
