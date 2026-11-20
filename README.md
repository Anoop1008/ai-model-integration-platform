# Production-Ready AI Model Integration Platform

A TensorFlow churn classifier served by a modular FastAPI inference pipeline.

**Live demo:** https://churn-inference-api.onrender.com (web page; Swagger UI at [`/docs`](https://churn-inference-api.onrender.com/docs)).

**New here?** Read [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for a plain-English walkthrough. Render free instance: it sleeps after inactivity, so the first request can take ~50 s.

> **The data is synthetic.** `train.py` generates 2,000 telecom customer rows with a fixed seed. Churn is a noisy logistic function of the features. No real customer data is used.

## Architecture and request flow

```
                         ┌──────────────── process startup (lifespan, once) ─────────────────┐
                         │ load churn_model.keras + preprocessor.joblib + training_stats.json │
                         │ compile tf.function graph, run one warm-up inference               │
                         └────────────────────────────────────────────────────────────────────┘

 POST /predict/churn
        │  rate limit per client IP (429)  ·  X-API-Key check if API_KEY is set (401)
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
                              INSERT into SQLite (data/predictions.db), same worker thread
                              read back with SQL: /metrics/recent (last 10),
                              /metrics/drift (last 1000, PSI), /metrics/summary (aggregates)
                                                           ▼
                                                  200 ChurnResponse JSON
```

| File | Role |
|---|---|
| `train.py` | Generates data, fits preprocessor (train split only) and model, writes `artifacts/` |
| `app/schemas.py` | Pydantic request/response contracts |
| `app/pipeline.py` | `InferencePipeline`: `preprocess` → `infer` → `postprocess`, warm-up |
| `app/drift.py` | PSI drift indicator |
| `app/store.py` | SQLite prediction log: insert, recent, drift window, SQL aggregates |
| `app/security.py` | API-key check, per-client rate limiter, Content-Security-Policy |
| `app/main.py` | FastAPI app, lifespan, endpoints, error handlers |
| `app/static/index.html`, `app.js` | Web pages (plain HTML/CSS/JS, served at `/`); the script is a separate file so the CSP can forbid inline scripts |
| `tests/` | pytest suite |
| `benchmarks/latency_check.py` | asyncio + httpx latency check |

## Model (real run of `python train.py`)

| Metric | Value |
|---|---|
| Rows (train / test) | 2,000 (1,600 / 400) |
| Churn rate | 0.321 |
| Test accuracy | 0.7825 |
| Test ROC-AUC | 0.8554 |

## Quality on independent data

`evaluation/evaluate_real.py` scores the deployed model on the **IBM Telco Customer Churn** dataset (7,043 customers), the standard public churn benchmark. IBM describes it as a fictional telco company, so it is realistic sample data rather than verified real customers, but it was made independently of our generator. It has no `support_tickets` column, so that field is filled with a constant (0, 1 or 2) to show its effect. Numbers from `evaluation/results/real_world.json`:

| Model | Accuracy | ROC-AUC | Churn precision | Churn recall |
|---|---|---|---|---|
| Always predict "stay" | 0.7346 | – | – | – |
| Model at 0.5 cut-off, tickets = 0 | 0.7677 | 0.8184 | 0.5561 | 0.6174 |
| Model at 0.5 cut-off, tickets = 1 | 0.7548 | 0.8190 | 0.5291 | 0.6913 |
| Model at 0.5 cut-off, tickets = 2 | 0.7267 | 0.8228 | 0.4908 | 0.8026 |
| Logistic regression trained on this data (5-fold CV, same 4 fields) | 0.7866 | 0.8279 | 0.6260 | 0.4864 |

What it shows: the model **ranks** customers almost as well as one trained on this data (ROC-AUC 0.82 vs 0.83), so the churn patterns built into the synthetic data are realistic. Its **0.5 cut-off is less well matched**: it flags more churners than really leave (predicted 29–43% vs actual 26.5%), and the guessed ticket count moves accuracy from 0.77 to 0.73. Retraining or recalibrating on real data would fix that. Our own PSI check flags this data as shifted from training (monthly charges PSI 0.3773, significant), which is the drift indicator working as intended.

### Churn cut-off (model version `churn-tf-v1.1`)

The 0.5 cut-off flagged too many customers, so `evaluation/calibrate_threshold.py` tunes it: the IBM data is split 50/50 (stratified), the cut-off that maximises accuracy is picked on one half (**0.71**), and results are reported on the other, untouched half (3,522 customers, 26.5% churn). From `evaluation/results/threshold_calibration.json`:

| Cut-off | Accuracy | Churn precision | Churn recall | Flagged as churn |
|---|---|---|---|---|
| 0.5 (before) | 0.7521 | 0.5248 | 0.7016 | 35.5% |
| **0.71 (deployed)** | **0.7868** | **0.6620** | **0.4021** | **16.1%** |

Trade-off, chosen deliberately: flagged customers are now right 66% of the time instead of 52%, but only 40% of real churners are caught instead of 70%. That suits expensive retention offers. If missing a churner is costlier, rerun with a lower cut-off (maximising F1 picked 0.46). The cut-off is stored in `artifacts/training_stats.json` as `decision_threshold`; rerun the script after `train.py`. `support_tickets` was filled with 1 for this, since the IBM data lacks it.

Response fields follow the cut-off: `prediction` is `Churn` when probability ≥ 0.71, `confidence_score` is the probability of the predicted class, and `risk_level` is Low < 0.3, Medium < 0.71, High ≥ 0.71 (so High always means Churn).

## Run locally

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py                 # optional: artifacts/ is already committed
uvicorn app.main:app --port 8000
```

Web page: http://localhost:8000 · Swagger UI: http://localhost:8000/docs

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
#  "risk_level":"High","inference_latency_ms":6.569,"model_version":"churn-tf-v1.1"}

curl http://localhost:8000/metrics/recent
curl http://localhost:8000/metrics/drift
curl http://localhost:8000/metrics/summary      # SQL aggregates over every stored prediction
curl http://localhost:8000/metrics/evaluation   # recorded benchmark + evaluation results behind the web pages
```

Web pages (one HTML file, routed in the browser): `/` dashboard, `/predict`, `/recent`, `/drift`, `/performance`, `/quality`. Every number on them comes from the API: live values from `/health` and `/metrics/*`, measured results from `/metrics/evaluation`, which reads the committed files in `benchmarks/results/` and `evaluation/results/`.

When the server runs with `API_KEY` set, add the key to every prediction and metrics call:

```bash
curl -X POST http://localhost:8000/predict/churn -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{...}'
```

Invalid input returns 422 with a consistent shape: `{"error": "validation_error", "message": ..., "details": [...]}`. Unhandled errors return 500 `{"error": "internal_error", ...}` without leaking internals. 401, 404 and 429 use the same shape (`unauthorized`, `not_found`, `rate_limited`).

## Security

| Control | How | Default |
|---|---|---|
| API-key authentication | `X-API-Key` header on `/predict/churn` and `/metrics/recent`, `/drift`, `/summary`; compared with `hmac.compare_digest` (constant time). Swagger UI shows an **Authorize** button. `/health`, the web pages and `/metrics/evaluation` stay public. | Off unless `API_KEY` is set |
| Rate limiting | Sliding window per client IP on `/predict/churn`; `429` with `Retry-After`. Checked before the key, so it also slows key guessing. The visitor IP comes from `CLIENT_IP_HEADER` (on Render `cf-connecting-ip`, which Cloudflare overwrites so clients can't fake it), else the TCP peer. `X-Forwarded-For` is never trusted: an earlier version did, and a burst with a faked header bypassed the limit live. | 60 per minute |
| Input validation | Pydantic: types, ranges, allowed contract values, unknown fields rejected (422) | On |
| SQL injection | All SQL uses `?` parameters; no user value is pasted into SQL text | On |
| Content-Security-Policy | Page may load scripts, styles, fonts and API calls only from its own origin; no inline scripts; `frame-ancestors 'none'` | On |
| Other headers | `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` on every response | On |
| No internals in errors | 500 returns a generic message; the stack trace goes to the server log only | On |
| Container | Runs as non-root `appuser` (uid 1000); only `/app/data` is writable | On |
| Secrets | Read from environment variables; nothing secret is in the repo | — |

On the public demo links `API_KEY` is not set, so visitors can try it without a key. Setting it in the Render dashboard (Environment) switches auth on without a code change; the web page then asks for the key once per tab.

Settings (environment variables):

| Variable | Meaning | Default |
|---|---|---|
| `API_KEY` | Required `X-API-Key` value; unset = auth off | unset |
| `RATE_LIMIT_PER_MINUTE` | Predictions per client IP per minute; `0` = off | `60` |
| `CLIENT_IP_HEADER` | Header a trusted proxy sets to the visitor IP (Render: `cf-connecting-ip`, set in `render.yaml`) | unset = TCP peer |
| `DB_PATH` | SQLite file for the prediction log | `data/predictions.db` |
| `FRAME_ANCESTORS` | Who may show the page in a frame (CSP) | `'none'` |
| `PORT` | Port uvicorn listens on | `8000` |

Limits: the rate-limit counts are per process (fine for one worker; use Redis for several); there are no user accounts, only one shared key.

## Prediction log (SQL)

Every prediction is inserted into a SQLite table `predictions` (inputs, outputs, model version, UTC time; indexed on time). SQLite is part of Python's standard library, so there is no extra service to run. The endpoints read it with SQL:

- `/metrics/recent`: `SELECT ... ORDER BY id DESC LIMIT 10`
- `/metrics/drift`: the last 1,000 inputs, then PSI
- `/metrics/summary`: `COUNT`, `AVG`, `SUM(prediction = 'Churn')`, `GROUP BY risk_level`, `GROUP BY contract_type`

The insert runs in the same worker thread as inference, with WAL mode and `synchronous=NORMAL` (no disk flush per commit). Measured locally, back to back against the previous version: p50 7.1–7.7 ms vs 6.5–6.8 ms at concurrency 1, and the same within noise at 10 and 25 (~100 ms p50, ~176 vs ~170 req/s). A first version using `@app.middleware("http")` for the headers and a second thread hop was clearly slower (p50 157 vs 84 ms at concurrency 10), which is why the headers use a plain ASGI middleware.

Locally the log survives restarts. On Render's free plan the disk is wiped on every restart or redeploy; durable history would need a managed database (PostgreSQL), which is a change to `app/store.py` only.

## Tests

```bash
pytest -v
```

26 tests: SQLite storage and `/metrics/summary` aggregates, API key (401 without/with wrong key, 200 with it, public endpoints stay public), rate limit (429 + `Retry-After`, per-client window), Content-Security-Policy and `nosniff`, 404 error shape, each of the 6 page routes served, `/metrics/evaluation` (all result files present, cut-off matches the API), web page served, health, valid prediction, 422 for negative tenure / bad contract type / missing field, preprocessing determinism, `/metrics/recent`, drift `insufficient_data` then a real status, PSI math, 20 concurrent requests, and the 500 error shape.

## Latency before / after

`benchmarks/latency_check.py` sends 50 `/predict/churn` requests per concurrency level and polls `/health` during the load. Single uvicorn worker, local Windows machine, Python 3.11. Numbers are copied from `benchmarks/results/*.json`. Run benchmarks against a server started with `RATE_LIMIT_PER_MINUTE=0`, or the rate limit answers most requests with 429.

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

This is a **basic indicator, not full monitoring**: numeric features only, window read from the SQLite log (wiped on redeploy on free hosting), no alerting, no label/accuracy tracking.

## Design decisions and limitations

- **Synthetic training data.** On an independent public benchmark it ranks well (ROC-AUC ~0.82). The churn cut-off was tuned on that benchmark to 0.71, which favours accuracy over catching every churner; see "Churn cut-off".
- **Security is basic but real**: one shared API key (off on the public demo), per-IP rate limiting, CSP; no user accounts. See "Security".
- **Single worker.** Each extra worker would load its own model copy, multiplying memory. Concurrency comes from the thread pool.
- **Basic drift only.** See above.
- **SQLite, not a server database.** Right for one process; on free hosting the file is wiped on redeploy. See "Prediction log (SQL)".
- **Not deployed on AWS.** Deployed on Render's free plan (Docker web service, 512 MB, sleeps when idle). Live inference latency there was 74.6 ms for one request versus ~5–7 ms locally, because the free instance has a small CPU share.
- **Pinned dependencies.** `preprocessor.joblib` must be loaded by the same scikit-learn version that saved it.
- **Memory and size.** The Docker image is 2.54 GB (TensorFlow dominates). `/health` reports memory because Render's free plan hides metrics: container numbers (`memory.cgroup_mb`, `cgroup_peak_mb`), the app's own heap (`rss_anon_mb`), and process memory (`process_mb`, `process_peak_mb`, also on Windows). The page leads with `rss_anon_mb`, because the cgroup total includes file cache the kernel frees under pressure (a fresh Docker run read 510 of 512 MB with 0 OOM kills). On Render: 301.0 MB idle, **304.2 MB peak** after 150 requests at concurrency up to 25, against a 512 MB limit. Process RSS reads ~620 MB, but most of that is TensorFlow's shared libraries mapped from disk, which the limit does not count.
- **Render free-tier latency** (`benchmarks/results/render-free.json`, from a client in a different region, so ~290 ms is network round trip): predict p50 291 / 321 / 929 ms at concurrency 1 / 10 / 25, 0 errors. The instance has 0.15 CPU.
