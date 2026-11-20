# Build Progress

Phase-by-phase log of what was built and whether it was verified. Every number below comes from a real command run on 2026-11-20.

| Phase | Status |
|---|---|
| 0. Environment check | ✅ Done (WSL repaired, Docker running; gh login still pending) |
| 1. Scaffold, data, training | ✅ Verified |
| 2. API | ✅ Verified |
| 3. Tests | ✅ Verified (11/11 pass) |
| 4. Latency before/after | ✅ Verified |
| 5. Docker | ✅ Verified |
| 6. Git and Render | ✅ Pushed to GitHub, live on Render |
| 7. README and summary | ✅ Done |

---

## Phase 0: Environment check

| Tool | Result |
|---|---|
| Python 3.11 | Not installed (system had 3.13.15 only). Installed **Python 3.11.17** with `uv python install 3.11`. Project venv: `.venv` |
| git | `git version 2.55.0.windows.5` |
| Docker | User installed Docker Desktop (per-user, `%LOCALAPPDATA%\Programs\DockerDesktop`). Client/server **29.9.3**. |
| WSL | Was broken (`Wsl/CallMsi/Install/REGDB_E_CLASSNOTREG`). Reinstalled `Microsoft.WSL` elevated via winget. Now WSL **3.0.5.0**, kernel 6.18.58.1-1, default version 2. |
| GitHub CLI | Not installed. Installed with `winget install GitHub.cli`. Not logged in yet. |

Git had no identity configured. Repo-local identity is `Anoop1008` with the GitHub noreply email, so no personal email is public. Commits were rewritten to this identity before the first push.

**User actions needed:**
- [x] Install Docker Desktop and WSL2.
- [x] Run `gh auth login` (logged in as Anoop1008).
- [x] Create a Render account and deploy the Blueprint.

## Phase 1: Scaffold, data, training ✅

Built: `train.py`, `artifacts/` (`churn_model.keras`, `preprocessor.joblib`, `training_stats.json`), `requirements.txt` (pinned), `requirements-dev.txt`, `.gitignore`, `.dockerignore`, `.gitattributes` (LF line endings).

- 2,000 synthetic rows, seed 42, churn rate 0.321.
- ColumnTransformer (StandardScaler + OneHotEncoder) fit on train split only.
- Keras Sequential: Dense(32) → Dropout(0.2) → Dense(16) → Dense(1, sigmoid), early stopping.
- `training_stats.json`: 10 quantile bins per numeric feature, proportions, mean churn probability 0.3146.

Output of `python train.py`:
```
rows=2000 train=1600 test=400 churn_rate=0.321
test_accuracy=0.7825 test_roc_auc=0.8554
```

## Phase 2: API ✅

Built: `app/schemas.py`, `app/pipeline.py`, `app/drift.py`, `app/main.py`.

- Lifespan loads model + preprocessor once, runs warm-up.
- `POST /predict/churn` offloads inference with `run_in_threadpool`.
- `GET /health`, `GET /metrics/recent` (ring buffer of 10), `GET /metrics/drift` (PSI, `insufficient_data` under 50).
- Consistent JSON for 422 and 500.

Smoke test against a live uvicorn server:
```
/health        -> {"status":"ok","model_loaded":true,"model_version":"churn-tf-v1","uptime_seconds":0.21}
/predict/churn -> {"prediction":"Churn","confidence_score":0.8624,"churn_probability":0.8624,"risk_level":"High",...}
invalid body   -> HTTP 422 {"error":"validation_error",...}
/metrics/drift -> {"status":"insufficient_data","n_samples":1,...}
```

## Phase 3: Tests ✅

`tests/test_api.py`, `tests/conftest.py`. Result of `pytest -v`: **11 passed**.

Covers: health; valid prediction; 422 for negative tenure, bad contract_type, missing field; preprocessing determinism; /metrics/recent; drift insufficient_data then `significant` on shifted inputs; PSI math; 20 concurrent requests all 200; 500 error shape.

One harmless warning: Starlette deprecation notice about `httpx` in `TestClient`.

## Phase 4: Latency before/after ✅

- Commit tagged `baseline-naive`: model loaded inside handler, inference inline in the async handler.
- Commit tagged `optimized`: lifespan load, warm-up, `run_in_threadpool`, plus `tf.function`.
- Results saved in `benchmarks/results/` (`baseline-naive.json`, `optimized-eager-keras.json`, `optimized.json`).

Finding during this phase: the first optimized run (eager Keras calls) regressed at concurrency 25 (p50 352.83 ms, 61.62 req/s). Root cause: eager Keras calls are Python-heavy and contend for the GIL across threads. Fix: compile the forward pass with `tf.function` (1.98 ms → 0.31 ms per call, same output). Tests still pass (11/11).

| Concurrency | predict p50 naive → optimized | /health p95 naive → optimized | req/s naive → optimized |
|---|---|---|---|
| 1 | 49.05 → 6.48 ms | 22.77 → 9.14 ms | 17.62 → 78.21 |
| 10 | 443.19 → 6.57 ms | 1174.69 → 16.67 ms | 21.90 → 144.40 |
| 25 | 1098.01 → 84.77 ms | 1226.93 → 67.71 ms | 20.10 → 183.37 |

## Phase 5: Docker ✅

Files: `Dockerfile` (python:3.11-slim, tensorflow-cpu, `--no-cache-dir`, non-root user, HEALTHCHECK on /health, `$PORT` default 8000, `TF_CPP_MIN_LOG_LEVEL=2`), `render.yaml`.

Verified with `docker build -t churn-api .` and `docker run -p 8000:8000 churn-api`:
```
/health        -> {"status":"ok","model_loaded":true,"model_version":"churn-tf-v1","uptime_seconds":0.77}
/predict/churn -> {"prediction":"Churn","confidence_score":0.8624,...,"inference_latency_ms":5.057,...}
invalid body   -> HTTP 422
docker exec churn id -> uid=1000(appuser)        (non-root)
HEALTHCHECK    -> health=healthy
PORT=10000     -> /health on :10000 returned 200 (Render-style $PORT honored)
```

| Measure | Value |
|---|---|
| Image size | 2.54 GB (`docker image ls`) |
| Idle container memory | 295.3 MiB (`docker stats`, after startup + 2 requests) |

Idle memory is under ~350 MB, so Render's free 512 MB instance should fit. Headroom is about 200 MB, so it should be watched under load. The earlier 367 MB Windows figure was the native process, not the container.

## Phase 6: Git and Render 🟡

- Local repo on `main` with commits per phase and tags `baseline-naive`, `optimized`.
- `render.yaml` written.
- Public repo created and pushed (with tags): https://github.com/Anoop1008/ai-model-integration-platform
- Render Blueprint deployed `churn-inference-api` (Docker, free plan) from commit `805266f`. Live URL: https://churn-inference-api.onrender.com

Live curl checks:
```
GET  /health        -> HTTP 200 {"status":"ok","model_loaded":true,"model_version":"churn-tf-v1","uptime_seconds":62.26}
POST /predict/churn -> HTTP 200 {"prediction":"Churn","confidence_score":0.8624,"churn_probability":0.8624,"risk_level":"High","inference_latency_ms":74.638,...}
POST invalid body   -> HTTP 422 {"error":"validation_error",...}
GET  /metrics/drift -> {"status":"insufficient_data","n_samples":1,...}
```
Same churn probability (0.8624) as local and Docker runs. Inference on Render's free CPU was 74.6 ms for one request (local ~5–7 ms).

### Render memory and load check

Render's free plan hides memory metrics ("Upgrade to any paid compute plan to view application metrics"). Fix: `/health` now returns `memory` from Linux cgroup v2 (`memory.current`, `memory.peak`) plus the process's anonymous RSS. In Docker, `cgroup_mb` matched `docker stats` (296.9 vs 296.8 MiB).

Why not plain RSS: it read ~620 MB, but ~340 MB of that is TensorFlow's shared libraries mapped from disk (`RssFile`), which are reclaimable and not what the 512 MB limit counts.

Live on Render (instance: 512 MB, 0.15 CPU), `benchmarks/latency_check.py --url https://churn-inference-api.onrender.com --label render-free`:

| Concurrency | predict p50 | predict p95 | errors | req/s |
|---|---|---|---|---|
| 1 | 291.05 ms | 342.01 ms | 0 | 3.23 |
| 10 | 320.73 ms | 489.25 ms | 0 | 24.96 |
| 25 | 928.67 ms | 1289.83 ms | 0 | 22.86 |

Memory from live `/health`: 301.0 MB idle, **304.2 MB peak** after the load. No restart (uptime kept rising). About 290 ms of each request is network round trip (`/health` alone p50 293 ms).

## Phase 7: README and summary ✅

`README.md` includes: live URL, architecture diagram, local run steps, curl examples, test instructions, real before/after table, drift explanation, design decisions and limitations.

Not verified: the README links rendering on GitHub, and Render behaviour beyond concurrency 25.
