# Build Progress

Phase-by-phase log of what was built and whether it was verified. Every number below comes from a real command run on 2026-11-20.

| Phase | Status |
|---|---|
| 0. Environment check | Partial: Docker missing (user action needed) |
| 1. Scaffold, data, training | ✅ Verified |
| 2. API | ✅ Verified |
| 3. Tests | ✅ Verified (11/11 pass) |
| 4. Latency before/after | ✅ Verified |
| 5. Docker | ⏸ Files written, NOT verified (Docker not installed) |
| 6. Git and Render | 🟡 Local git done; GitHub push and Render not started (needs approval) |
| 7. README and summary | 🟡 README written; Docker and Render sections need real numbers |

---

## Phase 0: Environment check

| Tool | Result |
|---|---|
| Python 3.11 | Not installed (system had 3.13.15 only). Installed **Python 3.11.17** with `uv python install 3.11`. Project venv: `.venv` |
| git | `git version 2.55.0.windows.5` |
| Docker | **Not installed** (`docker: command not found`) |
| GitHub CLI | Not installed. Installed with `winget install GitHub.cli`. Not logged in yet. |

Git had no identity configured. Set repo-local `user.name=shukl`, `user.email=<account email>`. Change with `git config user.name "..."` if wanted.

**User actions needed:**
- [ ] Install Docker Desktop (needs admin + WSL2) and start it.
- [ ] Run `gh auth login` (open a new terminal first so `gh` is on PATH).
- [ ] Create a Render account and connect it to GitHub.

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

## Phase 5: Docker ⏸ NOT VERIFIED

Written: `Dockerfile` (python:3.11-slim, tensorflow-cpu, `--no-cache-dir`, non-root user, HEALTHCHECK on /health, `$PORT` default 8000, `TF_CPP_MIN_LOG_LEVEL=2`), `render.yaml`.

Not built or run: Docker is not installed. Image size and container idle memory unknown.

⚠️ Memory warning: the local Windows server process used **367 MB** working set after startup and one request. This is above ~350 MB. Render's free instance (512 MB) may run out of memory. Container memory still needs measuring. Options if it is too high:
1. Convert the model to TFLite and serve with `tflite-runtime` / `ai-edge-litert` (drops full TensorFlow).
2. Use a paid Render instance with more RAM.
3. Swap the classifier to scikit-learn (smaller runtime), at the cost of the TensorFlow requirement.

## Phase 6: Git and Render 🟡

- Local repo on `main` with commits per phase and tags `baseline-naive`, `optimized`.
- `render.yaml` written.
- Not done: GitHub repo creation, push, Render deploy, live curl checks. These need your approval and login.

## Phase 7: README 🟡

`README.md` written: architecture diagram, run steps, curl examples, tests, real before/after table, drift explanation, design decisions and limitations. Pending: Docker image size, container memory, live URL.
