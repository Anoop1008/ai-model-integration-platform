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

---

## After the spec: frontend and walkthrough ✅

Built:
- `app/static/index.html`: one static page served by FastAPI at `/` (same origin, so no CORS, no build step, no new dependencies). Prediction form with two example presets, result card (prediction, risk, probability bar, confidence, latency), API status pill with live memory, last 10 predictions, drift table. Light and dark mode.
- `docs/HOW_IT_WORKS.md`: plain-English walkthrough of the request flow, every file, the four key ideas, and likely interview questions.
- New test `test_frontend_served`. `pytest`: **12 passed**.

Verified in a real browser (Chrome via DevTools, local server):
- Prediction works for both presets (high-risk 86.2% Churn/High, loyal 0.8% Retain/Low).
- Invalid input (tenure −5, monthly 900) shows the API's 422 messages for both fields.
- After 62 requests the drift table appears (overall `moderate`, from sampling noise with requests drawn from the training distribution).
- Phone width (390 px), light mode: page width 390 px, no horizontal scroll. Fixed a bug where the table widened the page.
- No console errors after adding an inline favicon.

Docker check: the page is served from the image (`GET /` → 200, 14,675 bytes).

Memory caveat found here: the first run of a freshly built image reported `cgroup_mb` 1139.9 MB, while later runs reported 293.6 MB (matching `docker stats` 293.9 MiB). The extra was reclaimable file cache from reading TensorFlow's files out of new image layers. With a real 512 MB limit (`docker run -m 512m`) the container ran at 294.4 MB and was not OOM-killed. `rss_anon_mb` (~279 MB) is the steadier number; the caveat is noted in `memory_mb()`.

---

## Checking the "not tested" items ✅

**Accessibility (screen reader proxy).** No real screen reader available. Ran Lighthouse on the live page with a prediction shown: **Accessibility 100**, Best Practices 100, SEO 75 → fixed the one failure (missing meta description).

**Phone width.** Skipped at the user's request (already checked locally at 390 px).

**Load above 25 concurrent (live Render).** `benchmarks/latency_check.py --levels 25,50,100 --requests 200`, results in `benchmarks/results/render-free-high-load.json`:

| Concurrency | Requests | predict p50 | predict p95 | errors | /health p50 | req/s |
|---|---|---|---|---|---|---|
| 25 | 200 | 922.18 ms | 1500.55 ms | 0 | 451.65 ms | 24.20 |
| 50 | 200 | 2024.85 ms | 3112.79 ms | 0 | 431.20 ms | 24.55 |
| 100 | 200 | 3383.24 ms | 5672.01 ms | 0 | 430.70 ms | 23.96 |

600 requests, 0 errors, no restart; memory peak 314.2 MB of 512 MB. Throughput is flat at ~24 req/s: the 0.15 CPU is the limit, so more concurrency only adds queueing time. `/health` p50 stayed ~430 ms (≈290 ms of that is network), so the event loop stayed responsive.

**Model quality on independent data.** `evaluation/evaluate_real.py` on IBM Telco Customer Churn (7,043 rows; IBM calls it a fictional telco, so realistic sample data, not verified real customers). `support_tickets` is missing there and filled with 0/1/2. Deployed model: ROC-AUC 0.8184–0.8228, accuracy 0.7267–0.7677. Majority-class accuracy 0.7346. Logistic regression trained on that data (5-fold CV): ROC-AUC 0.8279, accuracy 0.7866. Conclusion: ranking transfers well; the 0.5 threshold over-predicts churn (29–43% predicted vs 26.5% actual). PSI flags the data as shifted (monthly charges 0.3773). Full table in README.

---

## Churn cut-off tuned (model version `churn-tf-v1.1`) ✅

- `evaluation/calibrate_threshold.py`: split IBM Telco 50/50 (stratified, seed 42), pick the cut-off on the tune half, report on the holdout half (3,522 rows). Writes `decision_threshold` and `model_version` into `artifacts/training_stats.json`.
- Compared objectives on the same split: max accuracy → 0.71; max F1 → 0.46. User chose **0.71**.
- Holdout: accuracy 0.7521 → **0.7868**, precision 0.5248 → **0.6620**, recall 0.7016 → **0.4021**, flagged 35.5% → **16.1%** (actual 26.5%). ROC-AUC unchanged (0.8161).
- `app/pipeline.py`: reads the cut-off (default 0.5). `confidence_score` is now the probability of the predicted class (was `max(p, 1-p)`, wrong once the cut-off isn't 0.5). High risk now starts at the cut-off, so High always means Churn.
- Tests: updated version and confidence checks; new `test_threshold_between_05_and_cutoff` (0.6 → Retain/Medium/0.4, 0.71 → Churn/High). `pytest`: **13 passed**.

**Borderline display fix.** Just under the cut-off, the page showed "Likely to stay" with a bare low confidence (e.g. 29%), and 0.7095 rounded to "71.0%" next to a 71% cut-off. Now `/health` returns `decision_threshold`; when confidence is under 50% the page shows "Borderline: 70.95% churn, just under the 71% cut-off" and uses 2 decimals in the result and the table. Verified in the browser (borderline, clear churn, clear stay); no console errors. `pytest`: 13 passed.

## Final UI from design file

Rebuilt `app/static/index.html` from the supplied design (`Churn Inference Platform UI.html`, a bundled React mock with hard-coded numbers) as plain HTML/CSS/JS. No new dependencies.

- Live from the API: health status, model version, cut-off, memory (see below), prediction gauge with cut-off marker, risk badge, 422 errors, last 10 predictions, live PSI once 50+ requests exist.
- Static, copied from committed result files: latency bars and throughput chart (`benchmarks/results/baseline-naive.json`, `optimized.json`), cut-off trade-off (`evaluation/results/threshold_calibration.json`), ROC-AUC (`artifacts/training_stats.json`), and IBM Telco PSI shown until live drift has enough data (`evaluation/results/real_world.json`).
- Kept the borderline fix: 62.35% at a 71% cut-off reads "Borderline: 62.35% churn, just under the 71% cut-off".

Checked locally (1440 px, Chrome): high-risk example gives 86.2% / Likely to churn / High; tenure -1 shows the 422 detail; tenure 48 shows the borderline text. Lighthouse desktop: Accessibility, Best Practices, SEO and Agentic all 100. 13/13 tests pass. Works: yes.

### Follow-up: memory on every platform, self-hosted fonts, phone width

- **Memory showed "n/a" locally.** `/health` now also returns `process_mb` / `process_peak_mb`: `VmRSS`/`VmHWM` from `/proc` on Linux, the working set from the Win32 API (`GetProcessMemoryInfo` via `ctypes`, no psutil) on Windows. Checked: Windows `/health` 367.0 MB vs `tasklist` 375,812 K (= 367.0 MB). The test now requires a real number on every platform.
- **Headline memory number changed.** In Docker with `-m 512m`, `cgroup_mb` read 509.8 of 512 MB right after start. `memory.stat`: anon 251.6 MB, file cache 230.3 MB, `oom_kill` 0, container kept running after 100 requests. The kernel reclaims that cache, so a cgroup meter at 99.6% would look full when it is not. The page now leads with the app's own memory (`rss_anon_mb`, 279.6 / 512 MB in that run) and shows the container total in the note. On Windows it falls back to `process_mb`.
- **Fonts self-hosted.** Latin subsets of Nunito Sans and JetBrains Mono (variable woff2, 81 KB together, taken from the design file) live in `app/static/fonts/` with their OFL license, served via a `/static` mount with `font/woff2` content type. The page makes no third-party requests.
- **Phone width (375 px).** The sidebar becomes a compact header with a swipeable nav (866 px tall before, 324 px after). The throughput chart scrolls sideways inside its card instead of shrinking its text to ~5 px. No page-level horizontal scroll. Lighthouse mobile: Accessibility, Best Practices, SEO, Agentic all 100.
- Dark mode: not needed (per user). 13/13 tests pass. Works: yes.
