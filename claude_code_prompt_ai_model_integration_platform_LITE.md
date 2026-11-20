# Claude Code prompt (LITE, one-day version): Production-Ready AI Model Integration Platform

Paste everything between the two PROMPT lines as one message, inside your empty project folder.

=== PROMPT START ===

Build, test, and deploy a small but real project from scratch in the current empty directory. Work phase by phase; do not move on until the current phase is verified by real command output.

## What we are building

"Production-Ready AI Model Integration Platform": a trained TensorFlow classifier wrapped in a modular FastAPI inference pipeline.

Request flow: Pydantic v2 validation -> deterministic preprocessing (the same fitted scikit-learn transformer used in training) -> TensorFlow inference -> post-processing into clean JSON with a confidence score.

Production concerns that must be genuinely implemented and measured:
1. Load the model ONCE at server startup (FastAPI lifespan), never per request.
2. Offload CPU-heavy inference (run_in_threadpool) so it never blocks the event loop.
3. Containerize with Docker.
Plus a basic drift indicator, a pytest suite, and deployment to Render.

Scope is deliberately small. Do NOT add PostgreSQL, AWS/EC2, GitHub Actions, GHCR, or a regression model.

## Step 0: Environment check

Check what's installed. Install anything missing that you can, and tell me what I have to do myself. Check Python 3.11, git, Docker (and that the daemon is running). Show real version output. Give me a short checklist of anything only I can do, and wait for my confirmation.

## Rules

1. Run every command and read the real output. Never invent numbers; any metric in the README must come from an actual run.
2. Fix root causes; never weaken a test to pass it.
3. STOP and ask before `git push`, creating any cloud resource, or anything needing credentials. Never commit secrets.
4. Target Python 3.11. Show me a short plan first, then track phases in the task list.
5. Keep code simple and readable. I must be able to explain every file in an interview, so add brief comments explaining WHY (model loaded once, why threadpool, what PSI is).

## Phase 1: Scaffold, data, training

Files: app/main.py, app/schemas.py, app/pipeline.py, app/drift.py, train.py, tests/, benchmarks/latency_check.py, artifacts/, Dockerfile, render.yaml, requirements.txt, requirements-dev.txt, .gitignore, .dockerignore, README.md.

train.py:
- 2,000 synthetic telecom customer rows, fixed seed. Features: tenure_months (0-72), monthly_charges (18-120), total_charges, support_tickets (0-10), contract_type (month-to-month / one-year / two-year).
- Churn label = noisy logistic function of the features (short tenure, high charges, many tickets, month-to-month raise churn).
- One scikit-learn ColumnTransformer (StandardScaler + OneHotEncoder), fit on train split only, saved to artifacts/preprocessor.joblib.
- Compact TensorFlow Sequential classifier (Dense + Dropout + Sigmoid) saved to artifacts/churn_model.keras. Print test accuracy and ROC-AUC.
- Save artifacts/training_stats.json (per-feature bin edges and training bin proportions, mean churn probability).
- Commit artifacts/ so cloud builds work without retraining. README must say the data is synthetic.

## Phase 2: API

- schemas.py: ChurnRequest (tenure_months int 0-120, monthly_charges float >0 and <=500, total_charges float >=0, support_tickets int 0-50, contract_type Literal) with a Swagger example; ChurnResponse (request_id, prediction "Churn"/"Retain", confidence_score, churn_probability, risk_level Low/Medium/High, inference_latency_ms, model_version).
- pipeline.py: InferencePipeline with separate preprocess, infer, postprocess methods and latency timing. Warm-up inference at startup. Call the model as model(x, training=False).
- main.py: lifespan loads model + preprocessor once. GET /health (status, model_loaded, model_version, uptime_seconds). POST /predict/churn. GET /metrics/recent (last 10 predictions from an in-memory ring buffer). GET /metrics/drift (basic PSI per numeric feature over recent requests vs training_stats.json; "insufficient_data" under 50 requests; PSI <0.1 stable, 0.1-0.25 moderate, >0.25 significant; call it a basic indicator, not full monitoring). Consistent JSON for 422 and 500 errors.

## Phase 3: Tests

pytest covering: /health; valid prediction; 422 for negative tenure, bad contract_type, missing field; preprocessing determinism; /metrics/recent; /metrics/drift insufficient_data then real status; 20 concurrent requests all 200. Run pytest -v, all must pass.

## Phase 4: Quick latency before/after (keep it small)

1. First commit a deliberately naive version (model loaded inside the request handler). Tag baseline-naive.
2. benchmarks/latency_check.py (asyncio + httpx): at concurrency 1, 10, 25 report p50, p95, error count for /predict/churn, and /health latency during the load. Save JSON to benchmarks/results/.
3. Run it, then fix (lifespan load, threadpool offload, warm-up), tag optimized, run again.
4. Put a before/after table in the README using ONLY the saved numbers.

## Phase 5: Docker

Dockerfile: python:3.11-slim, tensorflow-cpu, pip --no-cache-dir, non-root user, HEALTHCHECK on /health, start command respecting $PORT (default 8000), TF_CPP_MIN_LOG_LEVEL=2. Build, run, and curl /health and /predict/churn. Report image size and container memory at idle. If idle memory is above ~350 MB warn me that Render's free 512 MB instance may fail and suggest options.

## Phase 6: Git and Render

- git init, sensible .gitignore (keep artifacts/ committed), clear commits per phase.
- render.yaml: Docker web service, healthCheckPath /health, honoring $PORT.
- Ask me before pushing. Create the GitHub repo with gh and push after I approve.
- Give me the exact Render dashboard steps. Ask me for the live URL, then run curl checks: /health returns model_loaded true, a valid prediction returns the required fields, an invalid payload returns 422.

## Phase 7: README and summary

README: architecture and request-flow diagram (ASCII), local run steps, curl examples, test instructions, the real before/after table, drift explanation, and "Design decisions and limitations" (synthetic data, no auth, single worker, basic drift only, no DB, not deployed on AWS).

Finish with a short summary: how to run locally, real accuracy and ROC-AUC, test count and result, image size, benchmark comparison, live URL, and anything unverified.

=== PROMPT END ===
