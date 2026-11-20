# How It Works

A plain-English walkthrough of this project, written so you can explain every part of it in an interview. All numbers come from real runs (see `PROGRESS.md` and `benchmarks/results/`).

## The one-sentence pitch

> A TensorFlow model that predicts whether a telecom customer will leave, served by a FastAPI service that loads the model once, keeps the server responsive under load, watches for input drift, and runs in Docker on Render.

## Follow one request

When someone presses **Predict churn** on the web page:

1. **Browser** (`app/static/index.html`) sends `POST /predict/churn` with five fields as JSON.
2. **Validation** (`app/schemas.py`): Pydantic checks types and ranges. Bad input (e.g. tenure `-5`) stops here with HTTP 422 and a list of what is wrong. The model never sees it.
3. **Thread pool hand-off** (`app/main.py`): the handler passes the work to a worker thread with `run_in_threadpool`, so the server can keep answering other requests.
4. **Preprocess** (`app/pipeline.py`): the same fitted scikit-learn `ColumnTransformer` used in training scales the numbers and one-hot encodes the contract type. Output: 7 numbers.
5. **Infer**: the TensorFlow model turns those 7 numbers into a churn probability between 0 and 1.
6. **Postprocess**: probability becomes a readable answer: `Churn`/`Retain`, a confidence score, a risk level (Low under 0.3, Medium under 0.71, High otherwise; 0.71 is also the churn cut-off), latency, and model version.
7. **Bookkeeping**: the request is added to two in-memory lists, one for "last 10 predictions" and one for drift.
8. **Response**: JSON goes back; the page shows it.

## Every file, in one line each

| File | What it does | Why it exists |
|---|---|---|
| `train.py` | Makes 2,000 synthetic customers, trains the model, saves artifacts | Reproducible training (fixed seed 42) |
| `artifacts/churn_model.keras` | The trained TensorFlow model | Committed so Render does not retrain |
| `artifacts/preprocessor.joblib` | The fitted scaler + encoder | Serving must transform data exactly like training did |
| `artifacts/training_stats.json` | Training data distribution, metrics | Reference point for drift |
| `app/schemas.py` | Input/output contracts | Reject bad input before it reaches the model |
| `app/pipeline.py` | preprocess → infer → postprocess | Each step separate, so each can be tested and timed |
| `app/drift.py` | PSI drift calculation | Basic warning when live inputs stop looking like training data |
| `app/main.py` | FastAPI app, endpoints, error handlers | Ties it together |
| `app/static/index.html` | The web page | Lets anyone use the model without curl |
| `tests/test_api.py` | 12 automated tests | Proves behaviour; catches regressions |
| `benchmarks/latency_check.py` | Load test | Measures the before/after |
| `Dockerfile` | Container recipe | Same environment everywhere |
| `render.yaml` | Render deployment config | Infrastructure as code |

## The four ideas worth knowing cold

### 1. Load the model once, at startup

Loading a model means reading files from disk and building it in memory. If you do that inside the request handler, **every** request pays that cost.

- **Where**: `lifespan()` in `app/main.py` runs once when the server starts and stores the model in `app.state`.
- **Proof**: the `baseline-naive` git tag loads per request. At 1 request at a time, predict p50 was 49.05 ms (naive) vs 6.48 ms (optimized).
- **Bonus**: a **warm-up** prediction at startup, because the very first TensorFlow call is slow (it builds its graph). Better the server pays that than the first user.

### 2. Don't block the event loop

FastAPI runs on one **event loop**: a single thread that juggles many requests by switching between them whenever one is *waiting* (on network, disk...). Model inference doesn't wait, it *computes*. If you compute directly in an `async def` handler, the loop is stuck and **no other request moves**, not even `/health`, which does nothing.

- **Fix**: `await run_in_threadpool(pipeline.predict, req)` moves the computation to a worker thread; the loop stays free.
- **Proof**: with 25 requests at once, `/health` p95 was **1226.93 ms** in the naive version vs **67.71 ms** optimized. `/health` does no work, so its slowness in the naive version is purely the blocked loop.

### 3. `tf.function`: the fix found by measuring

The first "optimized" version still got slow at 25 concurrent requests (predict p50 352.83 ms). Cause: calling a Keras model directly runs a lot of **Python** code per call, and Python's **GIL** (global interpreter lock) lets only one thread run Python at a time, so the worker threads queued up.

- **Fix**: wrap the forward pass in `tf.function`, which compiles it into a TensorFlow graph that runs mostly outside Python.
- **Proof**: one inference went from 1.98 ms to 0.31 ms; at 25 concurrent, predict p50 dropped to 84.77 ms and throughput rose from 61.62 to 183.37 req/s.
- **Interview value**: shows you measured instead of assuming the first fix worked.

### 4. Drift with PSI

A model is only right for data that looks like its training data. **PSI (Population Stability Index)** measures how far recent inputs have moved from training.

- Training values for each number (e.g. tenure) were split into 10 bins of equal size; `training_stats.json` stores the bin edges and the share in each bin (~10%).
- Recent requests are put into the same bins. If, say, 40% now fall in one bin that held 10%, the distribution has shifted.
- `PSI = Σ (actual% − expected%) × ln(actual% / expected%)`. Under 0.1 stable, 0.1–0.25 moderate, over 0.25 significant.
- Needs 50+ requests; fewer gives `insufficient_data` because small samples are noisy. Even at ~60 requests drawn from the *training* distribution, some features read "moderate" purely from sampling noise. More requests make it more reliable.
- It is a **basic indicator**: numeric features only, in memory, no alerts, and it checks inputs, not whether predictions were correct.

## Other decisions you may be asked about

- **Why fit the preprocessor on the train split only?** Fitting on all data leaks test-set information (its mean/std) into training, making test scores look better than reality.
- **Why save and reuse the fitted preprocessor?** If serving scaled data differently from training, the model would get numbers it never learned from, giving silently wrong predictions. Called *training/serving skew*.
- **Why pin exact library versions?** A `joblib` file saved by one scikit-learn version may not load correctly in another.
- **Why one worker?** Each uvicorn worker is a separate process with its own copy of the model. On a 512 MB instance, one is the safe choice; concurrency comes from the thread pool.
- **Why non-root in Docker?** If the app is compromised, the attacker can't modify system files.
- **Why is memory ~300 MB?** That is Python + TensorFlow + the model, loaded and ready. Live on Render: 301.0 MB idle, 304.2 MB peak under load, limit 512 MB. It can't be 0 while running.
- **Why does Render feel slower than local?** Two reasons: ~290 ms network round trip, and the free instance has 0.15 of a CPU.
- **Why does the page use `novalidate`?** So the API's Pydantic validation is the single source of truth and its 422 errors are visible.

## Likely questions, short answers

| Question | Answer |
|---|---|
| How accurate is it? | 0.7825 accuracy, 0.8554 ROC-AUC on its synthetic test set. On the independent IBM Telco benchmark (7,043 customers): ROC-AUC 0.82 vs 0.83 for a model trained on that data, accuracy 0.73–0.77 depending on the guessed ticket count. Ranks well. The cut-off was then tuned to 0.71 on half that data; on the other half accuracy rose 0.752 → 0.787. |
| What's ROC-AUC? | Chance the model ranks a random churner above a random non-churner. 0.5 = guessing, 1.0 = perfect. |
| What happens with bad input? | 422 with a consistent JSON body listing each problem; the model is never called. |
| What if the model crashes mid-request? | 500 with a generic JSON message; details go to server logs, not the client. |
| How do you know it works? | 12 pytest tests, including 20 concurrent requests and drift going from `insufficient_data` to `significant`. |
| What would you add for real production? | Authentication, rate limiting, a database for predictions, persistent monitoring with alerts, real data, model versioning/rollback, CI. |
| Why test on another dataset? | A test split from the same generator only shows the model learned the generator. Independent data shows whether the patterns transfer. |
| Why 0.71 and not 0.5? | 0.5 flagged 35% of customers when 26.5% churn. 0.71 maximised accuracy on held-out tuning data: flagged customers are right 66% of the time, but only 40% of churners are caught. A business choice: suits costly retention offers. |
| Why tune on one half and report on the other? | Picking a value and grading it on the same rows overstates how good it is. |
| Biggest limitation? | Synthetic data; in-memory state lost on restart; ~200 MB memory headroom on the free plan. |

## Try it yourself

- Live: https://churn-inference-api.onrender.com (page) and `/docs` (Swagger).
- Locally: `uvicorn app.main:app --port 8000`, open http://localhost:8000.
- See the naive version: `git checkout baseline-naive`, then `git checkout main` to return.
