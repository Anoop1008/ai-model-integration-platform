"""FastAPI app: validation -> preprocessing -> TF inference -> JSON response."""

import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.drift import compute_drift
from app.pipeline import InferencePipeline
from app.schemas import ChurnRequest, ChurnResponse

logger = logging.getLogger("churn-api")

RECENT_SIZE = 10
DRIFT_WINDOW = 1000
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE per process. Loading per request would add seconds of
    # latency to every call and repeatedly allocate memory.
    pipeline = InferencePipeline.load()
    pipeline.warm_up()
    app.state.pipeline = pipeline
    app.state.started_at = time.time()
    # In-memory ring buffers: oldest entries fall off automatically. Lost on restart.
    app.state.recent = deque(maxlen=RECENT_SIZE)
    app.state.drift_window = deque(maxlen=DRIFT_WINDOW)
    yield


app = FastAPI(title="Churn Inference API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request body failed validation",
            "details": jsonable_encoder(exc.errors()),
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    # Do not leak internals (stack traces, paths) to clients.
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "Internal server error", "details": None},
    )


def _read_mb(path: str, divisor: int) -> float | None:
    try:
        return round(int(Path(path).read_text().split()[0]) / divisor, 1)
    except (OSError, ValueError, IndexError):
        return None


def memory_mb() -> dict | None:
    """Container memory, read from Linux cgroup v2 and /proc. None off Linux (e.g. Windows dev).

    Render's free plan hides memory metrics, so the app reports its own. The 512 MB limit
    is enforced on the cgroup (cgroup_mb / cgroup_peak_mb), not on process RSS: RSS also
    counts TensorFlow's shared libraries mapped from disk (~340 MB), which are reclaimable.
    rss_anon_mb is the process's own heap, a fallback if cgroup files are not visible.
    Caveat: cgroup values can include reclaimable file cache (seen once at ~1140 MB on the
    first run of a fresh image locally); rss_anon_mb is the steadier number.
    """
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return None
    anon = next((ln.split(":", 1)[1] for ln in lines if ln.startswith("RssAnon:")), None)
    return {
        "cgroup_mb": _read_mb("/sys/fs/cgroup/memory.current", 1024 * 1024),
        "cgroup_peak_mb": _read_mb("/sys/fs/cgroup/memory.peak", 1024 * 1024),
        "rss_anon_mb": round(int(anon.split()[0]) / 1024, 1) if anon else None,
    }


@app.get("/", include_in_schema=False)
async def frontend():
    # Single static page served by the API itself: same origin, so no CORS setup.
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    return {
        "status": "ok" if pipeline else "starting",
        "model_loaded": pipeline is not None,
        "model_version": pipeline.model_version if pipeline else None,
        "decision_threshold": pipeline.threshold if pipeline else None,
        "uptime_seconds": round(time.time() - request.app.state.started_at, 2),
        "memory": memory_mb(),
    }


@app.post("/predict/churn", response_model=ChurnResponse)
async def predict_churn(req: ChurnRequest, request: Request) -> ChurnResponse:
    pipeline: InferencePipeline = request.app.state.pipeline
    # TF inference is CPU-bound and synchronous. Running it directly in this async
    # handler would block the event loop, stalling every other request (even
    # /health). run_in_threadpool moves it to a worker thread.
    result = await run_in_threadpool(pipeline.predict, req)
    request.app.state.recent.append({"input": req.model_dump(), "output": result.model_dump()})
    request.app.state.drift_window.append(req.model_dump())
    return result


@app.get("/metrics/recent")
async def metrics_recent(request: Request):
    recent = list(request.app.state.recent)
    return {"count": len(recent), "predictions": recent}


@app.get("/metrics/drift")
async def metrics_drift(request: Request):
    return compute_drift(list(request.app.state.drift_window), request.app.state.pipeline.training_stats)
