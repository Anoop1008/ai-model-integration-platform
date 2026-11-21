"""FastAPI app: validation -> preprocessing -> TF inference -> JSON response."""

import json
import logging
import mimetypes
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.drift import compute_drift
from app.pipeline import InferencePipeline
from app.schemas import ChurnRequest, ChurnResponse
from app.security import RateLimiter, page_csp, rate_limit, require_api_key
from app.store import PredictionStore

logger = logging.getLogger("churn-api")

RECENT_SIZE = 10
DRIFT_WINDOW = 1000
STATIC_DIR = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parent.parent
HTTP_ERRORS = {401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed", 429: "rate_limited"}
PAGES = ("/", "/predict", "/recent", "/drift", "/performance", "/quality")
RESULT_FILES = {
    "benchmark_baseline": "benchmarks/results/baseline-naive.json",
    "benchmark_optimized": "benchmarks/results/optimized.json",
    "benchmark_render": "benchmarks/results/render-free.json",
    "benchmark_render_high_load": "benchmarks/results/render-free-high-load.json",
    "threshold_calibration": "evaluation/results/threshold_calibration.json",
    "real_world": "evaluation/results/real_world.json",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE per process. Loading per request would add seconds of
    # latency to every call and repeatedly allocate memory.
    pipeline = InferencePipeline.load()
    pipeline.warm_up()
    app.state.pipeline = pipeline
    app.state.started_at = time.time()
    # Configuration comes from environment variables, so secrets never live in the code or the repo.
    app.state.store = PredictionStore(os.environ.get("DB_PATH", ROOT / "data" / "predictions.db"))
    app.state.api_key = os.environ.get("API_KEY") or None  # unset = auth off
    app.state.limiter = RateLimiter(int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60")))
    # Header a trusted proxy sets to the visitor's IP (Render: cf-connecting-ip). Unset = use the TCP peer.
    app.state.client_ip_header = os.environ.get("CLIENT_IP_HEADER", "").strip().lower() or None
    # Who may show the page in a frame (CSP frame-ancestors): nobody by default.
    app.state.csp = page_csp(os.environ.get("FRAME_ANCESTORS", "'none'"))
    yield
    app.state.store.close()


app = FastAPI(title="Churn Inference API", version="1.0.0", lifespan=lifespan)
# Self-hosted fonts: the page makes no third-party requests. Windows and slim images lack a woff2 MIME entry.
mimetypes.add_type("font/woff2", ".woff2")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SecurityHeaders:
    """Adds headers to every response. Plain ASGI middleware on purpose: @app.middleware("http")
    wraps every response in extra tasks and measurably slowed predictions under load."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")  # browsers must not guess content types
                headers.setdefault("Referrer-Policy", "no-referrer")
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(SecurityHeaders)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    # Same JSON shape as 422 and 500, so clients handle every error one way.
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": HTTP_ERRORS.get(exc.status_code, "http_error"), "message": str(exc.detail), "details": None},
        headers=getattr(exc, "headers", None),
    )


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


def _windows_process_mb() -> tuple[float, float]:
    """(working set, peak working set) of this process in MB, via the Win32 API (no psutil needed)."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                         "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
        ]

    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    # Explicit types: the process handle is pointer-sized; ctypes' default int would truncate it on 64-bit.
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    c = Counters(cb=ctypes.sizeof(Counters))
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return round(c.WorkingSetSize / 2**20, 1), round(c.PeakWorkingSetSize / 2**20, 1)


def memory_mb() -> dict | None:
    """Memory in MB. Container (cgroup) numbers on Linux; process numbers on Linux and Windows.

    Render's free plan hides memory metrics, so the app reports its own. The 512 MB limit
    is enforced on the cgroup (cgroup_mb / cgroup_peak_mb), not on process RSS: RSS also
    counts TensorFlow's shared libraries mapped from disk (~340 MB), which are reclaimable.
    process_mb / process_peak_mb (RSS, or the working set on Windows) are there so local
    runs outside a container still show a real number; they are not the Render limit.
    rss_anon_mb is the process's own heap. Caveat: cgroup values can include reclaimable
    file cache (seen once at ~1140 MB on the first run of a fresh image locally).
    """
    if sys.platform == "win32":
        try:
            process, peak = _windows_process_mb()
        except OSError:
            return None
        return {"cgroup_mb": None, "cgroup_peak_mb": None, "rss_anon_mb": None, "process_mb": process, "process_peak_mb": peak}
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return None  # shortcut: macOS has no /proc; add a branch if anyone develops there.
    status = dict(ln.split(":", 1) for ln in lines if ":" in ln)

    def kb_to_mb(key: str) -> float | None:
        return round(int(status[key].split()[0]) / 1024, 1) if key in status else None

    return {
        "cgroup_mb": _read_mb("/sys/fs/cgroup/memory.current", 1024 * 1024),
        "cgroup_peak_mb": _read_mb("/sys/fs/cgroup/memory.peak", 1024 * 1024),
        "rss_anon_mb": kb_to_mb("RssAnon"),
        "process_mb": kb_to_mb("VmRSS"),
        "process_peak_mb": kb_to_mb("VmHWM"),
    }


async def frontend(request: Request):
    # Single static page served by the API itself: same origin, so no CORS setup.
    # Every page route returns it; the page's script shows the section for the URL.
    return FileResponse(STATIC_DIR / "index.html", headers={"Content-Security-Policy": request.app.state.csp})


for _page in PAGES:
    app.add_api_route(_page, frontend, methods=["GET"], include_in_schema=False)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None  # a missing result file shows as "not available" on the page, not a 500


@app.get("/metrics/evaluation")
async def metrics_evaluation(request: Request):
    """Measured results behind the Performance and Model quality pages, read from the committed
    result files, so the page shows recorded measurements instead of numbers typed into HTML."""
    stats = request.app.state.pipeline.training_stats
    training = {k: stats.get(k) for k in ("model_version", "n_train_rows", "test_accuracy", "test_roc_auc", "decision_threshold")}
    return {"training": training, **{name: _read_json(ROOT / path) for name, path in RESULT_FILES.items()}}


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
        "rate_limit": {"per_minute": request.app.state.limiter.limit, "client_ip_header": request.app.state.client_ip_header},
    }


# Rate limit runs before the key check, so it also slows down anyone guessing keys.
@app.post("/predict/churn", response_model=ChurnResponse, dependencies=[Depends(rate_limit), Depends(require_api_key)])
async def predict_churn(req: ChurnRequest, request: Request) -> ChurnResponse:
    pipeline: InferencePipeline = request.app.state.pipeline
    # TF inference is CPU-bound and synchronous. Running it directly in this async
    # handler would block the event loop, stalling every other request (even
    # /health). run_in_threadpool moves it to a worker thread.
    # The SQLite insert blocks too; doing it in the same worker-thread call saves a second hop.
    return await run_in_threadpool(_predict_and_log, pipeline, request.app.state.store, req)


def _predict_and_log(pipeline: InferencePipeline, store: PredictionStore, req: ChurnRequest) -> ChurnResponse:
    result = pipeline.predict(req)
    store.save(req, result)
    return result


@app.get("/metrics/recent", dependencies=[Depends(require_api_key)])
async def metrics_recent(request: Request):
    recent = await run_in_threadpool(request.app.state.store.recent, RECENT_SIZE)
    return {"count": len(recent), "predictions": recent}


@app.get("/metrics/drift", dependencies=[Depends(require_api_key)])
async def metrics_drift(request: Request):
    rows = await run_in_threadpool(request.app.state.store.recent_inputs, DRIFT_WINDOW)
    return compute_drift(rows, request.app.state.pipeline.training_stats)


@app.get("/metrics/summary", dependencies=[Depends(require_api_key)])
async def metrics_summary(request: Request):
    """SQL aggregates over every stored prediction: totals, churn rate, by risk level and by contract."""
    return await run_in_threadpool(request.app.state.store.summary)
