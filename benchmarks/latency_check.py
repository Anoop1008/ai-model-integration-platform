"""Latency check against a running server.

For each concurrency level, fires N /predict/churn requests with at most C in flight,
while a background task polls /health. Reports p50/p95 and error count for both, and
saves JSON to benchmarks/results/<label>.json.

/health latency under load is the key signal: if inference blocks the event loop,
/health (which does no work) gets slow too.

Usage: python benchmarks/latency_check.py --label optimized [--url http://localhost:8000]
"""

import argparse
import asyncio
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

PAYLOAD = {
    "tenure_months": 5,
    "monthly_charges": 95.5,
    "total_charges": 477.5,
    "support_tickets": 4,
    "contract_type": "month-to-month",
}
LEVELS = [1, 10, 25]
RESULTS_DIR = Path(__file__).parent / "results"


def summarize(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"count": 0, "p50_ms": None, "p95_ms": None}
    arr = np.array(latencies_ms)
    return {
        "count": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }


async def timed(client: httpx.AsyncClient, method: str, path: str, **kw) -> tuple[float, bool]:
    start = time.perf_counter()
    try:
        r = await client.request(method, path, **kw)
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    return (time.perf_counter() - start) * 1000, ok


async def run_level(client: httpx.AsyncClient, concurrency: int, n_requests: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    predict_lat, errors = [], 0
    health_lat, health_errors = [], 0
    done = asyncio.Event()

    async def one_predict():
        nonlocal errors
        async with sem:
            ms, ok = await timed(client, "POST", "/predict/churn", json=PAYLOAD)
        if ok:
            predict_lat.append(ms)
        else:
            errors += 1

    async def poll_health():
        nonlocal health_errors
        while not done.is_set():
            ms, ok = await timed(client, "GET", "/health")
            if ok:
                health_lat.append(ms)
            else:
                health_errors += 1
            await asyncio.sleep(0.05)

    poller = asyncio.create_task(poll_health())
    start = time.perf_counter()
    await asyncio.gather(*(one_predict() for _ in range(n_requests)))
    wall = time.perf_counter() - start
    done.set()
    await poller

    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(n_requests / wall, 2),
        "predict": {**summarize(predict_lat), "errors": errors},
        "health_during_load": {**summarize(health_lat), "errors": health_errors},
    }


async def main(url: str, label: str, n_requests: int) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=120) as client:
        (await client.get("/health")).raise_for_status()
        results = []
        for c in LEVELS:
            res = await run_level(client, c, n_requests)
            results.append(res)
            p, h = res["predict"], res["health_during_load"]
            print(
                f"[{label}] c={c:>2} predict p50={p['p50_ms']}ms p95={p['p95_ms']}ms "
                f"errors={p['errors']} | health p50={h['p50_ms']}ms p95={h['p95_ms']}ms "
                f"| {res['throughput_rps']} req/s"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{label}.json"
    out.write_text(
        json.dumps(
            {
                "label": label,
                "url": url,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "machine": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
                "results": results,
            },
            indent=2,
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--requests", type=int, default=50, help="requests per concurrency level")
    args = ap.parse_args()
    asyncio.run(main(args.url, args.label, args.requests))
