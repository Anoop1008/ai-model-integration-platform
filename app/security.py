"""API-key authentication, per-client rate limiting and security headers."""

import hmac
import math
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, Response, Security
from fastapi.security import APIKeyHeader

# auto_error=False: the dependency decides, so auth can be switched off when no key is configured.
# Declaring it with Security() also adds an "Authorize" button to Swagger UI at /docs.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, key: str | None = Security(api_key_header)) -> None:
    expected = request.app.state.api_key
    if not expected:
        return  # API_KEY not set: auth is off (public demo)
    # compare_digest takes the same time wherever the strings differ, so the key cannot be guessed byte by byte.
    if key is None or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key (header X-API-Key)", headers={"WWW-Authenticate": "API-Key"})


class RateLimiter:
    """Sliding window: at most `limit` requests per `window` seconds per client. limit 0 = off.

    shortcut: counts live in this process's memory, which is right for one worker on one
    instance; with several workers or instances keep the counts in Redis instead.
    """

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str) -> tuple[int, int]:
        """(retry_after, remaining): retry_after is 0 if the request may go ahead, else the
        seconds to wait; remaining is how many more requests fit in the current window."""
        if self.limit <= 0:
            return 0, -1
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window - now)), 0
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle clients so the dict cannot grow without bound
                for k in [k for k, q in self._hits.items() if not q or q[-1] <= now - self.window]:
                    del self._hits[k]
            return 0, self.limit - len(hits)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def client_ip(request: Request) -> str:
    """Who the rate limit counts. X-Forwarded-For is never trusted: clients can write anything
    into it, and trusting it let a burst bypass the limit on Render by faking the header.
    Behind a proxy that sets its own header (Render sits behind Cloudflare, which overwrites
    CF-Connecting-IP), name that header in CLIENT_IP_HEADER. Only do that when the app is
    reachable solely through that proxy; otherwise clients could fake that header too."""
    header = request.app.state.client_ip_header
    value = request.headers.get(header, "").split(",")[0].strip() if header else ""
    return value or (request.client.host if request.client else "unknown")


def rate_limit(request: Request, response: Response) -> None:
    limiter = request.app.state.limiter
    retry_after, remaining = limiter.check(client_ip(request))
    if retry_after:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded; retry in {retry_after} s",
                            headers={"Retry-After": str(retry_after), "X-RateLimit-Limit": str(limiter.limit), "X-RateLimit-Remaining": "0"})
    if limiter.limit > 0:  # standard headers: clients can see their budget before they hit 429
        response.headers["X-RateLimit-Limit"] = str(limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)


def page_csp(frame_ancestors: str) -> str:
    """Content-Security-Policy for the web page: scripts, fonts and API calls only from this origin."""
    return "; ".join([
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",  # the page uses a few inline style attributes
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        f"frame-ancestors {frame_ancestors}",
    ])
