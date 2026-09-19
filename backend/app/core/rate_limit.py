"""
rate_limit.py — per-client throttling for the prediction endpoint.

/api/predict is public, unauthenticated, and costs roughly 25 seconds of CPU on
the free instance this runs on. One person with a loop could hold the service
down for everyone, and the URL is going on a conference slide.

Two separate budgets, because the two costs are different:

  * requests   — every call, cached or not. Cheap, but not free, and a flood of
                 them still saturates a single worker.
  * computes   — calls that actually run the model. This is the real resource.

A fixed window per client is enough here. There is one worker and one process,
so the counters are exact; nothing needs a shared store. Windows are kept in a
plain dict and pruned as they expire, so memory stays proportional to the number
of clients seen in the last hour rather than growing without bound.
"""

import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request

# (max events, window in seconds)
REQUEST_LIMIT = (30, 60)        # 30 calls a minute, cached or not
COMPUTE_LIMIT = (6, 60)         # 6 model runs a minute
COMPUTE_LIMIT_HOURLY = (40, 3600)

_PRUNE_AFTER = 3600.0


@dataclass
class _Windows:
    counts: dict[str, tuple[float, int]] = field(default_factory=dict)
    last_seen: float = 0.0


_clients: dict[str, _Windows] = {}
_lock = threading.Lock()


def client_key(request: Request) -> str:
    """
    Identify the caller. Render sits behind Cloudflare, so the socket address is
    a proxy; the forwarded headers carry the real client. They are trivially
    spoofable, which is fine: this limits accidental and casual abuse, and is not
    an authentication boundary.
    """
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _prune(now: float) -> None:
    stale = [k for k, w in _clients.items() if now - w.last_seen > _PRUNE_AFTER]
    for k in stale:
        del _clients[k]


def _check(key: str, bucket: str, limit: tuple[int, int], consume: bool) -> float:
    """
    Return 0.0 if the call is allowed, otherwise the seconds until the window
    resets. With consume=False the counter is only read, never incremented.
    """
    max_events, window = limit
    now = time.monotonic()
    with _lock:
        if len(_clients) > 512:
            _prune(now)
        w = _clients.setdefault(key, _Windows())
        w.last_seen = now
        start, count = w.counts.get(bucket, (now, 0))
        if now - start >= window:
            start, count = now, 0
        if count >= max_events:
            return window - (now - start)
        if consume:
            w.counts[bucket] = (start, count + 1)
        else:
            w.counts[bucket] = (start, count)
    return 0.0


def _deny(retry_after: float, what: str) -> None:
    wait = max(1, int(round(retry_after)))
    raise HTTPException(
        status_code=429,
        detail=f"Too many {what}. Try again in {wait} second{'s' if wait != 1 else ''}.",
        headers={"Retry-After": str(wait)},
    )


def check_request(request: Request) -> str:
    """Charge one request against the caller's request budget. Returns their key."""
    key = client_key(request)
    retry = _check(key, "req", REQUEST_LIMIT, consume=True)
    if retry:
        _deny(retry, "requests")
    return key


def check_compute(key: str) -> None:
    """
    Charge one model run. Called only when the scenario is not already cached,
    so repeating a scenario someone else has run never uses this budget.
    """
    for bucket, limit in (("cpu_m", COMPUTE_LIMIT), ("cpu_h", COMPUTE_LIMIT_HOURLY)):
        retry = _check(key, bucket, limit, consume=True)
        if retry:
            _deny(retry, "new scenarios")


def reset() -> None:
    """Test helper."""
    with _lock:
        _clients.clear()
