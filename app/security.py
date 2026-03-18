import os
import time
import asyncio
import logging
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, Security #type: ignore
from fastapi.security import APIKeyHeader #type: ignore

logger = logging.getLogger(__name__)


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_API_KEY: Optional[str] = os.getenv("API_KEY")


def get_api_key() -> Optional[str]:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.getenv("API_KEY")
    return _API_KEY


async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """
    Validate the X-API-Key header.

    - If API_KEY env var is not set, the app refuses to start in a way that
      makes the misconfiguration obvious (see lifespan in main.py).
    - In practice this dependency always has a key to check against.
    """
    expected = get_api_key()

    if not expected:
        logger.error(
            "API_KEY is not configured. "
            "Set the API_KEY environment variable before deploying."
        )
        raise HTTPException(
            status_code=503,
            detail="Server misconfiguration: authentication is not set up.",
        )

    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="Missing API key. Include an 'X-API-Key' header.",
        )

    if api_key != expected:
        logger.warning("Request rejected: invalid API key")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


class RateLimiter:
    """
    In-memory sliding window rate limiter keyed by client IP.

    Not suitable for multi-process deployments (use a Redis-backed limiter
    for horizontal scaling). Fine for a single-process deployment (Render free tier).
    """

    TRUST_PROXY: bool = os.getenv("TRUST_PROXY", "false").lower() == "true"

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP.
        """
        if self.TRUST_PROXY:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                # Take the leftmost IP (original client), not the last hop.
                return forwarded.split(",")[0].strip()

        return request.client.host if request.client else "unknown"

    def _cleanup(self, ip: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._requests[ip] = [ts for ts in self._requests[ip] if ts > cutoff]
        if not self._requests[ip]:
            del self._requests[ip]

    def check(self, request: Request) -> dict:
        now = time.time()
        ip = self._get_client_ip(request)

        self._cleanup(ip, now)

        current_count = len(self._requests.get(ip, []))

        if current_count >= self.max_requests:
            oldest = self._requests[ip][0]
            retry_after = int(self.window_seconds - (now - oldest)) + 1

            logger.warning(
                f"Rate limit exceeded: {current_count}/{self.max_requests} "
                f"requests in window"
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded. Max {self.max_requests} requests "
                    f"per {self.window_seconds}s. Retry after {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        self._requests.setdefault(ip, []).append(now)

        return {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(self.max_requests - current_count - 1),
            "X-RateLimit-Reset": str(int(now + self.window_seconds)),
        }


rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_MAX_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)


async def check_rate_limit(request: Request) -> dict:
    """FastAPI dependency that enforces per-IP rate limiting."""
    return rate_limiter.check(request)

MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "2"))

_inference_semaphore: Optional[asyncio.Semaphore] = None


def get_inference_semaphore() -> asyncio.Semaphore:
    """
    Lazy-initialize the semaphore inside the running event loop.

    Limits simultaneous model inferences to prevent CPU saturation on
    resource-constrained environments.
    """
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
        logger.info(
            f"Inference semaphore initialized (max={MAX_CONCURRENT_INFERENCES})"
        )
    return _inference_semaphore


INFERENCE_SLOT_TIMEOUT = float(os.getenv("INFERENCE_SLOT_TIMEOUT", "10"))
"""
How long (seconds) a request will wait for a free inference slot before
returning 503. Default is 10s, which is generous enough for CPU inference
on a local machine without letting requests pile up indefinitely.

Set to 0 in the environment to get the original fail-fast behavior.
"""


async def acquire_inference_slot() -> None:

    sem = get_inference_semaphore()
    timeout = INFERENCE_SLOT_TIMEOUT if INFERENCE_SLOT_TIMEOUT > 0 else None
    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            f"All {MAX_CONCURRENT_INFERENCES} inference slots occupied after "
            f"{INFERENCE_SLOT_TIMEOUT}s — request rejected"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Server busy. All {MAX_CONCURRENT_INFERENCES} inference slots "
                f"are in use. Please retry shortly."
            ),
            headers={"Retry-After": "5"},
        )


def release_inference_slot() -> None:
    """Release an inference slot back to the pool."""
    get_inference_semaphore().release()

MAX_FRAMES_PER_REQUEST = int(os.getenv("MAX_FRAMES_PER_REQUEST", "512"))
EXPECTED_FEATURE_DIM = 858