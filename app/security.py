import os
import time
import asyncio
import logging
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_API_KEY: Optional[str] = os.getenv("API_KEY")


def get_api_key() -> Optional[str]:
    """Return the configured API key (re-reads env on first call if needed)."""
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.getenv("API_KEY")
    return _API_KEY


async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    - If API_KEY env var is not set, authentication is DISABLED (dev mode).
    - If set, every request must include a matching X-API-Key header.

    Returns the validated key (or "dev-mode" when auth is disabled).
    """
    expected = get_api_key()

    if not expected:
        logger.debug("API key not configured - authentication disabled (dev mode)")
        return "dev-mode"

    if not api_key:
        logger.warning("Request missing X-API-Key header")
        raise HTTPException(
            status_code=403,
            detail="Missing API key. Include 'X-API-Key' header.",
        )

    if api_key != expected:
        logger.warning("Invalid API key received")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


# Configuration from environment
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


class RateLimiter:
    """
    In-memory sliding window rate limiter keyed by client IP.

    Not suitable for multi-process deployments (use Redis-backed
    alternative for horizontal scaling). Fine for single-process
    Render free tier.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # {ip: [timestamp, timestamp, ...]}
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For behind proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup(self, ip: str, now: float) -> None:
        """Remove timestamps outside the current window."""
        cutoff = now - self.window_seconds
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > cutoff
        ]
        if not self._requests[ip]:
            del self._requests[ip]

    def check(self, request: Request) -> dict:
        """
        Check rate limit for the given request.

        Returns a dict with rate-limit headers info.
        Raises HTTPException 429 if limit exceeded.
        """
        now = time.time()
        ip = self._get_client_ip(request)

        self._cleanup(ip, now)

        current_count = len(self._requests.get(ip, []))

        if current_count >= self.max_requests:
            oldest = self._requests[ip][0]
            retry_after = int(self.window_seconds - (now - oldest)) + 1

            logger.warning(
                f"Rate limit exceeded for IP {ip}: "
                f"{current_count}/{self.max_requests} requests"
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded. Maximum {self.max_requests} requests "
                    f"per {self.window_seconds} seconds. "
                    f"Retry after {retry_after} seconds."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        self._requests.setdefault(ip, []).append(now)

        return {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(self.max_requests - current_count - 1),
            "X-RateLimit-Reset": str(
                int(now + self.window_seconds)
            ),
        }

rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_MAX_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)


async def check_rate_limit(request: Request) -> dict:
    """FastAPI dependency that enforces rate limiting."""
    return rate_limiter.check(request)

MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "2"))

_inference_semaphore: Optional[asyncio.Semaphore] = None


def get_inference_semaphore() -> asyncio.Semaphore:
    """
    Lazy-initialize the semaphore (must happen inside the event loop).

    Limits the number of simultaneous model inferences to prevent CPU
    saturation on resource-constrained environments (e.g. Render free tier).
    """
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
        logger.info(
            f"Inference semaphore initialized: max {MAX_CONCURRENT_INFERENCES} "
            f"concurrent inferences"
        )
    return _inference_semaphore


async def acquire_inference_slot() -> None:
    """
    Try to acquire an inference slot without blocking indefinitely.

    If all slots are occupied, immediately returns 503 instead of
    queueing the request (fail-fast to avoid cascading timeouts).
    """
    sem = get_inference_semaphore()

    acquired = sem._value > 0
    if not acquired:
        logger.warning(
            f"All {MAX_CONCURRENT_INFERENCES} inference slots occupied. "
            f"Rejecting request."
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Server busy. All {MAX_CONCURRENT_INFERENCES} inference slots "
                f"are in use. Please retry shortly."
            ),
            headers={"Retry-After": "5"},
        )

    await sem.acquire()


def release_inference_slot() -> None:
    """Release an inference slot back to the pool."""
    sem = get_inference_semaphore()
    sem.release()

MAX_FRAMES_PER_REQUEST = int(os.getenv("MAX_FRAMES_PER_REQUEST", "512"))
EXPECTED_FEATURE_DIM = 858
