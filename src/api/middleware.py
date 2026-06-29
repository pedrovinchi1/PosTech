import logging
import time
from fastapi import Request

logger = logging.getLogger(__name__)


async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request",
        extra={
            "path": request.url.path,
            "method": request.method,
            "latency_ms": round(latency_ms, 2),
            "status_code": response.status_code,
        },
    )
    return response
