"""
metrics.py
----------
Prometheus metrics endpoint for the FastAPI service.

Goal:
Expose a /metrics endpoint that Prometheus can scrape.

Important:
This file only defines the endpoints and the metric objects.
In the next step, middleware will be wired in main.py to actually
increment counters and record latency for every request.
"""

# FastAPI routers lets keep endpoints modular (cleaner than main.py growing forever).
from fastapi import APIRouter

# Response is used because Prometheus expects plian text, not JSON.
from fastapi.responses import Response

# prometheus_client is the official Python client for Prometheus metrics.
# generate_latest renders all registered metrics into the text format Prometheus expects.
# CONTENT_TYPE_LATEST is the correct Content-Type header for Prometheus exposition.

# Counter counts things that only go up (requests, errors).
# Histogram tracks a ditribution (request duration, payload sizes).
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


# Router object that will be included in the main FastAPI app.
router = APIRouter()


# -----------------------
# Metric objects (global)
# -----------------------
# These are registered globally inside prometheus_client.
# Prometheus scrapes them by calling /metrics.
#
# These will be increment / observed from middleware in the next step.


HTTP_REQUESTS_TOTAL = Counter(
    name="cxr_http_requests_total",
    documentation="Total number of HTTP requests handled by the CXR API.",
    labelnames=["method", "path", "status_code"],
)


@router.get("/metrics")
def metrics():
    """
    Prometheus scrape endpoint.

    Returns:
    - Plain text in Prometheus exposition format.
    - Correct Content-Type, so Prometheus parses it properly.
    """
    payload = generate_latest() # bytes
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)