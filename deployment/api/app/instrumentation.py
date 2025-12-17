from prometheus_client import Counter, Histogram

# Total HTTP requests by route + method + status
HTTP_REQUESTS_TOTAL = Counter(
    "cxr_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

# Request latency by route + method
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "cxr_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

# Model inference latency
INFERENCE_DURATION_SECONDS = Histogram(
    "cxr_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_type"],
)

# Total predictions
PREDICTIONS_TOTAL = Counter(
    "cxr_predictions_total",
    "Total predictions",
    ["model_type"],
)
