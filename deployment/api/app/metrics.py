from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi import APIRouter, Response

router = APIRouter(tags=["metrics"])

@router.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
