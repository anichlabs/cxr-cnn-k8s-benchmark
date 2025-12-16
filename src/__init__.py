"""
src package
-----------
Reusable components of the project pipeline.

Important for deployment:
- Do NOT import training-only dependencies (e.g. sklearn) at package import time.
- Deployment code should import what it needs directly, e.g.:
    from src.transforms import build_shared_transforms
"""
