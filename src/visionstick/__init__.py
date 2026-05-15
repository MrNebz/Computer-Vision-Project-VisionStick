"""VisionStick V2 package."""

from .pipeline import (
    pipeline_close as pipeline_close,
    pipeline_create as pipeline_create,
    pipeline_open as pipeline_open,
    pipeline_process_frame as pipeline_process_frame,
    pipeline_run as pipeline_run,
)

__version__ = "2.0.0"

__all__ = [
    "pipeline_create",
    "pipeline_run",
    "pipeline_open",
    "pipeline_close",
    "pipeline_process_frame",
]
