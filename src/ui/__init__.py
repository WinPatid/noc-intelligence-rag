"""
NOC Intelligence — Stage 9 Streamlit UI Package
"""

from .presentation_adapter import (
    PresentationAdapter,
    SafePresentationView,
    SafeCitationView,
    SAFE_FALLBACK_TH,
    SAFE_FALLBACK_EN,
)

__all__ = [
    "PresentationAdapter",
    "SafePresentationView",
    "SafeCitationView",
    "SAFE_FALLBACK_TH",
    "SAFE_FALLBACK_EN",
]
