"""Model Diet public API."""

from modeldiet.api import RepeatedProfilingError, profile, profile_repeated
from modeldiet.result import ProfileResult, RepeatedProfileResult

__all__ = [
    "profile",
    "profile_repeated",
    "RepeatedProfilingError",
    "ProfileResult",
    "RepeatedProfileResult",
]
