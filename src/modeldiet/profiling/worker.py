"""Spawn-safe worker entry point for one isolated repeated profile."""

from __future__ import annotations

import traceback
from multiprocessing.connection import Connection
from typing import Any

from modeldiet.api import profile
from modeldiet.profiling.measurements import BaseMeasurement


def run_profile_worker(
    connection: Connection,
    model: Any,
    X: Any,
    measurements: tuple[BaseMeasurement, ...],
    on_error: str,
    include_environment: bool,
) -> None:
    """Execute a single profile and send a structured result to the parent."""
    try:
        result = profile(
            model,
            X,
            measurements=measurements,
            on_error=on_error,  # type: ignore[arg-type]
            include_environment=include_environment,
        )
        try:
            connection.send(("completed", result))
        except Exception as exc:
            connection.send(
                (
                    "failed",
                    type(exc).__name__,
                    f"Could not transfer the profile result to the parent process: {exc}",
                    traceback.format_exc(),
                )
            )
    except Exception as exc:
        try:
            connection.send(
                ("failed", type(exc).__name__, str(exc), traceback.format_exc())
            )
        except Exception:
            # The parent also watches the process exit code and reports EOF as
            # a worker failure if even this fallback cannot cross the pipe.
            pass
    finally:
        connection.close()
