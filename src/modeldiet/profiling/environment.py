import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import psutil

from modeldiet.profiling.measurements import BaseMeasurement


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect host and library metadata relevant to benchmark comparison."""
    cpu_frequency = psutil.cpu_freq()
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine_architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cpu_frequency_mhz": cpu_frequency.current if cpu_frequency else None,
        "numpy_version": np.__version__,
        "pandas_version": _package_version("pandas"),
        "sklearn_version": _package_version("scikit-learn"),
        "xgboost_version": _package_version("xgboost"),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


class EnvironmentMeasurement(BaseMeasurement):
    """Measurement strategy that captures the runtime environment metadata."""

    def measure(self, model: Any = None, X: Any = None) -> dict[str, Any]:
        del model, X
        return collect_environment()
