"""Small system snapshot recorded with every benchmark run."""

import platform

import psutil

GB = 1024**3


def snapshot() -> dict:
    mem = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / GB, 1),
        "ram_available_gb": round(mem.available / GB, 1),
    }