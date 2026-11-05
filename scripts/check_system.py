"""Report hardware relevant to running local LLMs."""

import platform
import shutil
import subprocess

import psutil

GB = 1024**3


def _powershell(command: str) -> str:
    """Run a PowerShell command and return stripped stdout ('' on failure)."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def get_cpu_info() -> dict:
    name = _powershell("(Get-CimInstance Win32_Processor).Name") or platform.processor()
    return {
        "name": name,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
    }


def get_ram_info() -> dict:
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / GB, 1),
        "available_gb": round(mem.available / GB, 1),
    }


def get_disk_info(path: str = "C:\\") -> dict:
    usage = shutil.disk_usage(path)
    return {"total_gb": round(usage.total / GB, 1), "free_gb": round(usage.free / GB, 1)}


def get_gpu_info() -> list[dict]:
    """Query NVIDIA GPUs via nvidia-smi; fall back to listing adapter names."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        gpus = []
        for line in out.splitlines():
            name, total, used, driver = [p.strip() for p in line.split(",")]
            gpus.append({"name": name, "vram_total_mb": int(total),
                         "vram_used_mb": int(used), "driver": driver})
        return gpus
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        names = _powershell("(Get-CimInstance Win32_VideoController).Name")
        return [{"name": n, "vram_total_mb": None, "note": "Check VRAM in Task Manager"}
                for n in names.splitlines() if n]


def main() -> None:
    print("=== LocalMind System Check ===")
    print(f"OS     : {platform.system()} {platform.release()}")
    print(f"Python : {platform.python_version()}")
    print(f"CPU    : {get_cpu_info()}")
    print(f"RAM    : {get_ram_info()}")
    print(f"Disk   : {get_disk_info()}")
    gpus = get_gpu_info()
    if not gpus:
        print("GPU    : none detected")
    for gpu in gpus:
        print(f"GPU    : {gpu}")


if __name__ == "__main__":
    main()