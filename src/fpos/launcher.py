from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


HEALTH_URL = "http://127.0.0.1:8000/health"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = "8000"


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_path() -> Path:
    return _repo_root() / "launcher.py"


def _run_server() -> int:
    import uvicorn
    from .api import app

    uvicorn.run(
        app,
        host=BACKEND_HOST,
        port=int(BACKEND_PORT),
        log_level="info",
        access_log=False,
    )
    return 0


def wait_for_health(timeout_seconds: int = 90) -> None:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None

    while time.time() < deadline:
        try:
            resp = requests.get(HEALTH_URL, timeout=2)
            if resp.ok:
                return
            last_error = f"health returned {resp.status_code}"
        except Exception as exc:  # pragma: no cover - startup best effort
            last_error = str(exc)
        time.sleep(0.5)

    raise RuntimeError(f"FPOS backend did not become healthy in time: {last_error}")


def start_backend_subprocess() -> subprocess.Popen[str]:
    if _is_frozen():
        cmd = [sys.executable, "--serve"]
    else:
        cmd = [sys.executable, str(_script_path()), "--serve"]

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    return subprocess.Popen(cmd, creationflags=creationflags)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--serve", action="store_true")
    args, _ = parser.parse_known_args(argv)

    if args.serve:
        return _run_server()

    from .cli import main as client_main

    backend = start_backend_subprocess()
    try:
        wait_for_health()
        return client_main()
    finally:
        if backend.poll() is None:
            backend.terminate()
            try:
                backend.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend.kill()
                backend.wait(timeout=5)
