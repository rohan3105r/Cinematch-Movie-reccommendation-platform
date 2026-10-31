"""Start CineMatch on an available local port and open it in a browser."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PORT_RANGE = range(8501, 8511)
STARTUP_TIMEOUT_SECONDS = 60


def find_available_port() -> int:
    for port in PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"Ports {PORT_RANGE.start}-{PORT_RANGE.stop - 1} are all in use. "
        "Close an existing CineMatch/Streamlit window and try again."
    )


def wait_until_ready(process: subprocess.Popen, url: str) -> bool:
    health_url = f"{url}/_stcore/health"
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200 and response.read().strip() == b"ok":
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    return False


def main() -> int:
    try:
        port = find_available_port()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        f"--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    print(f"Starting CineMatch at {url}", flush=True)

    try:
        process = subprocess.Popen(command, cwd=PROJECT_DIR)
    except OSError as exc:
        print(f"ERROR: Could not start Streamlit: {exc}", file=sys.stderr)
        return 1

    try:
        if not wait_until_ready(process, url):
            if process.poll() is None:
                print(
                    f"ERROR: CineMatch did not become ready within "
                    f"{STARTUP_TIMEOUT_SECONDS} seconds.",
                    file=sys.stderr,
                )
                process.terminate()
                process.wait(timeout=10)
            else:
                print(
                    f"ERROR: Streamlit exited with code {process.returncode}.",
                    file=sys.stderr,
                )
            return process.returncode or 1

        print(f"CineMatch is ready: {url}", flush=True)
        if not webbrowser.open(url):
            print("Open this address manually in your browser:", flush=True)
            print(url, flush=True)
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping CineMatch...")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return 0
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: CineMatch could not be monitored or stopped cleanly: {exc}", file=sys.stderr)
        if process.poll() is None:
            process.terminate()
            process.wait()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
