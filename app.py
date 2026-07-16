from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser

from dotenv import load_dotenv
from waitress import serve

from backend.api import create_app

load_dotenv()


def run_api() -> None:
    port = int(os.getenv("GRIDGUARD_API_PORT", "8000"))
    serve(create_app(), host="0.0.0.0", port=port, threads=4)


def open_browser(port: str) -> None:
    """Wait briefly for the server to start, then open the browser."""
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")


def main() -> None:
    api_thread = threading.Thread(target=run_api, daemon=True, name="gridguard-api")
    api_thread.start()
    time.sleep(0.4)

    streamlit_port = os.getenv("PORT", "8501")
    
    # Automatically launch the browser in a separate thread
    threading.Thread(target=open_browser, args=(streamlit_port,), daemon=True).start()

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.address=0.0.0.0",
        f"--server.port={streamlit_port}",
        "--server.headless=true",  # We explicitly handle opening the browser using webbrowser
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
