#!/usr/bin/env python
"""
Launch the Optuna Dashboard for a CellFoundry optimization study.

Requires the ``optuna-dashboard`` package::

    pip install optuna-dashboard

Usage::

    python optimizer/dashboard.py                              # auto-detect DB
    python optimizer/dashboard.py --storage sqlite:///my.db    # explicit DB
    python optimizer/dashboard.py --port 8080                  # custom port
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from time import sleep


def _find_default_db() -> str | None:
    """Try to locate an SQLite study database in typical locations."""
    search_dirs = [
        Path.cwd(),
        Path(__file__).resolve().parent,          # optimizer/
        Path(__file__).resolve().parent.parent,    # project root
    ]
    for d in search_dirs:
        dbs = sorted(d.glob("*.db"))
        if dbs:
            return f"sqlite:///{dbs[0]}"
    return None


def check_dashboard_available() -> bool:
    """Return True if ``optuna-dashboard`` is importable."""
    try:
        import optuna_dashboard  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Optuna Dashboard for CellFoundry"
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL, e.g. 'sqlite:///cellfoundry_cell_pop.db'. "
             "If omitted, auto-detects the first .db file in the project.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the dashboard web server (default: 8080).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the browser automatically.",
    )
    args = parser.parse_args()

    # --- Resolve storage URL ---
    storage = args.storage or _find_default_db()
    if storage is None:
        print("ERROR: No .db file found. Run an optimization first or specify --storage.")
        sys.exit(1)
    print(f"Storage: {storage}")

    # --- Check optuna-dashboard ---
    if not check_dashboard_available():
        print()
        print("optuna-dashboard is not installed.")
        print("Install it with:  pip install optuna-dashboard")
        print()
        ans = input("Install now? [y/N] ").strip().lower()
        if ans == "y":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna-dashboard"])
            print()
        else:
            sys.exit(1)

    # --- Launch ---
    url = f"http://{args.host}:{args.port}"
    print(f"Starting Optuna Dashboard at {url}")
    print("Press Ctrl+C to stop.\n")

    # Use the Python API directly — `python -m optuna_dashboard` is not
    # supported, and the `optuna-dashboard` CLI script may not be on PATH.
    try:
        from optuna_dashboard import run_server
    except ImportError:
        print("ERROR: could not import optuna_dashboard.run_server")
        sys.exit(1)

    if not args.no_browser:
        import threading

        def _open_browser() -> None:
            sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    try:
        run_server(storage, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")


if __name__ == "__main__":
    main()
