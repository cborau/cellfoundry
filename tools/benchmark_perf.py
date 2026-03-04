#!/usr/bin/env python
"""
benchmark_perf.py – Automated performance scaling study for cellfoundry.

Systematically runs model.py with different agent-count configurations and
records wall-clock time per step.

**Original project files are NEVER modified.**  The script copies the entire
project into a temporary working directory, patches only the copies, and
runs every simulation from there.

Usage
-----
    python tools/benchmark_perf.py --steps 10
    python tools/benchmark_perf.py --steps 20 --dry-run
    python tools/benchmark_perf.py --steps 10 --n 21 41 --n-cells 100 1000 --focad 5 25
    python tools/benchmark_perf.py --steps 10 --n 21 41 --cell-radius 5.0 8.412 15.0

The script:
  1. Copies all source files into a disposable working directory under
     ``tools/_benchmark_workdir/<timestamp>/`` (heavy folders like .git,
     result_files, results, __pycache__ are excluded).
  2. For each (N, N_CELLS, INIT_N_FOCAD_PER_CELL, CELL_RADIUS, network)
     combination it patches the *copies* of the .cpp and model.py files
     and runs the simulation as a subprocess from the working directory.
  3. Parses stdout for the ``[BENCHMARK]`` line emitted by model.py.
  4. When finished, the working directory is deleted (pass ``--keep-workdir``
     to keep it for inspection).
  5. Saves results to ``tools/benchmark_results.csv``.

What gets patched (in the working copy only)
--------------------------------------------
- ``model.py``  ``N = <val>``  (changes ECM grid density)
- 5 ``.cpp`` files  ``ECM_POPULATION_SIZE = <N^3>``  (template parameter)
- Runtime overrides via JSON: ``N_CELLS``, ``INIT_N_FOCAD_PER_CELL``,
  ``CELL_RADIUS``, ``STEPS``, ``SAVE_DATA_TO_FILE``, ``SAVE_PICKLE``,
  ``SHOW_PLOTS``, ``VISUALISATION``, ``NETWORK_FILE``
"""
from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the REAL project root)
# ---------------------------------------------------------------------------
TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
RESULTS_CSV = TOOLS_DIR / "benchmark_results.csv"

# Directories / patterns to SKIP when copying the project (saves time & disk)
COPY_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "result_files",
    "results",
    "manual_tests",
    "network_generator",
    "docs",
    "assets",
    "optimizer",
    "postprocessing",
    "tools",
    "_benchmark_workdir",
    "optuna_results",
    ".vscode",
    "node_modules",
}
COPY_EXCLUDE_SUFFIXES = {".db"}

# Relative paths (from project root) of .cpp files that contain
# ``const uint32_t ECM_POPULATION_SIZE = <val>;``
CPP_ECM_POP_RELNAMES = [
    "ecm_Csp_update.cpp",
    "ecm_ecm_interaction.cpp",
    "ecm_boundary_concentration_conditions.cpp",
    "cell_ecm_interaction_metabolism.cpp",
    "cell_move.cpp",
]

# Regex patterns
RE_N_PY = re.compile(r"^(N\s*=\s*)(\d+)", re.MULTILINE)
RE_ECM_POP_CPP = re.compile(
    r"(const\s+uint32_t\s+ECM_POPULATION_SIZE\s*=\s*)(\d+)(\s*;)"
)
RE_BENCHMARK = re.compile(
    r"\[BENCHMARK\]\s+EXECUTION_TIME=([\d.]+)\s+STEPS=(\d+)\s+TIME_PER_STEP=([\d.]+)"
    r"\s+INIT_TIME=([\d.]+)\s+SIMULATION_TIME=([\d.]+)"
    r"(?:\s+RTC_TIME=([\d.]+))?"
    r"(?:\s+INIT_FUNCTIONS_TIME=([\d.]+))?"
    r"(?:\s+EXIT_FUNCTIONS_TIME=([\d.]+))?"
)


# ---------------------------------------------------------------------------
# FNODE counting (reads pickle once per file, then caches)
# ---------------------------------------------------------------------------
_FNODE_COUNT_CACHE: dict[str, int] = {}


def _count_fnodes(pkl_path: Path) -> int:
    """Return the number of fibre-network nodes stored in *pkl_path*."""
    key = str(pkl_path.resolve())
    if key not in _FNODE_COUNT_CACHE:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        _FNODE_COUNT_CACHE[key] = int(data["node_coords"].shape[0])
    return _FNODE_COUNT_CACHE[key]


# ---------------------------------------------------------------------------
# Working-copy helpers
# ---------------------------------------------------------------------------

def _copy_project(dest: Path) -> None:
    """Shallow-copy the project tree into *dest*, skipping heavy dirs."""

    def _ignore(directory: str, entries: list[str]) -> set[str]:
        ignored: set[str] = set()
        for entry in entries:
            full = Path(directory) / entry
            if entry in COPY_EXCLUDE_DIRS:
                ignored.add(entry)
            elif full.is_file() and full.suffix in COPY_EXCLUDE_SUFFIXES:
                ignored.add(entry)
        return ignored

    shutil.copytree(PROJECT_ROOT, dest, ignore=_ignore, dirs_exist_ok=True)


class WorkingCopy:
    """Context-manager that creates (and optionally cleans up) a working copy.

    The copy lives under ``tools/_benchmark_workdir/<timestamp>/``.
    All patching and simulation runs happen inside this copy.
    """

    def __init__(self, dry: bool = False, keep: bool = False):
        self.dry = dry
        self.keep = keep
        self.workdir: Path | None = None

    # Convenient accessors for paths inside the working copy
    @property
    def model_py(self) -> Path:
        assert self.workdir is not None
        return self.workdir / "model.py"

    def cpp_ecm_pop_paths(self) -> list[Path]:
        assert self.workdir is not None
        return [self.workdir / name for name in CPP_ECM_POP_RELNAMES]

    def __enter__(self):
        if self.dry:
            return self
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workdir = TOOLS_DIR / "_benchmark_workdir" / ts
        print(f"[copy] Copying project → {self.workdir} …")
        _copy_project(self.workdir)
        print(f"[copy] Done.")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.dry or self.workdir is None:
            return False
        if self.keep:
            print(f"[keep] Working copy retained at {self.workdir}")
        else:
            print(f"[cleanup] Removing working copy {self.workdir} …")
            shutil.rmtree(self.workdir, ignore_errors=True)
            # Remove parent dir if empty
            parent = self.workdir.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            print("[cleanup] Done.")
        return False


# ---------------------------------------------------------------------------
# Patching helpers (always operate on the WORKING COPY)
# ---------------------------------------------------------------------------

def _patch_N_in_model(model_py: Path, new_n: int) -> None:
    """Rewrite ``N = <old>`` in the working copy's model.py."""
    text = model_py.read_text(encoding="utf-8")
    new_text, count = RE_N_PY.subn(rf"\g<1>{new_n}", text, count=1)
    if count == 0:
        raise RuntimeError(f"Could not find  N = <int>  in {model_py}")
    model_py.write_text(new_text, encoding="utf-8")


def _patch_ecm_pop_in_cpp(cpp_paths: list[Path], ecm_pop: int) -> None:
    """Rewrite ``ECM_POPULATION_SIZE = <old>`` in working-copy .cpp files."""
    for cpp in cpp_paths:
        if not cpp.exists():
            continue
        text = cpp.read_text(encoding="utf-8")
        new_text, count = RE_ECM_POP_CPP.subn(rf"\g<1>{ecm_pop}\3", text)
        if count:
            cpp.write_text(new_text, encoding="utf-8")


def _compute_ecm_pop(n: int) -> int:
    """For a cubical domain, ECM_POPULATION_SIZE = N^3."""
    return n ** 3


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def _run_single(
    wc: WorkingCopy,
    n: int,
    n_cells: int,
    init_focad: int,
    cell_radius: float,
    steps: int,
    network_file: str,
    run_index: int,
    total_runs: int,
    conda_env: str | None,
    dry: bool,
) -> dict:
    """Patch working copy, run model.py, parse output, return metrics dict."""

    ecm_pop = _compute_ecm_pop(n)
    search_radius = 3.0 * cell_radius  # MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION

    # Count FNODES from the network pickle (working copy or project root)
    if not dry and wc.workdir is not None:
        pkl_path = wc.workdir / network_file
    else:
        pkl_path = PROJECT_ROOT / network_file
    n_fnodes = _count_fnodes(pkl_path) if pkl_path.exists() else 0

    print(
        f"\n{'='*60}\n"
        f"  Run {run_index}/{total_runs}:  N={n}  N_CELLS={n_cells}  "
        f"FOCAD={init_focad}  CELL_RADIUS={cell_radius}  N_FNODES={n_fnodes}\n"
        f"  ECM_POPULATION_SIZE={ecm_pop}  SEARCH_RADIUS={search_radius:.2f}  STEPS={steps}\n"
        f"{'='*60}"
    )

    if dry:
        return _make_result(
            n, ecm_pop, n_cells, init_focad, n_fnodes, cell_radius,
            search_radius, steps, status="dry-run",
        )

    # 1. Patch the working copy
    _patch_N_in_model(wc.model_py, n)
    _patch_ecm_pop_in_cpp(wc.cpp_ecm_pop_paths(), ecm_pop)

    # 2. Write temporary overrides JSON inside the working copy
    overrides = {
        "N_CELLS": n_cells,
        "INIT_N_FOCAD_PER_CELL": init_focad,
        "CELL_RADIUS": cell_radius,
        "STEPS": steps,
        "SAVE_DATA_TO_FILE": False,
        "SAVE_PICKLE": False,
        "SHOW_PLOTS": False,
        "VISUALISATION": False,
        "NETWORK_FILE": network_file,
    }
    tmp_json = wc.workdir / "_benchmark_overrides.json"
    tmp_json.write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    # 3. Launch model.py from the working copy
    model_path = str(wc.model_py)
    overrides_path = str(tmp_json)

    if conda_env:
        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            sys.executable, model_path, "--overrides", overrides_path,
        ]
    else:
        cmd = [sys.executable, model_path, "--overrides", overrides_path]

    print(f"  cwd: {wc.workdir}")
    print(f"  cmd: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(wc.workdir),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max per run
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT (>1h)")
        return _make_result(n, ecm_pop, n_cells, init_focad, n_fnodes,
                            cell_radius, search_radius, steps,
                            status="TIMEOUT")

    # 4. Parse output
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        combined = (stdout + stderr).strip().splitlines()
        for line in combined[-20:]:
            print(f"    | {line}")
        return _make_result(n, ecm_pop, n_cells, init_focad, n_fnodes,
                            cell_radius, search_radius, steps,
                            status=f"ERROR({result.returncode})")

    m = RE_BENCHMARK.search(stdout)
    if m:
        total_time = float(m.group(1))
        actual_steps = int(m.group(2))
        time_per_step = float(m.group(3))
        init_time = float(m.group(4))
        sim_time = float(m.group(5))
        rtc_time = float(m.group(6)) if m.group(6) else None
        init_func_time = float(m.group(7)) if m.group(7) else None
        exit_func_time = float(m.group(8)) if m.group(8) else None
        detail_parts = []
        if rtc_time is not None:
            detail_parts.append(f"rtc={rtc_time:.2f}s")
        if init_func_time is not None:
            detail_parts.append(f"init_funcs={init_func_time:.2f}s")
        detail_str = f" ({', '.join(detail_parts)})" if detail_parts else ""
        print(f"  OK: {total_time:.2f}s total, "
              f"init={init_time:.2f}s{detail_str}, sim={sim_time:.2f}s, "
              f"{time_per_step:.4f}s/step ({actual_steps} steps)")
        return _make_result(
            n, ecm_pop, n_cells, init_focad, n_fnodes,
            cell_radius, search_radius, steps,
            total_time_s=total_time, time_per_step_s=time_per_step,
            init_time_s=init_time, simulation_time_s=sim_time,
            rtc_time_s=rtc_time, init_functions_time_s=init_func_time,
            exit_functions_time_s=exit_func_time,
            status="OK",
        )

    # Detect model.py's silent quit() on critical_error
    if re.search(r"critical.?error|must be higher than", stdout, re.IGNORECASE):
        print("  FAILED: model.py hit a critical-error check and quit()")
        # Show the relevant lines
        for line in stdout.strip().splitlines():
            if any(kw in line.lower() for kw in ("error", "must be", "critical")):
                print(f"    | {line}")
        return _make_result(n, ecm_pop, n_cells, init_focad, n_fnodes,
                            cell_radius, search_radius, steps,
                            status="CRITICAL_ERROR")

    print("  WARNING: could not parse timing from output")
    print("  (last 15 lines of stdout:)")
    for line in stdout.strip().splitlines()[-15:]:
        print(f"    | {line}")
    return _make_result(n, ecm_pop, n_cells, init_focad, n_fnodes,
                        cell_radius, search_radius, steps,
                        status="NO_TIMING")


def _make_result(
    n, ecm_pop, n_cells, init_focad, n_fnodes,
    cell_radius, search_radius, steps,
    total_time_s=None, time_per_step_s=None,
    init_time_s=None, simulation_time_s=None,
    rtc_time_s=None, init_functions_time_s=None,
    exit_functions_time_s=None,
    status="",
) -> dict:
    return {
        "N": n,
        "ECM_POPULATION_SIZE": ecm_pop,
        "N_CELLS": n_cells,
        "INIT_N_FOCAD_PER_CELL": init_focad,
        "FOCAD_count_init": n_cells * init_focad,
        "N_FNODES": n_fnodes,
        "CELL_RADIUS": cell_radius,
        "MAX_SEARCH_RADIUS": search_radius,
        "steps": steps,
        "init_time_s": init_time_s,
        "simulation_time_s": simulation_time_s,
        "rtc_time_s": rtc_time_s,
        "init_functions_time_s": init_functions_time_s,
        "exit_functions_time_s": exit_functions_time_s,
        "total_time_s": total_time_s,
        "time_per_step_s": time_per_step_s,
        "status": status,
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "run", "N", "ECM_POPULATION_SIZE", "N_CELLS", "INIT_N_FOCAD_PER_CELL",
    "FOCAD_count_init", "N_FNODES", "CELL_RADIUS", "MAX_SEARCH_RADIUS",
    "steps", "init_time_s", "simulation_time_s",
    "rtc_time_s", "init_functions_time_s", "exit_functions_time_s",
    "total_time_s", "time_per_step_s", "status", "timestamp",
]


def _write_csv(results: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run cellfoundry performance scaling benchmark.\n\n"
                    "Original project files are NEVER modified — the study "
                    "runs entirely inside a disposable working copy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/benchmark_perf.py --steps 10
  python tools/benchmark_perf.py --steps 20 --n 21 41 --n-cells 100 1000
  python tools/benchmark_perf.py --steps 10 --cell-radius 5.0 8.412 15.0
  python tools/benchmark_perf.py --steps 10 --dry-run
  python tools/benchmark_perf.py --steps 5 --keep-workdir   # inspect patched files
        """,
    )
    parser.add_argument(
        "--steps", type=int, required=True,
        help="Number of simulation steps per run.")
    parser.add_argument(
        "--n", type=int, nargs="+", default=[21, 41, 81],
        help="ECM grid sizes (N). ECM agents = N^3. Default: 21 41 81")
    parser.add_argument(
        "--n-cells", type=int, nargs="+", default=[100, 1000, 10000],
        help="Number of cells. Default: 100 1000 10000")
    parser.add_argument(
        "--focad", type=int, nargs="+", default=[5, 25, 50],
        help="INIT_N_FOCAD_PER_CELL values. Default: 5 25 50")
    parser.add_argument(
        "--cell-radius", type=float, nargs="+", default=[5.0, 8.412, 15.0],
        help="CELL_RADIUS values (µm). Affects MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION "
             "(= 3×CELL_RADIUS) and other derived parameters. Default: 8.412")
    parser.add_argument(
        "--network", type=str, nargs="+",
        default=["network_low_density.pkl", "network_medium_density.pkl", "network_high_density.pkl"],
        help="Network .pkl files. Default: network_low_density.pkl network_medium_density.pkl network_high_density.pkl")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path. Default: tools/benchmark_results.csv")
    parser.add_argument(
        "--conda-env", type=str, default=None,
        help="Conda environment name to activate for each run (optional).")
    parser.add_argument(
        "--keep-workdir", action="store_true",
        help="Keep the temporary working directory after completion "
             "(useful for inspecting patched files).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run matrix without copying or executing anything.")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else RESULTS_CSV

    # Build the full parameter grid
    grid = list(itertools.product(
        args.n, args.n_cells, args.focad, args.cell_radius, args.network))
    total = len(grid)

    print(f"Performance benchmark: {total} configurations, "
          f"{args.steps} steps each")
    print(f"  N:           {args.n}")
    print(f"  N_CELLS:     {args.n_cells}")
    print(f"  FOCAD:       {args.focad}")
    print(f"  CELL_RADIUS: {args.cell_radius}")
    print(f"  Network:     {args.network}")
    if args.dry_run:
        print("  (DRY RUN — nothing will be copied or executed)\n")

    results: list[dict] = []
    ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with WorkingCopy(dry=args.dry_run, keep=args.keep_workdir) as wc:
        for idx, (n, n_cells, focad, crad, net_file) in enumerate(grid, 1):
            row = _run_single(
                wc=wc,
                n=n,
                n_cells=n_cells,
                init_focad=focad,
                cell_radius=crad,
                steps=args.steps,
                network_file=net_file,
                run_index=idx,
                total_runs=total,
                conda_env=args.conda_env,
                dry=args.dry_run,
            )
            row["run"] = idx
            row["timestamp"] = ts_str
            results.append(row)

    # Write CSV (always to the real tools/ dir, not the working copy)
    _write_csv(results, out_path)

    # Quick summary
    print(f"\n{'='*60}")
    print("Summary:")
    ok = [r for r in results if r["status"] and r["status"].startswith("OK")]
    fail = [r for r in results
            if r["status"]
            and not r["status"].startswith("OK")
            and r["status"] != "dry-run"]
    print(f"  Completed: {len(ok)}/{total}")
    if fail:
        print(f"  Failed:    {len(fail)}")
        for r in fail:
            print(f"    Run {r['run']}: N={r['N']} CELLS={r['N_CELLS']} "
                  f"FOCAD={r['INIT_N_FOCAD_PER_CELL']} — {r['status']}")
    if ok:
        times = [r["time_per_step_s"] for r in ok
                 if r["time_per_step_s"] is not None]
        if times:
            print(f"  Time/step range: {min(times):.4f}s – {max(times):.4f}s")


if __name__ == "__main__":
    main()
