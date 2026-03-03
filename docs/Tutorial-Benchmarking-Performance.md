# Tutorial: Benchmarking Performance

This guide explains how to use the **`benchmark_perf.py`** script to run
automated performance-scaling studies on the cellfoundry simulation.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [How It Works](#how-it-works)
4. [Quick Start](#quick-start)
5. [Command-Line Reference](#command-line-reference)
6. [Parameter Grid](#parameter-grid)
7. [What Gets Patched](#what-gets-patched)
8. [Output Format](#output-format)
9. [Step-by-Step Examples](#step-by-step-examples)
   - [Minimal smoke test](#91-minimal-smoke-test)
   - [Dry-run to preview the matrix](#92-dry-run-to-preview-the-matrix)
   - [Scaling ECM grid size only](#93-scaling-ecm-grid-size-only)
   - [Full default benchmark (27 runs)](#94-full-default-benchmark-27-runs)
   - [Multiple network files](#95-multiple-network-files)
   - [Custom output path](#96-custom-output-path)
   - [Inspecting patched files](#97-inspecting-patched-files)
   - [Running inside a specific conda environment](#98-running-inside-a-specific-conda-environment)
10. [Understanding the Results CSV](#understanding-the-results-csv)
11. [Interpreting Status Codes](#interpreting-status-codes)
12. [Architecture & Safety Guarantees](#architecture--safety-guarantees)
13. [Excluded Directories](#excluded-directories)
14. [Troubleshooting](#troubleshooting)
15. [Advanced: Adding New Sweep Axes](#advanced-adding-new-sweep-axes)

---

## Overview

`benchmark_perf.py` automates performance-scaling experiments by sweeping
over combinations of:

| Axis | CLI flag | Default values | What it controls |
|------|----------|----------------|------------------|
| **N** (ECM grid) | `--n` | 21, 41, 81 | ECM agents = N³ (9 261 → 531 441) |
| **N_CELLS** | `--n-cells` | 100, 1 000, 1 000 000 | Number of cell agents |
| **INIT_N_FOCAD_PER_CELL** | `--focad` | 5, 25, 50 | Focal adhesions per cell |
| **Network file** | `--network` | `network_3d_small.pkl`, `network_3d_medium.pkl`, `network_3d_big.pkl` | Fibre network geometry |

With the defaults, the full Cartesian product is **3 × 3 × 3 × 3 = 81 runs**.

For each configuration the script records the wall-clock execution time and
time-per-step, then saves everything to a CSV file for later analysis.

---

## Prerequisites

- **Python 3.10+** with the cellfoundry dependencies installed
  (`pip install -r requirements.txt`).
- **FLAMEGPU2** Python bindings (`pyflamegpu`) available in the active
  environment.
- A working simulation — you should be able to run `python model.py` from
  the project root without errors before benchmarking.

> **Tip:** If you use conda, make sure the correct environment is activated
> before running the script, *or* use the `--conda-env` flag (see below).

---

## How It Works

```
Original project (NEVER touched)
│
├── model.py          ← stays untouched
├── *.cpp             ← stays untouched
├── network_3d_small.pkl
├── network_3d_medium.pkl
├── network_3d_big.pkl
└── ...

        ┌──────────────────────────────────┐
        │  1. COPY entire project into     │
        │     tools/_benchmark_workdir/    │
        │     <timestamp>/                 │
        └──────────┬───────────────────────┘
                   │
        ┌──────────▼───────────────────────┐
        │  2. PATCH the copies:            │
        │     • N = <val> in model.py      │
        │     • ECM_POPULATION_SIZE = N³   │
        │       in 5 .cpp files            │
        │     • Write JSON overrides for   │
        │       N_CELLS, FOCAD, STEPS, etc │
        └──────────┬───────────────────────┘
                   │
        ┌──────────▼───────────────────────┐
        │  3. RUN  python model.py         │
        │         --overrides <json>       │
        │     from the working copy dir    │
        └──────────┬───────────────────────┘
                   │
        ┌──────────▼───────────────────────┐
        │  4. PARSE [BENCHMARK] line from  │
        │     stdout → extract timings     │
        └──────────┬───────────────────────┘
                   │
        ┌──────────▼───────────────────────┐
        │  5. REPEAT for every config in   │
        │     the parameter grid           │
        └──────────┬───────────────────────┘
                   │
        ┌──────────▼───────────────────────┐
        │  6. CLEANUP: delete working copy │
        │     (or keep with --keep-workdir)│
        └──────────┬───────────────────────┘
                   │
        ┌──────────▼───────────────────────┐
        │  7. SAVE results CSV to          │
        │     tools/benchmark_results.csv  │
        └──────────────────────────────────┘
```

**Key safety property:** the original project directory is never modified.
All patching happens inside the disposable working copy. If the script
crashes, gets interrupted (Ctrl+C), or encounters an error, the original
files remain intact.

---

## Quick Start

```bash
# Preview what will be run (no files copied, no simulations launched)
python tools/benchmark_perf.py --steps 5 --dry-run

# Run a small test: 1 ECM size, 2 cell counts, 1 FOCAD count = 2 runs
python tools/benchmark_perf.py --steps 5 --n 21 --n-cells 100 500 --focad 10

# Full default sweep (27 runs)
python tools/benchmark_perf.py --steps 10
```

---

## Command-Line Reference

```
usage: benchmark_perf.py [-h] --steps STEPS
                         [--n N [N ...]]
                         [--n-cells N_CELLS [N_CELLS ...]]
                         [--focad FOCAD [FOCAD ...]]
                         [--network NETWORK [NETWORK ...]]
                         [--output OUTPUT]
                         [--conda-env CONDA_ENV]
                         [--keep-workdir]
                         [--dry-run]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--steps` | **Yes** | Number of simulation steps per run. |
| `--n` | No | ECM grid sizes. Each value produces N³ ECM agents. Default: `21 41 81`. |
| `--n-cells` | No | Cell agent counts. Default: `100 1000 1000000`. |
| `--focad` | No | Initial focal adhesions per cell. Default: `5 25 50`. |
| `--network` | No | Network `.pkl` file name(s). Default: `network_3d_small.pkl network_3d_medium.pkl network_3d_big.pkl`. |
| `--output` | No | Custom path for the results CSV. Default: `tools/benchmark_results.csv`. |
| `--conda-env` | No | Name of a conda environment to activate for each subprocess run. |
| `--keep-workdir` | No | Do not delete the working copy after completion. Useful for inspecting patched files. |
| `--dry-run` | No | Print the run matrix and exit. No files are copied or modified. |

---

## Parameter Grid

The script builds the **Cartesian product** of all axes:

```
total_runs = len(N_values) × len(N_CELLS_values) × len(FOCAD_values) × len(network_files)
```

Default grid (showing a subset — network axis omitted for brevity):

| N | ECM agents (N³) | N_CELLS | FOCAD | Total FOCAD agents (initial) |
|---|-----------------|---------|-------|-----------------------------|
| 21 | 9 261 | 100 | 5 | 500 |
| 21 | 9 261 | 100 | 25 | 2 500 |
| 21 | 9 261 | 100 | 50 | 5 000 |
| 21 | 9 261 | 1 000 | 5 | 5 000 |
| … | … | … | … | … |
| 81 | 531 441 | 1 000 000 | 50 | 50 000 000 |

Each row is run 3× (once per network file), giving **81 total runs**.

> **Warning:** Large configurations (N=81 with 1 000 000 cells) will require
> significant GPU memory and may take a very long time. Start small and
> scale up gradually.

---

## What Gets Patched

For each run, the script modifies **only the working copy**:

### Source-level patches (text replacement)

| File(s) | What changes | Example |
|---------|-------------|---------|
| `model.py` (line ~56) | `N = 21` → `N = 41` | ECM grid density |
| `ecm_Csp_update.cpp` | `ECM_POPULATION_SIZE = 9261` → `= 68921` | C++ template param |
| `ecm_ecm_interaction.cpp` | (same) | |
| `ecm_boundary_concentration_conditions.cpp` | (same) | |
| `cell_ecm_interaction_metabolism.cpp` | (same) | |
| `cell_move.cpp` | (same) | |

> **Why patch C++ files?**  `ECM_POPULATION_SIZE` is used as a template
> parameter for `getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>`.
> FLAMEGPU2 RTC compilation requires this value at compile time, so it
> cannot be passed as a runtime environment property.

### Runtime overrides (JSON)

A temporary `_benchmark_overrides.json` is written inside the working
copy and passed to `model.py --overrides`:

```json
{
  "N_CELLS": 1000,
  "INIT_N_FOCAD_PER_CELL": 25,
  "STEPS": 10,
  "SAVE_DATA_TO_FILE": false,
  "SAVE_PICKLE": false,
  "SHOW_PLOTS": false,
  "VISUALISATION": false,
  "NETWORK_FILE": "network_3d_small.pkl"
}
```

These suppress all file I/O and visualisation so that only compute time is
measured.

---

## Output Format

Results are saved to **`tools/benchmark_results.csv`** (or the path given
by `--output`).

### CSV columns

| Column | Type | Description |
|--------|------|-------------|
| `run` | int | Sequential run number (1-based). |
| `N` | int | ECM grid dimension used for this run. |
| `ECM_POPULATION_SIZE` | int | N³ — total ECM agent count. |
| `N_CELLS` | int | Number of cell agents. |
| `INIT_N_FOCAD_PER_CELL` | int | Focal adhesions seeded per cell. |
| `FOCAD_count_init` | int | `N_CELLS × INIT_N_FOCAD_PER_CELL`. |
| `network_file` | str | Network file used. |
| `steps` | int | Number of simulation steps requested. |
| `total_time_s` | float | Total wall-clock execution time (seconds). |
| `time_per_step_s` | float | `total_time_s / steps`. |
| `status` | str | Run outcome (see [Status Codes](#interpreting-status-codes)). |
| `timestamp` | str | When the benchmark batch was started. |

### Example CSV

```csv
run,N,ECM_POPULATION_SIZE,N_CELLS,INIT_N_FOCAD_PER_CELL,FOCAD_count_init,network_file,steps,total_time_s,time_per_step_s,status,timestamp
1,21,9261,100,5,500,network_3d_small.pkl,10,12.345678,1.234568,OK,2026-03-03 14:30:00
2,21,9261,100,5,500,network_3d_medium.pkl,10,13.456789,1.345679,OK,2026-03-03 14:30:00
3,21,9261,100,5,500,network_3d_big.pkl,10,14.567890,1.456789,OK,2026-03-03 14:30:00
```

---

## Step-by-Step Examples

### 9.1 Minimal smoke test

Run a single configuration with very few steps to verify the pipeline
works end-to-end:

```bash
python tools/benchmark_perf.py --steps 2 --n 21 --n-cells 100 --focad 5
```

This produces **1 run** (1×1×1×1). Expect it to finish in under a minute.

### 9.2 Dry-run to preview the matrix

Before committing to a long batch, inspect the planned configurations:

```bash
python tools/benchmark_perf.py --steps 10 --n 21 41 --n-cells 100 1000 --focad 5 25 --dry-run
```

Output:

```
Performance benchmark: 24 configurations, 10 steps each
  N:       [21, 41]
  N_CELLS: [100, 1000]
  FOCAD:   [5, 25]
  Network: ['network_3d_small.pkl', 'network_3d_medium.pkl', 'network_3d_big.pkl']
  (DRY RUN — nothing will be copied or executed)

============================================================
  Run 1/24:  N=21  N_CELLS=100  FOCAD=5  network=network_3d_small.pkl
  ECM_POPULATION_SIZE=9261  STEPS=10
============================================================
...
```

No files are created or modified.

### 9.3 Scaling ECM grid size only

To study how time scales with ECM agent count (holding cells and FOCAD
constant):

```bash
python tools/benchmark_perf.py --steps 20 --n 11 21 31 41 51 --n-cells 500 --focad 10
```

This gives 5 runs. Plot `ECM_POPULATION_SIZE` vs `time_per_step_s` from
the CSV.

### 9.4 Full default benchmark (81 runs)

```bash
python tools/benchmark_perf.py --steps 10
```

Uses the default grid: N ∈ {21, 41, 81}, N_CELLS ∈ {100, 1000, 1000000},
FOCAD ∈ {5, 25, 50}, network ∈ {small, medium, big}. → 81 runs.

> **Estimated time:** hours to days depending on GPU and the largest
> configurations. Consider starting with `--steps 3` for a rough estimate,
> then scale up.

### 9.5 Network file axis

Three fibre-network geometries are included by default:

| File | Description |
|------|-------------|
| `network_3d_small.pkl` | Sparse / low-density fibre network |
| `network_3d_medium.pkl` | Medium-density fibre network |
| `network_3d_big.pkl` | Dense / high-density fibre network |

All three are swept automatically. To benchmark with only one:

```bash
python tools/benchmark_perf.py --steps 10 --network network_3d_medium.pkl
```

Or supply your own files:

```bash
python tools/benchmark_perf.py --steps 10 \
    --network network_3d_medium.pkl my_custom_network.pkl
```

Network files must exist in the project root (they get copied into the
working directory).

### 9.6 Custom output path

```bash
python tools/benchmark_perf.py --steps 10 --output results/perf_study_march2026.csv
```

### 9.7 Inspecting patched files

To verify that source patching works correctly, use `--keep-workdir`:

```bash
python tools/benchmark_perf.py --steps 2 --n 41 --n-cells 100 --focad 5 --keep-workdir
```

After completion the script prints:

```
[keep] Working copy retained at C:\...\tools\_benchmark_workdir\20260303_143000
```

You can then open the working copy and inspect the patched `model.py` and
`.cpp` files. Delete the folder manually when done.

### 9.8 Running inside a specific conda environment

If you launch the benchmark from a different environment than the one where
FLAMEGPU2 is installed:

```bash
python tools/benchmark_perf.py --steps 10 --conda-env flamegpu_py310
```

Each subprocess will be launched via `conda run -n flamegpu_py310`.

---

## Understanding the Results CSV

### Loading in Python

```python
import pandas as pd

df = pd.read_csv("tools/benchmark_results.csv")

# Filter successful runs
ok = df[df["status"].str.startswith("OK")]

# Pivot: rows=N_CELLS, columns=N, values=time_per_step_s
pivot = ok.pivot_table(
    index="N_CELLS",
    columns="N",
    values="time_per_step_s",
    aggfunc="mean",
)
print(pivot)
```

### Quick matplotlib plot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for n_val, group in ok.groupby("N"):
    group = group.sort_values("N_CELLS")
    ax.plot(group["N_CELLS"], group["time_per_step_s"],
            marker="o", label=f"N={n_val} (ECM={n_val**3})")
ax.set_xlabel("N_CELLS")
ax.set_ylabel("Time per step (s)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_title("Cellfoundry scaling: time per step vs cell count")
plt.tight_layout()
plt.savefig("tools/benchmark_scaling.png", dpi=150)
plt.show()
```

---

## Interpreting Status Codes

| Status | Meaning |
|--------|---------|
| `OK` | Run completed successfully; timing parsed from `[BENCHMARK]` output line. |
| `OK(fallback)` | Run completed but the `[BENCHMARK]` line was not found; timing was extracted from the older `EXECUTION TIME: X.XX seconds` output instead. |
| `dry-run` | Dry-run mode — no simulation was launched. |
| `ERROR(<code>)` | The subprocess exited with a non-zero return code. Last 20 lines of output are printed to the console. |
| `TIMEOUT` | The run exceeded the 1-hour timeout. |
| `NO_TIMING` | The subprocess exited normally (code 0) but no timing information was found in stdout. |

---

## Architecture & Safety Guarantees

### Why a working copy?

Some parameters — notably `ECM_POPULATION_SIZE` — are hard-coded as C++
template arguments (`getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>`).
FLAMEGPU2's RTC compilation reads these values from the source text at
build time, so they must be changed in the `.cpp` files themselves.

Rather than modifying the original files (even with backup/restore), the
benchmark script **copies the entire project** into a temporary directory
and patches only the copies. This means:

- **Your source files are never touched** — not even temporarily.
- If the script crashes, is killed, or the machine loses power, nothing
  is lost.
- You can continue developing in the main project while a benchmark runs.
- Multiple benchmark runs could (in principle) run in parallel from
  separate working copies.

### What gets copied

Everything under the project root **except** heavy or irrelevant
directories:

| Excluded directory | Reason |
|-------------------|--------|
| `.git` | Large; not needed for simulation |
| `__pycache__` | Regenerated automatically |
| `result_files`, `results` | Potentially very large output data |
| `_benchmark_workdir` | Avoids recursive nesting |
| `_benchmark_backups` | Legacy backup folder |
| `.vscode` | Editor settings |
| `manual_tests`, `docs`, `assets` | Not needed at runtime |
| `optimizer`, `postprocessing` | Not needed for the simulation run |
| `network_generator` | Not needed at runtime |
| `optuna_results`, `node_modules` | Not needed at runtime |

Files with suffix `.db` are also excluded.

### The `[BENCHMARK]` output line

`model.py` emits a structured line at the end of each run:

```
[BENCHMARK] EXECUTION_TIME=12.345678 STEPS=10 TIME_PER_STEP=1.234568
```

The benchmark script parses this with a regex. If it is missing (e.g., in
an older version of model.py), the script falls back to parsing the legacy
`EXECUTION TIME: X.XX seconds` line.

---

## Excluded Directories

You can customise the set of excluded directories by editing the
`COPY_EXCLUDE_DIRS` set near the top of `benchmark_perf.py`:

```python
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
    "_benchmark_workdir",
    "_benchmark_backups",
    "optuna_results",
    ".vscode",
    "node_modules",
}
```

If your simulation requires files from one of these directories at runtime,
remove it from the set so it gets copied into the working directory.

---

## Troubleshooting

### "Could not find N = \<int\> in model.py"

The script expects a line matching `^N = <digits>` (at the start of a
line) in `model.py`. If `N` has been renamed or the formatting changed,
update `RE_N_PY` in `benchmark_perf.py`.

### Runs fail with `ERROR(1)` — import errors

The working copy may be missing a file or directory needed at import time.
Check whether a required module lives in one of the excluded directories
and remove that directory from `COPY_EXCLUDE_DIRS`.

Re-run with `--keep-workdir` and try running `python model.py` manually
from the working copy to diagnose.

### Runs succeed but status is `NO_TIMING`

`model.py` must print the `[BENCHMARK]` line. Verify it is present by
running `python model.py --overrides ... | findstr BENCHMARK` (Windows) or
`grep BENCHMARK` (Linux/macOS).

### Disk space concerns

Each working copy contains all `.cpp`, `.py`, `.pkl`, and `.vtk` files.
The `.pkl` network files can be tens of MB. If disk space is tight:

- Use `--keep-workdir` only when debugging.
- The working copy is deleted automatically after a normal run.
- Consider reducing the number of `.vtk`/`.pkl` files in the project root
  before benchmarking.

### Timeout (> 1 hour per run)

The default per-run timeout is 3600 seconds (1 hour). For very large
configurations you may need to increase this by editing `timeout=3600` in
the `subprocess.run()` call inside `_run_single()`.

---

## Advanced: Adding New Sweep Axes

To add a new parameter axis (e.g., `TIME_STEP`):

1. **Add a CLI argument** in `main()`:
   ```python
   parser.add_argument(
       "--time-step", type=float, nargs="+", default=[0.1],
       help="TIME_STEP values. Default: 0.1")
   ```

2. **Include it in the grid** product:
   ```python
   grid = list(itertools.product(
       args.n, args.n_cells, args.focad, args.network, args.time_step))
   ```

3. **Unpack it** in the loop and pass to `_run_single()`.

4. **Add it to the overrides dict** inside `_run_single()`:
   ```python
   overrides["TIME_STEP"] = time_step
   ```

5. **Add a column** to `FIELDNAMES` and `_make_result()`.

If the new parameter requires patching C++ source files (like
`ECM_POPULATION_SIZE`), add the corresponding regex and patching logic
following the existing `_patch_ecm_pop_in_cpp()` pattern.

---

*Last updated: March 2026*
