# Tutorial: Useful Tools

This page summarizes the most important practical scripts from `tools/`. Check these files' docs for further details.

## 1. `resize_array_variables.py` (most important)

Use this when changing dimensional constants such as `N_CELL_TYPES` and `N_SPECIES`.

What it does:

- updates `model.py` dimensions
- resizes known per-type/per-species arrays
- updates matching hard-coded constants in `.cpp` files
- creates backups of edited files

Examples:

```powershell
python tools/resize_array_variables.py --species 4
python tools/resize_array_variables.py --cell-types 4
python tools/resize_array_variables.py --cell-types 4 --species 4 --dry-run
```

## 2. `check_hard_coded_values.py`

This script checks consistency between literal constants in `model.py` and values repeated in Python/C++ files.

It is also called automatically by `model.py` during startup.

Example:

```powershell
python check_hard_coded_values.py 
```

Use it after major parameter-structure edits or merge conflicts.

## 3. `benchmark_perf.py`

Runs performance sweeps over combinations such as `N`, `N_CELLS`, FOCAD count, and optional networks.

Quick example:

```powershell
python tools/benchmark_perf.py --steps 10 --dry-run
```

See full guide: [Tutorial: Benchmarking Performance](Tutorial-Benchmarking-Performance).

## 4. `check_pickle.py`

Prints a structured summary of an output pickle (top-level keys, dataframe shapes, sample rows).

Example:

```powershell
python tools/check_pickle.py --pickle result_files/output_data_0.pickle
```

## 5. `inspect_pickle_key.py`

Loads a specific key from a pickle and drops into `breakpoint()` for manual debugging.

Example:

```powershell
python tools/inspect_pickle_key.py result_files/output_data_0.pickle --key CELL_METRICS_OVER_TIME --expect-df
```

Best used in debug mode from an IDE/terminal debugger session.

## 6. `delete_optuna_study.py`

Utility script to remove one Optuna study from an SQLite storage without deleting other studies in the same database.

Typical workflow:

1. Edit `storage` and `study_name` in the script.
2. Run:

```powershell
python tools/delete_optuna_study.py
```

Useful when one comparative study is invalid but other studies in the same DB must be kept.

## 7. `generate_vascular_network.py`

Generates a random connected vascular network inside user-defined cuboidal bounds and saves it as a pickle (optionally VTK).

Example:

```powershell
python tools/generate_vascular_network.py --bounds 50 -50 50 -50 50 -50 --diameter 5 --density 0.02 --resolution 5 --branching-probability 0.08 --nucleation-faces 1 1 0 0 0 0 --nucleation-per-face 8 --seed 1 --output vascular_network.pickle --save-vtk
```

## 8. `param_ui.py`

Desktop parameter editor for `model.py` built with PySide6. It helps edit and run with less manual source editing.

Run:

```powershell
python tools/param_ui.py
```

See: [Model Editor](Model-Editor).

## 9. `remove_anchors_from_cell_vtks.py`

Creates simplified VTK cell files without anchor points by trimming each `cells_t*.vtk` according to `output_data_0.pickle` cell counts.

Example:

```powershell
python tools/remove_anchors_from_cell_vtks.py --input-dir result_files
```

Output files are named like `no_anchor_cells_tXXXX.vtk`.
