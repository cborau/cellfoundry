# Tutorial — Parameter Optimization

This tutorial walks you through **automated parameter calibration** for CellFoundry simulations using [Optuna](https://optuna.org/).  By the end you will be able to:

1. Define which model parameters to explore and their bounds.
2. Choose one (or more) objective functions that measure how *wrong* a simulation is relative to target data.
3. Run a Bayesian optimization loop that launches GPU simulations automatically.
4. Inspect results with CSV exports, JSON best-parameter files, and a live web dashboard.

> **Prerequisites** — a working CellFoundry environment (`pyflamegpu`, CUDA, Python 3.10+).  
> Install the extra packages with:
> ```
> pip install optuna pyyaml
> pip install optuna-dashboard   # optional: live web dashboard
> ```

---

## 1. Project layout

All optimization components live in the `optimizer/` directory:

```
optimizer/
├── optimize.py                  # Main runner (launches trials)
├── objectives.py                # Objective functions (error metrics)
├── dashboard.py                 # Launch the Optuna web dashboard
├── optuna_config.yaml           # Example: single-objective cell population
├── optuna_config_organoid.yaml  # Example: multi-objective organoid size
└── reference_data/              # Target / reference CSV files
    ├── target_cell_count.csv
    ├── target_focad_per_cell.csv
    └── ...
```

The main simulation script (`model.py`) is **not modified** during optimization. Instead, it receives parameter overrides via command-line arguments (`--overrides`, `--result-dir`).

### Configuration checks and stopping on errors

Before creating/loading the study or launching trials, the runner checks hard-coded constants in core source files and the selected variant against the literal structural settings in `model.py`. This check is read-only. A mismatch, unreadable reference, or missing variant stops optimization with a nonzero exit and a diagnostic report. The model repeats the check at each trial's startup, including with empty JSON overrides. No optimizer path repairs source files or asks for keyboard input.

To review and repair a mismatch deliberately, run this separately from the repository root (replace the variant name as needed):

```powershell
python check_hard_coded_values.py --scan-root . --scan-root variants/radial_glia --no-recursive
```

This prompts before changing files. Append `--fail-on-mismatch` to check without writing, or `--fix` for explicitly requested repairs without a prompt. The reference remains core `model.py`; the checker does not derive new structural dimensions from variant parameters.

Do not put `BOUNDARY_COORDS`, `N`, `N_SPECIES`, `N_CELL_TYPES`, `MAX_CONNECTIVITY`, `N_ANCHOR_POINTS`, `MAX_VASC_CONNECTIVITY`, `ECM_AGENTS_PER_DIR`, or `ECM_POPULATION_SIZE` in the search space. Configure initial bounds and dimensions in the core and synchronize the kernels before starting optimization. Fixed variant/JSON overrides must agree with the core values, including indexed boundary changes. Bounds cannot safely change after the initial grid and assay setup; domain/resolution studies require separately prepared core builds. Unknown JSON parameter names and invalid array indices are errors, so a misspelling cannot silently disable a trial's intended override.

Configuration/checker errors, unexpected model exceptions, missing or unreadable output pickles, and unexpected objective exceptions **stop the whole study**. A failure during a trial is recorded as `FAIL`, retains its output directory, and no subsequent trials are launched. Earlier successful trials remain in the study. Optimization requires `SAVE_PICKLE=True`.

An explicitly detected infeasible parameter combination **prunes only that trial**, so optimization continues. This includes the FNODE displacement check when `ABORT_ON_UNSTABLE_FNODE_MOVE=True`, runtime multiscale diffusion data/CFL failures, and the configured subprocess timeout. Setting `ABORT_ON_UNSTABLE_FNODE_MOVE=False` retains the simulation's existing behavior: that guard does not abort or prune. Initial diffusion configuration validation still stops the study; a runtime rejection is not an automatic timestep adjustment or permission to evaluate partial output.

For a variant-specific numerical/biological rejection, use a host check at the appropriate point in the variant schedule:

```python
from helper_module import reject_trial

def check_growth(host):
    count = host.agent("CELL").count()
    if count > MAX_VALID_POPULATION:  # A defined validity limit for this model.
        reject_trial(f"Population {count} exceeds this assay's validity limit")
```

`reject_trial()` aborts the simulation. Outside optimization it raises a clear exception; under optimization it also writes a per-invocation `trial_rejection.json` signal because FLAMEGPU can wrap Python host exceptions. The optimizer records `prune_reason` and retains logs/partial files for diagnosis, without evaluating their objective. Use this only for understood infeasible states. An unclassified crash (including an arbitrary CUDA error) remains fatal: the runner cannot reliably infer whether it came from sampled parameters or a programming defect. Place guards before unsafe operations when a known parameter-dependent failure is possible. For GPU checks, publish an error flag and inspect it in a following host layer.

An objective can explicitly reject otherwise readable results with `raise TrialRejected(reason)` (imported from `helper_module`). Other objective exceptions remain fatal. Standard output and error are saved in `stdout.log` and `stderr.log`; rejection messages identify both paths. A fresh invocation token prevents an old rejection file from masking a new unrelated error when reusing a result directory.

The constants check covers the named literal assignments, not every possible model constraint. Variant validation and model startup check the effective configuration; numerical and biological validity still need suitable tests for the chosen model.

---

## 2. Configuration file (YAML)

Every optimization run is defined by a single YAML file.  Open the provided `optimizer/optuna_config.yaml` as a starting point.  The three main sections are:

### 2.1 Study settings

```yaml
study:
  name: "my_study"          # Human-readable study name
  n_trials: 50              # Total number of parameter evaluations
  sampler: TPE              # TPE | RANDOM | CMAES | NSGA-II
  seed: 42                  # Random seed (null = non-deterministic)
  direction: minimize       # minimize | maximize
  storage: "sqlite:///my_study.db"
```

- **`storage`** — an [SQLite](https://www.sqlite.org/) database file.  Optuna writes all results here;
  you can resume interrupted runs by re-running the same command (trials already
  completed are skipped).
- **`sampler`** — the search algorithm.  `TPE` (Tree-structured Parzen Estimator) is recommended for
  single-objective problems.  For multi-objective use `NSGA-II`.

### 2.2 Objective function

```yaml
# Single objective
objective:
  function: final_cell_count_error
  reference: "optimizer/reference_data/target_cell_count.csv"   # optional
  kwargs:
    normalize: true
```

The `function` field must match a name registered in `objectives.py`.  Available objectives:

| Function name | What it fits | Error metric |
|---|---|---|
| `stress_strain_curve_error` | Stress–strain curve | MSE (stress + 0.1 × strain) |
| `boundary_force_curve_error` | Boundary forces over time | MSE per force axis |
| `cell_population_error` | Alive-cell count time-series | RMSE |
| `focad_attached_ratio_error` | Focal-adhesion attachment ratio | RMSE |
| `poisson_ratio_error` | Poisson ratio (scalar or series) | |Δν| or RMSE |
| `matrix_remodeling_error` | Fibre remodeling metrics | RMSE |
| `final_cell_count_error` | Final alive-cell count (scalar) | |ΔN| |
| `final_focad_per_cell_error` | Final FOCAD per alive cell | |Δ| |
| `organoid_error` | Organoid / spheroid metrics (pickle) | |ΔR| or RMSE |

> **Adding your own** — define a function with signature `f(results: dict, reference_path: str, **kwargs) -> tuple[float, str | None]` in `objectives.py` and add it to `OBJECTIVE_REGISTRY`.

### 2.3 Parameters to tune

```yaml
parameters:
  N_CELLS:
    type: int           # int | float | categorical
    low: 1
    high: 10

  CELL_SPEED_REF:
    type: float
    low: 0.1
    high: 3.0
    log: false          # true → log-uniform sampling (good for orders-of-magnitude ranges)

  CELL_HYPOXIA_THRESHOLD:
    type: float
    low: 0.005
    high: 0.1
    log: true
```

Parameter names must exactly match the `UPPER_CASE` variable names in `model.py`.  If you tune a **base** parameter, derived parameters are automatically recomputed (you don't need to include them).

### 2.4 Model execution settings

```yaml
model:
  result_dir: "optuna_results"       # inside optimizer/
  timeout: 0                         # max seconds per trial (0 = unlimited)
  cleanup_trials: false              # delete trial dirs after evaluation?
  extra_overrides:                   # applied to EVERY trial
    STEPS: 200
    VISUALISATION: false
    SHOW_PLOTS: false
    SAVE_DATA_TO_FILE: false
    SAVE_PICKLE: true                # required — the optimizer reads pickle output
```

> **Tip:** set `SAVE_DATA_TO_FILE: false` unless your objective needs VTK files (e.g., `organoid_error_vtk`).  This skips writing large `.vtk` outputs and speeds up each trial.

---

## 3. Running an optimization

From the **project root**:

```bash
python -m optimizer.optimize --config optimizer/optuna_config.yaml
```

Or from the `optimizer/` directory:

```bash
cd optimizer
python optimize.py --config optuna_config.yaml
```

Each trial prints a one-line summary:

```
  [trial] Running: python model.py --overrides .../trial_00003/overrides.json --result-dir .../trial_00003
  [trial] Finished in 4.2s (exit code 0)
  [trial 3] final_cell_count_error=41.000000 (all: 20.00% off; type 0: 12.50% off)
```

At the end, the best parameters are saved:

```
  Best trial #15
  Objective: final_cell_count_error=0.000000 (0.00% off target)
  Parameters:
    N_CELLS: 3
    CELL_SPEED_REF: 0.573
    ...

Best parameters saved to optimizer/optuna_results/best_params.json
```

### 3.1 Resuming a run

Because Optuna stores all trials in the SQLite database, you can interrupt at any time with **Ctrl+C** and resume later by running the exact same command.  Already-completed trials are reused.

### 3.2 Re-running the model with best parameters

```bash
python model.py --overrides optimizer/optuna_results/best_params.json
```

This applies the optimized parameter values over the model defaults.  You can combine this with any normal model flags.

---

## 4. Multi-objective optimization

To optimize **multiple** objectives simultaneously, use `objectives` (plural) and `directions`:

```yaml
study:
  directions: [minimize, minimize, minimize]
  sampler: NSGA-II

objectives:
  # Reference CSV: compare final n_large_rg_clusters against the value in the file
  - function: rg_rosette_2d_error
    reference: "optimizer/reference_data/target_rg_n_large_clusters.csv"
    kwargs:
      metric: n_large_rg_clusters

  # Time-series CSV: RMSE of mean_rosette_maturity interpolated onto reference time grid
  - function: rg_rosette_2d_error
    reference: "optimizer/reference_data/target_rg_rosette_maturity.csv"
    kwargs:
      metric: mean_rosette_maturity
      smooth_window: 5

  # Inline scalar target: no CSV file needed (see section 7.1)
  - function: rg_rosette_2d_error
    reference: null
    kwargs:
      metric: mean_cluster_compactness
      target_value: 0.75
```

Optuna uses NSGA-II to maintain a **Pareto front** (a set of non-dominated solutions that represent the best trade-offs).  Results are saved to `pareto_trials.json`.

See `optimizer/optuna_config_organoid.yaml` for a full working example.

---

## 5. Live dashboard

Start the Optuna web dashboard to inspect your study interactively:

```bash
python optimizer/dashboard.py
```

This auto-detects the `.db` file and opens a browser at `http://127.0.0.1:8080`.  The dashboard shows:

- Trial history and parameter importance analysis
- Objective value convergence
- Parallel coordinate plots
- Hyperparameter relationships

Options:
```bash
python optimizer/dashboard.py --storage sqlite:///path/to/study.db --port 9090
```

---

## 6. Writing custom objective functions

Create a new function in `optimizer/objectives.py`:

```python
def my_custom_error(results: dict, reference_path: str = None, **kwargs) -> tuple[float, str | None]:
    """Compare some quantity from the simulation against a target."""
    cell_met = results["CELL_METRICS_OVER_TIME"]
    # ... compute your error ...
    return error, None
```

Then register it at the bottom of the file:

```python
OBJECTIVE_REGISTRY = {
    ...
    "my_custom_error": my_custom_error,
}
```

Now you can use `function: my_custom_error` in your YAML config.

### Function signature

| Argument | Type | Description |
|---|---|---|
| `results` | `dict` | Deserialized pickle output (`BPOS_OVER_TIME`, `CELL_METRICS_OVER_TIME`, etc.) |
| `reference_path` | `str` or `None` | Path to a reference CSV (set via `reference:` in YAML) |
| `**kwargs` | | Extra keyword arguments from the `kwargs:` block in YAML |
| `kwargs["trial_dir"]` | `str` | Path to the trial output directory (auto-injected) |

The function must return a tuple `(error, display_text)`.  The optimizer minimizes `error`, while `display_text` is printed in the console summaries.

If you want richer console output, return a display string as the second tuple element:

```python
def my_custom_error(results: dict, reference_path: str = None, **kwargs) -> tuple[float, str | None]:
    ref = _load_reference_csv(reference_path)   # raises if reference_path is None
    target = float(ref["my_metric"].iloc[0])
    simulated = float(results["CELL_METRICS_OVER_TIME"]["my_metric"].iloc[-1])
    error = abs(simulated - target)
    display_text = f"({100.0 * error / abs(target):.2f}% off target)"
    return error, display_text
```

If no extra display text is useful, return `None` as the second element.

---

## 7. Preparing reference data

Reference data lives in `optimizer/reference_data/`.  Each objective function documents the expected CSV format.  Typical examples:

**target_cell_count.csv** (for `final_cell_count_error`):
```csv
cell_type,target_count
-1,30
```
A `cell_type` of `-1` means *total alive cells*.  Use numeric type IDs for per-type targets.

**target_stress_strain.csv** (for `stress_strain_curve_error`):
```csv
strain,stress
0.00,0.0
0.02,0.5
0.05,1.2
...
```

**target_organoid_size.csv** (for `organoid_error`):
```csv
time,target_metric
0,20.0
72000,28.5
144000,36.0
...
```

**target_organoid_sphericity.csv** (for `organoid_error` with `metric: sphericity`):
```csv
time,target_metric
0,1.00
72000,0.97
144000,0.93
...
```

> **Important:** The `time` column in reference CSVs must be in the same
> units as the simulation time — typically **seconds** (matching
> `TIME_STEP`).  Do **not** use hours or other units.

### 7.1 Inline scalar targets (no CSV file needed)

When your target is a single number rather than a measured curve, you can skip
creating a CSV entirely.  Set `reference: null` and add `target_value` to
`kwargs`:

```yaml
- function: rg_rosette_2d_error
  reference: null
  kwargs:
    metric: mean_cluster_compactness
    target_value: 0.75    # desired per-rosette compactness (0=elongated, 1=circular)
```

The optimizer dispatch layer (`optimize.py`) intercepts `target_value`, writes
a temporary single-row CSV of the form:

```csv
mean_cluster_compactness
0.75
```

and passes it as `reference_path` to the objective function — then deletes
the file.  **This works for any registered objective function**, not just
`rg_rosette_2d_error`; no per-function changes are required.

> **Note:** `target_value` always compares the **final** simulation value
> (last row of the metrics DataFrame).  If you need a trajectory target,
> create a time-series CSV instead.

---

## 8. Tips and best practices

1. **Start with few trials** (`n_trials: 20`) and a short simulation (`STEPS: 100`) to verify the pipeline works before scaling up.

2. **Use log-scale sampling** (`log: true`) for parameters that span orders of magnitude (e.g., diffusion coefficients, threshold concentrations).

3. **Seed your runs** (`seed: 42`) for reproducibility.  Remove or set to `null` for production runs that should explore independently.

4. **Check `stderr.log`** in any trial directory if a trial fails (it contains the full model error output).

5. **Normalise multi-target objectives** (`normalize: true`) when comparing targets of different magnitudes.

6. **Avoid tuning derived parameters**: tune the base parameter and let the model's `recompute_derived_params()` handle the rest.

7. **Disk space**: each trial stores a pickle file (~1 MB) and optionally VTK files.  Set `cleanup_trials: true` in the YAML to delete trial directories after evaluation.

---

## Quick-start recipe

```bash
# 1. Activate your environment
conda activate flamegpu_py310

# 2. Run 20 trials with the example config
python -m optimizer.optimize --config optimizer/optuna_config.yaml

# 3. Inspect results
python optimizer/dashboard.py

# 4. Run a production simulation with the best parameters
python model.py --overrides optimizer/optuna_results/best_params.json
```
