"""
Objective (error) functions for Optuna-based parameter optimization.

Each function takes the simulation results dictionary (as saved in the pickle)
and a reference data source, and returns a scalar error value to minimize.

Add your own objective functions here following the same signature::

    def my_objective(results: dict, reference_path: str, **kwargs) -> float:
        ...
        return error

Then register the function name in the ``OBJECTIVE_REGISTRY`` at the bottom.

For objectives that need cell spatial data (e.g., organoid size) the function
receives an extra ``trial_dir`` kwarg pointing to the trial output directory
where VTK files are stored.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_reference_csv(path: str) -> pd.DataFrame:
    """Load a reference CSV.  Expects at least two columns: 'time' and the
    quantity being compared."""
    return pd.read_csv(path)


def _mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean-squared error between two 1-D arrays of equal length."""
    n = min(len(predicted), len(target))
    return float(np.mean((predicted[:n] - target[:n]) ** 2))


def _rmse(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(_mse(predicted, target)))


def _interpolate_to_match(series: pd.Series, target_len: int) -> np.ndarray:
    """Linearly interpolate *series* so it has exactly *target_len* points."""
    x_old = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, series.values)


# ---------------------------------------------------------------------------
# Built-in objectives
# ---------------------------------------------------------------------------

def stress_strain_curve_error(results: dict, reference_path: str, **kwargs) -> float:
    """Compare the simulated stress-strain curve with a reference CSV.

    The reference CSV must have columns ``strain`` and ``stress``.
    The simulated curve is built from ``BPOS_OVER_TIME`` (strain) and
    ``BFORCE_OVER_TIME`` (stress proxy = boundary normal force).

    Strain direction defaults to the first POISSON_DIRS axis; override with
    ``kwargs["strain_axis"]`` (0=x, 1=y, 2=z).
    """
    ref = _load_reference_csv(reference_path)
    if "strain" not in ref.columns or "stress" not in ref.columns:
        raise ValueError("Reference CSV must contain 'strain' and 'stress' columns")

    bpos = results.get("BPOS_OVER_TIME")
    bforce = results.get("BFORCE_OVER_TIME")
    if bpos is None or bforce is None:
        raise ValueError("Results must contain BPOS_OVER_TIME and BFORCE_OVER_TIME")

    axis = kwargs.get("strain_axis", 0)
    pos_col = bpos.columns[axis * 2]
    neg_col = bpos.columns[axis * 2 + 1]
    force_col = bforce.columns[axis * 2]

    # Engineering strain
    L0 = bpos.iloc[0][pos_col] - bpos.iloc[0][neg_col]
    sim_strain = ((bpos[pos_col] - bpos[neg_col]) - L0) / L0
    sim_stress = bforce[force_col]

    # Interpolate simulation to match reference length
    sim_strain_interp = _interpolate_to_match(sim_strain, len(ref))
    sim_stress_interp = _interpolate_to_match(sim_stress, len(ref))

    strain_err = _mse(sim_strain_interp, ref["strain"].values)
    stress_err = _mse(sim_stress_interp, ref["stress"].values)

    # Combined error (stress dominates since that's what we're fitting)
    return float(stress_err + 0.1 * strain_err)


def boundary_force_curve_error(results: dict, reference_path: str, **kwargs) -> float:
    """Compare boundary normal forces over time with a reference CSV.

    Reference CSV must have a ``time`` column and one or more of:
    ``fxpos``, ``fxneg``, ``fypos``, ``fyneg``, ``fzpos``, ``fzneg``.
    """
    ref = _load_reference_csv(reference_path)
    bforce = results.get("BFORCE_OVER_TIME")
    if bforce is None or len(bforce) == 0:
        raise ValueError("Results must contain non-empty BFORCE_OVER_TIME")

    total_error = 0.0
    n_cols = 0
    for col in ["fxpos", "fxneg", "fypos", "fyneg", "fzpos", "fzneg"]:
        if col in ref.columns and col in bforce.columns:
            sim_vals = _interpolate_to_match(bforce[col], len(ref))
            total_error += _mse(sim_vals, ref[col].values)
            n_cols += 1

    if n_cols == 0:
        raise ValueError("No matching force columns found between reference and results")
    return total_error / n_cols


def cell_population_error(results: dict, reference_path: str, **kwargs) -> float:
    """Compare the cell population time-series with a reference CSV.

    Reference CSV must have columns ``time`` and ``n_cells_alive``.
    """
    ref = _load_reference_csv(reference_path)
    cell_met = results.get("CELL_METRICS_OVER_TIME")
    if cell_met is None or len(cell_met) == 0:
        raise ValueError("Results must contain non-empty CELL_METRICS_OVER_TIME")
    if "n_cells_alive" not in ref.columns:
        raise ValueError("Reference CSV must contain 'n_cells_alive' column")

    sim_alive = _interpolate_to_match(cell_met["n_cells_alive"], len(ref))
    return _rmse(sim_alive, ref["n_cells_alive"].values)


def focad_attached_ratio_error(results: dict, reference_path: str, **kwargs) -> float:
    """Compare FOCAD attached ratio over time with a reference CSV.

    Reference CSV must have columns ``time`` and ``attached_ratio``.
    """
    ref = _load_reference_csv(reference_path)
    focad_met = results.get("FOCAD_METRICS_OVER_TIME")
    if focad_met is None or len(focad_met) == 0:
        raise ValueError("Results must contain non-empty FOCAD_METRICS_OVER_TIME")
    if "attached_ratio" not in ref.columns:
        raise ValueError("Reference CSV must contain 'attached_ratio' column")

    sim_ratio = _interpolate_to_match(focad_met["attached_ratio"], len(ref))
    return _rmse(sim_ratio, ref["attached_ratio"].values)


def poisson_ratio_error(results: dict, reference_path: str = None, **kwargs) -> float:
    """Match the final Poisson ratio to a scalar target.

    Either provide ``reference_path`` to a CSV with a ``poisson_ratio`` column,
    or pass ``kwargs["target_poisson"]`` as a scalar.
    """
    pr = results.get("POISSON_RATIO_OVER_TIME")
    if pr is None or len(pr) == 0:
        raise ValueError("Results must contain POISSON_RATIO_OVER_TIME")

    if "target_poisson" in kwargs:
        target = float(kwargs["target_poisson"])
        sim_final = float(pr.iloc[-1])
        return abs(sim_final - target)

    if reference_path:
        ref = _load_reference_csv(reference_path)
        if "poisson_ratio" not in ref.columns:
            raise ValueError("Reference CSV must contain 'poisson_ratio' column")
        sim_interp = _interpolate_to_match(pr.iloc[:, 0] if hasattr(pr, "iloc") else pr, len(ref))
        return _rmse(sim_interp, ref["poisson_ratio"].values)

    raise ValueError("Provide either reference_path or target_poisson kwarg")


def matrix_remodeling_error(results: dict, reference_path: str, **kwargs) -> float:
    """Compare FNODE (fibre network) matrix remodeling metrics with reference.

    Reference CSV should have ``time`` and one or more of:
    ``n_fnodes_total``, ``sum_degradation``, ``sum_reinforcement``,
    ``mean_elastic_energy``, ``net_remodeling_total``.
    """
    ref = _load_reference_csv(reference_path)
    fnode_met = results.get("FNODE_METRICS_OVER_TIME")
    if fnode_met is None or len(fnode_met) == 0:
        raise ValueError("Results must contain non-empty FNODE_METRICS_OVER_TIME")

    total_error = 0.0
    n_cols = 0
    compare_cols = ["n_fnodes_total", "sum_degradation", "sum_reinforcement",
                    "mean_elastic_energy", "net_remodeling_total"]
    for col in compare_cols:
        if col in ref.columns and col in fnode_met.columns:
            sim_vals = _interpolate_to_match(fnode_met[col], len(ref))
            total_error += _rmse(sim_vals, ref[col].values)
            n_cols += 1

    if n_cols == 0:
        raise ValueError("No matching remodeling columns found between reference and results")
    return total_error / n_cols


# ---------------------------------------------------------------------------
# Simple scalar objectives
# ---------------------------------------------------------------------------

def final_cell_count_error(results: dict, reference_path: str = None, **kwargs) -> float:
    """Compare the final cell count (total alive) against a scalar target.

    Supports per-cell-type targets when VTK data is available.

    **Option A — scalar target (no reference CSV):**
        Pass ``kwargs["target_cell_count"]`` (int/float).
        Returns absolute difference.

    **Option B — reference CSV:**
        The CSV must have a ``cell_type`` column and a ``target_count`` column.
        Each row defines the expected final alive-count for that cell type.
        A row with ``cell_type == -1`` (or ``"all"``) targets total alive count.

        Example CSV::

            cell_type,target_count
            -1,15
            0,10
            1,5

    Error is the sum of absolute differences (optionally normalised by the
    number of targets).  Set ``kwargs["normalize"] = True`` to divide by the
    number of target rows.
    """
    cell_met = results.get("CELL_METRICS_OVER_TIME")
    if cell_met is None or len(cell_met) == 0:
        raise ValueError("Results must contain non-empty CELL_METRICS_OVER_TIME")

    final_alive = int(cell_met["n_cells_alive"].iloc[-1])

    # --- Option A: scalar target ---
    if "target_cell_count" in kwargs:
        target = float(kwargs["target_cell_count"])
        return abs(final_alive - target)

    # --- Option B: reference CSV (may include per-type targets) ---
    if reference_path is None:
        raise ValueError("Provide either reference_path or target_cell_count kwarg")

    ref = _load_reference_csv(reference_path)
    if "target_count" not in ref.columns:
        raise ValueError("Reference CSV must contain a 'target_count' column")

    normalize = kwargs.get("normalize", False)
    total_error = 0.0
    n_targets = 0

    for _, row in ref.iterrows():
        ct = row.get("cell_type", -1)
        # -1 or "all" → compare total alive
        if ct == -1 or str(ct).strip().lower() == "all":
            total_error += abs(final_alive - float(row["target_count"]))
            n_targets += 1
        else:
            # Per-type columns are only present when the simulation exports
            # them (VTK path with per-type counting).  The column convention
            # is  n_alive_type_<N>.
            col_name = f"n_alive_type_{int(ct)}"
            if col_name in cell_met.columns:
                sim_val = int(cell_met[col_name].iloc[-1])
                total_error += abs(sim_val - float(row["target_count"]))
                n_targets += 1
            else:
                # Silently skip types we can't measure (avoids hard crash when
                # per-type tracking is not available).
                pass

    if n_targets == 0:
        raise ValueError("No usable target rows found in reference CSV")

    return total_error / n_targets if normalize else total_error


def final_focad_per_cell_error(results: dict, reference_path: str = None, **kwargs) -> float:
    """Compare the final average number of focal adhesions per alive cell.

    **Option A — scalar target:**
        Pass ``kwargs["target_focad_per_cell"]`` (float).
        Returns absolute difference.

    **Option B — reference CSV:**
        CSV with a single row containing ``target_focad_per_cell`` column.
    """
    cell_met = results.get("CELL_METRICS_OVER_TIME")
    focad_met = results.get("FOCAD_METRICS_OVER_TIME")
    if cell_met is None or len(cell_met) == 0:
        raise ValueError("Results must contain non-empty CELL_METRICS_OVER_TIME")
    if focad_met is None or len(focad_met) == 0:
        raise ValueError("Results must contain non-empty FOCAD_METRICS_OVER_TIME")

    final_alive = int(cell_met["n_cells_alive"].iloc[-1])
    final_total_focad = int(focad_met["total"].iloc[-1])
    sim_focad_per_cell = (final_total_focad / final_alive) if final_alive > 0 else 0.0

    # --- Option A: scalar target ---
    if "target_focad_per_cell" in kwargs:
        target = float(kwargs["target_focad_per_cell"])
        return abs(sim_focad_per_cell - target)

    # --- Option B: reference CSV ---
    if reference_path is None:
        raise ValueError("Provide either reference_path or target_focad_per_cell kwarg")

    ref = _load_reference_csv(reference_path)
    if "target_focad_per_cell" not in ref.columns:
        raise ValueError("Reference CSV must contain 'target_focad_per_cell' column")

    target = float(ref["target_focad_per_cell"].iloc[0])
    return abs(sim_focad_per_cell - target)


# ---------------------------------------------------------------------------
# VTK spatial helpers (for organoid / spheroid size measurements)
# ---------------------------------------------------------------------------

def _read_cell_positions_from_vtk(vtk_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read cell positions, ``dead`` flags and ``cell_type`` from a CELL VTK file.

    Returns
    -------
    positions : ndarray, shape (n_cells, 3)
        XYZ of each cell (only the cell-centre points, not anchor points).
    dead : ndarray, shape (n_cells,)
        0 = alive, != 0 = dead.
    cell_type : ndarray, shape (n_cells,)
        Integer cell-type label.
    """
    text = vtk_path.read_text()
    lines = text.splitlines()

    # --- Number of unique cells from POINT_DATA id scalars ---
    pd_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("POINT_DATA"))
    n_total_points = int(lines[pd_idx].split()[1])

    # Read the "id" scalar to figure out how many unique cells there are
    id_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("SCALARS id"))
    id_start = id_idx + 2  # skip SCALARS + LOOKUP_TABLE
    ids: list[int] = []
    for i in range(id_start, id_start + n_total_points):
        ids.append(int(float(lines[i].strip().split()[0])))
    # The first occurrence of each id corresponds to the cell centre
    seen: set[int] = set()
    cell_indices: list[int] = []
    for i, cid in enumerate(ids):
        if cid not in seen:
            seen.add(cid)
            cell_indices.append(i)
    n_cells = len(cell_indices)

    # --- Read positions (POINTS section) ---
    pts_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("POINTS"))
    pts_start = pts_idx + 1
    all_xyz: list[list[float]] = []
    for i in range(pts_start, pts_start + n_total_points):
        parts = lines[i].strip().split()
        all_xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
    all_xyz_arr = np.array(all_xyz)
    positions = all_xyz_arr[cell_indices]

    # --- Read dead scalar ---
    def _read_int_scalar(name: str) -> np.ndarray:
        idx = next((i for i, l in enumerate(lines) if l.strip().startswith(f"SCALARS {name}")), None)
        if idx is None:
            return np.zeros(n_cells, dtype=int)
        start = idx + 2
        vals = [int(float(lines[start + j].strip().split()[0])) for j in range(n_total_points)]
        return np.array([vals[ci] for ci in cell_indices], dtype=int)

    dead = _read_int_scalar("dead")
    cell_type = _read_int_scalar("cell_type")

    return positions, dead, cell_type


def _get_last_cell_vtk(trial_dir: str) -> Path:
    """Find the highest-timestep cells_tXXXX.vtk in *trial_dir*."""
    d = Path(trial_dir)
    vtks = sorted(d.glob("cells_t*.vtk"))
    if not vtks:
        raise FileNotFoundError(f"No cells_t*.vtk found in {trial_dir}")
    return vtks[-1]


# ---------------------------------------------------------------------------
# Organoid / spheroid size metrics
# ---------------------------------------------------------------------------

def compute_organoid_metrics(positions: np.ndarray) -> dict[str, float]:
    """Compute simple size metrics for a 3-D point cloud (alive-cell centres).

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        XYZ coordinates of alive cells.

    Returns
    -------
    dict with keys:
        ``radius_of_gyration`` : float
            RMS distance from the centroid, Rg = sqrt(mean(|r_i - r̄|²)).
            This is the simplest and most robust spheroid size metric.
        ``max_span`` : float
            Maximum Euclidean distance between any pair of cells (diameter).
        ``equivalent_sphere_radius`` : float
            Radius of a uniform sphere that has the same Rg:
            R_eq = Rg * sqrt(5/3).
        ``centroid`` : list[float]
            [x, y, z] centroid of the alive-cell cloud.
        ``n_alive`` : int
            Number of cells used in the computation.
    """
    n = len(positions)
    if n == 0:
        return {
            "radius_of_gyration": 0.0,
            "max_span": 0.0,
            "equivalent_sphere_radius": 0.0,
            "centroid": [0.0, 0.0, 0.0],
            "n_alive": 0,
        }

    centroid = positions.mean(axis=0)
    displacements = positions - centroid
    sq_dists = np.sum(displacements ** 2, axis=1)
    rg = float(np.sqrt(np.mean(sq_dists)))

    # Max span (diameter) — O(n²) but fine for small cell counts (<10k)
    if n <= 5000:
        from scipy.spatial.distance import pdist
        max_span = float(pdist(positions).max()) if n > 1 else 0.0
    else:
        # For very large clouds, approximate from PCA principal axis
        eigvals = np.linalg.eigvalsh(np.cov(displacements.T))
        max_span = 2.0 * float(np.sqrt(max(eigvals))) * 2.0  # rough 2-sigma diameter

    equivalent_r = rg * np.sqrt(5.0 / 3.0)

    return {
        "radius_of_gyration": rg,
        "max_span": max_span,
        "equivalent_sphere_radius": float(equivalent_r),
        "centroid": centroid.tolist(),
        "n_alive": n,
    }


def organoid_size_error(results: dict, reference_path: str = None, **kwargs) -> float:
    """Compare the spheroid size at the final timestep against a target.

    The size is measured from the point cloud of **alive** cell positions
    obtained from VTK output files.

    **Requires** ``SAVE_DATA_TO_FILE: true`` in model overrides so that
    ``cells_tXXXX.vtk`` files are produced.

    **Metric selection** via ``kwargs["metric"]``:
        - ``"radius_of_gyration"`` (default) — RMS distance from centroid
        - ``"max_span"`` — maximum inter-cell distance (diameter)
        - ``"equivalent_sphere_radius"`` — Rg * sqrt(5/3)

    **Option A — scalar target:**
        Pass ``kwargs["target_size"]`` (float, in model length units [µm]).

    **Option B — reference CSV:**
        CSV with a ``target_size`` column (single row).
    """
    trial_dir = kwargs.get("trial_dir")
    if trial_dir is None:
        raise ValueError(
            "organoid_size_error requires 'trial_dir' kwarg "
            "(automatically provided by the optimizer). "
            "Make sure SAVE_DATA_TO_FILE is true."
        )

    vtk_path = _get_last_cell_vtk(trial_dir)
    positions, dead, _ = _read_cell_positions_from_vtk(vtk_path)

    # Keep only alive cells
    alive_mask = dead == 0
    alive_pos = positions[alive_mask]

    metrics = compute_organoid_metrics(alive_pos)

    metric_name = kwargs.get("metric", "radius_of_gyration")
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric '{metric_name}'. Choose from: {list(metrics.keys())}")
    sim_value = metrics[metric_name]

    # --- Option A: scalar target ---
    if "target_size" in kwargs:
        target = float(kwargs["target_size"])
        return abs(sim_value - target)

    # --- Option B: reference CSV ---
    if reference_path is None:
        raise ValueError("Provide either reference_path or target_size kwarg")

    ref = _load_reference_csv(reference_path)
    if "target_size" not in ref.columns:
        raise ValueError("Reference CSV must contain a 'target_size' column")

    target = float(ref["target_size"].iloc[0])
    return abs(sim_value - target)


# ---------------------------------------------------------------------------
# Registry — maps string names (used in YAML config) to callables
# ---------------------------------------------------------------------------

OBJECTIVE_REGISTRY = {
    "stress_strain_curve_error": stress_strain_curve_error,
    "boundary_force_curve_error": boundary_force_curve_error,
    "cell_population_error": cell_population_error,
    "focad_attached_ratio_error": focad_attached_ratio_error,
    "poisson_ratio_error": poisson_ratio_error,
    "matrix_remodeling_error": matrix_remodeling_error,
    "final_cell_count_error": final_cell_count_error,
    "final_focad_per_cell_error": final_focad_per_cell_error,
    "organoid_size_error": organoid_size_error,
}
