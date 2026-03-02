"""
Objective (error) functions for Optuna-based parameter optimization.

Each function takes the simulation results dictionary (as saved in the pickle)
and a reference data source, and returns a scalar error value to minimize.

Add your own objective functions here following the same signature::

    def my_objective(results: dict, reference_path: str, **kwargs) -> float:
        ...
        return error

Then register the function name in the ``OBJECTIVE_REGISTRY`` at the bottom.
"""

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
# Registry — maps string names (used in YAML config) to callables
# ---------------------------------------------------------------------------

OBJECTIVE_REGISTRY = {
    "stress_strain_curve_error": stress_strain_curve_error,
    "boundary_force_curve_error": boundary_force_curve_error,
    "cell_population_error": cell_population_error,
    "focad_attached_ratio_error": focad_attached_ratio_error,
    "poisson_ratio_error": poisson_ratio_error,
    "matrix_remodeling_error": matrix_remodeling_error,
}
