"""
Objective (error) functions for Optuna-based parameter optimization.

Each function takes the simulation results dictionary (as saved in the pickle)
and a reference data source, and returns a tuple
``(error, display_text)`` where ``display_text`` is an optional string used
only for human-readable optimizer output.

Add your own objective functions here following the same signature::

    def my_objective(results: dict, reference_path: str, **kwargs) -> tuple[float, str | None]:
        ...
        return error, None

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


def _format_percent_text(error: float, target: float, label: str = "target") -> str:
    abs_target = abs(float(target))
    if abs_target <= 1e-12:
        return f"({label}~0; % error unavailable)"
    percent_error = 100.0 * abs(float(error)) / abs_target
    return f"({percent_error:.2f}% off {label})"


def _format_percent_detail(label: str, error: float, target: float) -> str:
    abs_target = abs(float(target))
    if abs_target <= 1e-12:
        return f"{label}: target~0"
    percent_error = 100.0 * abs(float(error)) / abs_target
    return f"{label}: {percent_error:.2f}% off"


def _get_boundary_surface_area(bpos: pd.DataFrame, axis: int) -> float:
    ortho_axes = [i for i in range(3) if i != axis]
    lengths = []
    for oa in ortho_axes:
        pos0 = float(bpos.iloc[0, oa * 2])
        neg0 = float(bpos.iloc[0, oa * 2 + 1])
        lengths.append(abs(pos0 - neg0))
    area = lengths[0] * lengths[1]
    if area < 1e-12:
        raise ValueError(
            f"Cross-sectional area is ~0 (ortho lengths: {lengths}). "
            "Check BPOS_OVER_TIME initial boundary positions."
        )
    return area


def _get_attachment_count_series(results: dict, axis: int) -> tuple[pd.Series, pd.Series]:
    counts = results.get("BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME")
    if counts is None or len(counts) == 0:
        raise ValueError(
            "Results must contain non-empty BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME "
            "for stress_area_mode='per_fibre_area'"
        )
    face_keys = [("n_bx_pos", "n_bx_neg"), ("n_by_pos", "n_by_neg"), ("n_bz_pos", "n_bz_neg")]
    pos_col, neg_col = face_keys[axis]
    if pos_col not in counts.columns or neg_col not in counts.columns:
        raise ValueError(f"Attachment count columns '{pos_col}'/'{neg_col}' not found")
    return counts[pos_col].astype(float), counts[neg_col].astype(float)


def _get_fibre_section_area(results: dict, fibre_section_area_um2: float | None = None) -> float:
    area = fibre_section_area_um2 if fibre_section_area_um2 is not None else results.get("FIBRE_SECTION_AREA_UM2")
    if area is None:
        raise ValueError(
            "No fibre section area available. Provide kwargs['fibre_section_area_um2'] "
            "or store FIBRE_SECTION_AREA_UM2 in the pickle."
        )
    area = float(area)
    if area <= 1e-12:
        raise ValueError(f"Invalid fibre section area: {area}")
    return area


def _raise_if_axis_selection_has_no_signal(
    sim_strain: pd.Series,
    sim_stress: pd.Series,
    ref: pd.DataFrame,
    force_type: str,
    strain_axis: int,
) -> None:
    """Fail fast when the chosen force/axis pair yields a flat zero signal.

    This usually indicates a misconfigured objective, e.g. reading the x-axis
    while the assay is actually loading the y-axis.
    """
    sim_strain_arr = np.asarray(sim_strain, dtype=float)
    sim_stress_arr = np.asarray(sim_stress, dtype=float)

    max_abs_sim_strain = float(np.max(np.abs(sim_strain_arr))) if sim_strain_arr.size else 0.0
    max_abs_sim_stress = float(np.max(np.abs(sim_stress_arr))) if sim_stress_arr.size else 0.0
    max_abs_ref_strain = float(np.max(np.abs(ref["strain"].to_numpy(dtype=float)))) if "strain" in ref.columns and len(ref) else 0.0

    ref_cols = [col for col in ("stress", "differential_modulus") if col in ref.columns]
    max_abs_ref_response = 0.0
    if ref_cols and len(ref):
        ref_response = ref[ref_cols[0]].to_numpy(dtype=float)
        max_abs_ref_response = float(np.max(np.abs(ref_response)))

    if (
        max_abs_sim_strain <= 1e-12
        and max_abs_sim_stress <= 1e-12
        and (max_abs_ref_strain > 1e-12 or max_abs_ref_response > 1e-12)
    ):
        raise ValueError(
            "Selected objective signal is identically zero. "
            f"force_type='{force_type}', strain_axis={strain_axis}. "
            "This usually means the objective is reading the wrong loaded axis "
            "or force type for this assay."
        )


# ---------------------------------------------------------------------------
# Built-in objectives
# ---------------------------------------------------------------------------

def _extract_sim_strain_stress(
    results: dict,
    force_type: str = "normal",
    strain_axis: int = 0,
    shear_component: int = 0,
    stress_area_mode: str = "boundary_surface",
    fibre_section_area_um2: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Extract simulated strain [dimensionless] and stress [kPa] from results.

    Stress is computed by dividing a boundary reaction force [nN] by an area [µm²].
    Since 1 nN/µm² = 1 kPa, the returned stress is directly in **kPa**.

    Two area normalization modes are supported:

    1. ``"boundary_surface"``
       Stress is normalized by the geometric surface area of the loaded boundary,
       computed from the *initial* boundary positions stored in
       ``BPOS_OVER_TIME`` (row 0). For a face whose normal is ``strain_axis``,
       this area is the product of the initial lengths along the two orthogonal
       axes.

    2. ``"per_fibre_area"``
       Stress is normalized by the total attached fibre cross-sectional area at
       each boundary, i.e. number of attached fibres multiplied by the fibre
       cross-sectional area. In this mode, the effective area may vary over time
       as boundary attachments change.

    For both normal and shear loading, stress is computed using the reaction
    forces from the two opposite boundaries normal to ``strain_axis``. The final
    stress is obtained from the average of their absolute values, which makes the
    measure symmetric and avoids sign-convention issues.

    Parameters
    ----------
    results : dict
        Simulation output (from pickle). Must contain ``BPOS_OVER_TIME`` and,
        depending on ``force_type``, either ``BFORCE_OVER_TIME`` or
        ``BFORCE_SHEAR_OVER_TIME``.
    force_type : {"normal", "shear"}
        Type of stress to extract.
    strain_axis : int
        Axis normal to the loaded boundary pair:
        0 = x, 1 = y, 2 = z.
    shear_component : int
        For shear only. Selects which of the two tangential directions on the
        chosen face is used:
        0 = first tangential direction, 1 = second tangential direction.
    stress_area_mode : {"boundary_surface", "per_fibre_area"}
        Area normalization mode. Use ``"boundary_surface"`` to divide by the
        geometric boundary area, or ``"per_fibre_area"`` to divide by the total
        attached fibre cross-sectional area.
    fibre_section_area_um2 : float | None
        Fibre cross-sectional area in µm². Only used when
        ``stress_area_mode="per_fibre_area"``. If ``None``, the value is
        obtained through ``_get_fibre_section_area(...)``.

    Returns
    -------
    sim_strain : pd.Series
        Simulated strain [dimensionless].
    sim_stress : pd.Series
        Simulated stress [kPa].
    """
    bpos = results.get("BPOS_OVER_TIME")
    if bpos is None:
        raise ValueError("Results must contain BPOS_OVER_TIME")

    axis = strain_axis

    cross_section_area = _get_boundary_surface_area(bpos, axis)
    count_pos = count_neg = None
    fibre_section_area = None
    if stress_area_mode == "per_fibre_area":
        count_pos, count_neg = _get_attachment_count_series(results, axis)
        fibre_section_area = _get_fibre_section_area(results, fibre_section_area_um2)
    elif stress_area_mode != "boundary_surface":
        raise ValueError("stress_area_mode must be 'boundary_surface' or 'per_fibre_area'")

    if force_type == "shear":
        bforce_shear = results.get("BFORCE_SHEAR_OVER_TIME")
        if bforce_shear is None or len(bforce_shear) == 0:
            raise ValueError(
                "Results must contain non-empty BFORCE_SHEAR_OVER_TIME for force_type='shear'"
            )

        axis_label = ["x", "y", "z"][axis]
        tangent_dirs = [d for d in ["x", "y", "z"] if d != axis_label]
        comp = shear_component

        if comp not in (0, 1):
            raise ValueError("shear_component must be 0 or 1")

        force_col_pos = f"f{axis_label}pos_{tangent_dirs[comp]}"
        force_col_neg = f"f{axis_label}neg_{tangent_dirs[comp]}"

        if force_col_pos not in bforce_shear.columns:
            raise ValueError(
                f"Shear force column '{force_col_pos}' not found in BFORCE_SHEAR_OVER_TIME"
            )
        if force_col_neg not in bforce_shear.columns:
            raise ValueError(
                f"Shear force column '{force_col_neg}' not found in BFORCE_SHEAR_OVER_TIME"
            )

        if stress_area_mode == "per_fibre_area":
            effective_area_pos = count_pos * fibre_section_area
            effective_area_neg = count_neg * fibre_section_area
            safe_pos = effective_area_pos.where(effective_area_pos > 1e-12, np.nan)
            safe_neg = effective_area_neg.where(effective_area_neg > 1e-12, np.nan)

            stress_pos = bforce_shear[force_col_pos].abs() / safe_pos
            stress_neg = bforce_shear[force_col_neg].abs() / safe_neg
            sim_stress = ((stress_pos + stress_neg) / 2.0).fillna(0.0)
        else:
            sim_stress = (
                bforce_shear[force_col_pos].abs() + bforce_shear[force_col_neg].abs()
            ) / (2.0 * cross_section_area)

        tang_axis_idx = ["x", "y", "z"].index(tangent_dirs[comp])
        pos_col_tang = bpos.columns[tang_axis_idx * 2]
        neg_col_tang = bpos.columns[tang_axis_idx * 2 + 1]
        pos_col_norm = bpos.columns[axis * 2]
        neg_col_norm = bpos.columns[axis * 2 + 1]

        L0_normal = bpos.iloc[0][pos_col_norm] - bpos.iloc[0][neg_col_norm]
        tang_disp = (bpos[pos_col_tang] - bpos[neg_col_tang]) - (
            bpos.iloc[0][pos_col_tang] - bpos.iloc[0][neg_col_tang]
        )
        sim_strain = tang_disp / L0_normal if abs(L0_normal) > 1e-12 else tang_disp * 0.0

    else:
        bforce = results.get("BFORCE_OVER_TIME")
        if bforce is None:
            raise ValueError("Results must contain BFORCE_OVER_TIME")

        pos_col = bpos.columns[axis * 2]
        neg_col = bpos.columns[axis * 2 + 1]
        force_col_pos = bforce.columns[axis * 2]
        force_col_neg = bforce.columns[axis * 2 + 1]

        L0 = bpos.iloc[0][pos_col] - bpos.iloc[0][neg_col]
        sim_strain = ((bpos[pos_col] - bpos[neg_col]) - L0) / L0

        if stress_area_mode == "per_fibre_area":
            effective_area_pos = count_pos * fibre_section_area
            effective_area_neg = count_neg * fibre_section_area
            safe_pos = effective_area_pos.where(effective_area_pos > 1e-12, np.nan)
            safe_neg = effective_area_neg.where(effective_area_neg > 1e-12, np.nan)

            stress_pos = bforce[force_col_pos].abs() / safe_pos
            stress_neg = bforce[force_col_neg].abs() / safe_neg
            sim_stress = ((stress_pos + stress_neg) / 2.0).fillna(0.0)
        else:
            sim_stress = (
                bforce[force_col_pos].abs() + bforce[force_col_neg].abs()
            ) / (2.0 * cross_section_area)

    return sim_strain, sim_stress


def stress_strain_curve_error(results: dict, reference_path: str, **kwargs) -> float:
    """Compare the simulated stress-strain curve with a reference CSV.

    Units
    -----
    - **strain** [dimensionless]: engineering strain (normal) or shear strain.
    - **stress** [kPa]: boundary reaction force [nN] / face area [µm²].
      (1 nN/µm² = 1 kPa.)

    The reference CSV must have columns ``strain`` (dimensionless) and
    ``stress`` (kPa).

    **Force type** — controlled by ``kwargs["force_type"]``:
        - ``"normal"`` (default): uses ``BFORCE_OVER_TIME`` (boundary normal
          force) and computes engineering strain from boundary positions.
        - ``"shear"``: uses ``BFORCE_SHEAR_OVER_TIME`` (boundary shear force)
          and computes shear strain from boundary positions.  For shear, the
          *strain axis* selects the face pair whose tangential force is read,
          and *shear_component* (0 or 1) picks which tangential direction.

    **Strain direction** defaults to axis 0 (x); override with
    ``kwargs["strain_axis"]`` (0=x, 1=y, 2=z).

    **Shear component** (only for ``force_type="shear"``):
        ``kwargs["shear_component"]`` — 0 or 1, selects which of the two
        tangential directions on the chosen face.  Default 0.

    **Error weighting**: ``kwargs["strain_weight"]`` (default 0.1) controls
    the relative weight of the strain term vs the stress term.
    """
    ref = _load_reference_csv(reference_path)
    if "strain" not in ref.columns or "stress" not in ref.columns:
        raise ValueError("Reference CSV must contain 'strain' and 'stress' columns")

    force_type = kwargs.get("force_type", "normal")
    axis = kwargs.get("strain_axis", 0)
    strain_weight = float(kwargs.get("strain_weight", 0.1))

    sim_strain, sim_stress = _extract_sim_strain_stress(
        results, force_type=force_type, strain_axis=axis,
        shear_component=kwargs.get("shear_component", 0),
        stress_area_mode=kwargs.get("stress_area_mode", "boundary_surface"),
        fibre_section_area_um2=kwargs.get("fibre_section_area_um2"),
    )
    _raise_if_axis_selection_has_no_signal(sim_strain, sim_stress, ref, force_type, axis)

    # Interpolate simulation to match reference length
    sim_strain_interp = _interpolate_to_match(sim_strain, len(ref))
    sim_stress_interp = _interpolate_to_match(sim_stress, len(ref))

    strain_err = _mse(sim_strain_interp, ref["strain"].values)
    stress_err = _mse(sim_stress_interp, ref["stress"].values)

    # Combined error (stress dominates since that's what we're fitting)
    return float(stress_err + strain_weight * strain_err), None


def shear_stress_strain_curve_error(results: dict, reference_path: str, **kwargs) -> float:
    """Convenience wrapper: calls ``stress_strain_curve_error`` with
    ``force_type='shear'``.  All other kwargs are forwarded."""
    kwargs.setdefault("force_type", "shear")
    return stress_strain_curve_error(results, reference_path, **kwargs)


def differential_modulus_error(results: dict, reference_path: str, **kwargs) -> float:
    """Compare the simulated differential modulus vs strain with a reference CSV.

    The *differential modulus* K(ε) = dσ/dε is the local slope of the
    stress-strain curve.  Many experimental papers report K vs ε instead of
    the raw σ–ε curve (e.g. Steinwachs et al., Nat. Methods 2016).

    Units
    -----
    - **strain** [dimensionless]
    - **differential_modulus** [kPa]:  dσ[kPa] / dε[-] = [kPa].

    The reference CSV must have columns ``strain`` (dimensionless) and
    ``differential_modulus`` (kPa).

    **Force type** — same as ``stress_strain_curve_error``:
        ``kwargs["force_type"]``: ``"normal"`` (default) or ``"shear"``.

    **Strain direction**: ``kwargs["strain_axis"]`` (0=x, 1=y, 2=z).

    **Shear component** (shear only): ``kwargs["shear_component"]`` (0 or 1).

    **Smoothing**: ``kwargs["smooth_window"]`` (int, default 5) — Savitzky-Golay
    window length for smoothing the numerical derivative.  Set to 0 or 1 to
    disable smoothing.  ``kwargs["smooth_polyorder"]`` (int, default 2) sets
    the polynomial order.

    **Error weighting**: ``kwargs["strain_weight"]`` (default 0.1) — relative
    weight of the strain-alignment MSE term.
    """
    ref = _load_reference_csv(reference_path)
    if "strain" not in ref.columns or "differential_modulus" not in ref.columns:
        raise ValueError(
            "Reference CSV must contain 'strain' and 'differential_modulus' columns"
        )

    force_type = kwargs.get("force_type", "normal")
    axis = kwargs.get("strain_axis", 0)
    strain_weight = float(kwargs.get("strain_weight", 0.0))

    sim_strain, sim_stress = _extract_sim_strain_stress(
        results, force_type=force_type, strain_axis=axis,
        shear_component=kwargs.get("shear_component", 0),
        stress_area_mode=kwargs.get("stress_area_mode", "boundary_surface"),
        fibre_section_area_um2=kwargs.get("fibre_section_area_um2"),
    )
    _raise_if_axis_selection_has_no_signal(sim_strain, sim_stress, ref, force_type, axis)

    # Convert to numpy
    strain_arr = sim_strain.values.astype(float)
    stress_arr = sim_stress.values.astype(float)

    # Numerical derivative  dσ/dε  (central differences where possible)
    d_stress = np.gradient(stress_arr)
    d_strain = np.gradient(strain_arr)
    # Avoid division by zero
    safe_mask = np.abs(d_strain) > 1e-15
    sim_K = np.zeros_like(stress_arr)
    sim_K[safe_mask] = d_stress[safe_mask] / d_strain[safe_mask]

    # Optional Savitzky-Golay smoothing
    smooth_window = int(kwargs.get("smooth_window", 5))
    smooth_polyorder = int(kwargs.get("smooth_polyorder", 2))
    if smooth_window > 2 and len(sim_K) >= smooth_window:
        from scipy.signal import savgol_filter
        # Window must be odd
        if smooth_window % 2 == 0:
            smooth_window += 1
        sim_K = savgol_filter(sim_K, smooth_window, min(smooth_polyorder, smooth_window - 1))

    # Interpolate to match reference length
    sim_strain_interp = _interpolate_to_match(pd.Series(strain_arr), len(ref))
    sim_K_interp = _interpolate_to_match(pd.Series(sim_K), len(ref))

    strain_err = _mse(sim_strain_interp, ref["strain"].values)
    K_err = _mse(sim_K_interp, ref["differential_modulus"].values)

    return float(K_err + strain_weight * strain_err), None


def shear_differential_modulus_error(results: dict, reference_path: str, **kwargs) -> float:
    """Convenience wrapper: calls ``differential_modulus_error`` with
    ``force_type='shear'``.  All other kwargs are forwarded."""
    kwargs.setdefault("force_type", "shear")
    return differential_modulus_error(results, reference_path, **kwargs)


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
    return total_error / n_cols, None


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
    return _rmse(sim_alive, ref["n_cells_alive"].values), None


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
    return _rmse(sim_ratio, ref["attached_ratio"].values), None


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
        error = abs(sim_final - target)
        return error, _format_percent_text(error, target)

    if reference_path:
        ref = _load_reference_csv(reference_path)
        if "poisson_ratio" not in ref.columns:
            raise ValueError("Reference CSV must contain 'poisson_ratio' column")
        sim_interp = _interpolate_to_match(pr.iloc[:, 0] if hasattr(pr, "iloc") else pr, len(ref))
        return _rmse(sim_interp, ref["poisson_ratio"].values), None

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
    return total_error / n_cols, None


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
        error = abs(final_alive - target)
        return error, _format_percent_text(error, target)

    # --- Option B: reference CSV (may include per-type targets) ---
    if reference_path is None:
        raise ValueError("Provide either reference_path or target_cell_count kwarg")

    ref = _load_reference_csv(reference_path)
    if "target_count" not in ref.columns:
        raise ValueError("Reference CSV must contain a 'target_count' column")

    normalize = kwargs.get("normalize", False)
    total_error = 0.0
    n_targets = 0
    display_parts = []

    for _, row in ref.iterrows():
        ct = row.get("cell_type", -1)
        # -1 or "all" → compare total alive
        if ct == -1 or str(ct).strip().lower() == "all":
            target = float(row["target_count"])
            error = abs(final_alive - target)
            total_error += error
            n_targets += 1
            display_parts.append(_format_percent_detail("all", error, target))
        else:
            # Per-type columns are only present when the simulation exports
            # them (VTK path with per-type counting).  The column convention
            # is  n_alive_type_<N>.
            col_name = f"n_alive_type_{int(ct)}"
            if col_name in cell_met.columns:
                sim_val = int(cell_met[col_name].iloc[-1])
                target = float(row["target_count"])
                error = abs(sim_val - target)
                total_error += error
                n_targets += 1
                display_parts.append(_format_percent_detail(f"type {int(ct)}", error, target))
            else:
                # Silently skip types we can't measure (avoids hard crash when
                # per-type tracking is not available).
                pass

    if n_targets == 0:
        raise ValueError("No usable target rows found in reference CSV")

    display_text = f"({'; '.join(display_parts)})" if display_parts else None
    return (total_error / n_targets if normalize else total_error), display_text


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
        error = abs(sim_focad_per_cell - target)
        return error, _format_percent_text(error, target)

    # --- Option B: reference CSV ---
    if reference_path is None:
        raise ValueError("Provide either reference_path or target_focad_per_cell kwarg")

    ref = _load_reference_csv(reference_path)
    if "target_focad_per_cell" not in ref.columns:
        raise ValueError("Reference CSV must contain 'target_focad_per_cell' column")

    target = float(ref["target_focad_per_cell"].iloc[0])
    error = abs(sim_focad_per_cell - target)
    return error, _format_percent_text(error, target)


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
        error = abs(sim_value - target)
        return error, _format_percent_text(error, target)

    # --- Option B: reference CSV ---
    if reference_path is None:
        raise ValueError("Provide either reference_path or target_size kwarg")

    ref = _load_reference_csv(reference_path)
    if "target_size" not in ref.columns:
        raise ValueError("Reference CSV must contain a 'target_size' column")

    target = float(ref["target_size"].iloc[0])
    error = abs(sim_value - target)
    return error, _format_percent_text(error, target)


# ---------------------------------------------------------------------------
# Registry — maps string names (used in YAML config) to callables
# ---------------------------------------------------------------------------

OBJECTIVE_REGISTRY = {
    "stress_strain_curve_error": stress_strain_curve_error,
    "shear_stress_strain_curve_error": shear_stress_strain_curve_error,
    "differential_modulus_error": differential_modulus_error,
    "shear_differential_modulus_error": shear_differential_modulus_error,
    "boundary_force_curve_error": boundary_force_curve_error,
    "cell_population_error": cell_population_error,
    "focad_attached_ratio_error": focad_attached_ratio_error,
    "poisson_ratio_error": poisson_ratio_error,
    "matrix_remodeling_error": matrix_remodeling_error,
    "final_cell_count_error": final_cell_count_error,
    "final_focad_per_cell_error": final_focad_per_cell_error,
    "organoid_size_error": organoid_size_error,
}
