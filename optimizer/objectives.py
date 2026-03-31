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


def _prepare_curve_for_interpolation(
    source_x: pd.Series | np.ndarray,
    source_y: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_x_arr = np.asarray(source_x, dtype=float)
    source_y_arr = np.asarray(source_y, dtype=float)

    if source_x_arr.shape != source_y_arr.shape:
        raise ValueError("source_x and source_y must have the same shape")
    if source_x_arr.ndim != 1:
        raise ValueError("source_x and source_y must be 1-D")

    finite_mask = np.isfinite(source_x_arr) & np.isfinite(source_y_arr)
    source_x_arr = source_x_arr[finite_mask]
    source_y_arr = source_y_arr[finite_mask]
    if source_x_arr.size == 0:
        raise ValueError("No finite points available for interpolation")

    order = np.argsort(source_x_arr, kind="stable")
    source_x_sorted = source_x_arr[order]
    source_y_sorted = source_y_arr[order]

    unique_x, inverse = np.unique(source_x_sorted, return_inverse=True)
    if unique_x.size != source_x_sorted.size:
        y_sum = np.zeros_like(unique_x)
        y_count = np.zeros_like(unique_x)
        np.add.at(y_sum, inverse, source_y_sorted)
        np.add.at(y_count, inverse, 1.0)
        source_y_sorted = y_sum / y_count
        source_x_sorted = unique_x

    return source_x_sorted, source_y_sorted


def _interpolate_response_to_reference_x(
    source_x: pd.Series | np.ndarray,
    source_y: pd.Series | np.ndarray,
    target_x: pd.Series | np.ndarray,
) -> np.ndarray:
    prepared_x, prepared_y = _prepare_curve_for_interpolation(source_x, source_y)
    target_x_arr = np.asarray(target_x, dtype=float)

    interp = np.full_like(target_x_arr, np.nan, dtype=float)
    finite_target = np.isfinite(target_x_arr)
    overlap_mask = (
        finite_target
        & (target_x_arr >= prepared_x[0])
        & (target_x_arr <= prepared_x[-1])
    )
    if np.any(overlap_mask):
        interp[overlap_mask] = np.interp(target_x_arr[overlap_mask], prepared_x, prepared_y)
    return interp


def _compute_reference_grid_coverage_error(
    source_x: pd.Series | np.ndarray,
    target_x: pd.Series | np.ndarray,
) -> float:
    prepared_x, _ = _prepare_curve_for_interpolation(source_x, source_x)
    target_x_arr = np.asarray(target_x, dtype=float)
    finite_target = target_x_arr[np.isfinite(target_x_arr)]
    if finite_target.size == 0:
        return 0.0

    below = np.clip(prepared_x[0] - finite_target, 0.0, None)
    above = np.clip(finite_target - prepared_x[-1], 0.0, None)
    return float(np.mean((below + above) ** 2))


def _filter_simulation_from_min_strain(
    strain: pd.Series | np.ndarray,
    *series: pd.Series | np.ndarray,
    min_sim_strain: float | None = None,
) -> tuple[np.ndarray, ...]:
    strain_arr = np.asarray(strain, dtype=float)
    mask = np.isfinite(strain_arr)
    if min_sim_strain is not None:
        mask &= strain_arr >= float(min_sim_strain)

    filtered = [strain_arr[mask]]
    for values in series:
        values_arr = np.asarray(values, dtype=float)
        if values_arr.shape != strain_arr.shape:
            raise ValueError("Simulation series must match the strain shape")
        filtered.append(values_arr[mask])

    if filtered[0].size == 0:
        raise ValueError(
            "No simulation points remain after applying the minimum strain cutoff"
        )

    return tuple(filtered)


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


def _relative_abs_error(predicted: float, target: float) -> float:
    abs_target = abs(float(target))
    if abs_target <= 1e-12:
        return abs(float(predicted) - float(target))
    return abs(float(predicted) - float(target)) / abs_target

def _abs_error(predicted: float, target: float, normalize: bool = False) -> float:
    error = abs(float(predicted) - float(target))
    if normalize:
        error = _relative_abs_error(predicted, target)
    return error


def _get_cell_speed_metrics_frame(results: dict) -> pd.DataFrame:
    metrics = results.get("CELL_SPEED_METRICS")
    if metrics is None or len(metrics) == 0:
        raise ValueError(
            "Results must contain non-empty CELL_SPEED_METRICS. "
            "Rerun the simulation with a build that exports per-cell speed summaries to the pickle."
        )

    if not isinstance(metrics, pd.DataFrame):
        metrics = pd.DataFrame(metrics)

    required = {"id", "cell_type", "trajectory_time", "vmean", "veff"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "CELL_SPEED_METRICS is missing required columns: "
            f"{sorted(missing)}"
        )

    if "dead" not in metrics.columns:
        metrics = metrics.copy()
        metrics["dead"] = 0

    return metrics


def _reduce_population_values(values: pd.Series | np.ndarray, reducer: str) -> float:
    values_arr = np.asarray(values, dtype=float)
    values_arr = values_arr[np.isfinite(values_arr)]
    if values_arr.size == 0:
        raise ValueError("No finite population values available for aggregation")

    if reducer == "mean":
        return float(np.mean(values_arr))
    if reducer == "median":
        return float(np.median(values_arr))
    raise ValueError("population_stat must be 'mean' or 'median'")


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


def _smooth_signal_savgol(
    values: pd.Series | np.ndarray,
    smooth_window: int = 0,
    smooth_polyorder: int = 2,
    *,
    label: str,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a 1-D signal when enabled."""
    values_arr = np.asarray(values, dtype=float)

    if values_arr.ndim != 1:
        raise ValueError(f"{label} must be 1-D for smoothing")
    if len(values_arr) == 0 or smooth_window <= 1:
        return values_arr.copy()

    max_window = len(values_arr) if len(values_arr) % 2 == 1 else len(values_arr) - 1
    if max_window < 3:
        return values_arr.copy()

    window = min(int(smooth_window), max_window)
    if window < 3:
        return values_arr.copy()
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values_arr.copy()

    polyorder = min(int(smooth_polyorder), window - 1)
    if polyorder < 0:
        raise ValueError(f"Invalid Savitzky-Golay polyorder for {label}: {smooth_polyorder}")

    try:
        from scipy.signal import savgol_filter

        return np.asarray(savgol_filter(values_arr, window, polyorder), dtype=float)
    except Exception as exc:
        raise ValueError(f"Savitzky-Golay smoothing failed for {label}") from exc


def _compute_differential_modulus(
    strain: pd.Series | np.ndarray,
    stress: pd.Series | np.ndarray,
    smooth_window: int = 5,
    smooth_polyorder: int = 2,
    modulus_smooth_window: int = 0,
    modulus_smooth_polyorder: int | None = None,
) -> np.ndarray:
    """Compute K(epsilon) = d(stress)/d(strain) with optional smoothing.

    ``smooth_window`` / ``smooth_polyorder`` are applied to the stress signal
    before differentiation. ``modulus_smooth_window`` optionally applies a
    second Savitzky-Golay pass to the resulting differential modulus.
    """
    strain_arr = np.asarray(strain, dtype=float)
    stress_arr = np.asarray(stress, dtype=float)

    if strain_arr.shape != stress_arr.shape:
        raise ValueError("strain and stress must have the same shape")
    if strain_arr.ndim != 1:
        raise ValueError("strain and stress must be 1-D")
    if len(strain_arr) == 0:
        return np.array([], dtype=float)

    invalid_strain = np.where(~np.isfinite(strain_arr))[0]
    if invalid_strain.size:
        first_idx = ", ".join(str(i) for i in invalid_strain[:5])
        raise ValueError(
            "Non-finite strain encountered while computing differential modulus: "
            f"count={invalid_strain.size}, first_indices=[{first_idx}]"
        )

    invalid_stress = np.where(~np.isfinite(stress_arr))[0]
    if invalid_stress.size:
        first_idx = ", ".join(str(i) for i in invalid_stress[:5])
        raise ValueError(
            "Non-finite stress encountered while computing differential modulus: "
            f"count={invalid_stress.size}, first_indices=[{first_idx}]"
        )

    stress_arr = _smooth_signal_savgol(
        stress_arr,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        label="stress",
    )

    d_stress = np.gradient(stress_arr)
    d_strain = np.gradient(strain_arr)

    sim_K = np.zeros_like(stress_arr)
    safe_mask = np.abs(d_strain) > 1e-15
    sim_K[safe_mask] = d_stress[safe_mask] / d_strain[safe_mask]

    invalid_modulus = np.where(~np.isfinite(sim_K))[0]
    if invalid_modulus.size:
        first_idx = ", ".join(str(i) for i in invalid_modulus[:5])
        raise ValueError(
            "Non-finite differential modulus encountered after differentiation: "
            f"count={invalid_modulus.size}, first_indices=[{first_idx}]"
        )

    if modulus_smooth_polyorder is None:
        modulus_smooth_polyorder = smooth_polyorder

    sim_K = _smooth_signal_savgol(
        sim_K,
        smooth_window=modulus_smooth_window,
        smooth_polyorder=modulus_smooth_polyorder,
        label="differential modulus",
    )

    return sim_K


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

    **Initial stabilization cutoff**: ``kwargs["min_sim_strain"]`` ignores
    simulation samples below this strain before computing the error.
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
    sim_strain_arr, sim_stress_arr = _filter_simulation_from_min_strain(
        sim_strain,
        sim_stress,
        min_sim_strain=kwargs.get("min_sim_strain"),
    )
    sim_strain = pd.Series(sim_strain_arr)
    sim_stress = pd.Series(sim_stress_arr)
    _raise_if_axis_selection_has_no_signal(sim_strain, sim_stress, ref, force_type, axis)

    ref_strain = ref["strain"].to_numpy(dtype=float)
    ref_stress = ref["stress"].to_numpy(dtype=float)
    sim_stress_interp = _interpolate_response_to_reference_x(sim_strain, sim_stress, ref_strain)
    overlap_mask = np.isfinite(sim_stress_interp)
    if not np.any(overlap_mask):
        raise ValueError("Simulation and reference strain ranges do not overlap")

    strain_err = _compute_reference_grid_coverage_error(sim_strain, ref_strain)
    stress_err = _mse(sim_stress_interp[overlap_mask], ref_stress[overlap_mask])

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

        **Smoothing**:
                - ``kwargs["smooth_window"]`` (int, default 5) and
                    ``kwargs["smooth_polyorder"]`` (int, default 2) apply
                    Savitzky-Golay smoothing to the **stress** signal *before*
                    differentiation.
                - ``kwargs["modulus_smooth_window"]`` (int, default 0) and
                    ``kwargs["modulus_smooth_polyorder"]`` optionally smooth the
                    resulting differential modulus *after* differentiation.
                - Set either window to 0 or 1 to disable that smoothing stage.

    **Error weighting**: ``kwargs["strain_weight"]`` (default 0.0) — weight of strain-range coverage penalty relative to K(ε) MSE”.

    **Initial stabilization cutoff**: ``kwargs["min_sim_strain"]`` ignores
    simulation samples below this strain before computing the modulus and error.
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
    sim_strain_arr, sim_stress_arr = _filter_simulation_from_min_strain(
        sim_strain,
        sim_stress,
        min_sim_strain=kwargs.get("min_sim_strain"),
    )
    sim_strain = pd.Series(sim_strain_arr)
    sim_stress = pd.Series(sim_stress_arr)
    _raise_if_axis_selection_has_no_signal(sim_strain, sim_stress, ref, force_type, axis)

    strain_arr = sim_strain.values.astype(float)
    smooth_window = int(kwargs.get("smooth_window", 5))
    smooth_polyorder = int(kwargs.get("smooth_polyorder", 2))
    modulus_smooth_window = int(kwargs.get("modulus_smooth_window", 0))
    modulus_smooth_polyorder = kwargs.get("modulus_smooth_polyorder")
    if modulus_smooth_polyorder is not None:
        modulus_smooth_polyorder = int(modulus_smooth_polyorder)
    sim_K = _compute_differential_modulus(
        strain=sim_strain,
        stress=sim_stress,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        modulus_smooth_window=modulus_smooth_window,
        modulus_smooth_polyorder=modulus_smooth_polyorder,
    )

    ref_strain = ref["strain"].to_numpy(dtype=float)
    ref_K = ref["differential_modulus"].to_numpy(dtype=float)
    sim_K_interp = _interpolate_response_to_reference_x(strain_arr, sim_K, ref_strain)
    overlap_mask = np.isfinite(sim_K_interp)
    if not np.any(overlap_mask):
        raise ValueError("Simulation and reference strain ranges do not overlap")

    strain_err = _compute_reference_grid_coverage_error(strain_arr, ref_strain)
    K_err = _mse(sim_K_interp[overlap_mask], ref_K[overlap_mask])

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


def cell_speed_error(results: dict, reference_path: str, **kwargs) -> float:
    """Match per-cell-type mean/effective speeds against a reference CSV.

    The reference CSV must contain a ``cell_type`` column and at least one of:
    ``target_vmean`` or ``target_veff``.

    Simulation values are read from ``CELL_SPEED_METRICS`` stored in the pickle.
    Those metrics are computed from compact per-cell trajectory summaries:

    - ``vmean``: cumulative path length / tracked lifetime [um/s]
    - ``veff``: net displacement / tracked lifetime [um/s]

    By default, only alive cells are considered. Set ``include_dead=True`` to
    include dead cells that remain in the final population. The population
    reduction is controlled by ``population_stat`` and can be ``"mean"``
    (default) or ``"median"``.

    If normalize is True, error terms are normalized by the corresponding target magnitude.
    If use_max_error is True, the final error is the maximum of the per-type errors instead of the average.
    """
    ref = _load_reference_csv(reference_path)
    if "cell_type" not in ref.columns:
        raise ValueError("Reference CSV must contain a 'cell_type' column")

    target_cols = [col for col in ("target_vmean", "target_veff") if col in ref.columns]
    if not target_cols:
        raise ValueError(
            "Reference CSV must contain at least one of 'target_vmean' or 'target_veff'"
        )

    speed_metrics = _get_cell_speed_metrics_frame(results)
    include_dead = bool(kwargs.get("include_dead", False))
    if not include_dead:
        speed_metrics = speed_metrics[speed_metrics["dead"] == 0]

    min_trajectory_time = kwargs.get("min_trajectory_time")
    if min_trajectory_time is not None:
        speed_metrics = speed_metrics[
            speed_metrics["trajectory_time"].astype(float) >= float(min_trajectory_time)
        ]

    if len(speed_metrics) == 0:
        raise ValueError("No cells remain after applying the selected speed-metric filters")

    population_stat = str(kwargs.get("population_stat", "mean")).strip().lower()
    normalize = bool(kwargs.get("normalize", True))
    use_max_error = bool(kwargs.get("use_max_error", True))

    total_error = 0.0
    max_error = 0.0
    n_terms = 0
    display_parts = []

    for _, row in ref.iterrows():
        cell_type = int(row["cell_type"])
        group = speed_metrics[speed_metrics["cell_type"].astype(int) == cell_type]
        if len(group) == 0:
            raise ValueError(f"No simulation cells found for cell_type={cell_type}")

        if "target_vmean" in target_cols and pd.notna(row.get("target_vmean")):
            sim_vmean = _reduce_population_values(group["vmean"], population_stat)
            target_vmean = float(row["target_vmean"])
            error_vmean = _abs_error(sim_vmean, target_vmean, normalize) 
            if error_vmean > max_error:
                max_error = error_vmean
            total_error += error_vmean
            n_terms += 1
            display_parts.append(
                _format_percent_detail(
                    f"type {cell_type} vmean",
                    sim_vmean - target_vmean,
                    target_vmean,
                )
            )

        if "target_veff" in target_cols and pd.notna(row.get("target_veff")):
            sim_veff = _reduce_population_values(group["veff"], population_stat)
            target_veff = float(row["target_veff"])
            error_veff = _abs_error(sim_veff, target_veff, normalize)
            if error_veff > max_error:
                max_error = error_veff
            total_error += error_veff
            n_terms += 1
            display_parts.append(
                _format_percent_detail(
                    f"type {cell_type} veff",
                    sim_veff - target_veff,
                    target_veff,
                )
            )

    if n_terms == 0:
        raise ValueError("No usable target speed values found in reference CSV")

    display_text = f"({'; '.join(display_parts)})" if display_parts else None
    if use_max_error:
        return max_error, display_text
    else:
        return total_error / n_terms, display_text


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
    "cell_speed_error": cell_speed_error,
    "organoid_size_error": organoid_size_error,
}
