"""
Compare cell migration results between one or two simulation conditions.

Features
--------
1 – Violin plots of cell speeds (vmean, veff) per cell type from pickle files.
     If two pickles are given, plots side-by-side (1×2 subplots) for comparison.
     Optional target CSV with expected median values (per cell type & metric)
     is shown as horizontal reference lines.

2 – Per-cell-type 3D trajectory plots and directionality-ratio curves from
     VTK files.  Uses a (rows × N_CELL_TYPES+1) subplot layout where:
       • First N_CELL_TYPES panels: 3D scatter of final positions with
         trajectories drawn from a common origin (initial position subtracted).
       • Last panel: directionality ratio (straight distance / total path)
         over time, one line per cell type.
     If two conditions are provided, a second row is added underneath.

Usage examples
--------------
# Single condition — violin only (no VTK directory provided):
python postprocessing/plot_migration_comparison.py ^
    --pickle1 result_files/speed_control/output_data_0_speed_control.pickle ^
    --target1 optimizer/reference_data/target_cell_speed_control.csv

# Two conditions — violin comparison:
python postprocessing/plot_migration_comparison.py ^
    --pickle1 result_files/speed_control/output_data_0_speed_control.pickle ^
    --pickle2 result_files/speed_chemokinesis/output_data_0_speed_chemokinesis.pickle ^
    --target1 optimizer/reference_data/target_cell_speed_control.csv ^
    --target2 optimizer/reference_data/target_cell_speed_chemokinesis.csv

# Two conditions — violin + trajectories:
python postprocessing/plot_migration_comparison.py 
    --pickle1 result_files/speed_control/output_data_0_speed_control.pickle 
    --pickle2 result_files/speed_chemokinesis/output_data_0_speed_chemokinesis.pickle 
    --target1 optimizer/reference_data/target_cell_speed_control.csv 
    --target2 optimizer/reference_data/target_cell_speed_chemokinesis.csv 
    --vtk-dir1 result_files/speed_control 
    --vtk-dir2 result_files/speed_chemokinesis 
    --max-trajectories 50
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "result_files"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Safe pickle loading (copied from existing postprocessing convention)
# ---------------------------------------------------------------------------

class _DummyModelParameterConfig:
    pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "helper_module" and name == "ModelParameterConfig":
            return _DummyModelParameterConfig
        if module.startswith("numpy._core"):
            remapped = "numpy.core" + module[len("numpy._core"):]
            return getattr(importlib.import_module(remapped), name)
        return super().find_class(module, name)


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    with path.open("rb") as f:
        return _SafeUnpickler(f).load()


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------

def _coerce_metrics_frame(metrics: Any) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        return metrics.copy()
    return pd.DataFrame(metrics)


def load_speed_metrics(results: dict[str, Any]) -> pd.DataFrame:
    """Extract CELL_SPEED_METRICS dataframe from a pickle dict."""
    raw = results.get("CELL_SPEED_METRICS")
    if raw is None or len(raw) == 0:
        raise ValueError("Pickle does not contain non-empty CELL_SPEED_METRICS")
    df = _coerce_metrics_frame(raw)
    if "dead" not in df.columns:
        df["dead"] = 0
    return df


def load_target_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "cell_type" not in df.columns:
        return None
    return df


# ---------------------------------------------------------------------------
# VTK helpers
# ---------------------------------------------------------------------------

def _read_scalar_float(lines: list[str], name: str, n: int) -> np.ndarray:
    header = f"SCALARS {name}"
    idx = next((i for i, l in enumerate(lines) if l.startswith(header)), None)
    if idx is None:
        raise ValueError(f"Scalar '{name}' not found in VTK")
    start = idx + 2  # skip SCALARS line + LOOKUP_TABLE line
    vals = np.empty(n, dtype=float)
    for k in range(n):
        vals[k] = float(lines[start + k].split()[0])
    return vals


def _count_cells(ids: np.ndarray) -> int:
    """Find the number of actual cells (before anchor points begin).

    In the VTK files, the first N entries are cells with unique IDs.
    After that, anchor points follow and re-use the same cell IDs.
    We detect N by finding the first index where an ID repeats.
    """
    seen: set[int] = set()
    for i, v in enumerate(ids):
        if v in seen:
            return i
        seen.add(int(v))
    return len(ids)  # all unique — every point is a cell


def _read_vtk_cells(path: Path) -> dict[str, np.ndarray]:
    """
    Parse a cells_tXXXX.vtk file, returning **only cell data** (no anchors).

    Returns dict with keys: 'x', 'y', 'z', 'id', 'cell_type', 'dead'.
    """
    text = path.read_text()
    lines = text.splitlines()

    # Read all points
    pts_idx = next(i for i, l in enumerate(lines) if l.startswith("POINTS"))
    n_points = int(lines[pts_idx].split()[1])
    coords = np.empty((n_points, 3), dtype=float)
    for k in range(n_points):
        parts = lines[pts_idx + 1 + k].split()
        coords[k, 0] = float(parts[0])
        coords[k, 1] = float(parts[1])
        coords[k, 2] = float(parts[2])

    ids_all = _read_scalar_float(lines, "id", n_points).astype(int)

    # Determine how many points are actual cells (rest are anchor points)
    n_cells = _count_cells(ids_all)

    ids = ids_all[:n_cells]
    coords = coords[:n_cells]
    dead = _read_scalar_float(lines, "dead", n_points).astype(int)[:n_cells]
    try:
        cell_types = _read_scalar_float(lines, "cell_type", n_points).astype(int)[:n_cells]
    except ValueError:
        cell_types = np.zeros(n_cells, dtype=int)

    return {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "id": ids,
        "cell_type": cell_types,
        "dead": dead,
    }


def _discover_vtk_files(vtk_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (timestep, path) for cells_tXXXX.vtk files."""
    pairs: list[tuple[int, Path]] = []
    for p in vtk_dir.glob("cells_t*.vtk"):
        m = re.search(r"t(\d+)\.vtk$", p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def build_trajectory_data(vtk_dir: Path) -> dict[str, Any]:
    """
    Build per-cell trajectories from VTK time series.

    Returns
    -------
    dict with:
        'timesteps'  : np.ndarray of shape (T,)
        'positions'  : dict  cell_id -> np.ndarray (T, 3)  — NaN where absent
        'cell_types' : dict  cell_id -> int  (last known type)
        'alive'      : dict  cell_id -> bool (alive at last timestep)
    """
    file_list = _discover_vtk_files(vtk_dir)
    if not file_list:
        raise FileNotFoundError(f"No cells_t*.vtk files found in {vtk_dir}")

    timesteps = np.array([t for t, _ in file_list])
    n_steps = len(timesteps)

    # First pass: discover all unique cell IDs and cell types
    all_ids: set[int] = set()
    type_map: dict[int, int] = {}
    alive_map: dict[int, bool] = {}

    # We read first, last and sample of frames to discover IDs efficiently,
    # but we need all frames for full trajectories.
    frame_data: list[dict[str, np.ndarray]] = []
    for _t, fpath in file_list:
        data = _read_vtk_cells(fpath)
        frame_data.append(data)
        for cid, ct, d in zip(data["id"], data["cell_type"], data["dead"]):
            all_ids.add(int(cid))
            type_map[int(cid)] = int(ct)
            alive_map[int(cid)] = (int(d) == 0)

    # Build position arrays
    positions: dict[int, np.ndarray] = {cid: np.full((n_steps, 3), np.nan) for cid in all_ids}
    for ti, data in enumerate(frame_data):
        for k in range(len(data["id"])):
            cid = int(data["id"][k])
            positions[cid][ti, :] = [data["x"][k], data["y"][k], data["z"][k]]

    return {
        "timesteps": timesteps,
        "positions": positions,
        "cell_types": type_map,
        "alive": alive_map,
    }


def compute_directionality_ratios(
    traj_data: dict[str, Any],
    speed_metrics: pd.DataFrame,
    time_step: float | None,
) -> dict[str, Any]:
    """
    Compute directionality ratio per cell over time.

    directionality_ratio(t) = straight_distance(0→t) / estimated_true_path(0→t)

    The straight distance is computed exactly from VTK positions.  The true
    cumulative path is estimated as ``vmean * elapsed_time`` using each cell's
    ``vmean`` from the pickle.  This avoids the artefact where coarsely-sampled
    VTK snapshots make the cumulative path appear much shorter than reality
    (because the cell wanders between snapshots but we only see the net
    displacement per interval).

    Returns dict with:
        'timesteps'  : np.ndarray (T,)
        'ratios'     : dict  cell_id -> np.ndarray (T,)  — NaN where undefined
        'cell_types' : dict  cell_id -> int
    """
    timesteps = traj_data["timesteps"]
    positions = traj_data["positions"]
    cell_types = traj_data["cell_types"]
    n_steps = len(timesteps)

    # Build lookup: cell_id -> vmean from pickle
    vmean_lookup: dict[int, float] = {}
    if "id" in speed_metrics.columns and "vmean" in speed_metrics.columns:
        for _, row in speed_metrics.iterrows():
            vmean_lookup[int(row["id"])] = float(row["vmean"])

    # Fallback time_step (seconds per simulation step)
    dt = time_step if time_step is not None else 1.0

    ratios: dict[int, np.ndarray] = {}
    for cid, pos in positions.items():
        r = np.full(n_steps, np.nan)
        valid = ~np.isnan(pos[:, 0])
        valid_indices = np.where(valid)[0]
        if len(valid_indices) < 2:
            ratios[cid] = r
            continue

        vmean = vmean_lookup.get(cid)
        first_idx = valid_indices[0]
        origin = pos[first_idx].copy()
        t0 = float(timesteps[first_idx])

        for ti in valid_indices:
            curr = pos[ti]
            straight = np.linalg.norm(curr - origin)
            elapsed = (float(timesteps[ti]) - t0) * dt

            if vmean is not None and vmean > 1e-15:
                # Use true path estimate from pickle vmean
                true_path = vmean * elapsed
            else:
                # Fallback: not in pickle — skip this cell
                continue

            if true_path > 1e-12:
                r[ti] = straight / true_path
            else:
                r[ti] = 1.0  # no time elapsed yet
        ratios[cid] = r

    return {
        "timesteps": timesteps,
        "ratios": ratios,
        "cell_types": cell_types,
    }


# ---------------------------------------------------------------------------
# D1 — Violin speed plots
# ---------------------------------------------------------------------------

CELL_TYPE_COLORS = [
    "#b6d73a",   
    "#3bb29f",   
    "#8a00a8",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _color_for(ct: int) -> str:
    return CELL_TYPE_COLORS[ct % len(CELL_TYPE_COLORS)]


def _plot_violin_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    value_col: str,
    ylabel: str,
    target_df: pd.DataFrame | None,
    title: str,
    show_legend: bool = True,
    type_labels: list[str] | None = None,
) -> None:
    """Draw violin + strip + mean/median + optional target for one speed metric."""
    cell_types = sorted(metrics["cell_type"].dropna().astype(int).unique().tolist())
    data = []
    for ct in cell_types:
        vals = metrics.loc[metrics["cell_type"].astype(int) == ct, value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        data.append(vals)
    positions = np.arange(len(cell_types), dtype=float)

    rng = np.random.default_rng(0)
    for pos, ct, vals in zip(positions, cell_types, data):
        color = _color_for(ct)
        if len(vals) >= 2 and np.ptp(vals) > 1e-12:
            vp = ax.violinplot(
                [vals], positions=[pos], widths=0.7,
                showmeans=False, showmedians=False, showextrema=False, bw_method=0.3,
            )
            for body in vp["bodies"]:
                body.set_alpha(0.22)
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_linewidth(1.5)
            # IQR indicator
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(pos, q25, q75, color="dimgray", linewidth=2.5, zorder=3)
            ax.hlines(q50, pos - 0.06, pos + 0.06, color="dimgray", linewidth=3, zorder=4)
        # Strip (jitter) points — random spread for natural look
        if len(vals):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter, vals,
                s=25, color=color, alpha=0.55, edgecolors="white",
                linewidths=0.4, zorder=2,
            )

    # Mean / median markers
    means = [float(np.mean(v)) if len(v) else np.nan for v in data]
    medians = [float(np.median(v)) if len(v) else np.nan for v in data]
    ax.scatter(positions, means, marker="o", s=50, color="black",
               label="mean", zorder=5, edgecolors="white", linewidths=0.8)
    ax.scatter(positions, medians, marker="D", s=40, color="tab:red",
               label="median", zorder=5, edgecolors="white", linewidths=0.8)

    # Target reference
    if target_df is not None:
        target_col = f"target_{value_col}"
        if target_col in target_df.columns:
            lookup = {
                int(row.cell_type): float(getattr(row, target_col))
                for row in target_df.itertuples()
                if pd.notna(getattr(row, target_col, np.nan))
            }
            for pos, ct in zip(positions, cell_types):
                if ct in lookup:
                    ax.hlines(
                        lookup[ct], pos - 0.35, pos + 0.35,
                        colors="tab:green", linewidths=2.0, linestyles="--",
                        label="target" if pos == positions[0] else None, zorder=6,
                    )

    # X-tick labels
    if type_labels is not None:
        tick_labels = [type_labels[ct] if ct < len(type_labels) else str(ct)
                       for ct in cell_types]
    else:
        tick_labels = [str(ct) for ct in cell_types]
    ax.set_xticks(positions, tick_labels)
    ax.set_xlabel("Cell type", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize=10)


def create_violin_figure(
    metrics1: pd.DataFrame,
    target1: pd.DataFrame | None,
    label1: str,
    metrics2: pd.DataFrame | None = None,
    target2: pd.DataFrame | None = None,
    label2: str | None = None,
    ymax_vmean: float | None = None,
    ymax_veff: float | None = None,
) -> plt.Figure:
    """
    Create the D1 violin-plot figure.

    If only one condition is provided, produces a (2×1) figure (vmean, veff).
    If two conditions are given, produces a (2×2) figure for side-by-side
    comparison — left column for condition 1, right for condition 2.

    Parameters
    ----------
    ymax_vmean, ymax_veff : float or None
        Upper y-axis limit for vmean / veff rows.  If None (default), auto-
        computed as the global maximum value across all conditions + 10% offset.
        All panels in the same row share [0, ymax].
    """
    two = metrics2 is not None

    # --- Auto-detect ymax if not provided ---
    all_vmean = metrics1["vmean"].to_numpy(dtype=float)
    all_veff = metrics1["veff"].to_numpy(dtype=float)
    if two:
        all_vmean = np.concatenate([all_vmean, metrics2["vmean"].to_numpy(dtype=float)])
        all_veff = np.concatenate([all_veff, metrics2["veff"].to_numpy(dtype=float)])
    if ymax_vmean is None:
        ymax_vmean = float(np.nanmax(all_vmean[np.isfinite(all_vmean)])) * 1.10
    if ymax_veff is None:
        ymax_veff = float(np.nanmax(all_veff[np.isfinite(all_veff)])) * 1.10

    ncols = 2 if two else 1
    fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 10), squeeze=False)
    fig.suptitle("Cell Speed Distributions", fontsize=14, y=1.01)

    _plot_violin_panel(axes[0, 0], metrics1, "vmean", "vmean [µm/s]", target1,
                       f"vmean — {label1}")
    _plot_violin_panel(axes[1, 0], metrics1, "veff", "veff [µm/s]", target1,
                       f"veff — {label1}")

    if two:
        _plot_violin_panel(axes[0, 1], metrics2, "vmean", "vmean [µm/s]", target2,
                           f"vmean — {label2}")
        _plot_violin_panel(axes[1, 1], metrics2, "veff", "veff [µm/s]", target2,
                           f"veff — {label2}")

    # Apply shared y-axis limits per row
    for col in range(ncols):
        axes[0, col].set_ylim(0, ymax_vmean)
        axes[1, col].set_ylim(0, ymax_veff)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# D2 — Trajectory + directionality ratio plots
# ---------------------------------------------------------------------------

def create_trajectory_figure(
    traj_data1: dict[str, Any],
    dir_data1: dict[str, Any],
    label1: str,
    traj_data2: dict[str, Any] | None = None,
    dir_data2: dict[str, Any] | None = None,
    label2: str | None = None,
    max_trajectories: int = 50,
    time_step: float | None = None,
) -> plt.Figure:
    """
    Create the D2 trajectory + directionality figure.

    Layout: (n_rows × N_CELL_TYPES+1).
      - n_rows=1 for one condition, n_rows=2 for two.
      - First N_CELL_TYPES columns: 3D scatter of trajectories from common origin.
      - Last column: directionality ratio over time.

    Parameters
    ----------
    max_trajectories : int
        Maximum number of trajectories to plot per cell type (random subset).
    time_step : float or None
        Seconds per simulation step. If provided, x-axis is in seconds.
    """

    # Determine cell types from the union of both conditions
    all_types1 = sorted(set(traj_data1["cell_types"].values()))
    all_types2 = sorted(set(traj_data2["cell_types"].values())) if traj_data2 else []
    all_types = sorted(set(all_types1) | set(all_types2))
    n_types = len(all_types)
    n_cols = n_types + 1
    n_rows = 2 if traj_data2 is not None else 1

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    datasets = [(traj_data1, dir_data1, label1)]
    if traj_data2 is not None:
        datasets.append((traj_data2, dir_data2, label2))

    # --- First pass: compute shifted trajectories & global axis limits ---
    global_max = 0.0  # symmetric limit for all 3D panels
    # Store per-dataset, per-cell-type: list of shifted arrays
    precomputed: list[dict[str, Any]] = []

    for traj_data, _dir_data, _label in datasets:
        positions = traj_data["positions"]
        cell_types = traj_data["cell_types"]
        alive_map = traj_data["alive"]

        ids_by_type: dict[int, list[int]] = {ct: [] for ct in all_types}
        for cid, ct in cell_types.items():
            if ct in ids_by_type and alive_map.get(cid, False):
                pos = positions[cid]
                valid = ~np.isnan(pos[:, 0])
                if np.sum(valid) >= 2:
                    ids_by_type[ct].append(cid)

        rng = np.random.default_rng(42)
        shifted_by_type: dict[int, list[np.ndarray]] = {ct: [] for ct in all_types}
        sampled_ids_by_type: dict[int, list[int]] = {ct: [] for ct in all_types}

        for ct in all_types:
            cids = ids_by_type.get(ct, [])
            if len(cids) > max_trajectories:
                cids = list(rng.choice(cids, size=max_trajectories, replace=False))
            sampled_ids_by_type[ct] = cids
            for cid in cids:
                pos = positions[cid]
                valid_idx = np.where(~np.isnan(pos[:, 0]))[0]
                origin = pos[valid_idx[0]]
                shifted = pos[valid_idx] - origin
                shifted_by_type[ct].append(shifted)
                amax = np.max(np.abs(shifted))
                if amax > global_max:
                    global_max = amax

        precomputed.append({
            "ids_by_type": ids_by_type,
            "sampled_ids_by_type": sampled_ids_by_type,
            "shifted_by_type": shifted_by_type,
        })

    # Add a small margin to the symmetric limit
    axis_lim = global_max * 1.05 if global_max > 0 else 1.0

    # --- Second pass: plot ---
    for row_idx, (traj_data, dir_data, label) in enumerate(datasets):
        cell_types = traj_data["cell_types"]
        pc = precomputed[row_idx]
        ids_by_type = pc["ids_by_type"]
        sampled_ids_by_type = pc["sampled_ids_by_type"]
        shifted_by_type = pc["shifted_by_type"]

        # --- 3D trajectory panels (one per cell type) ---
        for col_idx, ct in enumerate(all_types):
            ax = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1,
                                 projection="3d")
            color = _color_for(ct)

            for shifted in shifted_by_type.get(ct, []):
                ax.plot(shifted[:, 0], shifted[:, 1], shifted[:, 2],
                        color=color, alpha=0.35, linewidth=0.7)
                ax.scatter(
                    [shifted[-1, 0]], [shifted[-1, 1]], [shifted[-1, 2]],
                    color=color, s=12, alpha=0.7, edgecolors="none",
                )

            # Shared axis limits across all 3D panels
            ax.set_xlim(-axis_lim, axis_lim)
            ax.set_ylim(-axis_lim, axis_lim)
            ax.set_zlim(-axis_lim, axis_lim)

            ax.set_xlabel("Δx [µm]", fontsize=8)
            ax.set_ylabel("Δy [µm]", fontsize=8)
            ax.set_zlabel("Δz [µm]", fontsize=8)
            ax.tick_params(labelsize=7)
            n_shown = len(sampled_ids_by_type.get(ct, []))
            n_total = len(ids_by_type.get(ct, []))
            subset_info = f" ({n_shown}/{n_total})" if n_shown < n_total else ""
            ax.set_title(f"{label} — type {ct}{subset_info}", fontsize=9)

        # --- Directionality ratio panel ---
        ax_dir = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + n_cols)
        ratios = dir_data["ratios"]
        ts = dir_data["timesteps"]
        if time_step is not None:
            x_vals = ts.astype(float) * time_step
            x_label = "Time [s]"
        else:
            x_vals = ts.astype(float)
            x_label = "Timestep"

        for ct in all_types:
            color = _color_for(ct)
            ct_cids = [cid for cid, t in cell_types.items() if t == ct]
            if not ct_cids:
                continue
            # Stack ratios and compute per-timestep mean
            ratio_stack = np.array([ratios[cid] for cid in ct_cids if cid in ratios])
            if ratio_stack.size == 0:
                continue
            mean_ratio = np.nanmean(ratio_stack, axis=0)
            std_ratio = np.nanstd(ratio_stack, axis=0)
            valid = ~np.isnan(mean_ratio)
            ax_dir.plot(x_vals[valid], mean_ratio[valid], color=color,
                        linewidth=1.5, label=f"type {ct}")
            ax_dir.fill_between(
                x_vals[valid],
                (mean_ratio - std_ratio)[valid],
                (mean_ratio + std_ratio)[valid],
                color=color, alpha=0.15,
            )

        ax_dir.set_xlabel(x_label, fontsize=9)
        ax_dir.set_ylabel("Directionality ratio", fontsize=9)
        ax_dir.set_ylim(-0.05, 1.05)
        ax_dir.set_title(f"{label} — directionality", fontsize=9)
        ax_dir.grid(True, alpha=0.25)
        ax_dir.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined figure (violins + trajectories in one plot)
# ---------------------------------------------------------------------------

def create_combined_figure(
    metrics1: pd.DataFrame,
    target1: pd.DataFrame | None,
    label1: str,
    metrics2: pd.DataFrame | None,
    target2: pd.DataFrame | None,
    label2: str | None,
    traj_data1: dict[str, Any] | None = None,
    dir_data1: dict[str, Any] | None = None,
    traj_data2: dict[str, Any] | None = None,
    dir_data2: dict[str, Any] | None = None,
    *,
    max_trajectories: int = 100,
    time_step: float | None = None,
    ymax_vmean: float | None = None,
    ymax_veff: float | None = None,
    figsize: tuple[float, float] = (20, 15),
    type_labels: list[str] | None = None,
) -> plt.Figure:
    """
    Create a single combined migration figure.

    Layout (3 x N_cols when two conditions + VTK data):
      Row 0: vmean (cond1), veff (cond1), vmean (cond2), veff (cond2)
      Row 1: 3D trajectories per type + directionality (cond 1)
      Row 2: 3D trajectories per type + directionality (cond 2)

    Falls back to fewer rows when only one condition or no VTK data.
    """
    has_traj = traj_data1 is not None
    two_cond = metrics2 is not None

    # Determine cell types
    if has_traj:
        all_types1 = sorted(set(traj_data1["cell_types"].values()))
        all_types2 = sorted(set(traj_data2["cell_types"].values())) if traj_data2 else []
        all_types = sorted(set(all_types1) | set(all_types2))
    else:
        all_types = sorted(metrics1["cell_type"].dropna().astype(int).unique().tolist())

    n_types = len(all_types)
    n_violin_cols = 4 if two_cond else 2
    n_traj_cols = n_types + 1 if has_traj else 0
    n_cols = max(n_violin_cols, n_traj_cols) if n_traj_cols else n_violin_cols

    if has_traj:
        n_rows = 3 if (traj_data2 is not None) else 2
    else:
        n_rows = 1

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.35)

    # --- Auto-detect ymax ---
    all_vmean = metrics1["vmean"].to_numpy(dtype=float)
    all_veff = metrics1["veff"].to_numpy(dtype=float)
    if two_cond:
        all_vmean = np.concatenate([all_vmean, metrics2["vmean"].to_numpy(dtype=float)])
        all_veff = np.concatenate([all_veff, metrics2["veff"].to_numpy(dtype=float)])
    if ymax_vmean is None:
        ymax_vmean = float(np.nanmax(all_vmean[np.isfinite(all_vmean)])) * 1.10
    if ymax_veff is None:
        ymax_veff = float(np.nanmax(all_veff[np.isfinite(all_veff)])) * 1.10

    # === Row 0: Violin panels ===
    ax_v1 = fig.add_subplot(gs[0, 0])
    ax_v2 = fig.add_subplot(gs[0, 1])
    _plot_violin_panel(ax_v1, metrics1, "vmean", "vmean [µm/s]", target1,
                       f"vmean — {label1}", show_legend=True, type_labels=type_labels)
    _plot_violin_panel(ax_v2, metrics1, "veff", "veff [µm/s]", target1,
                       f"veff — {label1}", show_legend=False, type_labels=type_labels)
    ax_v1.set_ylim(0, ymax_vmean)
    ax_v2.set_ylim(0, ymax_veff)

    if two_cond:
        ax_v3 = fig.add_subplot(gs[0, 2])
        ax_v4 = fig.add_subplot(gs[0, 3])
        _plot_violin_panel(ax_v3, metrics2, "vmean", "vmean [µm/s]", target2,
                           f"vmean — {label2}", show_legend=False, type_labels=type_labels)
        _plot_violin_panel(ax_v4, metrics2, "veff", "veff [µm/s]", target2,
                           f"veff — {label2}", show_legend=False, type_labels=type_labels)
        ax_v3.set_ylim(0, ymax_vmean)
        ax_v4.set_ylim(0, ymax_veff)

    if not has_traj:
        fig.tight_layout()
        return fig

    # === Precompute trajectory data ===
    datasets = [(traj_data1, dir_data1, label1)]
    if traj_data2 is not None:
        datasets.append((traj_data2, dir_data2, label2))

    global_max = 0.0
    precomputed: list[dict[str, Any]] = []

    for traj_data, _dir_data, _label in datasets:
        positions = traj_data["positions"]
        cell_types_map = traj_data["cell_types"]
        alive_map = traj_data["alive"]

        ids_by_type: dict[int, list[int]] = {ct: [] for ct in all_types}
        for cid, ct in cell_types_map.items():
            if ct in ids_by_type and alive_map.get(cid, False):
                pos = positions[cid]
                valid = ~np.isnan(pos[:, 0])
                if np.sum(valid) >= 2:
                    ids_by_type[ct].append(cid)

        rng = np.random.default_rng(42)
        shifted_by_type: dict[int, list[np.ndarray]] = {ct: [] for ct in all_types}
        sampled_ids_by_type: dict[int, list[int]] = {ct: [] for ct in all_types}

        for ct in all_types:
            cids = ids_by_type.get(ct, [])
            if len(cids) > max_trajectories:
                cids = list(rng.choice(cids, size=max_trajectories, replace=False))
            sampled_ids_by_type[ct] = cids
            for cid in cids:
                pos = positions[cid]
                valid_idx = np.where(~np.isnan(pos[:, 0]))[0]
                origin = pos[valid_idx[0]]
                shifted = pos[valid_idx] - origin
                shifted_by_type[ct].append(shifted)
                amax = np.max(np.abs(shifted))
                if amax > global_max:
                    global_max = amax

        precomputed.append({
            "ids_by_type": ids_by_type,
            "sampled_ids_by_type": sampled_ids_by_type,
            "shifted_by_type": shifted_by_type,
        })

    axis_lim = global_max * 1.05 if global_max > 0 else 1.0

    # === Rows 1+: Trajectories + directionality ===
    for ds_idx, (traj_data, dir_data, label) in enumerate(datasets):
        row = ds_idx + 1
        cell_types_map = traj_data["cell_types"]
        pc = precomputed[ds_idx]

        # 3D trajectory panels
        for col_idx, ct in enumerate(all_types):
            ax = fig.add_subplot(gs[row, col_idx], projection="3d")
            color = _color_for(ct)

            # White background
            ax.xaxis.pane.set_facecolor("white")
            ax.yaxis.pane.set_facecolor("white")
            ax.zaxis.pane.set_facecolor("white")
            ax.xaxis.pane.set_edgecolor("lightgray")
            ax.yaxis.pane.set_edgecolor("lightgray")
            ax.zaxis.pane.set_edgecolor("lightgray")

            for shifted in pc["shifted_by_type"].get(ct, []):
                ax.plot(shifted[:, 0], shifted[:, 1], shifted[:, 2],
                        color=color, alpha=0.35, linewidth=0.7)
                ax.scatter(
                    [shifted[-1, 0]], [shifted[-1, 1]], [shifted[-1, 2]],
                    color=color, s=12, alpha=0.7, edgecolors="none",
                )

            ax.set_xlim(-axis_lim, axis_lim)
            ax.set_ylim(-axis_lim, axis_lim)
            ax.set_zlim(-axis_lim, axis_lim)

            ax.set_xlabel("Δx [µm]", fontsize=10)
            ax.set_ylabel("Δy [µm]", fontsize=10)
            ax.set_zlabel("Δz [µm]", fontsize=10)
            ax.tick_params(labelsize=9)

            type_lbl = (type_labels[ct] if type_labels and ct < len(type_labels)
                        else f"type {ct}")
            n_shown = len(pc["sampled_ids_by_type"].get(ct, []))
            n_total = len(pc["ids_by_type"].get(ct, []))
            subset_info = f" ({n_shown}/{n_total})" if n_shown < n_total else ""
            ax.set_title(f"{label} — {type_lbl}{subset_info}",
                         fontsize=11, fontweight="bold")

        # Directionality ratio panel
        dir_col = min(n_types, n_cols - 1)
        ax_dir = fig.add_subplot(gs[row, dir_col])
        ratios = dir_data["ratios"]
        ts = dir_data["timesteps"]
        if time_step is not None:
            x_vals = ts.astype(float) * time_step
            x_label = "Time [s]"
        else:
            x_vals = ts.astype(float)
            x_label = "Timestep"

        for ct in all_types:
            color = _color_for(ct)
            type_lbl = (type_labels[ct] if type_labels and ct < len(type_labels)
                        else f"type {ct}")
            ct_cids = [cid for cid, t in cell_types_map.items() if t == ct]
            if not ct_cids:
                continue
            ratio_stack = np.array([ratios[cid] for cid in ct_cids if cid in ratios])
            if ratio_stack.size == 0:
                continue
            mean_ratio = np.nanmean(ratio_stack, axis=0)
            std_ratio = np.nanstd(ratio_stack, axis=0)
            valid = ~np.isnan(mean_ratio)
            ax_dir.plot(x_vals[valid], mean_ratio[valid], color=color,
                        linewidth=1.5, label=type_lbl)
            ax_dir.fill_between(
                x_vals[valid],
                (mean_ratio - std_ratio)[valid],
                (mean_ratio + std_ratio)[valid],
                color=color, alpha=0.15,
            )

        ax_dir.set_xlabel(x_label, fontsize=12)
        ax_dir.set_ylabel("Directionality ratio", fontsize=12)
        ax_dir.set_ylim(-0.05, 1.05)
        ax_dir.set_title(f"{label} — directionality",
                         fontsize=11, fontweight="bold")
        ax_dir.tick_params(labelsize=11)
        ax_dir.grid(True, alpha=0.25)
        ax_dir.legend(loc="best", fontsize=10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare cell migration between one or two simulation conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Condition 1 (required)
    p.add_argument("--pickle1", required=True,
                   help="Pickle file for condition 1 (e.g. control)")
    p.add_argument("--label1", default="control",
                   help="Display label for condition 1 (default: 'control')")
    p.add_argument("--target1", default=None,
                   help="Target CSV for condition 1 (cell_type,target_vmean,target_veff)")
    p.add_argument("--vtk-dir1", default=None,
                   help="Directory with cells_tXXXX.vtk for condition 1")

    # Condition 2 (optional)
    p.add_argument("--pickle2", default=None,
                   help="Pickle file for condition 2")
    p.add_argument("--label2", default="condition",
                   help="Display label for condition 2 (default: 'condition')")
    p.add_argument("--target2", default=None,
                   help="Target CSV for condition 2 (required if --pickle2 is given)")
    p.add_argument("--vtk-dir2", default=None,
                   help="Directory with cells_tXXXX.vtk for condition 2")

    # Output / display
    p.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR),
                   help="Directory for output figures (default: postprocessing/results)")
    p.add_argument("--tag", default="comparison",
                   help="Suffix tag for output filenames")
    p.add_argument("--show", action="store_true",
                   help="Display figures interactively instead of only saving")
    p.add_argument("--max-trajectories", type=int, default=100,
                   help="Max trajectories to plot per cell type (default: 100)")
    p.add_argument("--ymax-vmean", type=float, default=None,
                   help="Upper y-axis limit for vmean violin panels (default: auto)")
    p.add_argument("--ymax-veff", type=float, default=None,
                   help="Upper y-axis limit for veff violin panels (default: auto)")
    p.add_argument("--figsize", type=float, nargs=2, default=[20, 15],
                   metavar=("W", "H"),
                   help="Figure size in inches (default: 20 15)")
    p.add_argument("--dpi", type=int, default=600,
                   help="Resolution for saved figure (default: 600)")
    p.add_argument("--type-labels", nargs="+", default=None,
                   help="Custom labels for cell types, e.g. --type-labels Epithelial Mesenchymal Stem")
    return p.parse_args()


def _get_time_step(results: dict[str, Any]) -> float | None:
    cfg = results.get("MODEL_CONFIG")
    if cfg is not None:
        return getattr(cfg, "TIME_STEP", None)
    return None


def main() -> None:
    args = parse_args()

    # --- Load condition 1 ---
    pkl1 = _load_pickle(Path(args.pickle1))
    metrics1 = load_speed_metrics(pkl1)
    target1 = load_target_csv(Path(args.target1)) if args.target1 else None
    time_step = _get_time_step(pkl1)

    # --- Load condition 2 (if provided) ---
    metrics2 = None
    target2 = None
    if args.pickle2:
        pkl2_data = _load_pickle(Path(args.pickle2))
        metrics2 = load_speed_metrics(pkl2_data)
        if args.target2:
            target2 = load_target_csv(Path(args.target2))
        else:
            print("WARNING: --pickle2 provided without --target2. "
                  "No target reference for condition 2.")

    # --- Load VTK data (if provided) ---
    traj1, dir1 = None, None
    traj2, dir2 = None, None
    vtk_dir1 = Path(args.vtk_dir1) if args.vtk_dir1 else None
    vtk_dir2 = Path(args.vtk_dir2) if args.vtk_dir2 else None

    if vtk_dir1 is not None:
        print(f"Loading VTK trajectories from {vtk_dir1} …")
        traj1 = build_trajectory_data(vtk_dir1)
        dir1 = compute_directionality_ratios(traj1, metrics1, time_step)

    if vtk_dir2 is not None and metrics2 is not None:
        print(f"Loading VTK trajectories from {vtk_dir2} …")
        traj2 = build_trajectory_data(vtk_dir2)
        dir2 = compute_directionality_ratios(traj2, metrics2, time_step)

    # === Create combined figure ===
    print("Creating combined migration figure …")
    fig = create_combined_figure(
        metrics1, target1, args.label1,
        metrics2, target2, args.label2,
        traj1, dir1,
        traj2, dir2,
        max_trajectories=args.max_trajectories,
        time_step=time_step,
        ymax_vmean=args.ymax_vmean,
        ymax_veff=args.ymax_veff,
        figsize=tuple(args.figsize),
        type_labels=args.type_labels,
    )

    # === Save / show ===
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"migration_combined_{args.tag}.png"
    fig.savefig(str(out_path), dpi=args.dpi)
    print(f"Saved → {out_path}")

    w, h = fig.get_size_inches()
    print(f"{w:.2f} x {h:.2f} inches after manual resize")

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("Done.")


if __name__ == "__main__":
    main()
