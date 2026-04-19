"""
Compare migration metrics, directionality, and diffusion-profile evolution for two simulation conditions in a single 3x4 figure.

The figure layout is fixed as follows:
  - Row 1: vmean (condition 1), vmean (condition 2), veff (condition 1), veff (condition 2)
  - Row 2: directionality ratio (condition 1), directionality ratio (condition 2), blank, blank
  - Row 3: scalar-profile evolution from ecm_data_t*.vtk in vtk-dir1, scalar-profile evolution from ecm_data_t*.vtk in vtk-dir2, blank, blank

The script reads:
  - pickle files for CELL_SPEED_METRICS
  - optional target CSV files for target_vmean / target_veff reference lines
  - cells_t*.vtk files for directionality-ratio calculation
  - ecm_data_t*.vtk files for scalar-profile evolution along a user-defined line

Both conditions are required for this comparison layout.

Example call (one line)
-----------------------
python postprocessing\plot_migration_diff_profiles_comparison.py --pickle1 result_files\homogeneous_diff\output_data_0.pickle --pickle2 result_files\heterogeneous_diff\output_data_0.pickle --vtk-dir1 result_files\homogeneous_diff --vtk-dir2 result_files\heterogeneous_diff --scalar-name concentration_species_0 --scalar-ylabel "[TGF-$\beta$1 ng/mL]" --x1 0.0 --y1 -500.0 --z1 0.0 --x2 0.0 --y2 0.0 --z2 0.0 --label1 hom --label2 het --type-labels shControl shSMAD2 shSMAD3 --figsize 10 7 --dpi 600 --show --no-titles
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_scalar_profiles_vtk import plot_scalar_line_evolution_comparison


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


CELL_TYPE_COLORS = [
    "#b6d73a",
    "#3bb29f",
    "#8a00a8",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


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


def _coerce_metrics_frame(metrics: Any) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        return metrics.copy()
    return pd.DataFrame(metrics)



def load_speed_metrics(results: dict[str, Any]) -> pd.DataFrame:
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



def _set_panel_title(ax: plt.Axes, title: str, show_title: bool, **kwargs: Any) -> None:
    ax.set_title(title if show_title else "", **kwargs)



def _format_large_yaxis(ax: plt.Axes) -> None:
    y_min, y_max = ax.get_ylim()
    formatter = ScalarFormatter(useMathText=True)
    max_abs = max(abs(y_min), abs(y_max))
    if max_abs != 0 and (max_abs >= 1000 or max_abs < 0.01):
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
    else:
        formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)



def _apply_layout(fig: plt.Figure, hspace: float, wspace: float, *, top: float = 0.95) -> None:
    fig.subplots_adjust(
        left=0.07,
        right=0.945,
        bottom=0.08,
        top=top,
        hspace=hspace,
        wspace=wspace,
    )



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
    xlabel: str = "Cell type",
    show_title: bool = True,
) -> None:
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
                [vals],
                positions=[pos],
                widths=0.7,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.3,
            )
            for body in vp["bodies"]:
                body.set_alpha(0.22)
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_linewidth(1.5)
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(pos, q25, q75, color="dimgray", linewidth=2.5, zorder=3)
            ax.hlines(q50, pos - 0.06, pos + 0.06, color="dimgray", linewidth=3, zorder=4)
        if len(vals):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=25,
                color=color,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )

    means = [float(np.mean(v)) if len(v) else np.nan for v in data]
    medians = [float(np.median(v)) if len(v) else np.nan for v in data]
    ax.scatter(
        positions,
        means,
        marker="o",
        s=50,
        color="black",
        label="mean",
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.scatter(
        positions,
        medians,
        marker="D",
        s=40,
        color="tab:red",
        label="median",
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )

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
                        lookup[ct],
                        pos - 0.35,
                        pos + 0.35,
                        colors="tab:green",
                        linewidths=2.0,
                        linestyles="--",
                        label="target" if pos == positions[0] else None,
                        zorder=6,
                    )

    if type_labels is not None:
        tick_labels = [type_labels[ct] if ct < len(type_labels) else str(ct) for ct in cell_types]
    else:
        tick_labels = [str(ct) for ct in cell_types]

    ax.set_xticks(positions, tick_labels)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    _set_panel_title(ax, title, show_title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize=10)



def _read_scalar_float(lines: list[str], name: str, n: int) -> np.ndarray:
    header = f"SCALARS {name}"
    idx = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if idx is None:
        raise ValueError(f"Scalar '{name}' not found in VTK")
    start = idx + 2
    vals = np.empty(n, dtype=float)
    for k in range(n):
        vals[k] = float(lines[start + k].split()[0])
    return vals



def _count_cells(ids: np.ndarray) -> int:
    seen: set[int] = set()
    for i, value in enumerate(ids):
        if value in seen:
            return i
        seen.add(int(value))
    return len(ids)



def _read_vtk_cells(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text()
    lines = text.splitlines()

    pts_idx = next(i for i, line in enumerate(lines) if line.startswith("POINTS"))
    n_points = int(lines[pts_idx].split()[1])
    coords = np.empty((n_points, 3), dtype=float)
    for k in range(n_points):
        parts = lines[pts_idx + 1 + k].split()
        coords[k, 0] = float(parts[0])
        coords[k, 1] = float(parts[1])
        coords[k, 2] = float(parts[2])

    ids_all = _read_scalar_float(lines, "id", n_points).astype(int)
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
    pairs: list[tuple[int, Path]] = []
    for path in vtk_dir.glob("cells_t*.vtk"):
        match = re.search(r"t(\d+)\.vtk$", path.name)
        if match:
            pairs.append((int(match.group(1)), path))
    pairs.sort(key=lambda item: item[0])
    return pairs



def build_trajectory_data(vtk_dir: Path) -> dict[str, Any]:
    file_list = _discover_vtk_files(vtk_dir)
    if not file_list:
        raise FileNotFoundError(f"No cells_t*.vtk files found in {vtk_dir}")

    timesteps = np.array([timestep for timestep, _ in file_list])
    n_steps = len(timesteps)

    all_ids: set[int] = set()
    type_map: dict[int, int] = {}
    alive_map: dict[int, bool] = {}
    frame_data: list[dict[str, np.ndarray]] = []

    for _timestep, fpath in file_list:
        data = _read_vtk_cells(fpath)
        frame_data.append(data)
        for cid, ct, dead in zip(data["id"], data["cell_type"], data["dead"]):
            all_ids.add(int(cid))
            type_map[int(cid)] = int(ct)
            alive_map[int(cid)] = int(dead) == 0

    positions: dict[int, np.ndarray] = {
        cid: np.full((n_steps, 3), np.nan, dtype=float) for cid in all_ids
    }
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
    timesteps = traj_data["timesteps"]
    positions = traj_data["positions"]
    cell_types = traj_data["cell_types"]
    n_steps = len(timesteps)

    vmean_lookup: dict[int, float] = {}
    if "id" in speed_metrics.columns and "vmean" in speed_metrics.columns:
        for _, row in speed_metrics.iterrows():
            vmean_lookup[int(row["id"])] = float(row["vmean"])

    dt = time_step if time_step is not None else 1.0

    ratios: dict[int, np.ndarray] = {}
    for cid, pos in positions.items():
        ratio = np.full(n_steps, np.nan)
        valid_indices = np.where(~np.isnan(pos[:, 0]))[0]
        if len(valid_indices) < 2:
            ratios[cid] = ratio
            continue

        vmean = vmean_lookup.get(cid)
        first_idx = valid_indices[0]
        origin = pos[first_idx].copy()
        t0 = float(timesteps[first_idx])

        for ti in valid_indices:
            curr = pos[ti]
            straight = np.linalg.norm(curr - origin)
            elapsed = (float(timesteps[ti]) - t0) * dt

            if vmean is None or vmean <= 1e-15:
                continue

            true_path = vmean * elapsed
            if true_path > 1e-12:
                ratio[ti] = straight / true_path
            else:
                ratio[ti] = 1.0
        ratios[cid] = ratio

    return {
        "timesteps": timesteps,
        "ratios": ratios,
        "cell_types": cell_types,
    }



def _plot_directionality_panel(
    ax: plt.Axes,
    dir_data: dict[str, Any],
    label: str,
    *,
    time_step: float | None,
    type_labels: list[str] | None,
    show_titles: bool,
) -> None:
    cell_types_map = dir_data["cell_types"]
    ratios = dir_data["ratios"]
    timesteps = dir_data["timesteps"]
    all_types = sorted(set(cell_types_map.values()))

    if time_step is not None:
        x_vals = timesteps.astype(float) * time_step / 3600.0
        x_label = "Time [h]"
    else:
        x_vals = timesteps.astype(float)
        x_label = "Timestep"

    for ct in all_types:
        color = _color_for(ct)
        type_lbl = type_labels[ct] if type_labels and ct < len(type_labels) else f"type {ct}"
        ct_cids = [cid for cid, value in cell_types_map.items() if value == ct]
        if not ct_cids:
            continue
        ratio_stack = np.array([ratios[cid] for cid in ct_cids if cid in ratios])
        if ratio_stack.size == 0:
            continue
        mean_ratio = np.nanmean(ratio_stack, axis=0)
        std_ratio = np.nanstd(ratio_stack, axis=0)
        valid = ~np.isnan(mean_ratio)
        ax.plot(x_vals[valid], mean_ratio[valid], color=color, linewidth=1.5, label=type_lbl)
        ax.fill_between(
            x_vals[valid],
            (mean_ratio - std_ratio)[valid],
            (mean_ratio + std_ratio)[valid],
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Directionality ratio", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    _set_panel_title(ax, f"{label} - directionality", show_titles, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=10)



def _style_scalar_profile_axis(ax: plt.Axes, *, title: str, show_titles: bool) -> None:
    _set_panel_title(ax, title, show_titles, fontsize=11, fontweight="bold")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25)



def _hide_panel(ax: plt.Axes) -> None:
    ax.axis("off")



def create_combined_figure(
    metrics1: pd.DataFrame,
    target1: pd.DataFrame | None,
    label1: str,
    metrics2: pd.DataFrame,
    target2: pd.DataFrame | None,
    label2: str,
    dir_data1: dict[str, Any] | None = None,
    dir_data2: dict[str, Any] | None = None,
    *,
    scalar_parent1: str | Path | None = None,
    scalar_parent2: str | Path | None = None,
    scalar_name: str | None = None,
    line_coords: tuple[float, float, float, float, float, float] | None = None,
    time_step: float | None = None,
    ymax_vmean: float | None = None,
    ymax_veff: float | None = None,
    figsize: tuple[float, float] = (20, 15),
    type_labels: list[str] | None = None,
    show_titles: bool = True,
    hspace: float = 0.35,
    wspace: float = 0.42,
    scalar_cmap: str = "viridis",
    scalar_linewidth: float = 1.5,
    scalar_alpha: float = 0.9,
    scalar_tol: float = 1e-8,
    scalar_ylabel: str | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4)

    all_vmean = np.concatenate([
        metrics1["vmean"].to_numpy(dtype=float),
        metrics2["vmean"].to_numpy(dtype=float),
    ])
    all_veff = np.concatenate([
        metrics1["veff"].to_numpy(dtype=float),
        metrics2["veff"].to_numpy(dtype=float),
    ])
    if ymax_vmean is None:
        ymax_vmean = float(np.nanmax(all_vmean[np.isfinite(all_vmean)])) * 1.10
    if ymax_veff is None:
        ymax_veff = float(np.nanmax(all_veff[np.isfinite(all_veff)])) * 1.10

    ax_v1 = fig.add_subplot(gs[0, 0])
    ax_v2 = fig.add_subplot(gs[0, 1])
    ax_v3 = fig.add_subplot(gs[0, 2])
    ax_v4 = fig.add_subplot(gs[0, 3])
    _plot_violin_panel(
        ax_v1,
        metrics1,
        "vmean",
        "vmean [µm/s]",
        target1,
        f"vmean - {label1}",
        show_legend=True,
        type_labels=type_labels,
        xlabel=label1,
        show_title=show_titles,
    )
    _plot_violin_panel(
        ax_v2,
        metrics2,
        "vmean",
        "vmean [µm/s]",
        target2,
        f"vmean - {label2}",
        show_legend=False,
        type_labels=type_labels,
        xlabel=label2,
        show_title=show_titles,
    )
    _plot_violin_panel(
        ax_v3,
        metrics1,
        "veff",
        "veff [µm/s]",
        target1,
        f"veff - {label1}",
        show_legend=False,
        type_labels=type_labels,
        xlabel=label1,
        show_title=show_titles,
    )
    _plot_violin_panel(
        ax_v4,
        metrics2,
        "veff",
        "veff [µm/s]",
        target2,
        f"veff - {label2}",
        show_legend=False,
        type_labels=type_labels,
        xlabel=label2,
        show_title=show_titles,
    )
    ax_v1.set_ylim(0, ymax_vmean)
    ax_v2.set_ylim(0, ymax_vmean)
    ax_v3.set_ylim(0, ymax_veff)
    ax_v4.set_ylim(0, ymax_veff)
    _format_large_yaxis(ax_v1)
    _format_large_yaxis(ax_v2)
    _format_large_yaxis(ax_v3)
    _format_large_yaxis(ax_v4)

    ax_dir1 = fig.add_subplot(gs[1, 0])
    ax_dir2 = fig.add_subplot(gs[1, 1])
    ax_blank_21 = fig.add_subplot(gs[1, 2])
    ax_blank_22 = fig.add_subplot(gs[1, 3])

    if dir_data1 is not None:
        _plot_directionality_panel(
            ax_dir1,
            dir_data1,
            label1,
            time_step=time_step,
            type_labels=type_labels,
            show_titles=show_titles,
        )
    else:
        _hide_panel(ax_dir1)

    if dir_data2 is not None:
        _plot_directionality_panel(
            ax_dir2,
            dir_data2,
            label2,
            time_step=time_step,
            type_labels=type_labels,
            show_titles=show_titles,
        )
    else:
        _hide_panel(ax_dir2)

    _hide_panel(ax_blank_21)
    _hide_panel(ax_blank_22)

    ax_scalar1 = fig.add_subplot(gs[2, 0])
    ax_scalar2 = fig.add_subplot(gs[2, 1])
    ax_blank_31 = fig.add_subplot(gs[2, 2])
    ax_blank_32 = fig.add_subplot(gs[2, 3])

    can_plot_scalar = (
        scalar_parent1 is not None
        and scalar_parent2 is not None
        and scalar_name is not None
        and line_coords is not None
    )
    if can_plot_scalar:
        x1, y1, z1, x2, y2, z2 = line_coords
        plot_scalar_line_evolution_comparison(
            parent_folder1=scalar_parent1,
            parent_folder2=scalar_parent2,
            scalar_name=scalar_name,
            x1=x1,
            y1=y1,
            z1=z1,
            x2=x2,
            y2=y2,
            z2=z2,
            tol=scalar_tol,
            cmap=scalar_cmap,
            linewidth=scalar_linewidth,
            alpha=scalar_alpha,
            title1=label1 if show_titles else "",
            title2=label2 if show_titles else "",
            axes=[ax_scalar1, ax_scalar2],
            ylabel=scalar_ylabel,
        )
        _style_scalar_profile_axis(ax_scalar1, title=label1, show_titles=show_titles)
        _style_scalar_profile_axis(ax_scalar2, title=label2, show_titles=show_titles)
    else:
        _hide_panel(ax_scalar1)
        _hide_panel(ax_scalar2)

    _hide_panel(ax_blank_31)
    _hide_panel(ax_blank_32)

    _apply_layout(fig, hspace, wspace)
    return fig



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare migration speed, directionality, and diffusion profiles for two conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pickle1", required=True, help="Pickle file for condition 1")
    parser.add_argument("--pickle2", required=True, help="Pickle file for condition 2")
    parser.add_argument("--label1", default="condition 1", help="Display label for condition 1")
    parser.add_argument("--label2", default="condition 2", help="Display label for condition 2")
    parser.add_argument("--target1", default=None, help="Target CSV for condition 1")
    parser.add_argument("--target2", default=None, help="Target CSV for condition 2")
    parser.add_argument("--vtk-dir1", default=None, help="Directory with cells_tXXXX.vtk and ecm_data_tXXXX.vtk for condition 1")
    parser.add_argument("--vtk-dir2", default=None, help="Directory with cells_tXXXX.vtk and ecm_data_tXXXX.vtk for condition 2")
    parser.add_argument("--scalar-name", default=None, help="Scalar name for row 3 diffusion-profile plots")
    parser.add_argument("--x1", type=float, default=None, help="First point x for scalar profile extraction")
    parser.add_argument("--y1", type=float, default=None, help="First point y for scalar profile extraction")
    parser.add_argument("--z1", type=float, default=None, help="First point z for scalar profile extraction")
    parser.add_argument("--x2", type=float, default=None, help="Second point x for scalar profile extraction")
    parser.add_argument("--y2", type=float, default=None, help="Second point y for scalar profile extraction")
    parser.add_argument("--z2", type=float, default=None, help="Second point z for scalar profile extraction")
    parser.add_argument("--scalar-tol", type=float, default=1e-8, help="Tolerance for scalar profile extraction")
    parser.add_argument("--scalar-cmap", default="viridis", help="Colormap for scalar profile evolution")
    parser.add_argument("--scalar-linewidth", type=float, default=1.5, help="Line width for scalar profiles")
    parser.add_argument("--scalar-alpha", type=float, default=0.9, help="Alpha for scalar profiles")
    parser.add_argument("--scalar-ylabel", default=None, help="Custom y-axis label for scalar profile plots (row 3)")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output figure")
    parser.add_argument("--tag", default="comparison", help="Suffix tag for output filename")
    parser.add_argument("--show", action="store_true", help="Display figure interactively")
    parser.add_argument("--ymax-vmean", type=float, default=None, help="Upper y-axis limit for vmean panels")
    parser.add_argument("--ymax-veff", type=float, default=None, help="Upper y-axis limit for veff panels")
    parser.add_argument("--figsize", type=float, nargs=2, default=[20, 15], metavar=("W", "H"), help="Figure size in inches")
    parser.add_argument("--dpi", type=int, default=600, help="Resolution for saved figure")
    parser.add_argument(
        "--type-labels",
        nargs="+",
        default=None,
        help="Custom labels for cell types, for example --type-labels Epithelial Mesenchymal Stem",
    )
    parser.add_argument("--no-titles", action="store_true", help="Remove subplot titles from all panels")
    parser.add_argument("--hspace", type=float, default=0.35, help="Vertical spacing between subplot rows")
    parser.add_argument("--wspace", type=float, default=0.42, help="Horizontal spacing between subplot columns")
    return parser.parse_args()



def _get_time_step(results: dict[str, Any]) -> float | None:
    cfg = results.get("MODEL_CONFIG")
    if cfg is not None:
        return getattr(cfg, "TIME_STEP", None)
    return None



def _parse_line_coords(args: argparse.Namespace) -> tuple[float, float, float, float, float, float] | None:
    coords = [args.x1, args.y1, args.z1, args.x2, args.y2, args.z2]
    if all(value is not None for value in coords):
        return (
            float(args.x1),
            float(args.y1),
            float(args.z1),
            float(args.x2),
            float(args.y2),
            float(args.z2),
        )
    return None



def main() -> None:
    args = parse_args()

    pkl1 = _load_pickle(Path(args.pickle1))
    pkl2 = _load_pickle(Path(args.pickle2))
    metrics1 = load_speed_metrics(pkl1)
    metrics2 = load_speed_metrics(pkl2)
    target1 = load_target_csv(Path(args.target1)) if args.target1 else None
    target2 = load_target_csv(Path(args.target2)) if args.target2 else None
    time_step = _get_time_step(pkl1)

    dir1 = None
    dir2 = None
    vtk_dir1 = Path(args.vtk_dir1) if args.vtk_dir1 else None
    vtk_dir2 = Path(args.vtk_dir2) if args.vtk_dir2 else None

    if vtk_dir1 is not None:
        print(f"Loading VTK trajectories from {vtk_dir1} ...")
        traj1 = build_trajectory_data(vtk_dir1)
        dir1 = compute_directionality_ratios(traj1, metrics1, time_step)

    if vtk_dir2 is not None:
        print(f"Loading VTK trajectories from {vtk_dir2} ...")
        traj2 = build_trajectory_data(vtk_dir2)
        dir2 = compute_directionality_ratios(traj2, metrics2, time_step)

    line_coords = _parse_line_coords(args)
    if args.scalar_name is not None and line_coords is None:
        raise ValueError(
            "Scalar profile plotting requires all six coordinates: --x1 --y1 --z1 --x2 --y2 --z2"
        )

    print("Creating combined migration and diffusion-profile figure ...")
    fig = create_combined_figure(
        metrics1=metrics1,
        target1=target1,
        label1=args.label1,
        metrics2=metrics2,
        target2=target2,
        label2=args.label2,
        dir_data1=dir1,
        dir_data2=dir2,
        scalar_parent1=vtk_dir1,
        scalar_parent2=vtk_dir2,
        scalar_name=args.scalar_name,
        line_coords=line_coords,
        time_step=time_step,
        ymax_vmean=args.ymax_vmean,
        ymax_veff=args.ymax_veff,
        figsize=tuple(args.figsize),
        type_labels=args.type_labels,
        show_titles=not args.no_titles,
        hspace=args.hspace,
        wspace=args.wspace,
        scalar_cmap=args.scalar_cmap,
        scalar_linewidth=args.scalar_linewidth,
        scalar_alpha=args.scalar_alpha,
        scalar_tol=args.scalar_tol,
        scalar_ylabel=args.scalar_ylabel,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"migration_diff_profiles_comparison_{args.tag}.png"
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved -> {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("Done.")


if __name__ == "__main__":
    main()
