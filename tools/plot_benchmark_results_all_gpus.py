from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Import the existing single-CSV utilities.
# This script is expected to live next to plot_benchmark_results.py inside tools/.
from plot_benchmark_results import (
    DEFAULT_OUTDIR,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_SMALL,
    FONT_SIZE_TEXT,
    FONT_SIZE_TICKS,
    FONT_SIZE_TITLE,
    FONT_SIZE_SUPTITLE,
    YLABEL_INIT,
    YLABEL_STEP,
    XLABEL_TOTAL_AGENTS,
    CONTOUR_COLS,
    _apply_plot_style,
    _draw_pie_scatter,
    _fit_multivariate_loglog,
    _interaction_term_name,
    _label,
    _load,
    _pie_agent_cols,
    _savefig,
    _set_compact_log_ticks,
    _varied_cols,
)


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CSVS = [
    TOOLS_DIR / "benchmark_results_gtx1050ti.csv",
    TOOLS_DIR / "benchmark_results_rtx5000.csv",
    TOOLS_DIR / "benchmark_results_A100.csv",
    TOOLS_DIR / "benchmark_results_h100-nvl.csv",
]
GPU_NAMES = ["GTX-1050Ti", "RTX-5000", "A100", "H100-NVL"]
AGENT_PIE_COLORS = ["#0e6377", "#87d1d5", "#f9cb37", "#f37c20"]


# Base variables for the multivariate plot
MULTIVARIATE_BASE_VARS = {
    "CELL": "x_1",
    "R_cell": "x_2",
    "FOCAD/CELL": "x_3",
    "ECM": "x_4",
    "FNODES": "x_5",
}

def _format_multivariate_term(term: str) -> str:
    """
    Converts terms like 'CELL x FNODES' into '$x_1*x_5$' and 'CELL^2' to '$x_1^2$'.
    Sorts keys by length descending to prevent 'CELL' from overwriting 'FOCAD/CELL'.
    """
    res = term
    for k in sorted(MULTIVARIATE_BASE_VARS.keys(), key=len, reverse=True):
        res = res.replace(k, MULTIVARIATE_BASE_VARS[k])
    # Replace the " x " interaction separator with an asterisk
    res = res.replace(r"\times", "*").replace(" x ", "*").replace("×", "*")
    return f"${res}$"


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _save_publication_set(fig: plt.Figure, outdir: Path, stem: str, dpi: int = 600) -> None:
    """Save PNG only.

    TIFF export was removed because it was failing silently for large figures.
    PNG at 600 dpi is saved here, and other publication formats can be derived
    externally if needed.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = outdir / f"{stem}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved {png_path}")


def _set_explicit_log_xticks(ax: plt.Axes, xlim: tuple[float, float]) -> None:
    """Force visible log x-ticks with explicit labels."""
    xmin, xmax = xlim
    if xmin <= 0 or xmax <= 0:
        return
    pmin = int(np.floor(np.log10(xmin)))
    pmax = int(np.ceil(np.log10(xmax)))
    ticks = [10.0 ** p for p in range(pmin, pmax + 1)]
    ticks = [t for t in ticks if xmin <= t <= xmax]
    if len(ticks) < 2:
        return
    ax.set_xticks(ticks)
    labels = []
    for t in ticks:
        p = int(round(np.log10(t)))
        labels.append(f"1e{p}")
    ax.set_xticklabels(labels)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all(csv_paths: Iterable[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for gpu_name, csv_path in zip(GPU_NAMES, csv_paths):
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for {gpu_name}: {csv_path}")
        df = _load(csv_path)
        df["GPU_NAME"] = gpu_name
        frames.append(df)
    return frames


# ---------------------------------------------------------------------------
# Shared legends
# ---------------------------------------------------------------------------

def _add_axis_pie_legend(ax: plt.Axes, pie_cols: list[str]) -> None:
    handles = [
        mpatches.Patch(color=AGENT_PIE_COLORS[i], label=_label(col))
        for i, col in enumerate(pie_cols)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.35,
    )


# ---------------------------------------------------------------------------
# 1. Combined init/step time scatter
# ---------------------------------------------------------------------------

def plot_time_scatter_all(frames: list[pd.DataFrame], outdir: Path, dpi: int = 600) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.2), sharex=False, sharey=False)
    pie_cols = _pie_agent_cols(frames[0])

    xmins: list[float] = []
    xmaxs: list[float] = []
    init_ymins: list[float] = []
    init_ymaxs: list[float] = []
    step_ymins: list[float] = []
    step_ymaxs: list[float] = []

    for df in frames:
        x = df["TOTAL_AGENTS"].to_numpy(dtype=float)
        init_y = df["init_time_s"].to_numpy(dtype=float)
        step_y = df["time_per_step_s"].to_numpy(dtype=float)

        mask_init = (x > 0) & np.isfinite(x) & np.isfinite(init_y) & (init_y > 0)
        mask_step = (x > 0) & np.isfinite(x) & np.isfinite(step_y) & (step_y > 0)

        if np.any(mask_init):
            xmins.append(float(np.min(x[mask_init])))
            xmaxs.append(float(np.max(x[mask_init])))
            init_ymins.append(float(np.min(init_y[mask_init])))
            init_ymaxs.append(float(np.max(init_y[mask_init])))
        if np.any(mask_step):
            xmins.append(float(np.min(x[mask_step])))
            xmaxs.append(float(np.max(x[mask_step])))
            step_ymins.append(float(np.min(step_y[mask_step])))
            step_ymaxs.append(float(np.max(step_y[mask_step])))

    global_xlim = (
        min(xmins) * 0.85 if xmins else 1.0,
        max(xmaxs) * 1.15 if xmaxs else 10.0,
    )
    global_init_ylim = (
        min(init_ymins) * 0.85 if init_ymins else 1e-3,
        max(init_ymaxs) * 1.15 if init_ymaxs else 1.0,
    )
    global_step_ylim = (
        min(step_ymins) * 0.85 if step_ymins else 1e-6,
        max(step_ymaxs) * 1.15 if step_ymaxs else 1.0,
    )

    for j, (gpu_name, df) in enumerate(zip(GPU_NAMES, frames)):
        x = df["TOTAL_AGENTS"].to_numpy(dtype=float)

        # Top row: init time
        ax = axes[0, j]
        y = df["init_time_s"].to_numpy(dtype=float)
        mask = (x > 0) & np.isfinite(x) & np.isfinite(y) & (y > 0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*global_xlim)
        ax.set_ylim(*global_init_ylim)
        ax.scatter(x[mask], y[mask], s=1, alpha=0)
        _draw_pie_scatter(ax, df.loc[mask], "TOTAL_AGENTS", "init_time_s", pie_cols, size_pt=7.3, alpha=0.80)
        ax.grid(True, which="both", ls=":", lw=0.35, alpha=0.6)
        ax.set_title(gpu_name, fontsize=FONT_SIZE_TITLE + 4, pad=8)
        if j == 0:
            ax.set_ylabel(YLABEL_INIT, fontsize=FONT_SIZE_LABEL + 3)
            _add_axis_pie_legend(ax, pie_cols)
        ax.set_xlabel(XLABEL_TOTAL_AGENTS, fontsize=FONT_SIZE_LABEL + 2)
        ax.tick_params(labelsize=FONT_SIZE_TICKS + 1, labelbottom=True)
        _set_compact_log_ticks(ax)
        _set_explicit_log_xticks(ax, global_xlim)

        # Bottom row: step time
        ax = axes[1, j]
        y = df["time_per_step_s"].to_numpy(dtype=float)
        mask = (x > 0) & np.isfinite(x) & np.isfinite(y) & (y > 0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*global_xlim)
        ax.set_ylim(*global_step_ylim)
        ax.scatter(x[mask], y[mask], s=1, alpha=0)
        _draw_pie_scatter(ax, df.loc[mask], "TOTAL_AGENTS", "time_per_step_s", pie_cols, size_pt=7.3, alpha=0.80)
        ax.grid(True, which="both", ls=":", lw=0.35, alpha=0.6)
        if j == 0:
            ax.set_ylabel(YLABEL_STEP, fontsize=FONT_SIZE_LABEL + 3)
        ax.set_xlabel(XLABEL_TOTAL_AGENTS, fontsize=FONT_SIZE_LABEL + 2)
        ax.tick_params(labelsize=FONT_SIZE_TICKS + 1, labelbottom=True)
        _set_compact_log_ticks(ax)
        _set_explicit_log_xticks(ax, global_xlim)

    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.98])
    _save_publication_set(fig, outdir, "time_scatter_all", dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# 2. Combined multivariate summary panels
# ---------------------------------------------------------------------------

def _ordered_term_data(fit: dict[str, object]) -> tuple[list[str], np.ndarray]:
    coeffs = np.asarray(fit["coeffs"])[1:]
    term_names = list(fit["term_names"])
    order = np.argsort(np.abs(coeffs))[::-1]
    ordered_terms = [term_names[idx] for idx in order]
    ordered_coeffs = coeffs[order]
    return ordered_terms, ordered_coeffs


def _draw_multivariate_block(
    fig: plt.Figure,
    gridspec_cell,
    df: pd.DataFrame,
    gpu_name: str,
    init_fit: dict | None,
    step_fit: dict | None,
    is_first: bool = False
) -> None:
    subgs = gridspec_cell.subgridspec(2, 2, wspace=0.26, hspace=0.28)
    pie_cols = _pie_agent_cols(df)

    if init_fit is None or step_fit is None:
        ax = fig.add_subplot(subgs[:, :])
        ax.axis("off")
        ax.text(0.5, 0.5, f"{gpu_name}\nFit unavailable", ha="center", va="center", fontsize=FONT_SIZE_TITLE + 2)
        return

    panel_specs = [
        (subgs[0, 0], init_fit, YLABEL_INIT, f"{gpu_name}\nInit-time fit", True),
        (subgs[1, 0], step_fit, YLABEL_STEP, "Step-time fit", False),
    ]

    for spec, fit, y_label, title, is_init_panel in panel_specs:
        ax = fig.add_subplot(spec)
        pred_y = np.asarray(fit["pred_y"], dtype=float)
        y = np.asarray(fit["y"], dtype=float)
        fit_rows = df.loc[np.asarray(fit["mask"], dtype=bool)].copy()

        ax.scatter(pred_y, y, s=0, alpha=0)
        fit_rows = fit_rows.reset_index(drop=True)
        fit_rows["_plot_x"] = pred_y
        fit_rows["_plot_y"] = y
        _draw_pie_scatter(ax, fit_rows, "_plot_x", "_plot_y", pie_cols, size_pt=5.3, alpha=0.82)

        lo = min(pred_y.min(), y.min())
        hi = max(pred_y.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _set_compact_log_ticks(ax)
        ax.grid(True, which="both", ls=":", lw=0.3, alpha=0.6)
        ax.set_xlabel(f"Predicted {y_label}", fontsize=FONT_SIZE_SMALL + 1)
        ax.set_ylabel(f"Observed {y_label}", fontsize=FONT_SIZE_SMALL + 1)
        ax.set_title(f"{title} ($R^2={fit['r2']:.3f}$)", fontsize=FONT_SIZE_SMALL + 2, pad=4)
        ax.tick_params(labelsize=FONT_SIZE_TEXT)
        
        # Add pie legend to the very first subplot
        if is_first and is_init_panel:
            _add_axis_pie_legend(ax, pie_cols)

    for spec, fit, title, is_init_term in [
        (subgs[0, 1], init_fit, "Init-term strengths", True),
        (subgs[1, 1], step_fit, "Step-term strengths", False),
    ]:
        ax = fig.add_subplot(spec)
        ordered_terms, ordered_coeffs = _ordered_term_data(fit)
        colours = ["C0" if coef >= 0 else "C3" for coef in ordered_coeffs]
        
        mapped_terms = [_format_multivariate_term(t) for t in ordered_terms]
        
        ax.barh(mapped_terms, ordered_coeffs, color=colours, edgecolor="k", lw=0.35)
        ax.axvline(0.0, color="0.35", lw=0.7)
        ax.grid(axis="x", ls=":", lw=0.3, alpha=0.6)
        ax.set_xlabel("Std. coeff.", fontsize=FONT_SIZE_SMALL + 1)
        ax.set_title(title, fontsize=FONT_SIZE_SMALL + 2, pad=4)
        ax.tick_params(axis="x", labelsize=FONT_SIZE_TEXT)
        ax.tick_params(axis="y", labelsize=max(7, FONT_SIZE_TEXT + 1))

        # Add the base term x_i legend to the first term subplot
        if is_first and is_init_term:
            handles = [Line2D([], [], color="none", label=f"${v}$ = {k}") for k, v in MULTIVARIATE_BASE_VARS.items()]
            ax.legend(
                handles=handles,
                loc="upper right",
                frameon=True,
                framealpha=0.85,
                handlelength=0,
                handletextpad=0,
                fontsize=FONT_SIZE_LEGEND,
                borderpad=0.4,
                labelspacing=0.2,
            )


def plot_multivariate_all(frames: list[pd.DataFrame], outdir: Path, dpi: int = 600) -> plt.Figure:
    
    # Generate fits
    fits = []
    for df in frames:
        init_fit = _fit_multivariate_loglog(df, "init_time_s")
        step_fit = _fit_multivariate_loglog(df, "time_per_step_s")
        fits.append((init_fit, step_fit))

    # Draw panels
    fig = plt.figure(figsize=(20, 18))
    outer = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.18)

    for idx, (gpu_name, df, (init_fit, step_fit)) in enumerate(zip(GPU_NAMES, frames, fits)):
        r, c = divmod(idx, 2)
        _draw_multivariate_block(fig, outer[r, c], df, gpu_name, init_fit, step_fit, is_first=(idx == 0))

    # We removed the global figure legend at the bottom; adjust margins back to normal
    fig.tight_layout(rect=[0.015, 0.015, 1.0, 0.98])
    _save_publication_set(fig, outdir, "multivariate_all", dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# 3. Combined contourf panels
# ---------------------------------------------------------------------------

def _draw_contour_block(fig: plt.Figure, gridspec_cell, df: pd.DataFrame, gpu_name: str) -> None:
    cols = _varied_cols(df, CONTOUR_COLS)
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    if not pairs:
        ax = fig.add_subplot(gridspec_cell)
        ax.axis("off")
        ax.text(0.5, 0.5, f"{gpu_name}\nNo contour pairs", ha="center", va="center", fontsize=FONT_SIZE_TITLE + 2)
        return

    # Increased wspace and hspace here to avoid overlapping
    subgs = gridspec_cell.subgridspec(2, 3, wspace=0.35, hspace=0.38)
    axes = [fig.add_subplot(subgs[i, j]) for i in range(2) for j in range(3)]

    for idx, (col_y, col_x) in enumerate(pairs):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        pivot = df.groupby([col_y, col_x])["time_per_step_s"].mean().reset_index()
        table = pivot.pivot(index=col_y, columns=col_x, values="time_per_step_s")
        table = table.sort_index(ascending=True)
        table = table[sorted(table.columns)]
        
        Z = table.values.astype(float)
        pos = Z[np.isfinite(Z) & (Z > 0)]
        
        zmin, zmax = 1e-6, 1.0
        if pos.size:
            zmin = float(np.min(pos))
            zmax = float(np.max(pos))

        use_log = (zmax / max(zmin, 1e-12)) > 10
        levels = np.logspace(np.log10(zmin), np.log10(zmax), 12) if use_log else 12
        norm = LogNorm(vmin=zmin, vmax=zmax) if use_log else None
        
        X_vals = np.array(table.columns, dtype=float)
        Y_vals = np.array(table.index, dtype=float)
        X_mesh, Y_mesh = np.meshgrid(X_vals, Y_vals)

        cf = ax.contourf(X_mesh, Y_mesh, Z, levels=levels, cmap="inferno", norm=norm)
        try:
            cs = ax.contour(X_mesh, Y_mesh, Z, levels=cf.levels, colors="w", linewidths=0.4, alpha=0.9)
            ax.clabel(cs, inline=True, fontsize=max(7, FONT_SIZE_TEXT - 1), fmt="%.2g")
        except Exception:
            pass

        ax.set_xlabel(_label(col_x), fontsize=FONT_SIZE_SMALL + 1)
        ax.set_ylabel(_label(col_y), fontsize=FONT_SIZE_SMALL + 1)
        ax.tick_params(labelsize=FONT_SIZE_TEXT)
        ax.ticklabel_format(style="scientific", scilimits=(0, 3), axis="both", useMathText=True)

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    # Block title using an invisible axis occupying the whole block
    ax_title = fig.add_subplot(gridspec_cell)
    ax_title.set_title(gpu_name, fontsize=FONT_SIZE_TITLE + 4, pad=18)
    ax_title.axis("off")


def plot_contourf_all(frames: list[pd.DataFrame], outdir: Path, dpi: int = 600) -> plt.Figure:
    fig = plt.figure(figsize=(22, 14))
    outer = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.22)

    for idx, (gpu_name, df) in enumerate(zip(GPU_NAMES, frames)):
        r, c = divmod(idx, 2)
        _draw_contour_block(fig, outer[r, c], df, gpu_name)

    fig.tight_layout(rect=[0.01, 0.02, 0.995, 0.985])
    _save_publication_set(fig, outdir, "contourf_all", dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create combined multi-GPU benchmark figures from 4 CSV files.",
    )
    parser.add_argument(
        "--csvs",
        nargs=4,
        default=[str(p) for p in DEFAULT_CSVS],
        help=(
            "Four benchmark CSV files in this order: "
            "GTX-1050Ti RTX-5000 A100 H100-NVL"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR / "all_gpus"),
        help="Directory for combined figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster DPI for PNG output. Default: 600",
    )
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    plt.rcParams["figure.max_open_warning"] = 0
    _apply_plot_style()

    csv_paths = [Path(p) for p in args.csvs]
    outdir = Path(args.outdir)

    print("Reading CSV files:")
    frames = _load_all(csv_paths)
    for gpu_name, csv_path, df in zip(GPU_NAMES, csv_paths, frames):
        print(f"  {gpu_name}: {csv_path} ({len(df)} successful runs)")

    print("\nGenerating combined figures:")
    figures = [
        plot_time_scatter_all(frames, outdir, dpi=args.dpi),
        plot_multivariate_all(frames, outdir, dpi=args.dpi),
        plot_contourf_all(frames, outdir, dpi=args.dpi),
    ]

    print(f"\nDone - {len(figures)} combined figures saved to {outdir}/")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()