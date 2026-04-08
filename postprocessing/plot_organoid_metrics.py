"""
Plot organoid metrics over time from simulation pickle files.

Reads ``ORGANOID_METRICS_OVER_TIME`` from one or two pickle files and
generates publication-quality time-series plots of:

  - Radius of gyration + Equivalent sphere radius (dual y-axis)
  - Sphericity
  - Maximum span (largest pairwise cell distance)
  - Number of alive cells (total + per cell type)

If two pickles are given, lines are overlaid on the same axes for
direct comparison.

Usage examples
--------------
# Single condition:
python postprocessing/plot_organoid_metrics.py ^
    --pickle1 result_files/organoid/output_data_0.pickle

# Two conditions for comparison:
python postprocessing/plot_organoid_metrics.py ^
    --pickle1 result_files/organoid_ctrl/output_data_0.pickle ^
    --pickle2 result_files/organoid_tgfb/output_data_0.pickle ^
    --label1 Control --label2 TGFb

# Custom limits and output:
python postprocessing/plot_organoid_metrics.py ^
    --pickle1 result_files/organoid/output_data_0.pickle ^
    --ylim-rg 0 200 --ylim-span 0 500 ^
    --figsize 10 8 --dpi 600 --output organoid_metrics.png

# Overlay a target CSV on panel 1 (Radius of gyration / ESR):
python postprocessing/plot_organoid_metrics.py ^
    --pickle1 result_files/organoid/output_data_0.pickle ^
    --target-csv optimizer/reference_data/target_organoid_size.csv

# Overlay a target CSV on panel 2 (Sphericity):
python postprocessing/plot_organoid_metrics.py ^
    --pickle1 result_files/organoid/output_data_0.pickle ^
    --target-csv optimizer/reference_data/target_organoid_sphericity.csv ^
    --target-panel 2
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Safe pickle loading
# ---------------------------------------------------------------------------

class _DummyModelParameterConfig:
    pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "helper_module" and name == "ModelParameterConfig":
            return _DummyModelParameterConfig
        # Handle pickles saved with old NumPy (numpy.core.*) on new NumPy (>=2.0)
        # where numpy.core is deprecated in favour of numpy._core.
        if module.startswith("numpy.core"):
            remapped = "numpy._core" + module[len("numpy.core"):]
            try:
                return getattr(importlib.import_module(remapped), name)
            except (ModuleNotFoundError, AttributeError):
                pass  # fall through to default
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
# Extract organoid metrics
# ---------------------------------------------------------------------------

def load_organoid_metrics(results: dict[str, Any]) -> pd.DataFrame:
    """Extract ORGANOID_METRICS_OVER_TIME dataframe from a pickle dict."""
    raw = results.get("ORGANOID_METRICS_OVER_TIME")
    if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
        raise ValueError("Pickle does not contain non-empty ORGANOID_METRICS_OVER_TIME")
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    return pd.DataFrame(raw)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Colour palette for per-type curves
TYPE_COLORS = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#8c564b"]

# Default colours for two conditions
COLORS_1 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
COLORS_2 = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b"]


def _time_to_hours(t: np.ndarray) -> np.ndarray:
    """Convert time in seconds to hours."""
    return t / 3600.0


def _parse_n_alive_type(series: pd.Series) -> np.ndarray:
    """Parse the comma-separated n_alive_type column into a 2-D array (rows × types)."""
    rows = []
    for val in series:
        if isinstance(val, str) and val:
            rows.append([int(x) for x in val.split(",")])
        else:
            rows.append([])
    if not rows or len(rows[0]) == 0:
        return np.empty((len(series), 0), dtype=int)
    return np.array(rows, dtype=int)


def plot_organoid_metrics(
    df1: pd.DataFrame,
    label1: str,
    df2: pd.DataFrame | None = None,
    label2: str | None = None,
    *,
    time_col: str = "time",
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 600,
    ylims: dict[str, tuple[float, float] | None] | None = None,
    xlim: tuple[float, float] | None = None,
    target_csv: Path | str | None = None,
    target_panel: int = 1,
) -> plt.Figure:
    """Create a 2×2 grid of time-series subplots for the organoid metrics.

    Layout:
      (1,1) Radius of gyration + Equivalent sphere radius (dual y-axis)
      (1,2) Sphericity
      (2,1) Max span
      (2,2) Number of alive cells (total + per cell type)

    Parameters
    ----------
    target_csv : path, optional
        CSV file with ``time,target_metric`` columns.  The ``time`` column
        is in seconds (converted to hours internally).  Data are shown as
        scatter points on the panel selected by *target_panel*.
    target_panel : int, optional
        Which panel receives the target scatter (1-4, default 1).
        1 = (1,1), 2 = (1,2), 3 = (2,1), 4 = (2,2).
    """
    ylims = ylims or {}

    # Load target CSV if provided
    target_x: np.ndarray | None = None
    target_y: np.ndarray | None = None
    if target_csv is not None:
        tgt = pd.read_csv(target_csv)
        if "time" not in tgt.columns or "target_metric" not in tgt.columns:
            raise ValueError("Target CSV must contain 'time' and 'target_metric' columns.")
        target_x = _time_to_hours(tgt["time"].values)
        target_y = tgt["target_metric"].values

    # Decide x-axis: prefer 'time' column (convert to hours); fall back to 'step'
    if time_col in df1.columns:
        x1 = _time_to_hours(df1[time_col].values)
        xlabel = "Time (h)"
    else:
        x1 = df1["step"].values
        xlabel = "Step"

    if df2 is not None:
        if time_col in df2.columns:
            x2 = _time_to_hours(df2[time_col].values)
        else:
            x2 = df2["step"].values

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # --- (1,1) Radius of gyration + Equivalent sphere radius (dual y) ---
    ax_rg = axes[0, 0]
    color_rg = "#1f77b4"
    color_esr = "#2ca02c"
    if "radius_of_gyration" in df1.columns:
        ln1 = ax_rg.plot(x1, df1["radius_of_gyration"].values, color=color_rg,
                         linewidth=1.5, label=f"Rg — {label1}")
    else:
        ln1 = []
    ax_esr = ax_rg.twinx()
    if "equivalent_sphere_radius" in df1.columns:
        ln2 = ax_esr.plot(x1, df1["equivalent_sphere_radius"].values, color=color_esr,
                          linewidth=1.5, linestyle="--", label=f"Equiv. R — {label1}")
    else:
        ln2 = []
    if df2 is not None:
        if "radius_of_gyration" in df2.columns:
            ln1 += ax_rg.plot(x2, df2["radius_of_gyration"].values, color=color_rg,
                              linewidth=1.5, linestyle=":", alpha=0.7, label=f"Rg — {label2}")
        if "equivalent_sphere_radius" in df2.columns:
            ln2 += ax_esr.plot(x2, df2["equivalent_sphere_radius"].values, color=color_esr,
                               linewidth=1.5, linestyle=":", alpha=0.7, label=f"Equiv. R — {label2}")
    ax_rg.set_xlabel(xlabel, fontsize=11)
    ax_rg.set_ylabel("Radius of gyration (µm)", fontsize=11, color=color_rg)
    ax_esr.set_ylabel("Equiv. sphere radius (µm)", fontsize=11, color=color_esr)
    ax_rg.tick_params(axis="y", labelcolor=color_rg, labelsize=10)
    ax_esr.tick_params(axis="y", labelcolor=color_esr, labelsize=10)
    ax_rg.tick_params(axis="x", labelsize=10)
    if "radius_of_gyration" in ylims and ylims["radius_of_gyration"] is not None:
        ax_rg.set_ylim(ylims["radius_of_gyration"])
    if xlim is not None:
        ax_rg.set_xlim(xlim)
    # Target scatter on panel 1 (plotted on Rg axis)
    if target_x is not None and target_panel == 1:
        # Preserve data-driven y-limits so scatter doesn't distort dual-axis alignment
        rg_ylim = ax_rg.get_ylim()
        esr_ylim = ax_esr.get_ylim()
        sc = ax_rg.scatter(target_x, target_y, marker="o", s=30, color="red",
                           zorder=5, label="target")
        ax_rg.set_ylim(rg_ylim)
        ax_esr.set_ylim(esr_ylim)
        ln1 = ln1 + [sc]
    lns = ln1 + ln2
    if lns:
        ax_rg.legend(lns, [l.get_label() for l in lns], fontsize=8, loc="upper left")

    # --- (1,2) Sphericity ---
    ax_sph = axes[0, 1]
    if "sphericity" in df1.columns:
        ax_sph.plot(x1, df1["sphericity"].values, color=COLORS_1[1], linewidth=1.5, label=label1)
    if df2 is not None and "sphericity" in df2.columns:
        ax_sph.plot(x2, df2["sphericity"].values, color=COLORS_2[1], linewidth=1.5,
                    linestyle="--", label=label2)
    ax_sph.set_xlabel(xlabel, fontsize=11)
    ax_sph.set_ylabel("Sphericity", fontsize=11)
    ax_sph.tick_params(labelsize=10)
    if "sphericity" in ylims and ylims["sphericity"] is not None:
        ax_sph.set_ylim(ylims["sphericity"])
    if target_x is not None and target_panel == 2:
        ax_sph.scatter(target_x, target_y, marker="o", s=30, color="red",
                       zorder=5, label="target")
    if xlim is not None:
        ax_sph.set_xlim(xlim)
    if df2 is not None or (target_x is not None and target_panel == 2):
        ax_sph.legend(fontsize=9)

    # --- (2,1) Max span ---
    ax_span = axes[1, 0]
    if "max_span" in df1.columns:
        ax_span.plot(x1, df1["max_span"].values, color=COLORS_1[2], linewidth=1.5, label=label1)
    if df2 is not None and "max_span" in df2.columns:
        ax_span.plot(x2, df2["max_span"].values, color=COLORS_2[2], linewidth=1.5,
                     linestyle="--", label=label2)
    ax_span.set_xlabel(xlabel, fontsize=11)
    ax_span.set_ylabel("Max span (µm)", fontsize=11)
    ax_span.tick_params(labelsize=10)
    if "max_span" in ylims and ylims["max_span"] is not None:
        ax_span.set_ylim(ylims["max_span"])
    if target_x is not None and target_panel == 3:
        ax_span.scatter(target_x, target_y, marker="o", s=30, color="red",
                        zorder=5, label="target")
    if xlim is not None:
        ax_span.set_xlim(xlim)
    if df2 is not None or (target_x is not None and target_panel == 3):
        ax_span.legend(fontsize=9)

    # --- (2,2) Number of alive cells (total + per type) ---
    ax_n = axes[1, 1]
    if "n_alive" in df1.columns:
        ax_n.plot(x1, df1["n_alive"].values, color=COLORS_1[3], linewidth=2, label=f"total — {label1}")
    # Per-type curves from n_alive_type column
    if "n_alive_type" in df1.columns:
        type_arr1 = _parse_n_alive_type(df1["n_alive_type"])
        for ti in range(type_arr1.shape[1]):
            ax_n.plot(x1, type_arr1[:, ti], color=TYPE_COLORS[ti % len(TYPE_COLORS)],
                      linewidth=1.2, label=f"type {ti} — {label1}")
    if df2 is not None:
        if "n_alive" in df2.columns:
            ax_n.plot(x2, df2["n_alive"].values, color=COLORS_2[3], linewidth=2,
                      linestyle="--", label=f"total — {label2}")
        if "n_alive_type" in df2.columns:
            type_arr2 = _parse_n_alive_type(df2["n_alive_type"])
            for ti in range(type_arr2.shape[1]):
                ax_n.plot(x2, type_arr2[:, ti], color=TYPE_COLORS[ti % len(TYPE_COLORS)],
                          linewidth=1.2, linestyle="--", alpha=0.7, label=f"type {ti} — {label2}")
    if target_x is not None and target_panel == 4:
        ax_n.scatter(target_x, target_y, marker="o", s=30, color="red",
                     zorder=5, label="target")
    ax_n.set_xlabel(xlabel, fontsize=11)
    ax_n.set_ylabel("Number of alive cells", fontsize=11)
    ax_n.tick_params(labelsize=10)
    if "n_alive" in ylims and ylims["n_alive"] is not None:
        ax_n.set_ylim(ylims["n_alive"])
    if xlim is not None:
        ax_n.set_xlim(xlim)
    ax_n.legend(fontsize=8, loc="best")

    # Secondary y-axis: scatter points at final time showing percentage of total
    if "n_alive" in df1.columns and len(df1) > 0:
        total_last = df1["n_alive"].iloc[-1]
        if total_last > 0:
            ax_pct = ax_n.twinx()
            ax_pct.set_ylabel("Final ratio (%)", fontsize=11)
            ax_pct.tick_params(labelsize=10)

            # Collect final values: (label, last_count, colour)
            final_pts: list[tuple[str, float, str]] = []
            final_pts.append(("total", total_last, COLORS_1[3]))
            if "n_alive_type" in df1.columns:
                type_arr1 = _parse_n_alive_type(df1["n_alive_type"])
                for ti in range(type_arr1.shape[1]):
                    final_pts.append((f"type {ti}", type_arr1[-1, ti],
                                      TYPE_COLORS[ti % len(TYPE_COLORS)]))

            # Plot scatter on percentage axis; sync so total aligns at 100%
            x_final = x1[-1]
            # Sync secondary axis: primary y-range maps proportionally
            y_lo, y_hi = ax_n.get_ylim()
            ax_pct.set_ylim(y_lo / total_last * 100.0, y_hi / total_last * 100.0)
            for lbl, val, clr in final_pts:
                pct = val / total_last * 100.0
                ax_pct.scatter(x_final, pct, marker="o", s=40, color=clr,
                               zorder=5, edgecolors="black", linewidths=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pickle1", type=str, default=None,
                    help="Path to the first simulation pickle file "
                         "(default: result_files/output_data_0.pickle).")
    p.add_argument("--pickle2", type=str, default=None,
                    help="Path to a second pickle for comparison (optional).")
    p.add_argument("--label1", type=str, default="Condition 1",
                    help="Legend label for the first condition.")
    p.add_argument("--label2", type=str, default="Condition 2",
                    help="Legend label for the second condition.")

    # Figure customisation
    p.add_argument("--figsize", type=float, nargs=2, default=[10, 8],
                    metavar=("W", "H"), help="Figure size in inches (default: 10 8).")
    p.add_argument("--dpi", type=int, default=600,
                    help="Resolution for saved figure (default: 600).")

    # Axis limits — each takes two floats (min max)
    p.add_argument("--ylim-rg", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Y-axis limits for radius of gyration.")
    p.add_argument("--ylim-sph", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Y-axis limits for sphericity.")
    p.add_argument("--ylim-span", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Y-axis limits for max span.")
    p.add_argument("--ylim-nalive", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Y-axis limits for number of alive cells.")
    p.add_argument("--xlim", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="X-axis limits (time or step) for all panels.")

    # Output
    p.add_argument("--output", type=str, default=None,
                    help="Output filename (default: organoid_metrics.png in results/).")
    p.add_argument("--show", action="store_true",
                    help="Show the figure interactively instead of saving.")

    # Target overlay
    p.add_argument("--target-csv", type=str, default=None,
                    help="CSV file with 'time,target_metric' columns to overlay as scatter.")
    p.add_argument("--target-panel", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Panel to overlay target scatter on (1-4, default: 1).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve pickle1 default
    if args.pickle1 is None:
        args.pickle1 = str(PROJECT_ROOT / "result_files" / "output_data_0.pickle")
        print(f"No --pickle1 given, using default: {args.pickle1}")

    # Load pickles
    results1 = _load_pickle(Path(args.pickle1))
    df1 = load_organoid_metrics(results1)

    df2 = None
    if args.pickle2:
        results2 = _load_pickle(Path(args.pickle2))
        df2 = load_organoid_metrics(results2)

    # Build ylims dict
    ylims: dict[str, tuple[float, float] | None] = {
        "radius_of_gyration":       tuple(args.ylim_rg) if args.ylim_rg else None,
        "sphericity":               tuple(args.ylim_sph) if args.ylim_sph else None,
        "max_span":                 tuple(args.ylim_span) if args.ylim_span else None,
        "n_alive":                  tuple(args.ylim_nalive) if args.ylim_nalive else None,
    }
    xlim = tuple(args.xlim) if args.xlim else None

    fig = plot_organoid_metrics(
        df1,
        args.label1,
        df2,
        args.label2,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        ylims=ylims,
        xlim=xlim,
        target_csv=args.target_csv,
        target_panel=args.target_panel,
    )

    if args.show:
        plt.show()
    else:
        out_dir = DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = args.output if args.output else "organoid_metrics.png"
        out_path = out_dir / out_name
        fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
