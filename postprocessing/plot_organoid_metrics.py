"""
Plot organoid metrics over time from simulation pickle files.

Reads ``ORGANOID_METRICS_OVER_TIME`` from one or two pickle files and
generates publication-quality time-series plots of:

  - Radius of gyration (Rg)
  - Equivalent sphere radius (Rg * sqrt(5/3))
  - Maximum span (largest pairwise cell distance)
  - Number of alive cells

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

# Metrics to plot — (column_name, axis_label, colour)
METRIC_DEFS = [
    ("radius_of_gyration",       "Radius of gyration (µm)"),
    ("equivalent_sphere_radius", "Equivalent sphere radius (µm)"),
    ("max_span",                 "Max span (µm)"),
    ("n_alive",                  "Number of alive cells"),
]

# Default colours for two conditions
COLORS_1 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
COLORS_2 = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b"]


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
) -> plt.Figure:
    """Create a 2×2 grid of time-series subplots for the organoid metrics."""
    ylims = ylims or {}

    # Decide x-axis: prefer 'time' column; fall back to 'step'
    if time_col in df1.columns:
        x1 = df1[time_col].values
        xlabel = "Time (s)"
    else:
        x1 = df1["step"].values
        xlabel = "Step"

    if df2 is not None:
        if time_col in df2.columns:
            x2 = df2[time_col].values
        else:
            x2 = df2["step"].values

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    for i, (col, ylabel) in enumerate(METRIC_DEFS):
        ax = axes[i]
        if col in df1.columns:
            ax.plot(x1, df1[col].values, color=COLORS_1[i], linewidth=1.5, label=label1)
        if df2 is not None and col in df2.columns:
            ax.plot(x2, df2[col].values, color=COLORS_2[i], linewidth=1.5,
                    linestyle="--", label=label2)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=10)

        # Apply custom y-limits if provided
        if col in ylims and ylims[col] is not None:
            ax.set_ylim(ylims[col])

        if xlim is not None:
            ax.set_xlim(xlim)

        if df2 is not None:
            ax.legend(fontsize=9)

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
    p.add_argument("--ylim-esr", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Y-axis limits for equivalent sphere radius.")
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
        "equivalent_sphere_radius": tuple(args.ylim_esr) if args.ylim_esr else None,
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
