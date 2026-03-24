"""
plot_benchmark_results.py – Visualise cellfoundry performance benchmark results.

Reads ``tools/benchmark_results.csv`` (produced by ``benchmark_perf.py``) and
generates a comprehensive set of publication-quality figures that characterise
how simulation cost scales with the number of agents and cell-radius parameter.

Usage
-----
    python tools/plot_benchmark_results.py                        # save only
    python tools/plot_benchmark_results.py --show                 # save + show
    python tools/plot_benchmark_results.py --csv path/to/other.csv
    python tools/plot_benchmark_results.py --outdir my_figs/

Figures produced
----------------
 1. **Pairwise heatmaps** – For every pair of swept columns, a heatmap of
     mean time/step.
 2. **Scaling curves (log-log)** – One panel per swept variable showing how
     time/step grows with that variable.
 3. **Total-agent scatter** – Time/step vs. total agents with pie-chart
     markers showing agent composition.
 4. **Per-variable scaling exponents** – Bar chart of estimated log-log slopes.
 5. **Cost breakdown** – Approximate contribution of each swept variable to
     total step time (marginal-difference pie chart).
 6. **Box-plots of time/step** – Distribution grouped by each swept variable.
 7. **Total wall-clock time bars** – Sorted horizontal bar chart of total
     simulation time per configuration.
 8. **Pairwise contourf panel** – Filled-contour combined figure for all
     pairs of swept columns.
 9. **Multi-panel summary** – One-page summary with mini scaling curves and
     total-agent scatter.
10. **Cell-radius scaling** – x = CELL, y = time/step, one curve per
     R_cell value (includes R_search in legend).
11a. **Init-time scatter** – Init time vs total agents with pie-chart markers.
11b. **Time breakdown bars** – Stacked horizontal bars per run with dual
      x-axes: init-function time on one axis, other components on another.
12. **Total-agent fit residuals** – Residual diagnostic for the log-log
     power-law fit used in the total-agent scatter.
13. **Multivariate fit diagnostics** – Actual-vs-predicted and residual
     diagnostics for log-log models using the swept benchmark dimensions.
14. **Multivariate fit summary panel** – 2x2 summary of init-time and
     step-time multivariate fits and standardized term strengths.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = TOOLS_DIR / "benchmark_results.csv"
DEFAULT_OUTDIR = TOOLS_DIR / "benchmark_plots"

# Global typography. Adjust BASE_FONT_SIZE to scale the whole plotting style.
BASE_FONT_SIZE = 14
FONT_SIZE_SMALL = BASE_FONT_SIZE - 1
FONT_SIZE_TEXT = BASE_FONT_SIZE - 2
FONT_SIZE_LABEL = BASE_FONT_SIZE + 1
FONT_SIZE_TITLE = BASE_FONT_SIZE + 2
FONT_SIZE_SUPTITLE = BASE_FONT_SIZE + 3
FONT_SIZE_LEGEND = BASE_FONT_SIZE
FONT_SIZE_TICKS = BASE_FONT_SIZE
FONT_SIZE_YTICKS_COMPACT = BASE_FONT_SIZE - 4

YLABEL_STEP = "Time/step (s)"
YLABEL_INIT = "Init time (s)"
XLABEL_TOTAL_AGENTS = "Total agents"

# Agent-count columns present in the CSV (ordered for display)
AGENT_COLS = [
    "ECM_POPULATION_SIZE",
    "N_CELLS",
    "FOCAD_count_init",
    "N_FNODES",
]

# Additional swept parameters that affect performance
PARAM_COLS = [
    "CELL_RADIUS",
]

# Independent benchmark inputs for regression. Use per-cell focal adhesions
# instead of total focal adhesions because FOCAD_count_init is derived from
# N_CELLS * INIT_N_FOCAD_PER_CELL.
REGRESSION_BASE_COLS = [
    "ECM_POPULATION_SIZE",
    "N_CELLS",
    "INIT_N_FOCAD_PER_CELL",
    "N_FNODES",
    "CELL_RADIUS",
]

CONTOUR_COLS = [
    "ECM_POPULATION_SIZE",
    "N_CELLS",
    "N_FNODES",
    "CELL_RADIUS",
]

# All columns that may be swept in benchmark runs
SWEEP_COLS = AGENT_COLS + PARAM_COLS

COL_LABELS = {
    "ECM_POPULATION_SIZE": "ECM",
    "N_CELLS": "CELL",
    "INIT_N_FOCAD_PER_CELL": "FOCAD/CELL",
    "FOCAD_count_init": "FOCAD",
    "N_FNODES": "FNODES",
    "CELL_RADIUS": "R_cell",
    "MAX_SEARCH_RADIUS": "R_search",
    "TOTAL_AGENTS": "Total agents",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_compact_log_ticks(ax: plt.Axes) -> None:
    """Use compact scientific notation on log axes and hide minor tick labels."""
    major_locator = LogLocator(base=10.0)
    minor_locator = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)

    formatter = LogFormatterSciNotation(base=10.0)
    formatter.labelOnlyBase = False

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(major_locator)
        axis.set_major_formatter(formatter)
        axis.set_minor_locator(minor_locator)
        axis.set_minor_formatter(NullFormatter())

def _label(col: str) -> str:
    return COL_LABELS.get(col, col)


def _apply_plot_style() -> None:
    """Set global matplotlib font sizes from the module constants."""
    plt.rcParams.update({
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE,
        "xtick.labelsize": FONT_SIZE_TICKS,
        "ytick.labelsize": FONT_SIZE_TICKS,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.titlesize": FONT_SIZE_SUPTITLE,
    })


def _has_varied_focad_init(df: pd.DataFrame) -> bool:
    """Return whether focal adhesions per cell were explicitly swept."""
    return (
        "INIT_N_FOCAD_PER_CELL" in df.columns
        and df["INIT_N_FOCAD_PER_CELL"].nunique(dropna=True) > 1
    )


def _varied_cols(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    """Return columns present in *df* with at least two distinct values."""
    cols = cols if cols is not None else SWEEP_COLS
    varied: list[str] = []
    focad_init_varied = _has_varied_focad_init(df)
    for col in cols:
        if col not in df.columns:
            continue
        if col == "FOCAD_count_init" and not focad_init_varied:
            continue
        if df[col].nunique(dropna=True) > 1:
            varied.append(col)
    return varied


def _varied_agent_cols(df: pd.DataFrame) -> list[str]:
    """Return agent-count columns that actually vary in the loaded dataframe."""
    return _varied_cols(df, AGENT_COLS)


def _pie_agent_cols(df: pd.DataFrame) -> list[str]:
    """Return all agent-count columns present in the dataframe for pie charts."""
    return [c for c in AGENT_COLS if c in df.columns]


def _format_run_label(row: pd.Series, varied_cols: list[str]) -> str:
    """Compact run label including only varied benchmark parameters."""
    parts: list[str] = []
    if "ECM_POPULATION_SIZE" in varied_cols and pd.notna(row.get("ECM_POPULATION_SIZE")):
        parts.append(f"ECM={int(row['ECM_POPULATION_SIZE'])}")
    if "N_CELLS" in varied_cols and pd.notna(row.get("N_CELLS")):
        parts.append(f"CELL={int(row['N_CELLS'])}")
    if "FOCAD_count_init" in varied_cols and pd.notna(row.get("FOCAD_count_init")):
        parts.append(f"FOCAD={int(row['FOCAD_count_init'])}")
    if "N_FNODES" in varied_cols and pd.notna(row.get("N_FNODES")):
        parts.append(f"FNODES={int(row['N_FNODES'])}")
    if "CELL_RADIUS" in varied_cols and pd.notna(row.get("CELL_RADIUS")):
        parts.append(f"R_cell={row['CELL_RADIUS']:g}")
    return " ".join(parts) if parts else f"run {int(row.name) + 1}"


def _load(csv_path: Path) -> pd.DataFrame:
    """Load CSV keeping only successful runs with valid timing."""
    df = pd.read_csv(csv_path)
    df = df[df["status"].str.startswith("OK", na=False)].copy()

    for col in (
        "time_per_step_s",
        "total_time_s",
        "init_time_s",
        "simulation_time_s",
        "rtc_time_s",
        "init_functions_time_s",
        "exit_functions_time_s",
        "CELL_RADIUS",
        "MAX_SEARCH_RADIUS",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Correct init_time_s by removing RTC compilation time, when available.
    # Some runs recompile and others do not, so RTC should not be counted as
    # model initialization cost for benchmarking.
    if "init_time_s" in df.columns and "rtc_time_s" in df.columns:
        rtc = df["rtc_time_s"].fillna(0.0)
        df["init_time_s_raw"] = df["init_time_s"]
        df["init_time_s"] = (df["init_time_s"] - rtc).clip(lower=0.0)

    df.dropna(subset=["time_per_step_s"], inplace=True)
    if df.empty:
        print("ERROR: no successful runs with timing data found in CSV.")
        sys.exit(1)
    # Derived column: total agent count
    df["TOTAL_AGENTS"] = (
        df["ECM_POPULATION_SIZE"]
        + df["N_CELLS"]
        + df["FOCAD_count_init"]
        + df["N_FNODES"]
    )
    return df


def _savefig(fig: plt.Figure, outdir: Path, name: str, dpi: int = 300) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 1. Pairwise heatmaps
# ---------------------------------------------------------------------------

def plot_pairwise_heatmaps(df: pd.DataFrame, outdir: Path) -> list[plt.Figure]:
    """One heatmap per pair of swept columns.  Colour = mean time/step."""
    cols = _varied_cols(df)
    figs: list[plt.Figure] = []
    if len(cols) < 2:
        print("  Skipping pairwise heatmaps (need ≥2 varied agent columns).")
        return figs

    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    for col_y, col_x in pairs:
        pivot = df.groupby([col_y, col_x])["time_per_step_s"].mean().reset_index()
        table = pivot.pivot(index=col_y, columns=col_x, values="time_per_step_s")
        # Sort axes numerically
        table = table.sort_index(ascending=True)
        table = table[sorted(table.columns)]

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        vals = table.values
        vmin = np.nanmin(vals[vals > 0]) if np.any(vals > 0) else 1e-3
        vmax = np.nanmax(vals) if np.any(~np.isnan(vals)) else 1.0
        # Use log scale if range spans > 1 order of magnitude
        use_log = (vmax / max(vmin, 1e-12)) > 10
        norm = LogNorm(vmin=vmin, vmax=vmax) if use_log else None

        im = ax.imshow(vals, aspect="auto", origin="lower", norm=norm,
                        cmap="YlOrRd")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([f"{v:g}" for v in table.columns], fontsize=FONT_SIZE_SMALL)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([f"{v:g}" for v in table.index], fontsize=FONT_SIZE_SMALL)
        ax.set_xlabel(_label(col_x), fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(_label(col_y), fontsize=FONT_SIZE_LABEL)

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(YLABEL_STEP, fontsize=FONT_SIZE_LABEL)

        # Annotate cells
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.3g}" if v < 100 else f"{v:.1f}"
                text_color = "white" if (use_log and v > np.sqrt(vmin * vmax)) or (not use_log and v > (vmin + vmax) / 2) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=FONT_SIZE_TEXT,
                        color=text_color)

        ax.set_title(f"Mean time/step:  {_label(col_y)}  vs  {_label(col_x)}",
                     fontsize=FONT_SIZE_TITLE, pad=8)
        fig.tight_layout()
        fname = f"heatmap_{col_y}_vs_{col_x}"
        _savefig(fig, outdir, fname)
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# 2. Scaling curves (log-log)
# ---------------------------------------------------------------------------

def plot_scaling_curves(df: pd.DataFrame, outdir: Path) -> list[plt.Figure]:
    """For each agent column, plot time/step vs. that agent count (log-log).

    Lines are split by the *second most varied* column to see interaction effects.
    """
    cols = _varied_cols(df)
    figs: list[plt.Figure] = []
    if not cols:
        print("  Skipping scaling curves (no varied agent columns).")
        return figs

    for main_col in cols:
        # Choose a secondary grouping variable (the other most-varied column)
        others = [c for c in cols if c != main_col]
        grp_col = others[0] if others else None

        fig, ax = plt.subplots(figsize=(6, 4.5))
        if grp_col:
            for label, sub in df.groupby(grp_col):
                agg = sub.groupby(main_col)["time_per_step_s"].agg(["mean", "std"]).reset_index()
                agg = agg.sort_values(main_col)
                ax.errorbar(
                    agg[main_col], agg["mean"], yerr=agg["std"],
                    marker="o", markersize=4, capsize=3,
                    label=f"{_label(grp_col)}={label:g}",
                )
        else:
            agg = df.groupby(main_col)["time_per_step_s"].agg(["mean", "std"]).reset_index()
            agg = agg.sort_values(main_col)
            ax.errorbar(agg[main_col], agg["mean"], yerr=agg["std"],
                        marker="o", markersize=5, capsize=3, color="C0")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(_label(main_col), fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(YLABEL_STEP, fontsize=FONT_SIZE_LABEL)
        ax.set_title(f"Scaling:  time/step  vs  {_label(main_col)}", fontsize=FONT_SIZE_TITLE)
        if grp_col:
            ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best", framealpha=0.9)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
        fig.tight_layout()
        _savefig(fig, outdir, f"scaling_{main_col}")
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Helper: draw a pie-chart marker at a data position
# ---------------------------------------------------------------------------

_PIE_COLORS = ["#0e6377", "#87d1d5", "#f9cb37", "#f37c20"]  # one per agent col: ECM, CELL, FOCAD, FNODES


def _add_pie_legend(ax: plt.Axes, pie_cols: list[str], existing_handles=None) -> None:
    """Append a mini legend for the agent-type pie slices."""
    import matplotlib.patches as mpatches
    pie_handles = [
        mpatches.Patch(color=_PIE_COLORS[i], label=_label(c))
        for i, c in enumerate(pie_cols)
    ]
    handles = (list(existing_handles) if existing_handles else []) + pie_handles
    ax.legend(handles=handles, fontsize=FONT_SIZE_LEGEND, loc="best", framealpha=0.9)


def _draw_pie_scatter(
    ax: plt.Axes,
    df_rows: pd.DataFrame,
    x_col: str,
    y_col: str,
    pie_cols: list[str],
    size_pt: float = 8.0,
    alpha: float = 0.82,
) -> None:
    """Draw pie-chart markers at data positions using AnnotationBbox.

    Each pie is rendered inside a fixed-size DrawingArea (in points) and
    anchored to the data-coordinate position, so it is immune to log-axis
    distortion.
    """
    cols = [c for c in pie_cols if c in df_rows.columns]
    if not cols:
        return

    diameter = size_pt * 2  # DrawingArea width/height in points

    for _, row in df_rows.iterrows():
        xv, yv = float(row[x_col]), float(row[y_col])
        fracs = np.array([max(row.get(c, 0), 0) for c in cols], dtype=float)
        total = fracs.sum()
        if total <= 0:
            continue
        fracs /= total

        da = DrawingArea(diameter, diameter, 0, 0)
        cx, cy = diameter / 2, diameter / 2  # centre of the drawing area

        theta1 = 0.0
        for frac, colour in zip(fracs, _PIE_COLORS[: len(cols)]):
            if frac < 1e-9:
                theta1 += frac * 360.0
                continue
            theta2 = theta1 + frac * 360.0
            wedge = Wedge(
                (cx, cy), size_pt, theta1, theta2,
                facecolor=colour, edgecolor="white", linewidth=0.3,
                alpha=alpha,
            )
            da.add_artist(wedge)
            theta1 = theta2

        ab = AnnotationBbox(
            da, (xv, yv),
            xycoords="data",
            frameon=False,
            pad=0,
        )
        ax.add_artist(ab)


def _fit_loglog_power_law(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit log10(y) = a log10(x) + b on positive finite samples."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xm, ym = x[mask], y[mask]
    if len(xm) < 3:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = np.polyfit(np.log10(xm), np.log10(ym), 1)
    return xm, ym, coeffs


def _annotate_horizontal_bars(ax: plt.Axes, bars, values: np.ndarray) -> None:
    """Write coefficient values next to horizontal bars.

    Large bars get labels outside the bar.
    Very small bars get labels inside the plotting area with a fixed minimum offset
    from zero to avoid overlapping the y-axis/category labels.
    """
    if len(values) == 0:
        return

    span = float(np.max(np.abs(values)))
    if span <= 0:
        span = 1.0

    outside_offset = max(0.01, 0.04 * span)
    min_abs_x = max(0.02, 0.12 * span)

    for bar, value in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2

        # Large enough bar: place label outside the bar
        if abs(value) >= min_abs_x:
            x_text = value + outside_offset if value >= 0 else -value + outside_offset
            ha = "left" if value >= 0 else "right"
        else:
            # Tiny bar: place label slightly away from zero to avoid overlap
            x_text = min_abs_x if value >= 0 else -min_abs_x
            ha = "left" if value >= 0 else "right"

        if abs(value) < 0.005:
            continue

        ax.text(
            x_text,
            y,
            f"{value:+.2f}",
            va="center",
            ha=ha,
            fontsize=FONT_SIZE_TEXT,
        )


def _draw_pie_scatter_xy(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    composition_rows: pd.DataFrame,
    pie_cols: list[str],
    size_pt: float = 8.0,
    alpha: float = 0.82,
) -> None:
    """Draw pie-chart markers from explicit x/y arrays and composition rows."""
    plot_rows = composition_rows.reset_index(drop=True).copy()
    plot_rows["_plot_x"] = np.asarray(x_vals, dtype=float)
    plot_rows["_plot_y"] = np.asarray(y_vals, dtype=float)
    _draw_pie_scatter(ax, plot_rows, "_plot_x", "_plot_y", pie_cols, size_pt=size_pt, alpha=alpha)


def _multivariate_fit_cols(df: pd.DataFrame) -> list[str]:
    """Return independent positive-valued swept columns for regression."""
    fit_cols: list[str] = []
    for col in REGRESSION_BASE_COLS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        valid = np.isfinite(values)
        if not valid.any():
            continue
        if values.nunique(dropna=True) < 2:
            continue
        if (values[valid] > 0).all():
            fit_cols.append(col)
    return fit_cols


def _interaction_term_name(col_a: str, col_b: str) -> str:
    """Human-readable label for a pairwise interaction term."""
    return f"{_label(col_a)} × {_label(col_b)}"


def _fit_multivariate_loglog(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: list[str] | None = None,
) -> dict[str, object] | None:
    """Fit log10(y) against z-scored log features and all pairwise interactions."""
    if y_col not in df.columns:
        return None

    feature_cols = feature_cols if feature_cols is not None else _multivariate_fit_cols(df)
    if not feature_cols:
        return None

    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    x_arrays = [pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in feature_cols]

    mask = np.isfinite(y) & (y > 0)
    for values in x_arrays:
        mask &= np.isfinite(values) & (values > 0)

    if mask.sum() < len(feature_cols) + 2:
        return None

    logx = np.column_stack([np.log10(values[mask]) for values in x_arrays])
    logy = np.log10(y[mask])

    means = logx.mean(axis=0)
    scales = logx.std(axis=0, ddof=0)
    keep = scales > 0
    if not np.any(keep):
        return None
    if not np.all(keep):
        logx = logx[:, keep]
        means = means[keep]
        scales = scales[keep]
        feature_cols = [col for col, use_col in zip(feature_cols, keep) if use_col]

    z_main = (logx - means) / scales
    term_arrays = [z_main]
    term_names = [_label(col) for col in feature_cols]

    interaction_arrays: list[np.ndarray] = []
    interaction_names: list[str] = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            interaction_arrays.append((z_main[:, i] * z_main[:, j])[:, None])
            interaction_names.append(_interaction_term_name(feature_cols[i], feature_cols[j]))
    if interaction_arrays:
        term_arrays.extend(interaction_arrays)
        term_names.extend(interaction_names)

    design_terms = np.column_stack(term_arrays)
    design = np.column_stack([np.ones(design_terms.shape[0]), design_terms])
    coeffs, _, _, _ = np.linalg.lstsq(design, logy, rcond=None)
    pred_logy = design @ coeffs
    residuals = logy - pred_logy
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "feature_cols": feature_cols,
        "term_names": term_names,
        "coeffs": coeffs,
        "logy": logy,
        "pred_logy": pred_logy,
        "residuals": residuals,
        "y": y[mask],
        "pred_y": 10 ** pred_logy,
        "r2": r2,
        "mask": mask,
    }


def _summarize_top_terms(term_names: list[str], term_coeffs: np.ndarray, top_n: int = 5) -> str:
    """Compact summary of the strongest signed coefficients."""
    if len(term_names) == 0:
        return "No fitted terms"
    order = np.argsort(np.abs(term_coeffs))[::-1][:top_n]
    lines = ["Strongest standardized terms:"]
    for idx in order:
        lines.append(f"{term_coeffs[idx]:+,.2f}  {term_names[idx]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Total-agent scatter with power-law fit  (pie-chart markers)
# ---------------------------------------------------------------------------

def plot_total_agent_scatter(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Time/step vs. TOTAL_AGENTS with pie-chart markers showing composition."""
    if "TOTAL_AGENTS" not in df.columns or df["TOTAL_AGENTS"].nunique() < 2:
        print("  Skipping total-agent scatter (not enough data).")
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    pie_cols = _pie_agent_cols(df)

    x = df["TOTAL_AGENTS"].values.astype(float)
    y = df["time_per_step_s"].values.astype(float)
    mask = (x > 0) & (y > 0)

    # Invisible scatter to set axis limits correctly
    ax.scatter(x[mask], y[mask], s=0, alpha=0)

    # Draw pie markers
    _draw_pie_scatter(ax, df[mask], "TOTAL_AGENTS", "time_per_step_s", pie_cols,
                      size_pt=6.5, alpha=0.75)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(XLABEL_TOTAL_AGENTS, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(YLABEL_STEP, fontsize=FONT_SIZE_LABEL)
    #ax.set_title("Overall computational scaling", fontsize=FONT_SIZE_TITLE)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    if pie_cols:
        _add_pie_legend(ax, pie_cols)

    fig.tight_layout()
    _savefig(fig, outdir, "total_agent_scatter")
    return fig


def plot_total_agent_fit_residuals(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Residual diagnostic for the log-log power-law fit vs TOTAL_AGENTS."""
    if "TOTAL_AGENTS" not in df.columns or df["TOTAL_AGENTS"].nunique() < 2:
        print("  Skipping total-agent fit residuals (not enough data).")
        return None

    x = df["TOTAL_AGENTS"].to_numpy(dtype=float)
    y = df["time_per_step_s"].to_numpy(dtype=float)
    fit_result = _fit_loglog_power_law(x, y)
    if fit_result is None:
        print("  Skipping total-agent fit residuals (need at least 3 positive samples).")
        return None

    xm, ym, coeffs = fit_result
    logx = np.log10(xm)
    residuals = np.log10(ym) - np.polyval(coeffs, logx)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(logx, residuals, s=28, alpha=0.8, color="C0", edgecolors="none")
    ax.axhline(0.0, color="r", ls="--", lw=1.0)
    ax.set_xlabel("log10(Total agents)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Residual in log10(Time/step)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        f"Residuals of total-agent power-law fit ($\\alpha={coeffs[0]:.2f}$)",
        fontsize=FONT_SIZE_TITLE,
    )
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)

    fig.tight_layout()
    _savefig(fig, outdir, "total_agent_fit_residuals")
    return fig


def plot_multivariate_fit_diagnostics(
    df: pd.DataFrame,
    y_col: str,
    outdir: Path,
    name: str,
    title: str,
    y_label: str,
) -> plt.Figure | None:
    """Actual-vs-predicted and residual diagnostics for a multivariate log-log fit."""
    fit = _fit_multivariate_loglog(df, y_col)
    if fit is None:
        print(f"  Skipping {name} (not enough positive data for multivariate fit).")
        return None

    coeffs = fit["coeffs"]
    feature_cols = fit["feature_cols"]
    term_names = fit["term_names"]
    pred_y = fit["pred_y"]
    y = fit["y"]
    pred_logy = fit["pred_logy"]
    residuals = fit["residuals"]
    r2 = fit["r2"]
    term_coeffs = coeffs[1:]
    order = np.argsort(np.abs(term_coeffs))[::-1]
    ordered_terms = [term_names[idx] for idx in order]
    ordered_coeffs = term_coeffs[order]

    fit_rows = df.loc[fit["mask"]].copy()
    pie_cols = _pie_agent_cols(df)

    fig, (ax_fit, ax_res, ax_terms) = plt.subplots(1, 3, figsize=(16.2, 4.8))

    ax_fit.scatter(pred_y, y, s=0, alpha=0)
    _draw_pie_scatter_xy(ax_fit, pred_y, y, fit_rows, pie_cols, size_pt=6.2, alpha=0.78)
    lo = min(pred_y.min(), y.min())
    hi = max(pred_y.max(), y.max())
    ax_fit.plot([lo, hi], [lo, hi], "r--", lw=1.0)
    ax_fit.set_xscale("log")
    ax_fit.set_yscale("log")
    ax_fit.set_xlabel(f"Predicted {y_label}", fontsize=FONT_SIZE_LABEL)
    ax_fit.set_ylabel(f"Observed {y_label}", fontsize=FONT_SIZE_LABEL)
    ax_fit.set_title(f"{title}: predicted vs observed ($R^2={r2:.3f}$)", fontsize=FONT_SIZE_TITLE)
    ax_fit.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax_fit.text(
        0.03, 0.97,
        "Independent inputs:\n"
        + "\n".join(_label(col) for col in feature_cols)
        + "\n\n"
        + _summarize_top_terms(term_names, term_coeffs),
        transform=ax_fit.transAxes,
        va="top",
        ha="left",
        fontsize=FONT_SIZE_TEXT,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    if pie_cols:
        _add_pie_legend(ax_fit, pie_cols)

    ax_res.scatter(pred_logy, residuals, s=28, alpha=0.8, color="C1", edgecolors="none")
    ax_res.axhline(0.0, color="r", ls="--", lw=1.0)
    ax_res.set_xlabel(f"Predicted log10({y_label})", fontsize=FONT_SIZE_LABEL)
    ax_res.set_ylabel(f"Residual in log10({y_label})", fontsize=FONT_SIZE_LABEL)
    ax_res.set_title(f"{title}: residuals", fontsize=FONT_SIZE_TITLE)
    ax_res.grid(True, ls=":", lw=0.4, alpha=0.6)

    colours = ["C0" if coef >= 0 else "C3" for coef in ordered_coeffs]
    bars = ax_terms.barh(ordered_terms, ordered_coeffs, color=colours, edgecolor="k", lw=0.4)
    ax_terms.axvline(0.0, color="0.3", lw=0.8)
    ax_terms.set_xlabel("Standardized coefficient", fontsize=FONT_SIZE_LABEL)
    ax_terms.set_title(f"{title}: term strengths", fontsize=FONT_SIZE_TITLE)
    ax_terms.grid(axis="x", ls=":", lw=0.4, alpha=0.6)
    ax_terms.tick_params(axis="y", labelsize=FONT_SIZE_TEXT)
    _annotate_horizontal_bars(ax_terms, bars, ordered_coeffs)

    fig.tight_layout()
    _savefig(fig, outdir, name)
    return fig


def plot_multivariate_summary_panel(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """2x2 summary of init-time and step-time multivariate fits and term strengths."""
    step_fit = _fit_multivariate_loglog(df, "time_per_step_s")
    init_fit = _fit_multivariate_loglog(df, "init_time_s")
    if step_fit is None or init_fit is None:
        print("  Skipping multivariate fit summary panel (fit unavailable).")
        return None

    pie_cols = _pie_agent_cols(df)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.0))
    panel_specs = [
        (axes[0, 0], init_fit, YLABEL_INIT, "Init-time fit"),
        (axes[1, 0], step_fit, YLABEL_STEP, "Step-time fit"),
    ]
    for ax, fit, y_label, title in panel_specs:
        fit_rows = df.loc[fit["mask"]].copy()
        pred_y = fit["pred_y"]
        y = fit["y"]
        ax.scatter(pred_y, y, s=0, alpha=0)
        _draw_pie_scatter_xy(ax, pred_y, y, fit_rows, pie_cols, size_pt=6.5, alpha=0.8)
        lo = min(pred_y.min(), y.min())
        hi = max(pred_y.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _set_compact_log_ticks(ax)
        ax.set_xlabel(f"Predicted {y_label}", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(f"Observed {y_label}", fontsize=FONT_SIZE_LABEL)
        ax.set_title(f"{title} ($R^2={fit['r2']:.3f}$)", fontsize=FONT_SIZE_TITLE)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
        _add_pie_legend(ax, pie_cols)

    term_specs = [
        (axes[0, 1], init_fit, "Init-time model term strengths"),
        (axes[1, 1], step_fit, "Step-time model term strengths"),
    ]
    for ax, fit, title in term_specs:
        coeffs = fit["coeffs"][1:]
        term_names = fit["term_names"]
        order = np.argsort(np.abs(coeffs))[::-1]
        ordered_terms = [term_names[idx] for idx in order]
        ordered_coeffs = coeffs[order]
        colours = ["C0" if coef >= 0 else "C3" for coef in ordered_coeffs]
        bars = ax.barh(ordered_terms, ordered_coeffs, color=colours, edgecolor="k", lw=0.4)
        ax.axvline(0.0, color="0.3", lw=0.8)
        ax.set_xlabel("Standardized coefficient", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.grid(axis="x", ls=":", lw=0.4, alpha=0.6)
        ax.tick_params(axis="y", labelsize=FONT_SIZE_SMALL)
        _annotate_horizontal_bars(ax, bars, ordered_coeffs)

    #fig.suptitle("Cellfoundry - Multivariate fit summary", fontsize=FONT_SIZE_SUPTITLE, y=1.01)
    fig.tight_layout()
    _savefig(fig, outdir, "multivariate_fit_summary_panel")
    return fig


# ---------------------------------------------------------------------------
# 4. Per-agent scaling exponents (bar chart)
# ---------------------------------------------------------------------------

def plot_scaling_exponents(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Estimate and compare the log-log slope for each swept variable."""
    cols = _varied_cols(df)
    if not cols:
        print("  Skipping scaling exponents (no varied columns).")
        return None

    exponents = {}
    for col in cols:
        agg = df.groupby(col)["time_per_step_s"].mean().reset_index()
        x = agg[col].values.astype(float)
        y = agg["time_per_step_s"].values.astype(float)
        mask = (x > 0) & (y > 0)
        if mask.sum() < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope, _ = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
        exponents[col] = slope

    if not exponents:
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [_label(c) for c in exponents]
    values = list(exponents.values())
    colours = ["C0", "C1", "C2", "C3"][:len(values)]
    bars = ax.barh(labels, values, color=colours, edgecolor="k", lw=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=FONT_SIZE_SMALL)
    ax.set_xlabel("Scaling exponent  $\\alpha$  ($t \\propto N^\\alpha$)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Per-agent-type scaling exponents", fontsize=FONT_SIZE_TITLE)
    ax.axvline(1.0, ls="--", lw=0.8, color="grey", label="Linear ($\\alpha=1$)")
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.grid(axis="x", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout()
    _savefig(fig, outdir, "scaling_exponents")
    return fig


# ---------------------------------------------------------------------------
# 5. Cost breakdown – stacked area / bar
# ---------------------------------------------------------------------------

def plot_cost_breakdown(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Pie chart showing relative time contribution when increasing each
    swept variable, estimated via marginal differences."""
    cols = _varied_cols(df)
    if len(cols) < 2:
        print("  Skipping cost breakdown (need ≥2 varied columns).")
        return None

    # For each agent type, compute the marginal increase in time/step when
    # that population goes from its minimum to its maximum, averaged over
    # all other settings.
    baseline = df["time_per_step_s"].min()
    marginals: dict[str, float] = {}
    for col in cols:
        lo = df[df[col] == df[col].min()]["time_per_step_s"].mean()
        hi = df[df[col] == df[col].max()]["time_per_step_s"].mean()
        marginals[col] = max(hi - lo, 0.0)

    total_marginal = sum(marginals.values())
    if total_marginal <= 0:
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [_label(c) for c in marginals]
    sizes = [marginals[c] for c in marginals]
    colours = plt.cm.Set2(np.linspace(0, 1, len(sizes)))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140,
            colors=colours, textprops={"fontsize": FONT_SIZE_SMALL})
    ax.set_title("Approximate cost attribution\n(marginal increase min → max)",
                fontsize=FONT_SIZE_TITLE)
    fig.tight_layout()
    _savefig(fig, outdir, "cost_breakdown")
    return fig


# ---------------------------------------------------------------------------
# 6. Box-plots per variable
# ---------------------------------------------------------------------------

def plot_boxplots(df: pd.DataFrame, outdir: Path) -> list[plt.Figure]:
    """Box-plot of time/step grouped by each swept variable."""
    cols = _varied_cols(df)
    figs: list[plt.Figure] = []
    if not cols:
        return figs

    for col in cols:
        groups = sorted(df[col].unique())
        data = [df[df[col] == g]["time_per_step_s"].values for g in groups]
        fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(groups) + 2), 4))
        bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markeredgecolor="black",
                                       markerfacecolor="gold", markersize=5))
        for patch, colour in zip(bp["boxes"],
                                  plt.cm.Blues(np.linspace(0.3, 0.85, len(groups)))):
            patch.set_facecolor(colour)
        ax.set_xticklabels([f"{g:g}" for g in groups], fontsize=FONT_SIZE_SMALL)
        ax.set_xlabel(_label(col), fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(YLABEL_STEP, fontsize=FONT_SIZE_LABEL)
        ax.set_title(f"Time/step distribution by {_label(col)}", fontsize=FONT_SIZE_TITLE)
        ax.grid(axis="y", ls=":", lw=0.4, alpha=0.6)
        fig.tight_layout()
        _savefig(fig, outdir, f"boxplot_{col}")
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# 7. Total wall-clock time bar chart
# ---------------------------------------------------------------------------

def plot_total_time_bars(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Bar chart of total simulation wall-clock time for each run, sorted."""
    if "total_time_s" not in df.columns or df["total_time_s"].isna().all():
        return None

    sorted_df = df.sort_values("total_time_s", ascending=True).reset_index(drop=True)
    varied_cols = _varied_cols(df)
    fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(sorted_df) + 2), 4.5))

    # Build a descriptive label for each run
    labels = [_format_run_label(r, varied_cols) for _, r in sorted_df.iterrows()]

    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(sorted_df)))
    ax.barh(range(len(sorted_df)), sorted_df["total_time_s"], color=colours,
            edgecolor="k", lw=0.3)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(labels, fontsize=FONT_SIZE_YTICKS_COMPACT)
    ax.set_xlabel("Total wall-clock time (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Total simulation time per configuration", fontsize=FONT_SIZE_TITLE)
    ax.grid(axis="x", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout()
    _savefig(fig, outdir, "total_time_bars")
    return fig


# ---------------------------------------------------------------------------
# 8. Pairwise filled-contour plots (combined figure)
# ---------------------------------------------------------------------------

def plot_surface_panel(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Single figure with one filled-contour (contourf) subplot per pair of
    swept columns.  Colour = mean time/step."""
    #cols = _varied_cols(df)
    cols = _varied_cols(df, CONTOUR_COLS)
    if len(cols) < 2:
        print("  Skipping contourf panel (need ≥2 varied agent columns).")
        return None

    pairs = [(cols[i], cols[j])
             for i in range(len(cols)) for j in range(i + 1, len(cols))]
    n = len(pairs)
    ncols_grid = min(n, 3)
    nrows_grid = (n + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(
        nrows_grid, ncols_grid,
        figsize=(5.0 * ncols_grid, 5.0 * nrows_grid),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for k, (col_y, col_x) in enumerate(pairs):
        ax = axes_flat[k]

        pivot = df.groupby([col_y, col_x])["time_per_step_s"].mean().reset_index()
        table = pivot.pivot(index=col_y, columns=col_x, values="time_per_step_s")
        table = table.sort_index(ascending=True)
        table = table[sorted(table.columns)]

        X_vals = np.array(table.columns, dtype=float)
        Y_vals = np.array(table.index, dtype=float)
        X_mesh, Y_mesh = np.meshgrid(X_vals, Y_vals)
        Z = table.values.astype(float)

        # Use log-spaced levels when the range exceeds one order of magnitude
        zmin = np.nanmin(Z[Z > 0]) if np.any(Z > 0) else 1e-3
        zmax = np.nanmax(Z) if np.any(~np.isnan(Z)) else 1.0
        use_log = (zmax / max(zmin, 1e-12)) > 10
        if use_log:
            levels = np.logspace(np.log10(zmin), np.log10(zmax), 12)
            norm = LogNorm(vmin=zmin, vmax=zmax)
        else:
            levels = 12
            norm = None

        cf = ax.contourf(
            X_mesh, Y_mesh, Z,
            levels=levels, cmap="inferno", norm=norm,
        )
        # Overlay contour lines with inline value labels
        cs = ax.contour(
            X_mesh, Y_mesh, Z,
            levels=cf.levels, colors="w", linewidths=0.5, alpha=0.95,
        )
        ax.clabel(cs, inline=True, fontsize=BASE_FONT_SIZE, fmt="%.2g")

        ax.set_xlabel(_label(col_x), fontsize=FONT_SIZE_LABEL + 2)
        ax.set_ylabel(_label(col_y), fontsize=FONT_SIZE_LABEL + 2)
        # ax.set_title(
        #     f"{_label(col_y)}  vs  {_label(col_x)}",
        #     fontsize=FONT_SIZE_TITLE + 2, pad=6,
        # )
        ax.tick_params(labelsize=FONT_SIZE_TICKS + 2)
        ax.ticklabel_format(style="scientific", scilimits=(0, 3),
                            axis="both", useMathText=True)

    # Hide unused axes
    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # fig.suptitle(
    #     "Cellfoundry – Pairwise Cost Contours (mean time/step)",
    #     fontsize=FONT_SIZE_SUPTITLE, y=1.01,
    # )
    fig.tight_layout()
    _savefig(fig, outdir, "contourf_panel")
    return fig


# ---------------------------------------------------------------------------
# 9. Multi-panel summary (combined figure)
# ---------------------------------------------------------------------------

def plot_summary_panel(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """A single multi-panel figure with the key insights for quick review."""
    cols = _varied_cols(df)
    n_panels = min(len(cols), 4) + 1  # scaling curves + total scatter
    if n_panels < 2:
        print("  Skipping summary panel (not enough varied columns).")
        return None

    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    idx = 0
    # Per-agent-type mini-scaling curves
    for col in cols[:4]:
        ax = axes[idx]; idx += 1
        agg = df.groupby(col)["time_per_step_s"].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values(col)
        ax.errorbar(agg[col], agg["mean"], yerr=agg["std"],
                     marker="o", markersize=4, capsize=3, color="C0")
        ax.set_xscale("log"); ax.set_yscale("log")
        _set_compact_log_ticks(ax)
        ax.set_xlabel(_label(col), fontsize=FONT_SIZE_SMALL)
        ax.set_ylabel(YLABEL_STEP, fontsize=FONT_SIZE_SMALL)
        ax.set_title(f"Scaling: {_label(col)}", fontsize=FONT_SIZE_SMALL)
        ax.grid(True, which="both", ls=":", lw=0.3, alpha=0.5)

    # Total-agent scatter
    if idx < len(axes):
        ax = axes[idx]; idx += 1
        x = df["TOTAL_AGENTS"].values.astype(float)
        y = df["time_per_step_s"].values.astype(float)
        mask = (x > 0) & (y > 0)
        ax.scatter(x[mask], y[mask], s=12, alpha=0.7, c="C1", edgecolors="none")
        ax.set_title("Total agents", fontsize=FONT_SIZE_SMALL)
        ax.set_xscale("log"); ax.set_yscale("log")
        _set_compact_log_ticks(ax)
        ax.set_xlabel("Total agents", fontsize=FONT_SIZE_SMALL)
        ax.set_ylabel(YLABEL_STEP, fontsize=FONT_SIZE_SMALL)
        ax.grid(True, which="both", ls=":", lw=0.3, alpha=0.5)

    # Hide unused axes
    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cellfoundry – Performance Scaling Summary", fontsize=FONT_SIZE_SUPTITLE,
                  y=1.01)
    fig.tight_layout()
    _savefig(fig, outdir, "summary_panel")
    return fig


# ---------------------------------------------------------------------------
# 10. Cell-radius scaling curves
# ---------------------------------------------------------------------------

def plot_cell_radius_scaling(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Time per step vs N_CELLS with one curve per CELL_RADIUS value.

    This is the key plot for understanding how the cell-cell interaction
    search radius affects performance scaling.
    """
    if "CELL_RADIUS" not in _varied_cols(df, ["CELL_RADIUS"]):
        print("  Skipping cell-radius scaling (need >=2 CELL_RADIUS values).")
        return None
    if "N_CELLS" not in df.columns or df["N_CELLS"].nunique() < 2:
        print("  Skipping cell-radius scaling (need >=2 N_CELLS values).")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for cr, group in df.groupby("CELL_RADIUS"):
        agg = (group.groupby("N_CELLS")["time_per_step_s"]
               .agg(["mean", "std"]).reset_index())
        agg = agg.sort_values("N_CELLS")
        if "MAX_SEARCH_RADIUS" in group.columns and group["MAX_SEARCH_RADIUS"].notna().any():
            sr = group["MAX_SEARCH_RADIUS"].iloc[0]
        else:
            sr = 3.0 * cr
        ax.errorbar(
            agg["N_CELLS"], agg["mean"], yerr=agg["std"],
            marker="o", markersize=5, capsize=3,
            label=f"R_cell={cr:g}  (R_search={sr:.1f})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CELL", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(YLABEL_STEP, fontsize=FONT_SIZE_LABEL)
    ax.set_title("Effect of R_cell / R_search on scaling", fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best", framealpha=0.9)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout()
    _savefig(fig, outdir, "cell_radius_scaling")
    return fig


# ---------------------------------------------------------------------------
# 11a. Init time vs total agents  (pie-chart markers + power-law fit)
# ---------------------------------------------------------------------------

def plot_init_time_scatter(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Init time vs. TOTAL_AGENTS with pie-chart markers showing composition."""
    has_init = "init_time_s" in df.columns and df["init_time_s"].notna().any()
    if not has_init:
        print("  Skipping init-time scatter (no init_time_s data).")
        return None
    if "TOTAL_AGENTS" not in df.columns or df["TOTAL_AGENTS"].nunique() < 2:
        print("  Skipping init-time scatter (not enough data).")
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    pie_cols = _pie_agent_cols(df)

    x = df["TOTAL_AGENTS"].values.astype(float)
    y = df["init_time_s"].values.astype(float)
    mask = (x > 0) & (y > 0)

    # Invisible scatter to set axis limits correctly
    ax.scatter(x[mask], y[mask], s=0, alpha=0)

    # Draw pie markers
    _draw_pie_scatter(ax, df[mask], "TOTAL_AGENTS", "init_time_s", pie_cols,
                      size_pt=6.5, alpha=0.75)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(XLABEL_TOTAL_AGENTS, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(YLABEL_INIT, fontsize=FONT_SIZE_LABEL)
    #ax.set_title("Initialization-time scaling", fontsize=FONT_SIZE_TITLE)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    if pie_cols:
        _add_pie_legend(ax, pie_cols)

    fig.tight_layout()
    _savefig(fig, outdir, "init_time_scatter")
    return fig


# ---------------------------------------------------------------------------
# 11b. Time breakdown stacked bars  (separate figure, dual axes:
#      init_functions on its own axis, others stacked on a second axis)
# ---------------------------------------------------------------------------

def plot_time_breakdown_bars(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Horizontal stacked-bar chart of time components per run.

    Because init-function time dominates, the figure uses dual x-axes:
    the top axis shows init-function time alone, and the bottom axis
    shows the remaining components (Python setup, RTC, stepping) stacked.
    """
    has_init = "init_time_s" in df.columns and df["init_time_s"].notna().any()
    has_sim = "simulation_time_s" in df.columns and df["simulation_time_s"].notna().any()
    if not (has_init and has_sim):
        print("  Skipping time-breakdown bars (columns missing or empty).")
        return None

    has_rtc = "rtc_time_s" in df.columns and df["rtc_time_s"].notna().any()
    has_ifunc = (
        "init_functions_time_s" in df.columns
        and df["init_functions_time_s"].notna().any()
    )

    sorted_df = df.sort_values("total_time_s", ascending=True).reset_index(drop=True)
    varied_cols = _varied_cols(df)
    n_runs = len(sorted_df)
    idx = np.arange(n_runs)

    fig, ax_ifunc = plt.subplots(
        figsize=(max(7, 0.22 * n_runs + 3), max(5, 0.15 * n_runs + 2)),
    )

    if has_rtc and has_ifunc:
        rtc = sorted_df["rtc_time_s"].fillna(0).values
        ifunc = sorted_df["init_functions_time_s"].fillna(0).values
        py_setup = (sorted_df["init_time_s"].values - rtc - ifunc).clip(min=0)
        stepping = sorted_df["simulation_time_s"].fillna(0).values

        # Left axis: init functions only (the dominant component)
        ax_ifunc.barh(
            idx - 0.17, ifunc, height=0.34,
            color="C2", label="Init functions", alpha=0.85,
        )
        ax_ifunc.set_xlabel("Init functions time (s)", fontsize=FONT_SIZE_LABEL, color="C2")
        ax_ifunc.tick_params(axis="x", labelcolor="C2")

        # Right axis: other components stacked
        ax_other = ax_ifunc.twiny()
        ax_other.barh(
            idx + 0.17, py_setup, height=0.34,
            color="C4", label="Python setup", alpha=0.85,
        )
        ax_other.barh(
            idx + 0.17, rtc, left=py_setup, height=0.34,
            color="C0", label="RTC compilation", alpha=0.85,
        )
        ax_other.barh(
            idx + 0.17, stepping, left=py_setup + rtc, height=0.34,
            color="C1", label="Stepping", alpha=0.85,
        )
        ax_other.set_xlabel(
            "Other components (s): Python setup + RTC + Stepping",
            fontsize=FONT_SIZE_LABEL, color="C0",
        )
        ax_other.tick_params(axis="x", labelcolor="C0")

        # Combined legend
        h1, l1 = ax_ifunc.get_legend_handles_labels()
        h2, l2 = ax_other.get_legend_handles_labels()
        ax_ifunc.legend(
            h1 + h2, l1 + l2, fontsize=FONT_SIZE_LEGEND,
            loc="lower right", framealpha=0.9,
        )
    else:
        # Fallback: simple two-component bars
        ax_ifunc.barh(
            idx, sorted_df["init_time_s"].values,
            color="C0", label="Initialization",
        )
        ax_ifunc.barh(
            idx, sorted_df["simulation_time_s"].values,
            left=sorted_df["init_time_s"].values,
            color="C1", label="Simulation",
        )
        ax_ifunc.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)
        ax_ifunc.legend(fontsize=FONT_SIZE_LEGEND)

    # Y-tick labels
    labels = [_format_run_label(r, varied_cols) for _, r in sorted_df.iterrows()]
    ax_ifunc.set_yticks(idx)
    ax_ifunc.set_yticklabels(labels, fontsize=FONT_SIZE_YTICKS_COMPACT)
    ax_ifunc.set_title("Time breakdown per run", fontsize=FONT_SIZE_TITLE)
    ax_ifunc.grid(axis="x", ls=":", lw=0.4, alpha=0.4)

    fig.tight_layout()
    _savefig(fig, outdir, "time_breakdown_bars")
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from benchmark_perf.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/plot_benchmark_results.py
  python tools/plot_benchmark_results.py --show
  python tools/plot_benchmark_results.py --csv results.csv --outdir figs/
        """,
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help=f"Path to the benchmark CSV. Default: {DEFAULT_CSV}")
    parser.add_argument(
        "--outdir", type=str, default=None,
        help=f"Directory for saved figures. Default: {DEFAULT_OUTDIR}")
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively after saving.")
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution for saved PNGs. Default: 300")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    # If --show not set, use a non-interactive backend for headless servers
    if not args.show:
        matplotlib.use("Agg")

    # Suppress "More than 20 figures" warning — we close all at the end
    plt.rcParams["figure.max_open_warning"] = 0
    _apply_plot_style()

    print(f"Reading {csv_path} …")
    df = _load(csv_path)
    print(f"  {len(df)} successful runs loaded.\n")

    print("Generating figures:")

    all_figs: list[plt.Figure] = []

    # 1 – Pairwise heatmaps
    all_figs.extend(plot_pairwise_heatmaps(df, outdir))

    # 2 – Scaling curves
    all_figs.extend(plot_scaling_curves(df, outdir))

    # 3 – Total-agent scatter
    fig = plot_total_agent_scatter(df, outdir)
    if fig:
        all_figs.append(fig)

    # 3b – Total-agent fit residuals
    fig = plot_total_agent_fit_residuals(df, outdir)
    if fig:
        all_figs.append(fig)

    # 4 – Scaling exponents bar chart
    fig = plot_scaling_exponents(df, outdir)
    if fig:
        all_figs.append(fig)

    # 5 – Cost breakdown
    fig = plot_cost_breakdown(df, outdir)
    if fig:
        all_figs.append(fig)

    # 6 – Box-plots
    all_figs.extend(plot_boxplots(df, outdir))

    # 7 – Total wall-clock time bars
    fig = plot_total_time_bars(df, outdir)
    if fig:
        all_figs.append(fig)

    # 8 – Pairwise surface panel
    fig = plot_surface_panel(df, outdir)
    if fig:
        all_figs.append(fig)

    # 9 – Multi-panel summary
    fig = plot_summary_panel(df, outdir)
    if fig:
        all_figs.append(fig)

    # 10 – Cell-radius scaling curves
    fig = plot_cell_radius_scaling(df, outdir)
    if fig:
        all_figs.append(fig)

    # 11a – Init time scatter (pie markers + power-law fit)
    fig = plot_init_time_scatter(df, outdir)
    if fig:
        all_figs.append(fig)

    # 11c – Multivariate step-time diagnostics
    fig = plot_multivariate_fit_diagnostics(
        df,
        y_col="time_per_step_s",
        outdir=outdir,
        name="multivariate_step_time_fit",
        title="Step-time multivariate fit",
        y_label="time/step (s)",
    )
    if fig:
        all_figs.append(fig)

    # 11d – Multivariate init-time diagnostics
    fig = plot_multivariate_fit_diagnostics(
        df,
        y_col="init_time_s",
        outdir=outdir,
        name="multivariate_init_time_fit",
        title="Init-time multivariate fit",
        y_label="init time (s)",
    )
    if fig:
        all_figs.append(fig)

    # 13b - Combined multivariate summary panel
    fig = plot_multivariate_summary_panel(df, outdir)
    if fig:
        all_figs.append(fig)

    # 11b – Time breakdown stacked bars (separate figure)
    fig = plot_time_breakdown_bars(df, outdir)
    if fig:
        all_figs.append(fig)

    print(f"\nDone – {len(all_figs)} figures saved to {outdir}/")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
