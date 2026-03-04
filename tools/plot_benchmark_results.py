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
 1. **Pairwise heatmaps** – For every pair of swept columns (agent counts
    + CELL_RADIUS), a heatmap of mean time/step.
 2. **Scaling curves (log-log)** – One panel per swept variable showing how
    time/step grows with that variable.
 3. **Total-agent scatter** – Time/step vs. total agents with a power-law fit.
 4. **Per-variable scaling exponents** – Bar chart of estimated log-log slopes.
 5. **Cost breakdown** – Approximate contribution of each swept variable to
    total step time (marginal-difference pie chart).
 6. **Box-plots of time/step** – Distribution grouped by each swept variable.
 7. **Total wall-clock time bars** – Sorted horizontal bar chart of total
    simulation time per configuration.
 8. **Pairwise contourf panel** – Filled-contour combined figure for all
    pairs of swept columns.
 9. **Multi-panel summary** – One-page summary with mini scaling curves and
    total-agent scatter + power-law fit.
10. **Cell-radius scaling** – x = N_CELLS, y = time/step, one curve per
    CELL_RADIUS value (includes MAX_SEARCH_RADIUS in legend).
11. **Init vs simulation time** – Two-panel figure showing initialization
    and simulation time vs agent count, plus a stacked-bar breakdown.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = TOOLS_DIR / "benchmark_results.csv"
DEFAULT_OUTDIR = TOOLS_DIR / "benchmark_plots"

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

# All columns that may be swept in benchmark runs
SWEEP_COLS = AGENT_COLS + PARAM_COLS

COL_LABELS = {
    "ECM_POPULATION_SIZE": "ECM agents ($N^3$)",
    "N_CELLS": "Cells",
    "FOCAD_count_init": "Focal adhesions",
    "N_FNODES": "Fibre-network nodes",
    "CELL_RADIUS": "Cell radius",
    "MAX_SEARCH_RADIUS": "Max search radius",
    "TOTAL_AGENTS": "Total agents",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(col: str) -> str:
    return COL_LABELS.get(col, col)


def _load(csv_path: Path) -> pd.DataFrame:
    """Load CSV keeping only successful runs with valid timing."""
    df = pd.read_csv(csv_path)
    df = df[df["status"].str.startswith("OK", na=False)].copy()
    for col in ("time_per_step_s", "total_time_s", "init_time_s",
                 "simulation_time_s", "CELL_RADIUS", "MAX_SEARCH_RADIUS"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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
    # Only pairs where at least one of the two has more than 1 unique value
    cols = [c for c in SWEEP_COLS if c in df.columns and df[c].nunique() > 1]
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
        ax.set_xticklabels([f"{v:g}" for v in table.columns], fontsize=8)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([f"{v:g}" for v in table.index], fontsize=8)
        ax.set_xlabel(_label(col_x), fontsize=10)
        ax.set_ylabel(_label(col_y), fontsize=10)

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Mean time / step (s)", fontsize=9)

        # Annotate cells
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.3g}" if v < 100 else f"{v:.1f}"
                text_color = "white" if (use_log and v > np.sqrt(vmin * vmax)) or (not use_log and v > (vmin + vmax) / 2) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color=text_color)

        ax.set_title(f"Mean time/step:  {_label(col_y)}  vs  {_label(col_x)}",
                      fontsize=10, pad=8)
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

    Lines are split by the *second most varied* column so the viewer can
    see interaction effects.
    """
    cols = [c for c in SWEEP_COLS if c in df.columns and df[c].nunique() > 1]
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
        ax.set_xlabel(_label(main_col), fontsize=10)
        ax.set_ylabel("Time / step (s)", fontsize=10)
        ax.set_title(f"Scaling:  time/step  vs  {_label(main_col)}", fontsize=10)
        if grp_col:
            ax.legend(fontsize=7, loc="best", framealpha=0.9)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
        fig.tight_layout()
        _savefig(fig, outdir, f"scaling_{main_col}")
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# 3. Total-agent scatter with power-law fit
# ---------------------------------------------------------------------------

def plot_total_agent_scatter(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Time/step vs. TOTAL_AGENTS with a simple power-law fit."""
    if "TOTAL_AGENTS" not in df.columns or df["TOTAL_AGENTS"].nunique() < 2:
        print("  Skipping total-agent scatter (not enough data).")
        return None

    fig, ax = plt.subplots(figsize=(6, 4.5))

    x = df["TOTAL_AGENTS"].values.astype(float)
    y = df["time_per_step_s"].values.astype(float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    ax.scatter(x, y, s=18, alpha=0.7, edgecolors="none", c="C0")

    # Power-law fit:  log(y) = a * log(x) + b  →  y = 10^b * x^a
    if len(x) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(np.log10(x), np.log10(y), 1)
        exponent, intercept = coeffs
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        y_fit = 10 ** intercept * x_fit ** exponent
        ax.plot(x_fit, y_fit, "r--", lw=1.5,
                label=f"Power-law fit:  $t \\propto N^{{{exponent:.2f}}}$")
        ax.legend(fontsize=9, loc="upper left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total agents (ECM + Cells + FOCAD + FNODES)", fontsize=10)
    ax.set_ylabel("Time / step (s)", fontsize=10)
    ax.set_title("Overall computational scaling", fontsize=11)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout()
    _savefig(fig, outdir, "total_agent_scatter")
    return fig


# ---------------------------------------------------------------------------
# 4. Per-agent scaling exponents (bar chart)
# ---------------------------------------------------------------------------

def plot_scaling_exponents(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Estimate and compare the log-log slope for each swept variable."""
    cols = [c for c in SWEEP_COLS if c in df.columns and df[c].nunique() > 1]
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
                f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Scaling exponent  $\\alpha$  ($t \\propto N^\\alpha$)", fontsize=10)
    ax.set_title("Per-agent-type scaling exponents", fontsize=11)
    ax.axvline(1.0, ls="--", lw=0.8, color="grey", label="Linear ($\\alpha=1$)")
    ax.legend(fontsize=8)
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
    cols = [c for c in SWEEP_COLS if c in df.columns and df[c].nunique() > 1]
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
           colors=colours, textprops={"fontsize": 9})
    ax.set_title("Approximate cost attribution\n(marginal increase min → max)",
                  fontsize=10)
    fig.tight_layout()
    _savefig(fig, outdir, "cost_breakdown")
    return fig


# ---------------------------------------------------------------------------
# 6. Box-plots per variable
# ---------------------------------------------------------------------------

def plot_boxplots(df: pd.DataFrame, outdir: Path) -> list[plt.Figure]:
    """Box-plot of time/step grouped by each swept variable."""
    cols = [c for c in SWEEP_COLS if c in df.columns and df[c].nunique() > 1]
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
        ax.set_xticklabels([f"{g:g}" for g in groups], fontsize=8)
        ax.set_xlabel(_label(col), fontsize=10)
        ax.set_ylabel("Time / step (s)", fontsize=10)
        ax.set_title(f"Time/step distribution by {_label(col)}", fontsize=10)
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
    fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(sorted_df) + 2), 4.5))

    # Build a descriptive label for each run
    labels = []
    for _, r in sorted_df.iterrows():
        lbl = f"ECM={int(r['ECM_POPULATION_SIZE'])} C={int(r['N_CELLS'])} F={int(r['FOCAD_count_init'])}"
        if "N_FNODES" in r and pd.notna(r["N_FNODES"]):
            lbl += f" FN={int(r['N_FNODES'])}"
        if "CELL_RADIUS" in r and pd.notna(r.get("CELL_RADIUS")):
            lbl += f" R={r['CELL_RADIUS']:g}"
        labels.append(lbl)

    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(sorted_df)))
    ax.barh(range(len(sorted_df)), sorted_df["total_time_s"], color=colours,
            edgecolor="k", lw=0.3)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Total wall-clock time (s)", fontsize=10)
    ax.set_title("Total simulation time per configuration", fontsize=10)
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
    cols = [c for c in SWEEP_COLS if c in df.columns and df[c].nunique() > 1]
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
        figsize=(5.8 * ncols_grid, 4.6 * nrows_grid),
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
            levels=levels, cmap="viridis", norm=norm,
        )
        # Overlay contour lines with inline value labels
        cs = ax.contour(
            X_mesh, Y_mesh, Z,
            levels=cf.levels, colors="k", linewidths=0.5, alpha=0.55,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt="%.2g")

        ax.set_xlabel(_label(col_x), fontsize=15)
        ax.set_ylabel(_label(col_y), fontsize=15)
        ax.set_title(
            f"{_label(col_y)}  vs  {_label(col_x)}",
            fontsize=15, pad=6,
        )
        ax.tick_params(labelsize=15)
        ax.ticklabel_format(style="scientific", scilimits=(0, 3),
                            axis="both", useMathText=True)

    # Hide unused axes
    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Cellfoundry – Pairwise Cost Contours (mean time/step)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, outdir, "contourf_panel")
    return fig


# ---------------------------------------------------------------------------
# 9. Multi-panel summary (combined figure)
# ---------------------------------------------------------------------------

def plot_summary_panel(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """A single multi-panel figure with the key insights for quick review."""
    cols = [c for c in SWEEP_COLS if c in df.columns and df[c].nunique() > 1]
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
        ax.set_xlabel(_label(col), fontsize=9)
        ax.set_ylabel("Time/step (s)", fontsize=9)
        ax.set_title(f"Scaling: {_label(col)}", fontsize=9)
        ax.grid(True, which="both", ls=":", lw=0.3, alpha=0.5)

    # Total-agent scatter
    if idx < len(axes):
        ax = axes[idx]; idx += 1
        x = df["TOTAL_AGENTS"].values.astype(float)
        y = df["time_per_step_s"].values.astype(float)
        mask = (x > 0) & (y > 0)
        ax.scatter(x[mask], y[mask], s=12, alpha=0.7, c="C1", edgecolors="none")
        if mask.sum() >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
            x_fit = np.logspace(np.log10(x[mask].min()), np.log10(x[mask].max()), 80)
            ax.plot(x_fit, 10**coeffs[1] * x_fit**coeffs[0], "r--", lw=1.2)
            ax.set_title(f"Total agents ($\\alpha={coeffs[0]:.2f}$)", fontsize=9)
        else:
            ax.set_title("Total agents", fontsize=9)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Total agents", fontsize=9)
        ax.set_ylabel("Time/step (s)", fontsize=9)
        ax.grid(True, which="both", ls=":", lw=0.3, alpha=0.5)

    # Hide unused axes
    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cellfoundry – Performance Scaling Summary", fontsize=13,
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
    if "CELL_RADIUS" not in df.columns or df["CELL_RADIUS"].nunique() < 1:
        print("  Skipping cell-radius scaling (no CELL_RADIUS data).")
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
            label=f"CELL_RADIUS={cr:g}  (search={sr:.1f})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N_CELLS", fontsize=11)
    ax.set_ylabel("Time / step (s)", fontsize=11)
    ax.set_title("Effect of cell radius / search radius on scaling", fontsize=12)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout()
    _savefig(fig, outdir, "cell_radius_scaling")
    return fig


# ---------------------------------------------------------------------------
# 11. Initialization vs simulation time
# ---------------------------------------------------------------------------

def plot_init_vs_sim_time(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """Two-panel figure: (1) init & sim time vs total agents,
    (2) stacked bar of init + simulation time per configuration."""
    has_init = "init_time_s" in df.columns and df["init_time_s"].notna().any()
    has_sim = "simulation_time_s" in df.columns and df["simulation_time_s"].notna().any()
    if not (has_init and has_sim):
        print("  Skipping init-vs-sim time (columns missing or empty).")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: init & simulation time vs total agent count
    ax = axes[0]
    ax.scatter(df["TOTAL_AGENTS"], df["init_time_s"], alpha=0.6, s=25,
               edgecolors="none", c="C0", label="Init time")
    ax.scatter(df["TOTAL_AGENTS"], df["simulation_time_s"], alpha=0.6, s=25,
               edgecolors="none", c="C1", label="Simulation time")
    ax.set_xlabel("Total agents (ECM + Cells + FOCAD + FNODES)", fontsize=10)
    ax.set_ylabel("Time (s)", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Init & simulation time vs agent count", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    # Right: stacked bar per run (sorted by total time)
    ax = axes[1]
    sorted_df = df.sort_values("total_time_s", ascending=True).reset_index(drop=True)
    idx = range(len(sorted_df))
    ax.barh(list(idx), sorted_df["init_time_s"].values,
            color="C0", label="Initialization")
    ax.barh(list(idx), sorted_df["simulation_time_s"].values,
            left=sorted_df["init_time_s"].values,
            color="C1", label="Simulation")
    # Build concise labels
    labels = []
    for _, r in sorted_df.iterrows():
        lbl = f"C={int(r['N_CELLS'])}"
        if "CELL_RADIUS" in r.index and pd.notna(r.get("CELL_RADIUS")):
            lbl += f" R={r['CELL_RADIUS']:g}"
        lbl += f" ECM={int(r['ECM_POPULATION_SIZE'])}"
        labels.append(lbl)
    ax.set_yticks(list(idx))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title("Time breakdown per run", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="x", ls=":", lw=0.4, alpha=0.6)

    fig.tight_layout()
    _savefig(fig, outdir, "init_vs_sim_time")
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

    # 11 – Init vs simulation time breakdown
    fig = plot_init_vs_sim_time(df, outdir)
    if fig:
        all_figs.append(fig)

    print(f"\nDone – {len(all_figs)} figures saved to {outdir}/")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
