"""
plot_benchmark_results.py – Visualise cellfoundry performance benchmark results.

Reads ``tools/benchmark_results.csv`` (produced by ``benchmark_perf.py``) and
generates a comprehensive set of publication-quality figures that characterise
how simulation cost scales with the number of agents.

Usage
-----
    python tools/plot_benchmark_results.py                        # save only
    python tools/plot_benchmark_results.py --show                 # save + show
    python tools/plot_benchmark_results.py --csv path/to/other.csv
    python tools/plot_benchmark_results.py --outdir my_figs/

Figures produced
----------------
1. **Pairwise heatmaps** – For every pair of swept agent-count columns,
   a heatmap of mean time/step (averaged over the remaining axes).
2. **Scaling curves (log-log)** – One panel per agent type showing how
   time/step grows with that agent count (each line = one level of a
   secondary variable).
3. **Total-agent scatter** – Time/step vs. total agents (ECM + cells +
   FOCAD + FNODES) with a power-law fit, revealing whether cost is
   sub-/super-linear overall.
4. **Per-agent-type scaling exponents** – Bar chart of estimated scaling
   exponents (slope in log-log space) for each agent type.
5. **Stacked cost breakdown** – Approximate contribution of each agent
   population to the total step time (rough proportional attribution via
   a simple regression decomposition).
6. **Box-plots of time/step** – Distribution of time/step grouped by each
   swept variable, useful for spotting variability and outliers.
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
AGENT_LABELS = {
    "ECM_POPULATION_SIZE": "ECM agents ($N^3$)",
    "N_CELLS": "Cells",
    "FOCAD_count_init": "Focal adhesions",
    "N_FNODES": "Fibre-network nodes",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(col: str) -> str:
    return AGENT_LABELS.get(col, col)


def _load(csv_path: Path) -> pd.DataFrame:
    """Load CSV keeping only successful runs with valid timing."""
    df = pd.read_csv(csv_path)
    df = df[df["status"].str.startswith("OK", na=False)].copy()
    df["time_per_step_s"] = pd.to_numeric(df["time_per_step_s"], errors="coerce")
    df["total_time_s"] = pd.to_numeric(df["total_time_s"], errors="coerce")
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


def _savefig(fig: plt.Figure, outdir: Path, name: str, dpi: int = 200) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 1. Pairwise heatmaps
# ---------------------------------------------------------------------------

def plot_pairwise_heatmaps(df: pd.DataFrame, outdir: Path) -> list[plt.Figure]:
    """One heatmap per pair of agent-count columns.  Colour = mean time/step."""
    # Only pairs where at least one of the two has more than 1 unique value
    cols = [c for c in AGENT_COLS if c in df.columns and df[c].nunique() > 1]
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
    cols = [c for c in AGENT_COLS if c in df.columns and df[c].nunique() > 1]
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
    """Estimate and compare the log-log slope for each agent type."""
    cols = [c for c in AGENT_COLS if c in df.columns and df[c].nunique() > 1]
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
    """Stacked bar showing relative time contribution when increasing each
    agent population in turn, estimated via marginal differences."""
    cols = [c for c in AGENT_COLS if c in df.columns and df[c].nunique() > 1]
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
    cols = [c for c in AGENT_COLS if c in df.columns and df[c].nunique() > 1]
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
# 8. Multi-panel summary (combined figure)
# ---------------------------------------------------------------------------

def plot_summary_panel(df: pd.DataFrame, outdir: Path) -> plt.Figure | None:
    """A single multi-panel figure with the key insights for quick review."""
    cols = [c for c in AGENT_COLS if c in df.columns and df[c].nunique() > 1]
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
        "--dpi", type=int, default=200,
        help="Resolution for saved PNGs. Default: 200")
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

    # 8 – Multi-panel summary
    fig = plot_summary_panel(df, outdir)
    if fig:
        all_figs.append(fig)

    print(f"\nDone – {len(all_figs)} figures saved to {outdir}/")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
