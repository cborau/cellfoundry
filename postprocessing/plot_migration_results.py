"""
Plot CELL migration results from simulation pickle metrics.

Usage:
    python postprocessing/plot_migration_results.py
    python postprocessing/plot_migration_results.py --pickle result_files/output_data_0.pickle
    python postprocessing/plot_migration_results.py --target optimizer/reference_data/target_cell_speed.csv
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "result_files"
DEFAULT_PICKLE_PATH = DEFAULT_RESULTS_DIR / "output_data_0.pickle"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _DummyModelParameterConfig:
    pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "helper_module" and name == "ModelParameterConfig":
            return _DummyModelParameterConfig
        if module.startswith("numpy._core"):
            remapped_module = "numpy.core" + module[len("numpy._core"):]
            return getattr(importlib.import_module(remapped_module), name)
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot migration metrics from CellFoundry pickle outputs")
    parser.add_argument("--pickle", default=str(DEFAULT_PICKLE_PATH), help="Path to simulation pickle file")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output plots and CSVs")
    parser.add_argument("--target", "--target-csv", dest="target_csv", default=None, help="Optional target CSV with cell_type,target_vmean,target_veff")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--tag", default="latest", help="Suffix tag for output file names")
    return parser.parse_args()


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    try:
        with path.open("rb") as f:
            return _SafeUnpickler(f).load()
    except Exception as exc:
        raise RuntimeError(f"Failed to load pickle '{path}': {exc}") from exc


def _coerce_metrics_frame(metrics: Any) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        return metrics.copy()
    return pd.DataFrame(metrics)


def load_metrics_from_pickle(results: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = results.get("CELL_SPEED_METRICS")
    if metrics is None or len(metrics) == 0:
        raise ValueError("Pickle file does not contain non-empty CELL_SPEED_METRICS")

    df = _coerce_metrics_frame(metrics)
    df = df.copy()
    if "dead" not in df.columns:
        df["dead"] = 0
    if "effective_displacement" not in df.columns and {"veff", "trajectory_time"}.issubset(df.columns):
        df["effective_displacement"] = df["veff"].astype(float) * df["trajectory_time"].astype(float)
    if "path_efficiency" not in df.columns:
        path = df["trajectory_length"].astype(float).to_numpy() if "trajectory_length" in df.columns else np.zeros(len(df), dtype=float)
        disp = df["effective_displacement"].astype(float).to_numpy() if "effective_displacement" in df.columns else np.zeros(len(df), dtype=float)
        eff = np.zeros(len(df), dtype=float)
        mask = path > 1e-12
        eff[mask] = disp[mask] / path[mask]
        df["path_efficiency"] = eff

    meta = {
        "source": "pickle",
        "time_step": _get_model_attr(results, "TIME_STEP", None),
        "save_every_n_steps": _get_model_attr(results, "SAVE_EVERY_N_STEPS", None),
    }
    return df, meta


def _get_model_attr(results: dict[str, Any], attr: str, default: Any) -> Any:
    model_config = results.get("MODEL_CONFIG") if results else None
    if model_config is None:
        return default
    return getattr(model_config, attr, default)


def load_target_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "cell_type" not in df.columns:
        return None
    return df


def build_aggregate_table(metrics: pd.DataFrame) -> pd.DataFrame:
    def _count_alive(series: pd.Series) -> int:
        return int(np.sum(series.to_numpy(dtype=int) == 0))

    def _count_dead(series: pd.Series) -> int:
        return int(np.sum(series.to_numpy(dtype=int) != 0))

    agg = metrics.groupby("cell_type", dropna=False).agg(
        n_cells=("id", "count"),
        n_alive=("dead", _count_alive),
        n_dead=("dead", _count_dead),
        vmean_mean=("vmean", "mean"),
        vmean_median=("vmean", "median"),
        veff_mean=("veff", "mean"),
        veff_median=("veff", "median"),
        trajectory_time_mean=("trajectory_time", "mean"),
        path_efficiency_mean=("path_efficiency", "mean"),
    )
    return agg.reset_index()


def _plot_distribution_panel(ax: plt.Axes, metrics: pd.DataFrame, value_col: str, ylabel: str, target_df: pd.DataFrame | None) -> None:
    cell_types = sorted(metrics["cell_type"].dropna().astype(int).unique().tolist())
    data = []
    for ct in cell_types:
        vals = metrics.loc[metrics["cell_type"].astype(int) == ct, value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        data.append(vals)
    positions = np.arange(len(cell_types), dtype=float)

    for pos, ct, vals in zip(positions, cell_types, data):
        color = f"C{ct % 10}"
        if len(vals) >= 2 and np.ptp(vals) > 1e-12:
            violin = ax.violinplot([vals], positions=[pos], widths=0.9, showmeans=False, showmedians=False, showextrema=False, bw_method=0.4)
            for body in violin["bodies"]:
                body.set_alpha(0.45)
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_linewidth(1.0)
        elif len(vals) == 1:
            ax.add_patch(plt.Rectangle((pos - 0.22, vals[0] * 0.9995), 0.44, max(abs(vals[0]) * 0.001, 1e-9), facecolor=color, edgecolor="black", alpha=0.45))
        elif len(vals) > 1:
            center = float(vals[0])
            spread = max(abs(center) * 0.015, 1e-6)
            ax.add_patch(plt.Rectangle((pos - 0.22, center - 0.5 * spread), 0.44, spread, facecolor=color, edgecolor="black", alpha=0.45))

        if len(vals):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=10, color=color, alpha=0.18, linewidths=0, zorder=2)

    means = [float(np.mean(vals)) if len(vals) else 0.0 for vals in data]
    medians = [float(np.median(vals)) if len(vals) else 0.0 for vals in data]
    ax.scatter(positions, means, marker="o", color="black", label="mean", zorder=3)
    ax.scatter(positions, medians, marker="D", color="tab:red", label="median", zorder=3)

    if target_df is not None:
        target_col = f"target_{value_col}"
        if target_col in target_df.columns:
            target_lookup = {int(row.cell_type): float(getattr(row, target_col)) for row in target_df.itertuples() if pd.notna(getattr(row, target_col, np.nan))}
            target_vals = [target_lookup.get(ct, np.nan) for ct in cell_types]
            ax.scatter(positions, target_vals, marker="_", s=400, linewidths=2.0, color="tab:green", label="target", zorder=4)

    ax.set_xticks(positions, [str(ct) for ct in cell_types])
    ax.set_xlabel("Cell type")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def create_summary_figure(metrics: pd.DataFrame, aggregate_df: pd.DataFrame, target_df: pd.DataFrame | None, meta: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Migration Summary", fontsize=14)

    _plot_distribution_panel(axes[0, 0], metrics, "vmean", "Mean speed [um/s]", target_df)
    axes[0, 0].set_title("Per-type vmean distribution")
    axes[0, 0].legend(loc="best")

    _plot_distribution_panel(axes[0, 1], metrics, "veff", "Effective speed [um/s]", target_df)
    axes[0, 1].set_title("Per-type veff distribution")
    axes[0, 1].legend(loc="best")

    ax = axes[1, 0]
    for cell_type, group in metrics.groupby(metrics["cell_type"].astype(int)):
        ax.scatter(
            group["vmean"],
            group["veff"],
            s=35,
            alpha=0.75,
            label=f"type {cell_type}",
        )
    limit = float(max(metrics["vmean"].max(), metrics["veff"].max(), 1e-6))
    ax.plot([0.0, limit], [0.0, limit], "k--", linewidth=1.0, label="veff = vmean")
    ax.set_xlabel("vmean [um/s]")
    ax.set_ylabel("veff [um/s]")
    ax.set_title("Per-cell vmean vs veff")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax2 = ax.twinx()
    x = np.arange(len(aggregate_df), dtype=float)
    width = 0.18
    vmean_mean = ax.bar(x - 1.5 * width, aggregate_df["vmean_mean"], width=width, label="vmean mean", color="#355C7D", alpha=0.9)
    vmean_median = ax.bar(x - 0.5 * width, aggregate_df["vmean_median"], width=width, label="vmean median", color="#6C8EAD", alpha=0.9, hatch="//")
    veff_mean = ax2.bar(x + 0.5 * width, aggregate_df["veff_mean"], width=width, label="veff mean", color="#C06C2B", alpha=0.9)
    veff_median = ax2.bar(x + 1.5 * width, aggregate_df["veff_median"], width=width, label="veff median", color="#E09F5A", alpha=0.9, hatch="//")
    ax.set_xticks(x, [str(int(ct)) for ct in aggregate_df["cell_type"]])
    ax.set_xlabel("Cell type")
    ax.set_ylabel("vmean [um/s]", color="#355C7D")
    ax2.set_ylabel("veff [um/s]", color="#C06C2B")
    ax.tick_params(axis="y", colors="#355C7D")
    ax2.tick_params(axis="y", colors="#C06C2B")
    ax.set_title("Aggregate speeds by cell type")
    ax.grid(True, axis="y", alpha=0.25)
    handles = [vmean_mean[0], vmean_median[0], veff_mean[0], veff_median[0]]
    labels = ["vmean mean", "vmean median", "veff mean", "veff median"]
    ax.legend(handles, labels, loc="best")

    fig.tight_layout()
    return fig


def create_diagnostics_figure(metrics: pd.DataFrame, aggregate_df: pd.DataFrame, meta: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Migration Diagnostics", fontsize=14)

    ax = axes[0, 0]
    for cell_type, group in metrics.groupby(metrics["cell_type"].astype(int)):
        ax.hist(group["path_efficiency"], bins=20, alpha=0.45, label=f"type {cell_type}")
    ax.set_xlabel("Path efficiency = displacement / path length")
    ax.set_ylabel("Cell count")
    ax.set_title("Path efficiency distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0, 1]
    max_val = float(max(metrics["trajectory_length"].max(), metrics["effective_displacement"].max(), 1e-6))
    for cell_type, group in metrics.groupby(metrics["cell_type"].astype(int)):
        ax.scatter(
            group["trajectory_length"],
            group["effective_displacement"],
            s=35,
            alpha=0.75,
            label=f"type {cell_type}",
        )
    ax.plot([0.0, max_val], [0.0, max_val], "k--", linewidth=1.0, label="displacement = path")
    ax.set_xlabel("Trajectory length [um]")
    ax.set_ylabel("Effective displacement [um]")
    ax.set_title("Path length vs displacement")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 0]
    for cell_type, group in metrics.groupby(metrics["cell_type"].astype(int)):
        values = group["trajectory_time"].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(values, np.full(len(values), cell_type, dtype=float) + jitter, s=26, alpha=0.5, label=f"type {cell_type}")
        ax.scatter([float(np.mean(values))], [cell_type], s=90, marker="|", linewidths=2.0, color="black", zorder=4)
    max_time = float(np.nanmax(metrics["trajectory_time"].to_numpy(dtype=float))) if len(metrics) else 0.0
    ax.set_xlim(left=0.0, right=max(max_time * 1.05, 1.0))
    ax.set_yticks(sorted(metrics["cell_type"].dropna().astype(int).unique().tolist()))
    ax.set_xlabel("Tracked time [s]")
    ax.set_ylabel("Cell type")
    ax.set_title("Tracked time by cell type")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 1]
    x = np.arange(len(aggregate_df), dtype=float)
    ax.bar(x, aggregate_df["n_alive"], label="alive")
    ax.bar(x, aggregate_df["n_dead"], bottom=aggregate_df["n_alive"], label="dead")
    ax.set_xticks(x, [str(int(ct)) for ct in aggregate_df["cell_type"]])
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Cell count")
    ax.set_title(f"Final alive/dead counts by type ({meta.get('source')})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


def save_outputs(metrics: pd.DataFrame, aggregate_df: pd.DataFrame, summary_fig: plt.Figure, diagnostics_fig: plt.Figure, outdir: Path, tag: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outdir / f"migration_metrics_per_cell_{tag}.csv", index=False)
    aggregate_df.to_csv(outdir / f"migration_metrics_by_type_{tag}.csv", index=False)
    summary_fig.savefig(outdir / f"migration_summary_{tag}.png", dpi=200, bbox_inches="tight")
    diagnostics_fig.savefig(outdir / f"migration_diagnostics_{tag}.png", dpi=200, bbox_inches="tight")


def print_summary(metrics: pd.DataFrame, aggregate_df: pd.DataFrame, meta: dict[str, Any], target_df: pd.DataFrame | None) -> None:
    print("\nMigration metrics summary")
    print("=" * 80)
    print("Source: CELL_SPEED_METRICS from pickle")
    print(f"Cells: {len(metrics)}")
    if meta.get("time_step") is not None:
        print(f"Time step used: {meta['time_step']}")
    print()
    print(aggregate_df.to_string(index=False))

    if target_df is not None:
        merged = aggregate_df.merge(target_df, on="cell_type", how="left")
        print("\nTargets")
        print("-" * 80)
        print(merged[[col for col in merged.columns if col in {"cell_type", "vmean_mean", "veff_mean", "target_vmean", "target_veff"}]].to_string(index=False))


def main() -> None:
    args = parse_args()
    pickle_path = Path(args.pickle).resolve()
    outdir = Path(args.outdir).resolve()
    target_csv = Path(args.target_csv).resolve() if args.target_csv else None

    results = _load_pickle(pickle_path)
    metrics, meta = load_metrics_from_pickle(results)

    metrics = metrics.copy()
    if "path_efficiency" not in metrics.columns:
        path = metrics["trajectory_length"].astype(float).to_numpy()
        disp = metrics["effective_displacement"].astype(float).to_numpy()
        efficiency = np.zeros(len(metrics), dtype=float)
        mask = path > 1e-12
        efficiency[mask] = disp[mask] / path[mask]
        metrics["path_efficiency"] = efficiency

    aggregate_df = build_aggregate_table(metrics)
    target_df = load_target_csv(target_csv) if target_csv is not None else None

    print_summary(metrics, aggregate_df, meta, target_df)

    summary_fig = create_summary_figure(metrics, aggregate_df, target_df, meta)
    diagnostics_fig = create_diagnostics_figure(metrics, aggregate_df, meta)
    save_outputs(metrics, aggregate_df, summary_fig, diagnostics_fig, outdir, args.tag)

    if args.show:
        plt.show()
    else:
        plt.close(summary_fig)
        plt.close(diagnostics_fig)


if __name__ == "__main__":
    main()