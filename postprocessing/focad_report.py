#!/usr/bin/env python3
"""
Generate FOCAD reports from  pickle outputs.

This script:
1) Loads `output_data_*.pickle`
2) Exports FOCAD metrics/polarity time series to CSV
3) Builds summary CSV with final-20% statistics and front-vs-rear diagnostics
4) Saves convenient plots for quick inspection

Usage:
    python postprocessing/focad_report.py
    python postprocessing/focad_report.py --pickle result_files/output_data_0.pickle
    python postprocessing/focad_report.py --tag mytag --outdir result_files/reports
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


class DummyModelParameterConfig:
    pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "helper_module" and name == "ModelParameterConfig":
            return DummyModelParameterConfig
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FOCAD reports and plots from CellFoundry pickle output")
    parser.add_argument(
        "--pickle",
        default="result_files/output_data_0.pickle",
        help="Path to simulation pickle file",
    )
    parser.add_argument(
        "--outdir",
        default="result_files",
        help="Directory where CSV summaries and plots will be saved",
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="Tag suffix used in output filenames (e.g., phase3, phase4, runA)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )
    parser.add_argument(
        "--n-cells",
        type=float,
        default=None,
        help="Override number of cells used to compute avg FOCAD per cell (defaults to MODEL_CONFIG.N_CELLS if available)",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with path.open("rb") as f:
        return SafeUnpickler(f).load()


def compute_last20_stats(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if df is None or len(df) == 0:
        return out

    tail = df.iloc[int(0.8 * len(df)) :]
    for col in ("attached", "total", "attached_ratio", "mean_f_mag"):
        if col in tail.columns:
            out[f"{prefix}_{col}_mean_last20"] = float(tail[col].mean())
            out[f"{prefix}_{col}_std_last20"] = float(tail[col].std(ddof=1)) if len(tail) > 1 else 0.0
    if "avg_focad_per_cell" in tail.columns:
        out[f"{prefix}_avg_focad_per_cell_mean_last20"] = float(tail["avg_focad_per_cell"].mean())
        out[f"{prefix}_avg_focad_per_cell_std_last20"] = float(tail["avg_focad_per_cell"].std(ddof=1)) if len(tail) > 1 else 0.0
    return out


def resolve_n_cells(data: dict[str, Any], override_n_cells: float | None) -> float | None:
    if override_n_cells is not None:
        return float(override_n_cells)

    model_cfg = data.get("MODEL_CONFIG", None)
    if model_cfg is None:
        return None

    n_cells = getattr(model_cfg, "N_CELLS", None)
    if n_cells is None:
        return None

    try:
        n_cells_f = float(n_cells)
    except (TypeError, ValueError):
        return None

    if n_cells_f <= 0:
        return None
    return n_cells_f


def compute_polarity_summary(fpol: pd.DataFrame, fmet: pd.DataFrame) -> dict[str, float | bool]:
    out: dict[str, float | bool] = {}
    if fpol is None or len(fpol) == 0:
        return out

    valid = fpol[(fpol["front_count"] > 0) & (fpol["rear_count"] > 0)].copy()
    n = len(valid)
    out["valid_steps"] = int(n)
    if n == 0:
        return out

    valid["ratio_diff"] = valid["front_attached_ratio"] - valid["rear_attached_ratio"]

    mean_diff = float(valid["ratio_diff"].mean())
    std_diff = float(valid["ratio_diff"].std(ddof=1)) if n > 1 else 0.0
    t_like = mean_diff / (std_diff / math.sqrt(n)) if (n > 1 and std_diff > 0) else float("nan")
    p_one_sided = 0.5 * math.erfc(t_like / math.sqrt(2.0)) if (n > 1 and std_diff > 0) else float("nan")

    out.update(
        {
            "front_attached_ratio_mean": float(valid["front_attached_ratio"].mean()),
            "rear_attached_ratio_mean": float(valid["rear_attached_ratio"].mean()),
            "front_attached_ratio_weighted": float(valid["front_attached"].sum() / valid["front_count"].sum()),
            "rear_attached_ratio_weighted": float(valid["rear_attached"].sum() / valid["rear_count"].sum()),
            "front_minus_rear_attached_ratio_mean": mean_diff,
            "frontness_front_mean": float(valid["frontness_front_mean"].mean()),
            "frontness_rear_mean": float(valid["frontness_rear_mean"].mean()),
            "k_on_eff_front_mean": float(valid["k_on_eff_front_mean"].mean()),
            "k_on_eff_rear_mean": float(valid["k_on_eff_rear_mean"].mean()),
            "k_off_0_eff_front_mean": float(valid["k_off_0_eff_front_mean"].mean()),
            "k_off_0_eff_rear_mean": float(valid["k_off_0_eff_rear_mean"].mean()),
            "t_like_front_minus_rear": t_like,
            "p_one_sided_front_gt_rear": p_one_sided,
        }
    )

    if fmet is not None and len(fmet) > 0:
        out["has_nan_attached_ratio"] = bool(fmet["attached_ratio"].isna().any())
        out["has_nan_mean_f_mag"] = bool(fmet["mean_f_mag"].isna().any())
        out["max_mean_f_mag"] = float(fmet["mean_f_mag"].max())
        out["min_mean_f_mag"] = float(fmet["mean_f_mag"].min())

    return out


def make_plots(fmet: pd.DataFrame, fpol: pd.DataFrame, outdir: Path, tag: str, show: bool) -> None:
    if len(fmet) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fmet["step"], fmet["attached_ratio"], label="attached_ratio", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Attached ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(fmet["step"], fmet["mean_f_mag"], label="mean_f_mag", color="tab:orange", linewidth=2)
    ax2.set_ylabel("Mean |F| (nN)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"fa_metrics_trends_{tag}.png", dpi=180)

    if "avg_focad_per_cell" in fmet.columns:
        avg_window = max(5, min(25, len(fmet) // 6 if len(fmet) > 0 else 5))
        avg_series = fmet["avg_focad_per_cell"]
        avg_median = avg_series.rolling(window=avg_window, min_periods=1).median()
        avg_p25 = avg_series.rolling(window=avg_window, min_periods=1).quantile(0.25)
        avg_p75 = avg_series.rolling(window=avg_window, min_periods=1).quantile(0.75)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            fmet["step"],
            avg_series,
            label="avg_focad_per_cell (raw)",
            linewidth=1.4,
            alpha=0.35,
            color="tab:blue",
        )
        ax.fill_between(
            fmet["step"],
            avg_p25,
            avg_p75,
            alpha=0.22,
            color="tab:blue",
            label=f"IQR (rolling, w={avg_window})",
        )
        ax.plot(
            fmet["step"],
            avg_median,
            label=f"median (rolling, w={avg_window})",
            linewidth=2.4,
            color="tab:blue",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Average FOCAD per cell")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"fa_avg_focad_per_cell_{tag}.png", dpi=180)

    window = max(3, min(15, len(fmet) // 5 if len(fmet) > 0 else 3))
    smooth_attached = fmet["attached_ratio"].rolling(window=window, min_periods=1).mean()
    smooth_force = fmet["mean_f_mag"].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fmet["step"], fmet["attached_ratio"], alpha=0.35, linewidth=1.5, label="attached_ratio (raw)")
    ax.plot(fmet["step"], smooth_attached, linewidth=2.5, label=f"attached_ratio (rolling mean, w={window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Attached ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(fmet["step"], fmet["mean_f_mag"], color="tab:orange", alpha=0.35, linewidth=1.5, label="mean_f_mag (raw)")
    ax2.plot(
        fmet["step"],
        smooth_force,
        color="tab:red",
        linewidth=2.5,
        label=f"mean_f_mag (rolling mean, w={window})",
    )
    ax2.set_ylabel("Mean |F| (nN)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"fa_metrics_trends_smoothed_{tag}.png", dpi=180)

    if len(fpol) > 0:
        valid = fpol[(fpol["front_count"] > 0) & (fpol["rear_count"] > 0)].copy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(valid["step"], valid["front_attached_ratio"], label="front_attached_ratio", linewidth=2)
        ax.plot(valid["step"], valid["rear_attached_ratio"], label="rear_attached_ratio", linewidth=2)
        ax.plot(
            valid["step"],
            valid["front_attached_ratio"] - valid["rear_attached_ratio"],
            label="front - rear",
            linewidth=2,
            linestyle="--",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Attached ratio")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"fa_polarity_attached_ratio_{tag}.png", dpi=180)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(valid["step"], valid["k_on_eff_front_mean"], label="k_on_eff_front", linewidth=2)
        ax.plot(valid["step"], valid["k_on_eff_rear_mean"], label="k_on_eff_rear", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Effective k_on [1/s]")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"fa_polarity_kon_{tag}.png", dpi=180)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(valid["step"], valid["k_off_0_eff_front_mean"], label="k_off_0_eff_front", linewidth=2)
        ax.plot(valid["step"], valid["k_off_0_eff_rear_mean"], label="k_off_0_eff_rear", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Effective k_off_0 [1/s]")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"fa_polarity_koff_{tag}.png", dpi=180)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")

        ax = axes[0, 0]
        ax.plot(fmet["step"], fmet["attached_ratio"], label="attached_ratio", linewidth=2)
        ax.set_ylabel("Attached ratio")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(fmet["step"], fmet["mean_f_mag"], label="mean_f_mag", color="tab:orange", linewidth=2)
        ax2.set_ylabel("Mean |F| (nN)")
        ax.set_title("Global FOCAD Trends")

        ax = axes[0, 1]
        ax.plot(valid["step"], valid["front_attached_ratio"], label="front", linewidth=2)
        ax.plot(valid["step"], valid["rear_attached_ratio"], label="rear", linewidth=2)
        ax.plot(valid["step"], valid["front_attached_ratio"] - valid["rear_attached_ratio"], label="front-rear", linestyle="--", linewidth=2)
        ax.set_ylabel("Attached ratio")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        ax.set_title("Front/Rear Attachment")

        ax = axes[1, 0]
        ax.plot(valid["step"], valid["k_on_eff_front_mean"], label="k_on front", linewidth=2)
        ax.plot(valid["step"], valid["k_on_eff_rear_mean"], label="k_on rear", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("k_on_eff [1/s]")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        ax.set_title("Effective Attachment Rates")

        ax = axes[1, 1]
        ax.plot(valid["step"], valid["k_off_0_eff_front_mean"], label="k_off front", linewidth=2)
        ax.plot(valid["step"], valid["k_off_0_eff_rear_mean"], label="k_off rear", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("k_off_0_eff [1/s]")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        ax.set_title("Effective Detachment Rates")

        fig.suptitle("CellFoundry FOCAD Dashboard", fontsize=14)
        fig.tight_layout()
        fig.savefig(outdir / f"fa_dashboard_{tag}.png", dpi=180)

    if show:
        plt.show()
    else:
        plt.close("all")


def write_plot_explanations(outdir: Path, tag: str) -> Path:
    text = f"""CellFoundry FOCAD Plot Guide ({tag})
=================================

These plots summarize focal-adhesion (FOCAD) mechanics over simulation time.

1) fa_metrics_trends_{tag}.png
-----------------------------
- Blue axis (left): attached_ratio = attached FAs / total FAs.
- Orange axis (right): mean_f_mag = average traction magnitude per FOCAD [nN].
- Use this to check global FOCAD engagement and force levels.

2) fa_metrics_trends_smoothed_{tag}.png
--------------------------------------
- Same metrics as above, but with rolling averages overlaid.
- Faint lines: raw step-to-step values.
- Thick lines: smoothed trends (rolling mean window shown in legend).
- Useful for seeing long-timescale drift and comparing conditions with less noise.

2b) fa_avg_focad_per_cell_{tag}.png
----------------------------------
- Time series of average FOCAD count per cell.
- Computed as total FOCAD / N_CELLS (from MODEL_CONFIG.N_CELLS or --n-cells override).
- Shows raw series, rolling median, and shaded interquartile range (IQR, 25th–75th percentiles).
- Useful to verify that birth/death dynamics stay in expected bounds over time.

3) fa_polarity_attached_ratio_{tag}.png
--------------------------------------
- front_attached_ratio: attachment fraction among front-facing adhesions.
- rear_attached_ratio: attachment fraction among rear-facing adhesions.
- front - rear: asymmetry score (>0 means front more attached than rear).
- This is the key Phase-3 polarity behavior check.

4) fa_polarity_kon_{tag}.png
---------------------------
- Effective attachment rates (k_on_eff) for front vs rear groups.
- Confirms polarity bias in attachment kinetics.

5) fa_polarity_koff_{tag}.png
----------------------------
- Effective baseline detachment rates (k_off_0_eff) for front vs rear.
- Lower front k_off with higher rear k_off generally supports persistent front adhesions.

CSV files generated alongside plots
---------------------------------
- fa_metrics_{tag}.csv: per-step global FOCAD metrics.
- fa_polarity_{tag}.csv: per-step front/rear split metrics.
- fa_summary_{tag}.csv: condensed summary for final-20% statistics and polarity checks.

Important interpretation notes
-----------------------------
- These are model-level diagnostics, not direct experimental observables.
- A stable run should avoid NaNs and unrealistic force growth.
- Front/rear asymmetry can depend on network geometry, initialization, and parameter regime.
"""
    out = outdir / f"fa_plot_explanations_{tag}.txt"
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    pickle_path = Path(args.pickle)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_pickle(pickle_path)

    fmet = data.get("FOCAD_METRICS_OVER_TIME", pd.DataFrame()).copy().reset_index(drop=True)
    fpol = data.get("FOCAD_POLARITY_METRICS_OVER_TIME", pd.DataFrame()).copy().reset_index(drop=True)

    if len(fmet) > 0:
        fmet.insert(0, "step", range(1, len(fmet) + 1))
    if len(fpol) > 0:
        fpol.insert(0, "step", range(1, len(fpol) + 1))

    n_cells = resolve_n_cells(data, args.n_cells)
    if len(fmet) > 0 and n_cells is not None and n_cells > 0 and "total" in fmet.columns:
        fmet["avg_focad_per_cell"] = fmet["total"] / n_cells

    metrics_csv = outdir / f"fa_metrics_{args.tag}.csv"
    polarity_csv = outdir / f"fa_polarity_{args.tag}.csv"
    summary_csv = outdir / f"fa_summary_{args.tag}.csv"

    fmet.to_csv(metrics_csv, index=False)
    fpol.to_csv(polarity_csv, index=False)

    summary = {}
    summary.update(compute_last20_stats(fmet, args.tag))
    summary.update(compute_polarity_summary(fpol, fmet))
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    make_plots(fmet, fpol, outdir, args.tag, args.show)
    explanation_txt = write_plot_explanations(outdir, args.tag)

    print(f"Saved: {metrics_csv}")
    print(f"Saved: {polarity_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {explanation_txt}")
    if n_cells is not None:
        print(f"Avg FOCAD/cell computed with N_CELLS={n_cells:g}")
    else:
        print("Avg FOCAD/cell not computed: N_CELLS unavailable (use --n-cells to override)")
    print(f"Saved plots with tag '{args.tag}' in: {outdir}")


if __name__ == "__main__":
    main()
