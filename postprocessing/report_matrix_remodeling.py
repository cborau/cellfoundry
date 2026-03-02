"""
Build matrix remodeling reports from pickle or VTK outputs.

Tracks per-step:
- total FNODE count
- new (secreted) FNODEs per step and cumulative
- mean / total degradation
- mean / total reinforcement
- net remodeling (reinforcement - degradation)
- mean / total elastic energy
- positive-only means (from VTK source only)
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class _DummyModelParameterConfig:
    pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "helper_module" and name == "ModelParameterConfig":
            return _DummyModelParameterConfig
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matrix remodeling reports from pickle or VTK files"
    )
    parser.add_argument("--source", choices=["pickle", "vtk"], default="pickle",
                        help="Data source: 'pickle' (default) or 'vtk'")
    parser.add_argument("--pickle", default="../result_files/output_data_0.pickle",
                        help="Path to simulation pickle file (used when --source pickle)")
    parser.add_argument("--indir", default="../result_files",
                        help="Directory containing fibre_network_data_tXXXX.vtk (used when --source vtk)")
    parser.add_argument("--outdir", default="results", help="Directory for CSV/plots")
    parser.add_argument("--tag", default="latest", help="Tag suffix for output filenames")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    return parser.parse_args()


def _read_scalar_float(lines: list[str], name: str, n: int) -> list[float]:
    """Read a VTK SCALARS float field."""
    header = f"SCALARS {name}"
    idx = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if idx is None:
        return [0.0] * n
    start = idx + 2  # skip SCALARS + LOOKUP_TABLE
    out: list[float] = []
    for i in range(start, start + n):
        token = lines[i].strip().split()[0]
        out.append(float(token))
    return out


def _read_scalar_int(lines: list[str], name: str, n: int) -> list[int]:
    """Read a VTK SCALARS int field."""
    header = f"SCALARS {name}"
    idx = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if idx is None:
        return [0] * n
    start = idx + 2  # skip SCALARS + LOOKUP_TABLE
    out: list[int] = []
    for i in range(start, start + n):
        token = lines[i].strip().split()[0]
        out.append(int(float(token)))
    return out


def read_fnode_vtk(path: Path) -> dict:
    """Parse a fibre_network_data_tXXXX.vtk and return per-node data."""
    lines = [line.strip() for line in path.read_text().splitlines()]

    # Find POINT_DATA to get point count
    point_data_idx = next(
        (i for i, l in enumerate(lines) if l.startswith("POINT_DATA")), None
    )
    if point_data_idx is None:
        return {"n_total": 0, "n_secreted": 0, "degradation": [], "reinforcement": [],
                "elastic_energy": [], "is_corner": []}

    n_points = int(lines[point_data_idx].split()[1])

    is_corner = _read_scalar_int(lines, "is_corner", n_points)
    elastic_energy = _read_scalar_float(lines, "elastic_energy", n_points)
    degradation = _read_scalar_float(lines, "degradation", n_points)
    reinforcement = _read_scalar_float(lines, "reinforcement", n_points)
    secreted = _read_scalar_int(lines, "secreted", n_points)

    # Filter out corner points
    fnode_mask = [c == 0 for c in is_corner]
    deg_fnodes = [d for d, m in zip(degradation, fnode_mask) if m]
    reinf_fnodes = [r for r, m in zip(reinforcement, fnode_mask) if m]
    ee_fnodes = [e for e, m in zip(elastic_energy, fnode_mask) if m]
    sec_fnodes = [s for s, m in zip(secreted, fnode_mask) if m]

    n_total = sum(fnode_mask)
    n_secreted = sum(sec_fnodes)

    return {
        "n_total": n_total,
        "n_secreted": n_secreted,
        "degradation": deg_fnodes,
        "reinforcement": reinf_fnodes,
        "elastic_energy": ee_fnodes,
    }


def build_timeseries_from_pickle(pickle_path: Path) -> pd.DataFrame:
    """Build timeseries from FNODE_METRICS_OVER_TIME stored in pickle."""
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    with pickle_path.open("rb") as f:
        data = _SafeUnpickler(f).load()

    fnode_met = data.get("FNODE_METRICS_OVER_TIME")
    if fnode_met is None or (isinstance(fnode_met, list) and len(fnode_met) == 0):
        raise ValueError("FNODE_METRICS_OVER_TIME not found or empty in pickle. "
                         "Re-run the simulation to generate FNODE metrics, or use --source vtk.")
    if isinstance(fnode_met, list):
        fnode_met = pd.DataFrame(fnode_met)

    df = fnode_met.copy().reset_index(drop=True)
    df.insert(0, "step", range(1, len(df) + 1))

    # Compute new_fnodes_this_step from secreted cumulative
    if "n_fnodes_secreted_cumulative" in df.columns:
        df["new_fnodes_this_step"] = df["n_fnodes_secreted_cumulative"].diff().fillna(0).clip(lower=0).astype(int)
    else:
        df["new_fnodes_this_step"] = 0

    # Rename sum columns to total for consistency
    rename_map = {}
    for col in ("sum_degradation", "sum_reinforcement", "sum_elastic_energy"):
        target = col.replace("sum_", "total_")
        if col in df.columns and target not in df.columns:
            rename_map[col] = target
    df.rename(columns=rename_map, inplace=True)

    # Positive-only means are NOT available from pickle (no per-agent data)
    for col in ("n_degradation_positive", "mean_degradation_positive",
                "n_reinforcement_positive", "mean_reinforcement_positive",
                "n_elastic_energy_positive", "mean_elastic_energy_positive"):
        if col not in df.columns:
            df[col] = np.nan

    return df


def build_timeseries_from_vtk(vtk_files: list[Path]) -> pd.DataFrame:
    rows = []
    prev_n_secreted = 0

    for vtk in vtk_files:
        step_match = re.search(r"t(\d+)\.vtk$", vtk.name)
        step = int(step_match.group(1)) if step_match else -1

        data = read_fnode_vtk(vtk)
        n_total = data["n_total"]
        n_secreted = data["n_secreted"]
        deg = np.array(data["degradation"])
        reinf = np.array(data["reinforcement"])
        ee = np.array(data["elastic_energy"])

        new_secreted_this_step = max(0, n_secreted - prev_n_secreted)

        # Positive-only filters
        deg_pos = deg[deg > 0] if len(deg) > 0 else np.array([])
        reinf_pos = reinf[reinf > 0] if len(reinf) > 0 else np.array([])
        ee_pos = ee[ee > 0] if len(ee) > 0 else np.array([])

        row = {
            "step": step,
            "n_fnodes_total": n_total,
            "n_fnodes_secreted_cumulative": n_secreted,
            "new_fnodes_this_step": new_secreted_this_step,
            "mean_degradation": float(np.mean(deg)) if len(deg) > 0 else 0.0,
            "total_degradation": float(np.sum(deg)),
            "mean_reinforcement": float(np.mean(reinf)) if len(reinf) > 0 else 0.0,
            "total_reinforcement": float(np.sum(reinf)),
            "net_remodeling_total": float(np.sum(reinf) - np.sum(deg)),
            "mean_elastic_energy": float(np.mean(ee)) if len(ee) > 0 else 0.0,
            "total_elastic_energy": float(np.sum(ee)),
            # Positive-only means (averaged only over agents with value > 0)
            "n_degradation_positive": len(deg_pos),
            "mean_degradation_positive": float(np.mean(deg_pos)) if len(deg_pos) > 0 else 0.0,
            "n_reinforcement_positive": len(reinf_pos),
            "mean_reinforcement_positive": float(np.mean(reinf_pos)) if len(reinf_pos) > 0 else 0.0,
            "n_elastic_energy_positive": len(ee_pos),
            "mean_elastic_energy_positive": float(np.mean(ee_pos)) if len(ee_pos) > 0 else 0.0,
        }
        rows.append(row)
        prev_n_secreted = n_secreted

    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def save_summary(df: pd.DataFrame, outdir: Path, tag: str) -> None:
    if df.empty:
        return
    final = df.iloc[-1]
    summary = {
        "final_step": int(final["step"]),
        "final_n_fnodes_total": int(final["n_fnodes_total"]),
        "total_secreted_fnodes": int(final["n_fnodes_secreted_cumulative"]),
        "final_total_degradation": float(final["total_degradation"]),
        "final_total_reinforcement": float(final["total_reinforcement"]),
        "final_net_remodeling": float(final["net_remodeling_total"]),
        "final_total_elastic_energy": float(final["total_elastic_energy"]),
        "final_mean_degradation_positive": float(final["mean_degradation_positive"]),
        "final_mean_reinforcement_positive": float(final["mean_reinforcement_positive"]),
        "final_mean_elastic_energy_positive": float(final["mean_elastic_energy_positive"]),
    }
    pd.DataFrame([summary]).to_csv(outdir / f"matrix_remodeling_summary_{tag}.csv", index=False)


def make_plots(df: pd.DataFrame, outdir: Path, tag: str, show: bool) -> None:
    # --- Plot 1: New FNODEs per step and cumulative ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_step = "tab:blue"
    color_cum = "tab:orange"
    ax1.bar(df["step"], df["new_fnodes_this_step"], color=color_step, alpha=0.6, label="New FNODEs / step")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("New FNODEs per step", color=color_step)
    ax1.tick_params(axis="y", labelcolor=color_step)
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["n_fnodes_secreted_cumulative"], color=color_cum, linewidth=2, label="Cumulative secreted")
    ax2.set_ylabel("Cumulative secreted FNODEs", color=color_cum)
    ax2.tick_params(axis="y", labelcolor=color_cum)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / f"matrix_new_fnodes_{tag}.png", dpi=180)

    # --- Plot 2: Total degradation and reinforcement over time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["total_degradation"], label="Total degradation", linewidth=2, color="tab:red")
    ax.plot(df["step"], df["total_reinforcement"], label="Total reinforcement", linewidth=2, color="tab:green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Summed value across all FNODEs")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"matrix_degradation_reinforcement_{tag}.png", dpi=180)

    # --- Plot 3: Net remodeling and total elastic energy ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["step"], df["net_remodeling_total"], label="Net remodeling (reinf - deg)", linewidth=2, color="tab:purple")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Net remodeling", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["total_elastic_energy"], label="Total elastic energy", linewidth=2, color="tab:brown")
    ax2.set_ylabel("Total elastic energy", color="tab:brown")
    ax2.tick_params(axis="y", labelcolor="tab:brown")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"matrix_net_remodeling_energy_{tag}.png", dpi=180)

    # --- Plot 4: Total FNODE count over time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["n_fnodes_total"], label="Total FNODEs", linewidth=2, color="tab:cyan")
    ax.set_xlabel("Step")
    ax.set_ylabel("FNODE count")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"matrix_fnode_count_{tag}.png", dpi=180)

    # --- Plot 5: Positive-only mean degradation, reinforcement, elastic energy ---
    has_positive_data = (
        "mean_degradation_positive" in df.columns
        and not df["mean_degradation_positive"].isna().all()
    )
    if has_positive_data:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df["step"], df["mean_degradation_positive"], label="Mean degradation (>0 only)",
                 linewidth=2, color="tab:red")
        ax1.plot(df["step"], df["mean_reinforcement_positive"], label="Mean reinforcement (>0 only)",
                 linewidth=2, color="tab:green")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Active mean value")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(df["step"], df["mean_elastic_energy_positive"], label="Mean elastic energy (>0 only)",
                 linewidth=2, color="tab:brown", linestyle="--")
        ax2.set_ylabel("Mean elastic energy (>0 only)", color="tab:brown")
        ax2.tick_params(axis="y", labelcolor="tab:brown")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"matrix_positive_means_{tag}.png", dpi=180)

    if show:
        plt.show()
    else:
        plt.close("all")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.source == "vtk":
        indir = Path(args.indir)
        vtk_files = sorted(indir.glob("fibre_network_data_t*.vtk"))
        if not vtk_files:
            raise FileNotFoundError(f"No fibre_network_data_t*.vtk files found in {indir}")
        print(f"[VTK] Found {len(vtk_files)} fibre network VTK files in {indir}")
        df = build_timeseries_from_vtk(vtk_files)
    else:
        pickle_path = Path(args.pickle)
        print(f"[Pickle] Loading FNODE metrics from {pickle_path}")
        df = build_timeseries_from_pickle(pickle_path)

    csv_path = outdir / f"matrix_remodeling_timeseries_{args.tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Timeseries saved to {csv_path}")

    save_summary(df, outdir, args.tag)
    make_plots(df, outdir, args.tag, args.show)
    print(f"Plots saved to {outdir}")


if __name__ == "__main__":
    main()
