"""
Radial glia differentiation reporter.

Reads ``cells_tXXXX.vtk`` output files from a ``radial_glia`` simulation run,
extracts per-cell RG-specific scalars and vectors, and produces:

  - A ``rg_differentiation_<tag>.csv`` summary table (one row per snapshot)
  - A ``rg_differentiation_<tag>.png`` 4-panel figure:

      Panel 1 — Cell type composition (iPSC / NEP / RG counts)
      Panel 2 — Commitment: mean ``rg_commit_level`` per type + committed fraction
      Panel 3 — Epithelialization: mean ``epithelialization_level`` + ``rosette_maturity``
      Panel 4 — Apical polarity: mean |apz| for RG cells + mean ``morphogen_local``

  If a results pickle (``output_data_0.pickle`` or provided via ``--pickle``)
  is found, an additional ``rg_rosette_metrics_<tag>.png`` figure is produced
  with 4 panels covering RG population dynamics, cluster structure evolution,
  cluster size metrics, and rosette assembly quality.

Requires output from the ``radial_glia`` variant (VTK files must contain the
``rg_commit_level`` scalar; the script exits with a clear message otherwise).

Cell type index mapping
-----------------------
  0 → iPSC  (initial pluripotent stem cell)
  1 → NEP   (neuroepithelial progenitor)
  2 → RG    (radial glia)

Usage examples
--------------
# Minimal – reads result_files/, writes postprocessing/results/
python postprocessing/report_rg_differentiation.py

# Custom run directory with 60 s timestep:
python postprocessing/report_rg_differentiation.py \\
    --indir result_files/radial_glia --dt 60 --show

# Include rosette cluster metrics figure:
python postprocessing/report_rg_differentiation.py \\
    --indir result_files/radial_glia \\
    --pickle result_files/output_data_0.pickle --show

# Two-run comparison is not built in; run the script twice with different
# --tag values and compare the CSVs / figures manually.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import pickle as _pickle

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "result_files"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

# Cell type index → readable label / colour
_CELL_TYPE_LABELS = {0: "iPSC", 1: "NEP", 2: "RG"}
_CELL_TYPE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}

# VTK scalar names expected from the radial_glia variant
_RG_FLOAT_SCALARS = [
    "rg_commit_level",
    "epithelialization_level",
    "rosette_maturity",
    "rg_neighbour_density",
    "morphogen_local",
]
_RG_INT_SCALARS = ["rg_committed"]


# ---------------------------------------------------------------------------
# VTK reading helpers
# ---------------------------------------------------------------------------

def _point_data_n(lines: list[str]) -> int:
    """Return N from the ``POINT_DATA N`` header line."""
    for line in lines:
        if line.startswith("POINT_DATA"):
            return int(line.split()[1])
    raise ValueError("POINT_DATA header not found in VTK file")


def _read_scalar(lines: list[str], name: str, n: int, *, dtype: type = float) -> np.ndarray:
    """Read a SCALARS block by *name*; return ndarray of length *n*."""
    header = f"SCALARS {name}"
    idx = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if idx is None:
        raise KeyError(f"Scalar '{name}' not found")
    # Skip the LOOKUP_TABLE line
    start = idx + 2
    tokens: list[str] = []
    i = start
    while len(tokens) < n and i < len(lines):
        stripped = lines[i].strip()
        if stripped:
            tokens.extend(stripped.split())
        i += 1
    if len(tokens) < n:
        raise ValueError(f"Scalar '{name}': expected {n} values, got {len(tokens)}")
    return np.array(tokens[:n], dtype=dtype)


def _read_vector3(lines: list[str], name: str, n: int) -> np.ndarray:
    """Read a VECTORS block by *name*; return ndarray of shape *(n, 3)*."""
    header = f"VECTORS {name}"
    idx = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if idx is None:
        raise KeyError(f"Vector '{name}' not found")
    result = np.zeros((n, 3), dtype=float)
    count = 0
    i = idx + 1
    while count < n and i < len(lines):
        stripped = lines[i].strip()
        if stripped:
            parts = stripped.split()
            if len(parts) >= 3:
                result[count] = [float(parts[0]), float(parts[1]), float(parts[2])]
                count += 1
        i += 1
    return result


def _read_vtk_snapshot(path: Path) -> dict:
    """
    Parse a ``cells_tXXXX.vtk`` file.

    Returns a dict of per-cell numpy arrays keyed by field name.  Only *alive*
    cells are included (``dead == 0``).  Anchor-point duplicates are removed by
    keeping the first occurrence of each cell ``id``.
    """
    lines = [line.rstrip() for line in path.read_text().splitlines()]
    n_pts = _point_data_n(lines)

    ids  = _read_scalar(lines, "id",        n_pts, dtype=int)
    dead = _read_scalar(lines, "dead",      n_pts, dtype=int)
    ct   = _read_scalar(lines, "cell_type", n_pts, dtype=int)

    # RG float scalars (may be absent in non-radial_glia VTKs)
    rg_data: dict[str, np.ndarray | None] = {}
    for name in _RG_FLOAT_SCALARS:
        try:
            rg_data[name] = _read_scalar(lines, name, n_pts, dtype=float)
        except KeyError:
            rg_data[name] = None
    for name in _RG_INT_SCALARS:
        try:
            rg_data[name] = _read_scalar(lines, name, n_pts, dtype=int)
        except KeyError:
            rg_data[name] = None

    # Apical vector (VECTORS block)
    try:
        apical = _read_vector3(lines, "apical_vector", n_pts)
    except KeyError:
        apical = None

    # Keep first occurrence of each id → real cells (anchor duplicates share id)
    _, first_idx = np.unique(ids, return_index=True)
    first_idx = np.sort(first_idx)
    alive_mask = dead[first_idx] == 0

    out: dict = {
        "id":        ids[first_idx][alive_mask],
        "cell_type": ct[first_idx][alive_mask],
    }
    for name, arr in rg_data.items():
        out[name] = arr[first_idx][alive_mask] if arr is not None else None
    out["apical_vector"] = apical[first_idx][alive_mask] if apical is not None else None

    return out


# ---------------------------------------------------------------------------
# Per-step aggregation
# ---------------------------------------------------------------------------

def _safe_mean(arr: np.ndarray | None, mask: np.ndarray | None = None) -> float:
    if arr is None:
        return float("nan")
    data = arr[mask] if mask is not None else arr
    return float(data.mean()) if len(data) > 0 else 0.0


def _aggregate(snap: dict, step: int, dt_s: float) -> dict:
    ct = snap["cell_type"]
    n  = len(ct)

    row: dict = {
        "step":   step,
        "time_h": step * dt_s / 3600.0,
        "n_total": n,
    }

    # --- Per-type cell counts ---
    for t, label in _CELL_TYPE_LABELS.items():
        row[f"n_{label.lower()}"] = int((ct == t).sum())

    row["f_nep"] = row["n_nep"] / n if n > 0 else 0.0
    row["f_rg"]  = row["n_rg"]  / n if n > 0 else 0.0

    rg_mask  = (ct == 2)
    nep_mask = (ct == 1)

    # --- Commitment (rg_commit_level) ---
    cl = snap.get("rg_commit_level")
    row["commit_mean"]      = _safe_mean(cl)
    row["commit_ipsc_mean"] = _safe_mean(cl, ct == 0)
    row["commit_nep_mean"]  = _safe_mean(cl, nep_mask)
    row["commit_rg_mean"]   = _safe_mean(cl, rg_mask)

    # --- Committed fraction ---
    cc = snap.get("rg_committed")
    row["committed_frac"] = float(cc.mean()) if (cc is not None and n > 0) else 0.0

    # --- Epithelialization ---
    el = snap.get("epithelialization_level")
    row["epithelial_mean"]     = _safe_mean(el)
    row["epithelial_nep_mean"] = _safe_mean(el, nep_mask)
    row["epithelial_rg_mean"]  = _safe_mean(el, rg_mask)

    # --- Rosette maturity ---
    rm = snap.get("rosette_maturity")
    row["rosette_maturity_mean"]     = _safe_mean(rm)
    row["rosette_maturity_nep_mean"] = _safe_mean(rm, nep_mask)
    row["rosette_maturity_rg_mean"]  = _safe_mean(rm, rg_mask)

    # --- RG neighbour density (meaningful only for RG cells) ---
    nd = snap.get("rg_neighbour_density")
    row["rg_neighbour_density_mean"] = _safe_mean(nd, rg_mask)

    # --- Morphogen (local sp2 concentration at cell location) ---
    ml = snap.get("morphogen_local")
    row["morphogen_mean"]    = _safe_mean(ml)
    row["morphogen_rg_mean"] = _safe_mean(ml, rg_mask)

    # --- Apical z-alignment: mean |apz| for NEP and RG cells ---
    # |apz| = 1 → apical vector points along z (perfect epithelial polarity)
    # |apz| = 0 → apical vector lies in the xy-plane (no z-polarity)
    av = snap.get("apical_vector")
    if av is not None:
        if rg_mask.any():
            row["apical_z_abs_rg_mean"] = float(np.abs(av[rg_mask, 2]).mean())
        else:
            row["apical_z_abs_rg_mean"] = float("nan")
        if nep_mask.any():
            row["apical_z_abs_nep_mean"] = float(np.abs(av[nep_mask, 2]).mean())
        else:
            row["apical_z_abs_nep_mean"] = float("nan")
    else:
        row["apical_z_abs_rg_mean"]  = float("nan")
        row["apical_z_abs_nep_mean"] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(df: pd.DataFrame, out_path: Path, show: bool) -> None:
    x      = df["time_h"]
    xlabel = "Time (h)"

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("Radial glia differentiation — simulation summary",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Cell type composition ───────────────────────────────────────
    ax = axes[0]
    for t, label in _CELL_TYPE_LABELS.items():
        col = f"n_{label.lower()}"
        if col in df.columns:
            ax.plot(x, df[col], label=label, color=_CELL_TYPE_COLORS[t], linewidth=1.8)
    ax.set_ylabel("Cell count")
    ax.set_title("Cell type composition")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 2: Commitment ───────────────────────────────────────────────────
    ax  = axes[1]
    ax2 = ax.twinx()
    for t, label in _CELL_TYPE_LABELS.items():
        col = f"commit_{label.lower()}_mean"
        if col in df.columns and not df[col].isna().all():
            ax.plot(x, df[col], label=f"{label}",
                    color=_CELL_TYPE_COLORS[t], linewidth=1.5)
    if "committed_frac" in df.columns:
        ax2.plot(x, df["committed_frac"] * 100.0,
                 color="dimgray", linestyle="--", linewidth=1.2, label="Committed (%)")
        ax2.set_ylabel("Committed cells (%)", color="dimgray", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="dimgray")
        ax2.set_ylim(0, 105)
    ax.set_ylabel("Mean rg_commit_level")
    ax.set_ylim(0, 1.05)
    ax.set_title("RG commitment (rg_commit_level per cell type)")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 3: Epithelialization & rosette maturity ─────────────────────────
    ax = axes[2]
    _ep_nep = df.get("epithelial_nep_mean")
    _ep_rg  = df.get("epithelial_rg_mean")
    _rm_nep = df.get("rosette_maturity_nep_mean")
    _rm_rg  = df.get("rosette_maturity_rg_mean")
    if _ep_nep is not None and not _ep_nep.isna().all():
        ax.plot(x, _ep_nep, label="Epithelialization (NEP)",
                color=_CELL_TYPE_COLORS[1], linestyle="-", linewidth=1.5)
    if _ep_rg is not None and not _ep_rg.isna().all():
        ax.plot(x, _ep_rg, label="Epithelialization (RG)", color="#C44E52", linewidth=1.8)
    if _rm_nep is not None and not _rm_nep.isna().all():
        ax.plot(x, _rm_nep, label="Rosette maturity (NEP)",
                color=_CELL_TYPE_COLORS[1], linestyle=":", linewidth=1.2)
    if _rm_rg is not None and not _rm_rg.isna().all():
        ax.plot(x, _rm_rg, label="Rosette maturity (RG)", color="#8172B2",
                linestyle="--", linewidth=1.5)
    ax.set_ylabel("Level [0–1]")
    ax.set_ylim(0, 1.05)
    ax.set_title("Epithelialization & rosette maturity (NEP and RG)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 4: Apical polarity & morphogen ──────────────────────────────────
    ax  = axes[3]
    ax2 = ax.twinx()
    _ap_nep = df.get("apical_z_abs_nep_mean")
    _ap_rg  = df.get("apical_z_abs_rg_mean")
    if _ap_nep is not None and not _ap_nep.isna().all():
        ax.plot(x, _ap_nep, label="|apz| NEP",
                color=_CELL_TYPE_COLORS[1], linestyle="-", linewidth=1.5)
    if _ap_rg is not None and not _ap_rg.isna().all():
        ax.plot(x, _ap_rg, label="|apz| RG", color=_CELL_TYPE_COLORS[2], linewidth=1.8)
    ax.set_ylabel("Mean |apical z|")
    ax.set_ylim(0, 1.05)
    ax.set_title("Apical z-polarity & morphogen_local (NEP and RG)")
    _mg = df.get("morphogen_rg_mean")
    if _mg is not None and not _mg.isna().all():
        ax2.plot(x, _mg, color="#CCB974", linestyle=":", linewidth=1.5,
                 label="Morphogen (RG)")
        ax2.set_ylabel("Mean morphogen_local (RG)", color="#9C8040", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#9C8040")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, frameon=False, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  Figure   → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rosette metrics plotting (RG_ROSETTE_METRICS_OVER_TIME)
# ---------------------------------------------------------------------------

def _plot_rosette_metrics(rm: pd.DataFrame, out_path: Path, show: bool) -> None:
    """4-panel figure from the ``RG_ROSETTE_METRICS_OVER_TIME`` DataFrame.

    Loaded from the simulation results pickle (key ``RG_ROSETTE_METRICS_OVER_TIME``).

    Panels
    ------
    1 — RG population dynamics (n_alive_total, n_alive_rg, rg_fraction %)
    2 — Cluster structure over time (n_rg_clusters, n_large_rg_clusters,
        large_cluster_fraction %)
    3 — Cluster size evolution (mean_cluster_size, large_cluster_mean_size,
        largest_cluster_size)
    4 — Rosette assembly quality (mean_rosette_maturity, mean_apz,
        rg_assembly_compactness, mean_cluster_compactness)

    The ``time`` column is expected in seconds and is converted to hours for
    display.
    """
    x = rm["time"] / 3600.0 if "time" in rm.columns else rm.index.astype(float)
    xlabel = "Time (h)"

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("RG rosette cluster metrics — simulation summary",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: RG population dynamics ──────────────────────────────────────
    ax = axes[0]
    ax2 = ax.twinx()
    if "n_alive_total" in rm.columns:
        ax.plot(x, rm["n_alive_total"], color="#AAAAAA", linewidth=1.5,
                linestyle="--", label="Total alive")
    if "n_alive_rg" in rm.columns:
        ax.plot(x, rm["n_alive_rg"], color=_CELL_TYPE_COLORS[2], linewidth=1.8,
                label="RG alive")
    if "rg_fraction" in rm.columns:
        ax2.plot(x, rm["rg_fraction"] * 100.0, color="#C44E52", linestyle=":",
                 linewidth=1.4, label="RG fraction (%)")
        ax2.set_ylabel("RG fraction (%)", color="#C44E52", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        ax2.set_ylim(0, 105)
    ax.set_ylabel("Cell count")
    ax.set_title("RG population dynamics")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 2: Cluster structure over time ──────────────────────────────────
    ax = axes[1]
    ax2 = ax.twinx()
    if "n_rg_clusters" in rm.columns:
        ax.plot(x, rm["n_rg_clusters"], color="#4C72B0", linewidth=1.5,
                label="All clusters (DBSCAN)")
    if "n_large_rg_clusters" in rm.columns:
        ax.plot(x, rm["n_large_rg_clusters"], color="#8172B2", linewidth=1.8,
                label="Large clusters (\u2265 MIN_ROSETTE_SIZE)")
    if "large_cluster_fraction" in rm.columns:
        ax2.plot(x, rm["large_cluster_fraction"] * 100.0, color="#CCB974",
                 linestyle=":", linewidth=1.4, label="Fraction in large clusters (%)")
        ax2.set_ylabel("Fraction in large clusters (%)", color="#9C8040", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#9C8040")
        ax2.set_ylim(0, 105)
    ax.set_ylabel("Cluster count")
    ax.set_title("Cluster structure over time")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 3: Cluster size evolution ──────────────────────────────────────
    ax = axes[2]
    if "mean_cluster_size" in rm.columns:
        ax.plot(x, rm["mean_cluster_size"], color="#4C72B0", linewidth=1.5,
                linestyle="-", label="Mean cluster size (all)")
    if "large_cluster_mean_size" in rm.columns:
        ax.plot(x, rm["large_cluster_mean_size"], color="#8172B2", linewidth=1.8,
                linestyle="-", label="Mean size (large clusters)")
    if "largest_cluster_size" in rm.columns:
        ax.plot(x, rm["largest_cluster_size"], color="#C44E52", linewidth=1.5,
                linestyle="--", label="Largest cluster size")
    ax.set_ylabel("Cells per cluster")
    ax.set_title("Cluster size evolution")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 4: Assembly quality ─────────────────────────────────────────────
    ax = axes[3]
    ax2 = ax.twinx()
    if "mean_rosette_maturity" in rm.columns:
        ax.plot(x, rm["mean_rosette_maturity"], color=_CELL_TYPE_COLORS[2],
                linewidth=1.8, label="Rosette maturity")
    if "mean_apz" in rm.columns:
        ax.plot(x, rm["mean_apz"], color="#DD8452", linewidth=1.5, linestyle="--",
                label="Mean |apz|")
    ax.set_ylabel("Maturity / |apz| [0\u20131]")
    ax.set_ylim(0, 1.05)
    if "rg_assembly_compactness" in rm.columns:
        ax2.plot(x, rm["rg_assembly_compactness"], color="#4C72B0", linewidth=1.5,
                 linestyle=":", label="Assembly compactness")
    if "mean_cluster_compactness" in rm.columns:
        ax2.plot(x, rm["mean_cluster_compactness"], color="#9C8040", linewidth=1.5,
                 linestyle="-.", label="Mean cluster compactness")
    ax2.set_ylabel("Compactness [0\u20131]", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax.set_title("Rosette assembly quality")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, frameon=False, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  Figure   → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Radial glia differentiation report from cells_tXXXX.vtk files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--indir", default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing cells_tXXXX.vtk output files",
    )
    parser.add_argument(
        "--outdir", default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for CSV and figure",
    )
    parser.add_argument(
        "--tag", default="rg",
        help="Tag suffix appended to output file names",
    )
    parser.add_argument(
        "--dt", type=float, default=60.0,
        help="Simulation time step in seconds (used to convert steps → hours)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the figure interactively after saving",
    )
    parser.add_argument(
        "--pickle", default=None,
        help=(
            "Path to the simulation results pickle (output_data_0.pickle). "
            "If not provided, the script looks for 'output_data_0.pickle' inside "
            "--indir automatically. Required for the rosette cluster metrics figure."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    indir  = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted(
        indir.glob("cells_t*.vtk"),
        key=lambda p: int(re.search(r"t(\d+)\.vtk$", p.name).group(1)),
    )
    if not vtk_files:
        sys.exit(f"ERROR: No cells_t*.vtk files found in '{indir}'")

    print(f"Found {len(vtk_files)} VTK snapshot(s) in '{indir}'")

    rows: list[dict] = []
    for vtk in vtk_files:
        step = int(re.search(r"t(\d+)\.vtk$", vtk.name).group(1))
        snap = _read_vtk_snapshot(vtk)

        # Validate that this is a radial_glia run
        if snap.get("rg_commit_level") is None:
            sys.exit(
                f"ERROR: '{vtk.name}' does not contain 'rg_commit_level'.\n"
                "Make sure --indir points to output from the 'radial_glia' variant."
            )

        row = _aggregate(snap, step, args.dt)
        rows.append(row)

        print(
            f"  {vtk.name}  step={step:6d}  t={row['time_h']:5.1f}h"
            f"  total={row['n_total']:4d}"
            f"  iPSC={row['n_ipsc']:3d}  NEP={row['n_nep']:3d}  RG={row['n_rg']:3d}"
            f"  commit={row['commit_mean']:.3f}"
            f"  f_rg={row['f_rg']:.2f}"
            f"  |apz|={row['apical_z_abs_rg_mean']:.3f}"
            if not np.isnan(row["apical_z_abs_rg_mean"])
            else
            f"  {vtk.name}  step={step:6d}  t={row['time_h']:5.1f}h"
            f"  total={row['n_total']:4d}"
            f"  iPSC={row['n_ipsc']:3d}  NEP={row['n_nep']:3d}  RG={row['n_rg']:3d}"
            f"  commit={row['commit_mean']:.3f}"
            f"  f_rg={row['f_rg']:.2f}"
        )

    df = pd.DataFrame(rows)

    csv_path = outdir / f"rg_differentiation_{args.tag}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  CSV      → {csv_path}")

    fig_path = outdir / f"rg_differentiation_{args.tag}.png"
    _plot(df, fig_path, args.show)

    # -- Rosette metrics figure (from pickle) ---------------------------------
    pickle_path: Path | None = None
    if args.pickle:
        pickle_path = Path(args.pickle)
    else:
        candidate = indir / "output_data_0.pickle"
        if candidate.exists():
            pickle_path = candidate

    if pickle_path is not None:
        print(f"\nLoading rosette metrics from pickle '{pickle_path}'")
        with open(str(pickle_path), "rb") as _f:
            _results = _pickle.load(_f)
        rm_df = _results.get("RG_ROSETTE_METRICS_OVER_TIME")
        if rm_df is None or (hasattr(rm_df, "__len__") and len(rm_df) == 0):
            print(
                "  Note: pickle does not contain RG_ROSETTE_METRICS_OVER_TIME "
                "(not a radial_glia run or SAVE_PICKLE was False)."
            )
        else:
            if not isinstance(rm_df, pd.DataFrame):
                rm_df = pd.DataFrame(rm_df)
            rm_fig_path = outdir / f"rg_rosette_metrics_{args.tag}.png"
            _plot_rosette_metrics(rm_df, rm_fig_path, args.show)
    else:
        print(
            "\nNote: no results pickle found. "
            "Use --pickle to add the rosette cluster metrics figure."
        )

    print("Done.")


if __name__ == "__main__":
    main()
