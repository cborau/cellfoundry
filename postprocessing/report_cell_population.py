"""
Build CELL population reports from pickle or VTK outputs.

Tracks per-step:
- total cells
- alive/dead cells (from dead flag)
- new cell ids (proliferation proxy, VTK only)
- lost cell ids (disappearance/death proxy, VTK only)
- newly dead by cause (from dead_by, VTK only)
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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "result_files"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


CAUSE_LABELS = {
    -1: "none",
    0: "hypoxia",
    1: "starvation",
    2: "mechanical",
    3: "cumulative_damage",
}


class _DummyModelParameterConfig:
    pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "helper_module" and name == "ModelParameterConfig":
            return _DummyModelParameterConfig
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CELL population/death reports from pickle or VTK files")
    parser.add_argument("--source", choices=["pickle", "vtk"], default="pickle",
                        help="Data source: 'pickle' (default) or 'vtk'")
    parser.add_argument("--pickle", default=str(DEFAULT_RESULTS_DIR / "output_data_0.pickle"),
                        help="Path to simulation pickle file (used when --source pickle)")
    parser.add_argument("--indir", default=str(DEFAULT_RESULTS_DIR), help="Directory containing cells_tXXXX.vtk (used when --source vtk)")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/plots")
    parser.add_argument("--tag", default="latest", help="Tag suffix for output filenames")
    parser.add_argument("--dt", type=float, default=None,
                        help="Simulation time step in seconds. When provided, x-axis is shown in time units instead of steps.")
    parser.add_argument("--time-unit", choices=["hours", "mins", "seconds"], default="hours",
                        help="Time unit for x-axis when --dt is given (default: hours).")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    return parser.parse_args()


def _read_scalar(lines: list[str], name: str, n: int) -> list[int]:
    header = f"SCALARS {name}"
    idx = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if idx is None:
        raise ValueError(f"Scalar '{name}' not found")
    start = idx + 2  # skip SCALARS + LOOKUP_TABLE
    out: list[int] = []
    for i in range(start, start + n):
        token = lines[i].strip().split()[0]
        out.append(int(float(token)))
    return out


def read_cells_vtk(path: Path) -> dict[int, tuple[int, int, int]]:
    """Return ``{cell_id: (dead, dead_by, cell_type)}`` from a VTK file."""
    lines = [line.strip() for line in path.read_text().splitlines()]
    point_data_idx = next(i for i, l in enumerate(lines) if l.startswith("POINT_DATA"))
    n_points = int(lines[point_data_idx].split()[1])

    ids = _read_scalar(lines, "id", n_points)
    dead = _read_scalar(lines, "dead", n_points)
    dead_by = _read_scalar(lines, "dead_by", n_points)

    # cell_type may not be present in older VTK outputs — default to 0
    try:
        cell_types = _read_scalar(lines, "cell_type", n_points)
    except ValueError:
        cell_types = [0] * n_points

    by_cell: dict[int, tuple[int, int, int]] = {}
    for cid, d, db, ct in zip(ids, dead, dead_by, cell_types):
        if cid not in by_cell:
            by_cell[cid] = (d, db, ct)
    return by_cell


def build_timeseries_from_vtk(vtk_files: list[Path]) -> pd.DataFrame:
    rows = []
    prev_ids: set[int] | None = None
    prev_dead_map: dict[int, int] = {}
    all_cell_types: set[int] = set()  # collect every cell_type seen

    for vtk in vtk_files:
        step_match = re.search(r"t(\d+)\.vtk$", vtk.name)
        step = int(step_match.group(1)) if step_match else -1

        cell_map = read_cells_vtk(vtk)
        ids = set(cell_map.keys())

        alive = sum(1 for d, _, _ in cell_map.values() if d == 0)
        dead = sum(1 for d, _, _ in cell_map.values() if d != 0)

        cause_counts = {k: 0 for k in (0, 1, 2, 3)}
        for d, db, _ in cell_map.values():
            if d != 0 and db in cause_counts:
                cause_counts[db] += 1

        # --- Per-cell-type alive/total counts ---
        type_alive: dict[int, int] = {}
        type_total: dict[int, int] = {}
        for d, _, ct in cell_map.values():
            all_cell_types.add(ct)
            type_total[ct] = type_total.get(ct, 0) + 1
            if d == 0:
                type_alive[ct] = type_alive.get(ct, 0) + 1

        if prev_ids is None:
            new_ids = len(ids)
            lost_ids = 0
            newly_dead_ids = 0
            newly_dead_by = {k: 0 for k in (0, 1, 2, 3)}
        else:
            new_set = ids - prev_ids
            lost_set = prev_ids - ids
            new_ids = len(new_set)
            lost_ids = len(lost_set)

            newly_dead_by = {k: 0 for k in (0, 1, 2, 3)}
            newly_dead_ids = 0
            for cid, (d, db, _) in cell_map.items():
                prev_d = prev_dead_map.get(cid, 0)
                if d != 0 and prev_d == 0:
                    newly_dead_ids += 1
                    if db in newly_dead_by:
                        newly_dead_by[db] += 1

        row = {
            "step": step,
            "n_cells_total": len(ids),
            "n_cells_alive": alive,
            "n_cells_dead": dead,
            "new_cell_ids": new_ids,
            "lost_cell_ids": lost_ids,
            "newly_dead_ids": newly_dead_ids,
            "newly_dead_hypoxia": newly_dead_by[0],
            "newly_dead_starvation": newly_dead_by[1],
            "newly_dead_mechanical": newly_dead_by[2],
            "newly_dead_cumulative_damage": newly_dead_by[3],
            "dead_hypoxia_total": cause_counts[0],
            "dead_starvation_total": cause_counts[1],
            "dead_mechanical_total": cause_counts[2],
            "dead_cumulative_damage_total": cause_counts[3],
        }
        # Per-type columns
        for ct in sorted(all_cell_types):
            row[f"n_alive_type_{ct}"] = type_alive.get(ct, 0)
            row[f"n_total_type_{ct}"] = type_total.get(ct, 0)

        rows.append(row)

        prev_ids = ids
        prev_dead_map = {cid: d for cid, (d, _, _) in cell_map.items()}

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    # Back-fill per-type columns for early rows where a type wasn't yet seen
    type_cols = [c for c in df.columns if c.startswith("n_alive_type_") or c.startswith("n_total_type_")]
    df[type_cols] = df[type_cols].fillna(0).astype(int)

    return df


def build_timeseries_from_pickle(pickle_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Build timeseries from CELL_METRICS_OVER_TIME stored in pickle.

    Returns ``(cell_df, organoid_df)`` where *organoid_df* is the
    ``ORGANOID_METRICS_OVER_TIME`` frame (may be ``None`` if absent).
    """
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    with pickle_path.open("rb") as f:
        data = _SafeUnpickler(f).load()

    cell_met = data.get("CELL_METRICS_OVER_TIME")
    if cell_met is None or (isinstance(cell_met, list) and len(cell_met) == 0):
        raise ValueError("CELL_METRICS_OVER_TIME not found or empty in pickle. "
                         "Re-run the simulation to generate CELL metrics, or use --source vtk.")
    if isinstance(cell_met, list):
        cell_met = pd.DataFrame(cell_met)

    df = cell_met.copy().reset_index(drop=True)
    if "step" not in df.columns:
        df.insert(0, "step", range(1, len(df) + 1))

    # Per-ID and per-cause columns are not available from pickle
    for col in ("new_cell_ids", "lost_cell_ids", "newly_dead_ids",
                "newly_dead_hypoxia", "newly_dead_starvation",
                "newly_dead_mechanical", "newly_dead_cumulative_damage",
                "dead_hypoxia_total", "dead_starvation_total",
                "dead_mechanical_total", "dead_cumulative_damage_total"):
        if col not in df.columns:
            df[col] = np.nan

    # Extract ORGANOID_METRICS_OVER_TIME (for per-type curves)
    org_raw = data.get("ORGANOID_METRICS_OVER_TIME")
    organoid_df = None
    if org_raw is not None and (not hasattr(org_raw, "__len__") or len(org_raw) > 0):
        organoid_df = pd.DataFrame(org_raw) if not isinstance(org_raw, pd.DataFrame) else org_raw.copy()

    return df, organoid_df


def save_summary(df: pd.DataFrame, outdir: Path, tag: str) -> None:
    final = df.iloc[-1]
    summary = {
        "final_step": int(final["step"]),
        "final_n_cells_total": int(final["n_cells_total"]),
        "final_n_cells_alive": int(final["n_cells_alive"]),
        "final_n_cells_dead": int(final["n_cells_dead"]),
        "total_new_cell_ids": int(df["new_cell_ids"].sum()),
        "total_lost_cell_ids": int(df["lost_cell_ids"].sum()),
        "total_newly_dead_ids": int(df["newly_dead_ids"].sum()),
        "total_newly_dead_hypoxia": int(df["newly_dead_hypoxia"].sum()),
        "total_newly_dead_starvation": int(df["newly_dead_starvation"].sum()),
        "total_newly_dead_mechanical": int(df["newly_dead_mechanical"].sum()),
        "total_newly_dead_cumulative_damage": int(df["newly_dead_cumulative_damage"].sum()),
    }
    pd.DataFrame([summary]).to_csv(outdir / f"cell_population_summary_{tag}.csv", index=False)


# Colour palette for per-type curves
_TYPE_COLORS = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#8c564b"]


def _parse_n_alive_type(series: pd.Series) -> np.ndarray:
    """Parse comma-separated n_alive_type column into a 2-D array (rows × types)."""
    rows = []
    for val in series:
        if isinstance(val, str) and val:
            rows.append([int(x) for x in val.split(",")])
        else:
            rows.append([])
    if not rows or len(rows[0]) == 0:
        return np.empty((len(series), 0), dtype=int)
    return np.array(rows, dtype=int)


def _steps_to_time(steps: np.ndarray, dt: float | None, unit: str) -> tuple[np.ndarray, str]:
    """Convert step numbers to time values and return (values, xlabel)."""
    if dt is None:
        return steps.astype(float), "Step"
    t_sec = steps * dt
    if unit == "seconds":
        return t_sec, "Time (s)"
    elif unit == "mins":
        return t_sec / 60.0, "Time (min)"
    else:  # hours
        return t_sec / 3600.0, "Time (h)"


def make_plots(df: pd.DataFrame, outdir: Path, tag: str, show: bool,
               organoid_df: pd.DataFrame | None = None,
               dt: float | None = None, time_unit: str = "hours") -> None:
    # Compute x values for the cell_metrics dataframe
    steps_df = df["step"].values
    x_df, xlabel = _steps_to_time(steps_df, dt, time_unit)

    # Compute x values for the organoid dataframe (per-type alive curves)
    x_org: np.ndarray | None = None
    if organoid_df is not None and "n_alive_type" in organoid_df.columns:
        if "step" in organoid_df.columns:
            org_steps = organoid_df["step"].values
        else:
            org_steps = np.arange(len(organoid_df))
        x_org, _ = _steps_to_time(org_steps, dt, time_unit)

    # ---- 1. Overall population trends + per-type curves -----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_df, df["n_cells_total"], label="total", linewidth=2)
    ax.plot(x_df, df["n_cells_alive"], label="alive", linewidth=2)
    ax.plot(x_df, df["n_cells_dead"], label="dead", linewidth=2)

    # Per-type alive from VTK columns
    alive_type_cols = sorted([c for c in df.columns if c.startswith("n_alive_type_")])
    if alive_type_cols:
        for col in alive_type_cols:
            ct_label = col.replace("n_alive_type_", "alive type ")
            ci = int(col.rsplit("_", 1)[-1])
            ax.plot(x_df, df[col], color=_TYPE_COLORS[ci % len(_TYPE_COLORS)],
                    linewidth=1.5, linestyle="--", label=ct_label)
    # Per-type alive from organoid_df (pickle path)
    elif x_org is not None and organoid_df is not None and "n_alive_type" in organoid_df.columns:
        type_arr = _parse_n_alive_type(organoid_df["n_alive_type"])
        for ti in range(type_arr.shape[1]):
            ax.plot(x_org, type_arr[:, ti], color=_TYPE_COLORS[ti % len(_TYPE_COLORS)],
                    linewidth=1.5, linestyle="--", label=f"alive type {ti}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CELL count")
    ax.set_title("Cell Population Over Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"cell_population_trends_{tag}.png", dpi=180)

    # ---- 2. Per-cell-type alive population (separate plot) --------------
    has_type_data = bool(alive_type_cols) or (
        organoid_df is not None and "n_alive_type" in organoid_df.columns
    )
    if has_type_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        if alive_type_cols:
            for col in alive_type_cols:
                ct_label = col.replace("n_alive_type_", "type ")
                ci = int(col.rsplit("_", 1)[-1])
                ax.plot(x_df, df[col], color=_TYPE_COLORS[ci % len(_TYPE_COLORS)],
                        linewidth=2, label=ct_label)
        elif x_org is not None and organoid_df is not None and "n_alive_type" in organoid_df.columns:
            type_arr = _parse_n_alive_type(organoid_df["n_alive_type"])
            for ti in range(type_arr.shape[1]):
                ax.plot(x_org, type_arr[:, ti], color=_TYPE_COLORS[ti % len(_TYPE_COLORS)],
                        linewidth=2, label=f"type {ti}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Alive CELL count")
        ax.set_title("Cell Population Over Time — Per Cell Type")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"cell_population_per_type_{tag}.png", dpi=180)

    # ---- 3. Per-cell-type stacked area plot -----------------------------
    if alive_type_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [col.replace("n_alive_type_", "type ") for col in alive_type_cols]
        ax.stackplot(x_df, *[df[col] for col in alive_type_cols], labels=labels, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Alive CELL count")
        ax.set_title("Cell Population (Stacked) — Per Cell Type")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"cell_population_stacked_{tag}.png", dpi=180)
    elif organoid_df is not None and "n_alive_type" in organoid_df.columns:
        type_arr = _parse_n_alive_type(organoid_df["n_alive_type"])
        if type_arr.shape[1] > 0:
            if x_org is None:
                org_steps = organoid_df["step"].values if "step" in organoid_df.columns else np.arange(len(organoid_df))
                x_org, _ = _steps_to_time(org_steps, dt, time_unit)
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f"type {ti}" for ti in range(type_arr.shape[1])]
            ax.stackplot(x_org, *[type_arr[:, ti] for ti in range(type_arr.shape[1])],
                         labels=labels, alpha=0.7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Alive CELL count")
            ax.set_title("Cell Population (Stacked) — Per Cell Type")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(outdir / f"cell_population_stacked_{tag}.png", dpi=180)

    # ---- 4. Events plot (VTK only) --------------------------------------
    if not df["new_cell_ids"].isna().all():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_df, df["new_cell_ids"], label="new_cell_ids", linewidth=2)
        ax.plot(x_df, df["newly_dead_ids"], label="newly_dead_ids", linewidth=2)
        ax.plot(x_df, df["lost_cell_ids"], label="lost_cell_ids", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Events per step")
        ax.set_title("Cell Population Events Per Step")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"cell_population_events_{tag}.png", dpi=180)

    if show:
        plt.show()
    else:
        plt.close("all")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    organoid_df = None
    if args.source == "vtk":
        indir = Path(args.indir)
        vtk_files = sorted(indir.glob("cells_t*.vtk"))
        if not vtk_files:
            raise FileNotFoundError(f"No cells_t*.vtk files found in {indir}")
        print(f"[VTK] Found {len(vtk_files)} cell VTK files in {indir}")
        df = build_timeseries_from_vtk(vtk_files)
    else:
        pickle_path = Path(args.pickle)
        print(f"[Pickle] Loading CELL metrics from {pickle_path}")
        df, organoid_df = build_timeseries_from_pickle(pickle_path)

    df.to_csv(outdir / f"cell_population_timeseries_{args.tag}.csv", index=False)
    save_summary(df, outdir, args.tag)
    make_plots(df, outdir, args.tag, args.show, organoid_df=organoid_df,
               dt=args.dt, time_unit=args.time_unit)
    print(f"Results saved to {outdir}")


if __name__ == "__main__":
    main()
