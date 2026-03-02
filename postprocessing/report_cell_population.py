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
    parser.add_argument("--pickle", default="../result_files/output_data_0.pickle",
                        help="Path to simulation pickle file (used when --source pickle)")
    parser.add_argument("--indir", default="../result_files", help="Directory containing cells_tXXXX.vtk (used when --source vtk)")
    parser.add_argument("--outdir", default="results", help="Directory for CSV/plots")
    parser.add_argument("--tag", default="latest", help="Tag suffix for output filenames")
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


def read_cells_vtk(path: Path) -> dict[int, tuple[int, int]]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    point_data_idx = next(i for i, l in enumerate(lines) if l.startswith("POINT_DATA"))
    n_points = int(lines[point_data_idx].split()[1])

    ids = _read_scalar(lines, "id", n_points)
    dead = _read_scalar(lines, "dead", n_points)
    dead_by = _read_scalar(lines, "dead_by", n_points)

    by_cell: dict[int, tuple[int, int]] = {}
    for cid, d, db in zip(ids, dead, dead_by):
        if cid not in by_cell:
            by_cell[cid] = (d, db)
    return by_cell


def build_timeseries_from_vtk(vtk_files: list[Path]) -> pd.DataFrame:
    rows = []
    prev_ids: set[int] | None = None
    prev_dead_map: dict[int, int] = {}

    for vtk in vtk_files:
        step_match = re.search(r"t(\d+)\.vtk$", vtk.name)
        step = int(step_match.group(1)) if step_match else -1

        cell_map = read_cells_vtk(vtk)
        ids = set(cell_map.keys())

        alive = sum(1 for d, _ in cell_map.values() if d == 0)
        dead = sum(1 for d, _ in cell_map.values() if d != 0)

        cause_counts = {k: 0 for k in (0, 1, 2, 3)}
        for d, db in cell_map.values():
            if d != 0 and db in cause_counts:
                cause_counts[db] += 1

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
            for cid, (d, db) in cell_map.items():
                prev_d = prev_dead_map.get(cid, 0)
                if d != 0 and prev_d == 0:
                    newly_dead_ids += 1
                    if db in newly_dead_by:
                        newly_dead_by[db] += 1

        rows.append(
            {
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
        )

        prev_ids = ids
        prev_dead_map = {cid: d for cid, (d, _) in cell_map.items()}

    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def build_timeseries_from_pickle(pickle_path: Path) -> pd.DataFrame:
    """Build timeseries from CELL_METRICS_OVER_TIME stored in pickle."""
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
    df.insert(0, "step", range(1, len(df) + 1))

    # Per-ID and per-cause columns are not available from pickle
    for col in ("new_cell_ids", "lost_cell_ids", "newly_dead_ids",
                "newly_dead_hypoxia", "newly_dead_starvation",
                "newly_dead_mechanical", "newly_dead_cumulative_damage",
                "dead_hypoxia_total", "dead_starvation_total",
                "dead_mechanical_total", "dead_cumulative_damage_total"):
        if col not in df.columns:
            df[col] = np.nan

    return df


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


def make_plots(df: pd.DataFrame, outdir: Path, tag: str, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["n_cells_total"], label="total", linewidth=2)
    ax.plot(df["step"], df["n_cells_alive"], label="alive", linewidth=2)
    ax.plot(df["step"], df["n_cells_dead"], label="dead", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("CELL count")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"cell_population_trends_{tag}.png", dpi=180)

    # Events plot only when per-ID data is available (VTK source)
    if not df["new_cell_ids"].isna().all():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["step"], df["new_cell_ids"], label="new_cell_ids", linewidth=2)
        ax.plot(df["step"], df["newly_dead_ids"], label="newly_dead_ids", linewidth=2)
        ax.plot(df["step"], df["lost_cell_ids"], label="lost_cell_ids", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Events per step")
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
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        df = build_timeseries_from_pickle(pickle_path)

    df.to_csv(outdir / f"cell_population_timeseries_{args.tag}.csv", index=False)
    save_summary(df, outdir, args.tag)
    make_plots(df, outdir, args.tag, args.show)
    print(f"Results saved to {outdir}")


if __name__ == "__main__":
    main()
