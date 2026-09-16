"""Inspect diffusion in CellFoundry legacy ASCII VTK outputs.

Examples (run from the repository root)::

    python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --show
    python postprocessing/plot_diffusion_results.py --species 0 2 --probe 0 0 0 --probe 20 0 0
    python postprocessing/plot_diffusion_results.py --datasets cells --agent-id 1332
    python postprocessing/plot_diffusion_results.py --plots traces summary profiles --time-step 2
    python postprocessing/plot_diffusion_results.py --plots maps --planes z=0 --steps 1 10 50 --smooth

By default, discover all available ECM/cell/vascular datasets and species, sample
nearest to the origin at each saved step, and save traces and min/mean/max plots
plus CSVs in <results-dir>/diffusion_plots. Use --help for all options.

Files are processed one snapshot at a time. ECM corner markers and cell anchors
are excluded. Physical time is the filename step times TIME_STEP, with no extra
save-interval multiplier. Without a time step, plots use simulation main steps.
Map defaults: all species, central X/Y/Z planes, and all saved times. Each species
gets a figure with plane rows and time columns, paginated at six columns.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.vtk_timeseries_loader import parse_legacy_vtk_ascii
from postprocessing.diffusion_plane_maps import export_plane_maps, resolve_planes

PATTERNS = {"ecm": "ecm_data_t*.vtk", "cells": "cells_t*.vtk", "vascular": "vasc_data_t*.vtk"}
SPECIES_RE = re.compile(r"concentration_species_(\d+)$")
XYZ = ["x", "y", "z"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "--folder", type=Path, default=PROJECT_ROOT / "result_files",
                        help="Folder containing VTK files (default: repository result_files)")
    parser.add_argument("--datasets", nargs="+", choices=list(PATTERNS),
                        help="Datasets to include (default: all available)")
    fields = parser.add_mutually_exclusive_group()
    fields.add_argument("--species", nargs="+", type=int, help="Zero-based species indices (default: discover all)")
    fields.add_argument("--variables", nargs="+", help="Named point scalars, e.g. concentration_species_0 damage")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--probe", nargs=3, type=float, action="append", metavar=("X", "Y", "Z"),
                           help="Nearest-point probe; repeat for multiple locations (default: 0 0 0)")
    selection.add_argument("--point-id", type=int, help="Original zero-based VTK point index; ordering may change")
    selection.add_argument("--agent-id", type=int, help="Stable exported id; use with --datasets cells or vascular")
    parser.add_argument("--nearest-mode", choices=["per_time", "fixed_id"], default="per_time",
                        help="Resample each step, or follow the initial nearest agent/point (default: per_time)")
    parser.add_argument("--reference-step", type=int,
                        help="Saved step for fixed_id selection (default: first selected step)")
    parser.add_argument("--plots", nargs="+", choices=["traces", "summary", "profiles", "maps"],
                        default=["traces", "summary"], help="Plot types (default: traces summary)")
    parser.add_argument("--subplot-mode", choices=["by_species", "by_dataset"], default="by_species")
    parser.add_argument("--profile-axis", choices=XYZ, default="x", help="ECM profile axis (default: x)")
    parser.add_argument("--profile-steps", nargs="+", type=int,
                        help="Saved steps for ECM profiles (default: first and last selected steps)")
    parser.add_argument("--planes", "--plane", nargs="+", action="extend", default=None,
                        help="ECM maps: e.g. z=0 x=-20; at most one per axis (default: x=0 y=0 z=0)")
    parser.add_argument("--map-layout", choices=["by_plane", "by_species"], default="by_species",
                        help="Figure per plane with species rows, or per species with plane rows; time columns")
    parser.add_argument("--map-columns", type=int, default=6,
                        help="Maximum time columns per map page; all selected steps are included (default: 6)")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=False,
                        help="Bilinear map display smoothing; default: one grid pixel per ECM agent")
    parser.add_argument("--start-step", type=int, help="First main step to read, inclusive")
    parser.add_argument("--end-step", type=int, help="Last main step to read, inclusive")
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--steps", nargs="+", type=int, help="Exact saved main steps to read (default: all)")
    steps.add_argument("--every", type=int, default=1, help="Read every Nth available file per dataset (default: 1)")
    parser.add_argument("--time-step", type=float, help="Seconds per main step; overrides TIME_STEP in JSON")
    parser.add_argument("--config", type=Path,
                        help="Run's JSON overrides; default: parameters.json in the results folder, if present")
    parser.add_argument("--outdir", type=Path, help="Output directory (default: <results-dir>/diffusion_plots)")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png"])
    parser.add_argument("--show", action="store_true", help="Also open figures interactively")
    parser.add_argument("--list", action="store_true", help="List datasets, saved steps and scalars without plotting")
    return parser


def validate_args(args) -> None:
    if args.every < 1:
        raise ValueError("--every must be at least 1")
    if args.map_columns < 1:
        raise ValueError("--map-columns must be at least 1")
    if args.planes is not None:
        resolve_planes(args.planes)
    if args.steps is not None and any(step < 0 for step in args.steps):
        raise ValueError("--steps must be nonnegative")
    for name in ("start_step", "end_step", "reference_step", "point_id", "agent_id"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.start_step is not None and args.end_step is not None and args.start_step > args.end_step:
        raise ValueError("--start-step must not exceed --end-step")
    if args.species is not None and any(s < 0 for s in args.species):
        raise ValueError("--species indices must be nonnegative")
    if args.probe and not np.isfinite(args.probe).all():
        raise ValueError("--probe coordinates must be finite")
    if args.reference_step is not None and (args.nearest_mode != "fixed_id" or
                                             args.point_id is not None or args.agent_id is not None):
        raise ValueError("--reference-step requires nearest-point tracking with --nearest-mode fixed_id")


def saved_step(path: Path) -> int:
    match = re.search(r"_t(\d+)\.vtk$", path.name)
    if match is None:
        raise ValueError(f"Cannot read main step from filename: {path.name}")
    return int(match.group(1))


def discover_series(args) -> dict[str, list[Path]]:
    if not args.results_dir.is_dir():
        raise ValueError(f"Results directory does not exist: {args.results_dir}")
    series = {}
    datasets = args.datasets or (["ecm"] if set(args.plots) == {"maps"} else PATTERNS)
    for name in dict.fromkeys(datasets):
        paths = sorted(args.results_dir.glob(PATTERNS[name]), key=saved_step)
        paths = [p for p in paths if (args.start_step is None or saved_step(p) >= args.start_step)
                 and (args.end_step is None or saved_step(p) <= args.end_step)][::args.every]
        if paths and args.steps is not None:
            missing = set(args.steps) - {saved_step(p) for p in paths}
            if missing:
                raise ValueError(f"Requested {name} steps are unavailable in the selected range: {sorted(missing)}")
            paths = [p for p in paths if saved_step(p) in args.steps]
        if paths:
            steps = [saved_step(p) for p in paths]
            if len(set(steps)) != len(steps):
                raise ValueError(f"Multiple {name} files have the same step in {args.results_dir}")
            series[name] = paths
        elif args.datasets:
            raise ValueError(f"No {name} files match {PATTERNS[name]} and the selected step range")
    if not series:
        raise ValueError(f"No diffusion VTK files found in {args.results_dir}. "
                         "Use --results-dir to select a run folder; subfolders are not searched.")
    return series


def read_snapshot(path: Path, dataset: str) -> pd.DataFrame:
    points, scalars = parse_legacy_vtk_ascii(str(path))
    frame = pd.DataFrame(points, columns=XYZ)
    frame["point_id"] = np.arange(len(points))
    for name, values in scalars.items():
        if values.ndim == 1:
            frame[name] = values
    if "is_corner" in frame:
        frame = frame.loc[frame["is_corner"] == 0]
    if dataset == "cells" and "id" in frame:
        # The writer emits all cell centers first, then anchors with repeated ids.
        frame = frame.drop_duplicates("id", keep="first")
    if not np.isfinite(frame[XYZ].to_numpy()).all():
        raise ValueError(f"Non-finite point coordinates in {path}")
    return frame.reset_index(drop=True)


def resolve_time_step(args) -> float | None:
    config = args.config
    if config is None and (args.results_dir / "parameters.json").is_file():
        config = args.results_dir / "parameters.json"
    dt = args.time_step
    if dt is None and config is not None:
        parameters = json.loads(config.read_text(encoding="utf-8-sig"))
        if not isinstance(parameters, dict):
            raise ValueError(f"Expected a JSON object in {config}")
        if "TIME_STEP" not in parameters:
            raise ValueError(f"TIME_STEP missing from {config}; pass --time-step or use a run's overrides JSON")
        dt = parameters["TIME_STEP"]
    if dt is not None:
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("TIME_STEP / --time-step must be finite and positive")
        print(f"Time axis: seconds (main step x {dt:g}; VTK suffixes already include the save interval).")
        return float(dt)
    print("Time axis: simulation main step. Use --time-step or --config for seconds.")
    return None


def choose_variables(args, snapshots: dict[str, pd.DataFrame]) -> list[str]:
    available = set().union(*(set(frame.columns) - {*XYZ, "point_id"} for frame in snapshots.values()))
    if args.variables:
        variables = list(dict.fromkeys(args.variables))
    elif args.species is not None:
        variables = list(dict.fromkeys(f"concentration_species_{s}" for s in args.species))
    else:
        variables = sorted((v for v in available if SPECIES_RE.fullmatch(v)),
                           key=lambda v: int(SPECIES_RE.fullmatch(v).group(1)))
    missing = set(variables) - available
    if missing:
        raise ValueError(f"Scalars not found: {', '.join(sorted(missing))}. Use --list to inspect available scalars.")
    if not variables:
        raise ValueError("No concentration_species_* scalars found. Use --variables to choose other point scalars.")
    return variables


def nearest_row(frame: pd.DataFrame, xyz) -> pd.Series | None:
    if frame.empty:
        return None
    distances = np.sum((frame[XYZ].to_numpy() - np.asarray(xyz)) ** 2, axis=1)
    return frame.iloc[int(np.argmin(distances))]


def tracking_key(frame: pd.DataFrame, xyz) -> tuple[str, float]:
    row = nearest_row(frame, xyz)
    if row is None:
        raise ValueError("Cannot choose a fixed_id probe from an empty reference snapshot")
    column = "id" if "id" in frame else "point_id"
    if not np.isfinite(row[column]) or frame[column].duplicated().any():
        raise ValueError(f"Cannot track using {column}: expected finite, unique values")
    return column, row[column]


def sample_probe(frame: pd.DataFrame, xyz, key: tuple[str, float] | None) -> pd.Series | None:
    if key is None:
        return nearest_row(frame, xyz)
    column, value = key
    if column not in frame:
        raise ValueError(f"No '{column}' scalar in this dataset; --agent-id requires cell or vascular outputs")
    match = frame.loc[frame[column] == value]
    return None if match.empty else match.iloc[0]


def scalar_summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {"count": len(values), "finite_count": len(finite),
            "nonfinite_count": len(values) - len(finite), "negative_count": int((finite < 0).sum()),
            "min": float(finite.min()) if len(finite) else np.nan,
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "max": float(finite.max()) if len(finite) else np.nan}


def profile_line(frame: pd.DataFrame, axis: str, xyz) -> pd.DataFrame:
    """Select the nearest actual grid line; never average two equidistant lines."""
    if frame.empty:
        return frame
    transverse = [c for c in XYZ if c != axis]
    target = np.asarray(xyz)[[XYZ.index(c) for c in transverse]]
    points = frame[transverse].to_numpy()
    closest = points[int(np.argmin(np.sum((points - target) ** 2, axis=1)))]
    keep = np.all(np.isclose(points, closest, rtol=0, atol=1e-7), axis=1)
    line = frame.loc[keep].sort_values(axis)
    if len(line) < 2 or line[axis].duplicated().any():
        raise ValueError("ECM profiles require an axis-aligned grid with at least two points on a line. "
                         "Use traces/summary for irregular or rotated grids.")
    return line


def collect_data(args, series, snapshots, variables, dt):
    traces, summaries, profiles = [], [], []
    probes = args.probe or [[0.0, 0.0, 0.0]]
    want_traces = "traces" in args.plots
    want_profiles = "profiles" in args.plots
    if want_profiles and "ecm" not in series:
        raise ValueError("--plots profiles requires ECM output (include --datasets ecm)")
    for dataset, paths in series.items():
        fields = [v for v in variables if v in snapshots[dataset]]
        if not fields:
            if want_profiles and dataset == "ecm":
                raise ValueError("None of the selected scalars are available for ECM profiles")
            print(f"Skipping {dataset}: none of the selected scalars are present.")
            continue
        absent = [v for v in variables if v not in fields]
        if absent:
            print(f"{dataset}: unavailable scalars skipped: {', '.join(absent)}")
        keys = [None] * len(probes)
        labels = [f"({', '.join(f'{x:g}' for x in xyz)})" for xyz in probes]
        if want_traces:
            if args.point_id is not None or args.agent_id is not None:
                column = "id" if args.agent_id is not None else "point_id"
                value = args.agent_id if args.agent_id is not None else args.point_id
                keys = [(column, value)]
                labels = [f"{column}={value}"]
            elif args.nearest_mode == "fixed_id":
                ref = paths[0]
                if args.reference_step is not None:
                    ref = next((p for p in paths if saved_step(p) == args.reference_step), None)
                if ref is None:
                    raise ValueError(f"Reference step {args.reference_step} is not among selected {dataset} files")
                reference = snapshots[dataset] if ref == paths[0] else read_snapshot(ref, dataset)
                keys = [tracking_key(reference, xyz) for xyz in probes]
                labels = [f"{label}, {key[0]}={key[1]:g}" for label, key in zip(labels, keys)]
                print(f"{dataset}: fixed tracking at step {saved_step(ref)}: {'; '.join(labels)}")
                if "id" not in reference:
                    print(f"{dataset}: no exported id; fixed_id follows the VTK point index (requires stable ordering).")
        profile_steps = set()
        if want_profiles and dataset == "ecm":
            profile_steps = set(args.profile_steps or [saved_step(paths[0]), saved_step(paths[-1])])
            missing = profile_steps - {saved_step(p) for p in paths}
            if missing:
                raise ValueError(f"Profile steps are not among selected ECM files: {sorted(missing)}")
        missing_probes = [0] * len(probes)
        for i, path in enumerate(paths):
            step = saved_step(path)
            frame = snapshots[dataset] if i == 0 else read_snapshot(path, dataset)
            missing = set(fields) - set(frame.columns)
            if missing:
                raise ValueError(f"Scalars {sorted(missing)} missing from {path}")
            base = {"dataset": dataset, "step": step}
            if dt is not None:
                base["time"] = step * dt
            for variable in fields:
                summaries.append({**base, "variable": variable,
                                  **scalar_summary(frame[variable].to_numpy())})
            if want_traces:
                for j, (xyz, key, label) in enumerate(zip(probes, keys, labels)):
                    row = sample_probe(frame, xyz, key)
                    if row is None:
                        missing_probes[j] += 1
                    sample = {c: row[c] if row is not None and c in row else np.nan
                              for c in ["point_id", "id", *XYZ]}
                    distance = np.nan
                    if row is not None and args.point_id is None and args.agent_id is None:
                        distance = float(np.linalg.norm(row[XYZ].to_numpy(dtype=float) - xyz))
                    for variable in fields:
                        traces.append({**base, "probe": label, **sample, "distance": distance,
                                       "variable": variable, "value": row[variable] if row is not None else np.nan})
            if step in profile_steps:
                seen_lines = set()
                for xyz in probes:
                    line = profile_line(frame, args.profile_axis, xyz)
                    signature = tuple(line["point_id"])
                    if signature in seen_lines:
                        continue
                    seen_lines.add(signature)
                    for variable in fields:
                        for _, row in line.iterrows():
                            profiles.append({**base, "probe": str(tuple(xyz)), "variable": variable,
                                             "point_id": row["point_id"], **{c: row[c] for c in XYZ},
                                             "value": row[variable]})
            if (i + 1) % 100 == 0:
                print(f"Reading {dataset}: {i + 1}/{len(paths)} snapshots...", flush=True)
        for label, count in zip(labels, missing_probes):
            if count == len(paths):
                raise ValueError(f"{dataset}: probe {label} matched no points in any selected snapshot")
            if count:
                print(f"{dataset}: probe {label} missing in {count}/{len(paths)} snapshots; traces contain gaps.")
        print(f"Read {dataset}: {len(paths)} snapshots, steps {saved_step(paths[0])}..{saved_step(paths[-1])}.")
    return pd.DataFrame(traces), pd.DataFrame(summaries), pd.DataFrame(profiles)


def short_name(variable: str) -> str:
    match = SPECIES_RE.fullmatch(variable)
    return f"Species {match.group(1)}" if match else variable


def plot_time_series(table, kind, subplot_mode, time_column, plt):
    facet = "variable" if subplot_mode == "by_species" else "dataset"
    other = "dataset" if facet == "variable" else "variable"
    panels = table[facet].unique()
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.3 * len(panels)), sharex=True, squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        subset = table.loc[table[facet] == panel]
        grouping = [other, "probe"] if kind == "traces" else [other]
        for group, rows in subset.groupby(grouping, sort=False):
            rows = rows.sort_values("step")
            parts = group if isinstance(group, tuple) else (group,)
            label = " | ".join(short_name(str(p)) for p in parts)
            y = rows["value" if kind == "traces" else "mean"].to_numpy(dtype=float, copy=True)
            y[~np.isfinite(y)] = np.nan
            line, = ax.plot(rows[time_column], y, label=label, marker="." if len(rows) < 15 else None)
            if kind == "summary":
                ax.fill_between(rows[time_column].to_numpy(), rows["min"].to_numpy(), rows["max"].to_numpy(),
                                color=line.get_color(), alpha=0.16)
        ax.set_title(short_name(panel))
        ax.set_ylabel("Concentration" if all(SPECIES_RE.fullmatch(v) for v in subset["variable"]) else "Scalar value")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    axes[-1, 0].set_xlabel("Time (s)" if time_column == "time" else "Simulation main step")
    fig.suptitle("Probe traces" if kind == "traces" else "Mean concentration / scalar (shading: min to max)")
    fig.tight_layout()
    return fig


def plot_profiles(table, axis, plt):
    variables = table["variable"].unique()
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 3.3 * len(variables)), sharex=True, squeeze=False)
    transverse = [c for c in XYZ if c != axis]
    for ax, variable in zip(axes.flat, variables):
        for (step, probe), rows in table.loc[table["variable"] == variable].groupby(["step", "probe"], sort=False):
            rows = rows.sort_values(axis)
            coords = ", ".join(f"{c}={rows[c].iloc[0]:g}" for c in transverse)
            label = f"step {step}, {coords}"
            if "time" in rows:
                label += f" ({rows['time'].iloc[0]:g} s)"
            ax.plot(rows[axis], rows["value"], ".-", label=label)
        ax.set_title(short_name(variable))
        ax.set_ylabel("Concentration" if SPECIES_RE.fullmatch(variable) else "Scalar value")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    axes[-1, 0].set_xlabel(f"{axis} (simulation length units)")
    fig.suptitle("ECM profiles along the nearest grid line")
    fig.tight_layout()
    return fig


def run(args) -> Path | None:
    validate_args(args)
    series = discover_series(args)
    snapshots = {name: read_snapshot(paths[0], name) for name, paths in series.items()}
    for name, frame in snapshots.items():
        fields = [c for c in frame if c not in [*XYZ, "point_id"]]
        print(f"{name}: {len(series[name])} files, {len(frame)} physical points in first selected snapshot; "
              f"steps {saved_step(series[name][0])}..{saved_step(series[name][-1])}.")
        if args.list:
            print(f"  Scalars: {', '.join(fields)}")
            print(f"  Saved steps: {', '.join(str(saved_step(p)) for p in series[name])}")
    if args.list:
        return None
    variables = choose_variables(args, snapshots)
    if "maps" in args.plots:
        if "ecm" not in snapshots:
            raise ValueError("--plots maps requires ECM output (include --datasets ecm)")
        if not any(variable in snapshots["ecm"] for variable in variables):
            raise ValueError("None of the selected scalars are available for ECM maps")
    dt = resolve_time_step(args)
    print(f"Scalars: {', '.join(variables)}")
    traces, summaries, profiles = collect_data(args, series, snapshots, variables, dt)
    print("Concentration/scalar ranges across selected snapshots (finite values):")
    for (dataset, variable), rows in summaries.groupby(["dataset", "variable"], sort=False):
        print(f"  {dataset} / {short_name(variable)}: {rows['min'].min():.6g} .. {rows['max'].max():.6g}; "
              f"negative={rows['negative_count'].sum()}, non-finite={rows['nonfinite_count'].sum()} point-samples")
    outdir = args.outdir or args.results_dir / "diffusion_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time_column = "time" if dt is not None else "step"
    tables = {"traces": traces, "summary": summaries, "profiles": profiles}
    for kind, table in tables.items():
        if not table.empty:
            table.to_csv(outdir / f"diffusion_{kind}.csv", index=False)
        if kind not in args.plots or table.empty:
            continue
        fig = plot_profiles(table, args.profile_axis, plt) if kind == "profiles" else plot_time_series(
            table, kind, args.subplot_mode, time_column, plt)
        for extension in dict.fromkeys(args.formats):
            path = outdir / f"diffusion_{kind}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            print(f"Saved {path}")
        if not args.show:
            plt.close(fig)
    print(f"CSV data saved to {outdir.resolve()}")
    if "maps" in args.plots:
        export_plane_maps(args, series["ecm"], snapshots["ecm"], variables, summaries, dt, outdir,
                          read_snapshot=read_snapshot, saved_step=saved_step)
    if args.show:
        plt.show()
    return outdir


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
