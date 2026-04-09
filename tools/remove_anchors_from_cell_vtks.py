#!/usr/bin/env python3
"""
Trim the VTK cell files obtained in the simulation using per-timepoint cell counts from
output_data_0.pickle. This removes the anchor points (for easier visualization in Paraview).

For each VTK file matching ``cells_t*.vtk`` in the input folder, the script:
  - loads ``CELL_POPULATION_METRICS_OVER_TIME`` from ``output_data_0.pickle``
  - extracts the timestep from the filename (e.g. cells_t0005.vtk -> step 5)
  - finds the matching dataframe row where ``step == <that timestep>``
  - reads ``n_cells_total`` for that row
  - keeps only the first ``n_cells_total`` points/cells and associated point data
  - writes a new VTK file named ``no_anchor_cells_tXXXX.vtk``

Expected input format
---------------------
- Legacy VTK ASCII
- DATASET UNSTRUCTURED_GRID
- One point per cell
- POINT_DATA count matches POINTS count
- Point attributes stored as SCALARS and/or VECTORS

Examples
--------
python tools/remove_anchors_from_cell_vtks.py
python tools/remove_anchors_from_cell_vtks.py --input-dir result_files
python tools/remove_anchors_from_cell_vtks.py --pickle-path result_files/output_data_0.pickle
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import pickle
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PICKLE_PATH = PROJECT_ROOT / "result_files" / "output_data_0.pickle"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "result_files"


# ---------------------------------------------------------------------------
# Safe pickle loading
# ---------------------------------------------------------------------------

class _DummyModelParameterConfig:
    pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "helper_module" and name == "ModelParameterConfig":
            return _DummyModelParameterConfig
        if module.startswith("numpy.core"):
            remapped = "numpy._core" + module[len("numpy.core"):]
            try:
                return getattr(importlib.import_module(remapped), name)
            except (ModuleNotFoundError, AttributeError):
                pass
        return super().find_class(module, name)


def _load_pickle(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    with path.open("rb") as f:
        return _SafeUnpickler(f).load()


def load_cell_population_metrics(results: dict[str, Any]) -> pd.DataFrame:
    """Extract CELL_METRICS_OVER_TIME as a dataframe."""
    raw = results.get("CELL_METRICS_OVER_TIME")
    if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
        raise ValueError(
            "Pickle does not contain non-empty CELL_METRICS_OVER_TIME"
        )
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        df = pd.DataFrame(raw)

    required_cols = {"step", "n_cells_total"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "CELL_POPULATION_METRICS_OVER_TIME is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    return df


# ---------------------------------------------------------------------------
# VTK parsing
# ---------------------------------------------------------------------------

@dataclass
class ScalarBlock:
    header: str
    lookup_table: str
    values: list[str]


@dataclass
class VectorBlock:
    header: str
    values: list[str]


@dataclass
class VTKData:
    preamble: list[str]
    points_header: str
    points: list[str]
    cells_header: str
    cells: list[str]
    cell_types_header: str
    cell_types: list[str]
    point_data_header: str
    point_blocks: list[ScalarBlock | VectorBlock]


def _find_line_index(lines: list[str], prefix: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise ValueError(f"Could not find section starting with '{prefix}'")


def _parse_count_from_header(header: str, expected_prefix: str) -> int:
    parts = header.split()
    if len(parts) < 2 or parts[0] != expected_prefix:
        raise ValueError(f"Malformed header: '{header}'")
    return int(parts[1])


def parse_legacy_ascii_vtk(path: pathlib.Path) -> VTKData:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    points_idx = _find_line_index(lines, "POINTS ")
    cells_idx = _find_line_index(lines, "CELLS ", start=points_idx + 1)
    cell_types_idx = _find_line_index(lines, "CELL_TYPES ", start=cells_idx + 1)
    point_data_idx = _find_line_index(lines, "POINT_DATA ", start=cell_types_idx + 1)

    points_header = lines[points_idx]
    cells_header = lines[cells_idx]
    cell_types_header = lines[cell_types_idx]
    point_data_header = lines[point_data_idx]

    n_points = _parse_count_from_header(points_header, "POINTS")
    n_cells = _parse_count_from_header(cells_header, "CELLS")
    n_cell_types = _parse_count_from_header(cell_types_header, "CELL_TYPES")
    n_point_data = _parse_count_from_header(point_data_header, "POINT_DATA")

    if not (n_points == n_cells == n_cell_types == n_point_data):
        raise ValueError(
            f"Inconsistent counts in '{path.name}': "
            f"POINTS={n_points}, CELLS={n_cells}, CELL_TYPES={n_cell_types}, POINT_DATA={n_point_data}"
        )

    preamble = lines[:points_idx]

    points = lines[points_idx + 1 : cells_idx]
    if len(points) != n_points:
        raise ValueError(
            f"Expected {n_points} point rows in '{path.name}', found {len(points)}"
        )

    cells = lines[cells_idx + 1 : cell_types_idx]
    if len(cells) != n_cells:
        raise ValueError(
            f"Expected {n_cells} cell rows in '{path.name}', found {len(cells)}"
        )

    cell_types = lines[cell_types_idx + 1 : point_data_idx]
    if len(cell_types) != n_cell_types:
        raise ValueError(
            f"Expected {n_cell_types} CELL_TYPES rows in '{path.name}', found {len(cell_types)}"
        )

    point_blocks: list[ScalarBlock | VectorBlock] = []
    i = point_data_idx + 1

    while i < len(lines):
        line = lines[i]

        if line.startswith("SCALARS "):
            header = line
            if i + 1 >= len(lines) or not lines[i + 1].startswith("LOOKUP_TABLE "):
                raise ValueError(f"Missing LOOKUP_TABLE after SCALARS in '{path.name}'")
            lookup_table = lines[i + 1]
            values_start = i + 2
            values_end = values_start + n_point_data
            values = lines[values_start:values_end]
            if len(values) != n_point_data:
                raise ValueError(
                    f"Expected {n_point_data} scalar values after '{header}' in '{path.name}'"
                )
            point_blocks.append(ScalarBlock(header, lookup_table, values))
            i = values_end
            continue

        if line.startswith("VECTORS "):
            header = line
            values_start = i + 1
            values_end = values_start + n_point_data
            values = lines[values_start:values_end]
            if len(values) != n_point_data:
                raise ValueError(
                    f"Expected {n_point_data} vector values after '{header}' in '{path.name}'"
                )
            point_blocks.append(VectorBlock(header, values))
            i = values_end
            continue

        raise ValueError(
            f"Unsupported POINT_DATA block in '{path.name}' at line {i + 1}: '{line}'"
        )

    return VTKData(
        preamble=preamble,
        points_header=points_header,
        points=points,
        cells_header=cells_header,
        cells=cells,
        cell_types_header=cell_types_header,
        cell_types=cell_types,
        point_data_header=point_data_header,
        point_blocks=point_blocks,
    )


def _point_dtype_from_header(points_header: str) -> str:
    parts = points_header.split()
    if len(parts) >= 3:
        return parts[2]
    return "float"


def trim_vtk_data(data: VTKData, ncells: int) -> VTKData:
    if ncells < 0:
        raise ValueError("ncells must be >= 0")

    keep = min(ncells, len(data.points))
    point_dtype = _point_dtype_from_header(data.points_header)

    trimmed_blocks: list[ScalarBlock | VectorBlock] = []
    for block in data.point_blocks:
        if isinstance(block, ScalarBlock):
            trimmed_blocks.append(
                ScalarBlock(
                    header=block.header,
                    lookup_table=block.lookup_table,
                    values=block.values[:keep],
                )
            )
        elif isinstance(block, VectorBlock):
            trimmed_blocks.append(
                VectorBlock(
                    header=block.header,
                    values=block.values[:keep],
                )
            )
        else:
            raise TypeError(f"Unexpected block type: {type(block)}")

    return VTKData(
        preamble=data.preamble,
        points_header=f"POINTS {keep} {point_dtype}",
        points=data.points[:keep],
        cells_header=f"CELLS {keep} {2 * keep}",
        cells=[f"1 {i}" for i in range(keep)],
        cell_types_header=f"CELL_TYPES {keep}",
        cell_types=data.cell_types[:keep],
        point_data_header=f"POINT_DATA {keep}",
        point_blocks=trimmed_blocks,
    )


def write_legacy_ascii_vtk(path: pathlib.Path, data: VTKData) -> None:
    out_lines: list[str] = []

    out_lines.extend(data.preamble)
    out_lines.append(data.points_header)
    out_lines.extend(data.points)

    out_lines.append(data.cells_header)
    out_lines.extend(data.cells)

    out_lines.append(data.cell_types_header)
    out_lines.extend(data.cell_types)

    out_lines.append(data.point_data_header)
    for block in data.point_blocks:
        if isinstance(block, ScalarBlock):
            out_lines.append(block.header)
            out_lines.append(block.lookup_table)
            out_lines.extend(block.values)
        elif isinstance(block, VectorBlock):
            out_lines.append(block.header)
            out_lines.extend(block.values)
        else:
            raise TypeError(f"Unexpected block type: {type(block)}")

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def extract_step_from_vtk_filename(path: pathlib.Path) -> int:
    """
    Extract timestep from names like:
      cells_t0001.vtk -> 1
      cells_t25.vtk   -> 25
    """
    m = re.fullmatch(r"cells_t(\d+)\.vtk", path.name)
    if not m:
        raise ValueError(
            f"Filename does not match expected pattern 'cells_t####.vtk': {path.name}"
        )
    return int(m.group(1))


def build_step_to_ncells_map(df: pd.DataFrame) -> dict[int, int]:
    step_to_ncells: dict[int, int] = {}

    for idx, row in df.iterrows():
        step_raw = row["step"]
        ncells_raw = row["n_cells_total"]

        try:
            step = int(step_raw)
        except Exception as exc:
            raise ValueError(
                f"Invalid step value at dataframe row {idx}: {step_raw!r}"
            ) from exc

        try:
            ncells = int(ncells_raw)
        except Exception as exc:
            raise ValueError(
                f"Invalid n_cells_total value at dataframe row {idx}: {ncells_raw!r}"
            ) from exc

        if step in step_to_ncells:
            raise ValueError(
                f"Duplicate step value in CELL_POPULATION_METRICS_OVER_TIME: step={step}"
            )

        step_to_ncells[step] = ncells

    return step_to_ncells


def output_name_for(input_path: pathlib.Path) -> str:
    m = re.match(r"cells_(t\d+)\.vtk$", input_path.name)
    if m:
        return f"no_anchor_cells_{m.group(1)}.vtk"
    return f"no_anchor_{input_path.name}"


def process_file(path: pathlib.Path, ncells: int) -> pathlib.Path:
    data = parse_legacy_ascii_vtk(path)
    trimmed = trim_vtk_data(data, ncells)
    out_path = path.with_name(output_name_for(path))
    write_legacy_ascii_vtk(out_path, trimmed)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process all legacy ASCII VTK files in a folder, matching each file "
            "timestep to the 'step' column in CELL_POPULATION_METRICS_OVER_TIME."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing the original VTK files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--pickle-path",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to output_data_0.pickle "
            "(default: <input-dir>/output_data_0.pickle if present, "
            f"otherwise {DEFAULT_PICKLE_PATH})"
        ),
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    vtk_files = sorted(input_dir.glob("cells_t*.vtk"))
    if not vtk_files:
        raise FileNotFoundError(f"No files matching 'cells_t*.vtk' found in: {input_dir}")

    if args.pickle_path is not None:
        pickle_path = args.pickle_path.resolve()
    else:
        candidate = input_dir / "output_data_0.pickle"
        pickle_path = candidate if candidate.exists() else DEFAULT_PICKLE_PATH.resolve()

    results = _load_pickle(pickle_path)
    df = load_cell_population_metrics(results)
    step_to_ncells = build_step_to_ncells_map(df)

    vtk_steps: list[int] = []
    for vtk_path in vtk_files:
        vtk_steps.append(extract_step_from_vtk_filename(vtk_path))

    missing_steps = [step for step in vtk_steps if step not in step_to_ncells]
    if missing_steps:
        raise ValueError(
            "Some VTK files do not have a matching row in "
            "CELL_POPULATION_METRICS_OVER_TIME.step: "
            + ", ".join(str(s) for s in missing_steps)
        )

    print(f"Input directory: {input_dir}")
    print(f"Pickle path: {pickle_path}")
    print(f"Found {len(vtk_files)} VTK file(s)")
    print("Matching each filename timestep against dataframe column 'step'")

    for vtk_path in vtk_files:
        step = extract_step_from_vtk_filename(vtk_path)
        ncells = step_to_ncells[step]
        out_path = process_file(vtk_path, ncells)
        print(
            f"[OK] {vtk_path.name} -> {out_path.name}  |  step={step}  |  kept {ncells} cells"
        )

    print("Done.")


if __name__ == "__main__":
    main()