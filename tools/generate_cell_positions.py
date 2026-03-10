#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helper_module import saveCellInitializationCache  


DEFAULT_COUNTS = [1000, 10000, 100000, 1000000]
DEFAULT_BOUNDARY_COORDS = [500.0, -500.0, 500.0, -500.0, 500.0, -500.0]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cached cell positions and orientations for cellfoundry.",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=DEFAULT_COUNTS,
        help="Cell counts to pre-generate.",
    )
    parser.add_argument(
        "--boundary-coords",
        type=float,
        nargs=6,
        metavar=("X_POS", "X_NEG", "Y_POS", "Y_NEG", "Z_POS", "Z_NEG"),
        default=DEFAULT_BOUNDARY_COORDS,
        help="Boundary coordinates in the same format used by model.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where cache pickle files will be stored. Defaults to the folder containing model.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary_coords = [float(v) for v in args.boundary_coords]
    counts = [int(v) for v in args.counts]

    print(f"Output directory: {output_dir}")
    print(f"Boundary coords: {boundary_coords}")

    batch_start = time.perf_counter()
    for n_cells in counts:
        print(f"\nGenerating cache for N_CELLS={n_cells}...")
        item_start = time.perf_counter()
        output_path, cache_data = saveCellInitializationCache(n_cells, boundary_coords, output_dir)
        timings = cache_data["generation_time_seconds"]
        item_total = time.perf_counter() - item_start
        print(f"  Saved: {output_path}")
        print(
            f"  Timing: positions={timings['positions']:.3f}s, "
            f"orientations={timings['orientations']:.3f}s, total={item_total:.3f}s"
        )

    batch_total = time.perf_counter() - batch_start
    print(f"\nCompleted cache generation in {batch_total:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())