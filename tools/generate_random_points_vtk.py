from __future__ import annotations

import argparse
from pathlib import Path
import random


def generate_random_points(
    n_cells: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> list[tuple[float, float, float]]:
    """Generate uniformly distributed random points inside a box."""
    return [
        (
            random.uniform(xmin, xmax),
            random.uniform(ymin, ymax),
            random.uniform(zmin, zmax),
        )
        for _ in range(n_cells)
    ]


def generate_random_scalars(n_cells: int) -> list[float]:
    """Generate one dummy random scalar value per point."""
    return [random.random() for _ in range(n_cells)]


def write_legacy_vtk(
    output_path: Path,
    points: list[tuple[float, float, float]],
    scalars: list[float],
    scalar_name: str = "dummy_scalar",
) -> None:
    """Write points and one scalar field to a legacy ASCII VTK file."""
    n_points = len(points)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Random cell points\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {n_points} float\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        # One vertex cell per point
        f.write(f"CELLS {n_points} {2 * n_points}\n")
        for i in range(n_points):
            f.write(f"1 {i}\n")

        # VTK_VERTEX = 1
        f.write(f"CELL_TYPES {n_points}\n")
        for _ in range(n_points):
            f.write("1\n")

        # Point data
        f.write(f"POINT_DATA {n_points}\n")
        f.write(f"SCALARS {scalar_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for value in scalars:
            f.write(f"{value:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a legacy VTK file with random cell positions and one dummy scalar field."
    )

    # Defaults values
    parser.add_argument("--n-cells", type=int, default=1000000, help="Number of random points")
    parser.add_argument("--xmin", type=float, default=-500.0, help="Minimum x")
    parser.add_argument("--xmax", type=float, default=500.0, help="Maximum x")
    parser.add_argument("--ymin", type=float, default=-500.0, help="Minimum y")
    parser.add_argument("--ymax", type=float, default=500.0, help="Maximum y")
    parser.add_argument("--zmin", type=float, default=-500.0, help="Minimum z")
    parser.add_argument("--zmax", type=float, default=500.0, help="Maximum z")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dummy_cells.vtk"),
        help="Output VTK filename",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--scalar-name",
        type=str,
        default="dummy_scalar",
        help="Name of the scalar field stored in the VTK",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_cells <= 0:
        raise ValueError("n_cells must be a positive integer")

    if args.xmin > args.xmax:
        raise ValueError("xmin must be <= xmax")
    if args.ymin > args.ymax:
        raise ValueError("ymin must be <= ymax")
    if args.zmin > args.zmax:
        raise ValueError("zmin must be <= zmax")

    random.seed(args.seed)

    points = generate_random_points(
        n_cells=args.n_cells,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        zmin=args.zmin,
        zmax=args.zmax,
    )
    scalars = generate_random_scalars(args.n_cells)

    write_legacy_vtk(
        output_path=args.output,
        points=points,
        scalars=scalars,
        scalar_name=args.scalar_name,
    )

    print(f"Current working directory: {Path.cwd()}")
    print(f"VTK file written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()