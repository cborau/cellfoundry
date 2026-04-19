"""
Extract scalar profiles along a line from all ``ecm_data_tX.vtk`` files in one
or two folders and plot their evolution over time.

The script parses legacy ASCII VTK files without using VTK libraries. For each
file, it extracts the values of a selected point scalar along the segment
defined by two input coordinates. The resulting profiles are stacked into a 2D
array with shape ``(n_timesteps, n_points_along_line)``.

If two folders are provided, the script generates a 1x2 comparison figure with
one panel per folder.

Usage example
-------------
python ./postprocessing/extract_scalar_profiles_vtk.py --parent-folder1 "C:/Users/PC/Documents/FLAMEGPU2/cellfoundry/result_files/homogeneous_diff" --parent-folder2 "C:/Users/PC/Documents/FLAMEGPU2/cellfoundry/result_files/heterogeneous_diff" --scalar-name concentration_species_0 --x1 0.0 --y1 -500.0 --z1 0.0 --x2 0.0 --y2 0.0 --z2 0.0 --show
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


def _read_numeric_block(
    lines: list[str],
    start_idx: int,
    n_values: int,
    dtype: type[float] | type[int] = float,
) -> tuple[np.ndarray, int]:
    """
    Read a block of numeric values spanning one or more lines.

    Parameters
    ----------
    lines
        Full list of file lines.
    start_idx
        Line index where numeric data starts.
    n_values
        Number of numeric values to read.
    dtype
        Output numeric type.

    Returns
    -------
    values
        One-dimensional array of length ``n_values``.
    next_idx
        Line index after the last consumed line.
    """
    tokens: list[str] = []
    idx = start_idx

    while len(tokens) < n_values and idx < len(lines):
        stripped = lines[idx].strip()
        if stripped:
            tokens.extend(stripped.split())
        idx += 1

    if len(tokens) < n_values:
        raise ValueError(
            f"Not enough numeric data. Expected {n_values}, found {len(tokens)}."
        )

    values = np.asarray(tokens[:n_values], dtype=dtype)
    return values, idx


def _read_points(lines: list[str]) -> np.ndarray:
    """
    Read the ``POINTS`` block from a legacy ASCII VTK file.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` with point coordinates.
    """
    pts_idx = next((i for i, l in enumerate(lines) if l.startswith("POINTS")), None)
    if pts_idx is None:
        raise ValueError("POINTS section not found in VTK file.")

    parts = lines[pts_idx].split()
    if len(parts) < 3:
        raise ValueError("Malformed POINTS header.")

    n_points = int(parts[1])
    flat_coords, _ = _read_numeric_block(lines, pts_idx + 1, 3 * n_points, float)
    return flat_coords.reshape(n_points, 3)


def _read_point_scalar(lines: list[str], scalar_name: str, n_points: int) -> np.ndarray:
    """
    Read a scalar array from the ``POINT_DATA`` section of a legacy ASCII VTK file.

    Parameters
    ----------
    lines
        Full list of file lines.
    scalar_name
        Name of the scalar array to extract.
    n_points
        Number of points in the dataset.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N,)`` with scalar values.
    """
    point_data_idx = next(
        (i for i, l in enumerate(lines) if l.strip().startswith("POINT_DATA")),
        None,
    )
    if point_data_idx is None:
        raise ValueError("POINT_DATA section not found in VTK file.")

    scalar_idx = None
    scalar_pattern = re.compile(
        rf"^SCALARS\s+{re.escape(scalar_name)}\s+(\S+)(?:\s+(\d+))?\s*$"
    )

    for i in range(point_data_idx + 1, len(lines)):
        if scalar_pattern.match(lines[i].strip()):
            scalar_idx = i
            break

    if scalar_idx is None:
        available = []
        for i in range(point_data_idx + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith("SCALARS"):
                parts = stripped.split()
                if len(parts) >= 2:
                    available.append(parts[1])
        raise ValueError(
            f"Scalar '{scalar_name}' not found. Available point scalars: {available}"
        )

    match = scalar_pattern.match(lines[scalar_idx].strip())
    if match is None:
        raise ValueError(f"Malformed SCALARS header for '{scalar_name}'.")

    n_components = int(match.group(2)) if match.group(2) is not None else 1
    if n_components != 1:
        raise ValueError(
            f"Scalar '{scalar_name}' has {n_components} components. "
            "Only 1-component point scalars are supported."
        )

    lookup_idx = scalar_idx + 1
    while lookup_idx < len(lines) and not lines[lookup_idx].strip():
        lookup_idx += 1

    if lookup_idx >= len(lines) or not lines[lookup_idx].strip().startswith("LOOKUP_TABLE"):
        raise ValueError(f"LOOKUP_TABLE line not found after SCALARS {scalar_name}.")

    values, _ = _read_numeric_block(lines, lookup_idx + 1, n_points, float)
    return values


def _select_points_on_segment(
    points: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select points lying on the segment p1 -> p2 within a tolerance.

    Returns
    -------
    mask
        Boolean mask over points.
    distance_along
        Projected distance from p1 along the segment direction.
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length <= 0.0:
        raise ValueError("The two points are identical. Segment length is zero.")

    unit_dir = direction / length
    rel = points - p1

    distance_along = rel @ unit_dir
    projected = p1 + distance_along[:, None] * unit_dir[None, :]
    distance_to_line = np.linalg.norm(points - projected, axis=1)

    mask = (
        (distance_to_line <= tol)
        & (distance_along >= -tol)
        & (distance_along <= length + tol)
    )

    return mask, distance_along


def extract_scalar_profile_along_line(
    filename: str | Path,
    scalar_name: str,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    *,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the scalar profile along a line segment from one legacy ASCII VTK file.

    Parameters
    ----------
    filename
        Path to the VTK file.
    scalar_name
        Name of the point scalar to extract.
    x1, y1, z1
        Coordinates of the first point.
    x2, y2, z2
        Coordinates of the second point.
    tol
        Geometric tolerance used to determine whether a point belongs to the
        line segment.

    Returns
    -------
    distances : numpy.ndarray
        Distance of each intersected mesh point from the first point, sorted
        along the segment.
    values : numpy.ndarray
        Scalar values at the intersected mesh points, sorted from the first
        point to the second point.
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"VTK file not found: {path}")

    lines = path.read_text().splitlines()

    points = _read_points(lines)
    n_points = points.shape[0]
    scalar_values = _read_point_scalar(lines, scalar_name, n_points)

    p1 = np.array([x1, y1, z1], dtype=float)
    p2 = np.array([x2, y2, z2], dtype=float)

    mask, distance_along = _select_points_on_segment(points, p1, p2, tol)

    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)

    selected_dist = distance_along[mask]
    selected_vals = scalar_values[mask]

    order = np.argsort(selected_dist)
    selected_dist = selected_dist[order]
    selected_vals = selected_vals[order]

    return selected_dist, selected_vals


def discover_ecm_vtk_files(parent_folder: str | Path) -> list[tuple[int, Path]]:
    """
    Discover and sort all ``ecm_data_tX.vtk`` files in a folder.

    Parameters
    ----------
    parent_folder
        Folder containing files named like ``ecm_data_t0001.vtk``.

    Returns
    -------
    list of (timestep, path)
        Sorted by timestep.
    """
    parent = Path(parent_folder)
    if not parent.is_dir():
        raise NotADirectoryError(f"Folder not found: {parent}")

    pairs: list[tuple[int, Path]] = []
    for path in parent.glob("ecm_data_t*.vtk"):
        match = re.search(r"ecm_data_t(\d+)\.vtk$", path.name)
        if match:
            pairs.append((int(match.group(1)), path))

    pairs.sort(key=lambda x: x[0])

    if not pairs:
        raise FileNotFoundError(f"No ecm_data_t*.vtk files found in {parent}")

    return pairs


def build_scalar_line_matrix(
    parent_folder: str | Path,
    scalar_name: str,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    *,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read all ``ecm_data_tX.vtk`` files in a folder and build a scalar-evolution matrix.

    For each file, the scalar profile along the same line segment is extracted.
    The resulting curves are stacked into a 2D array where each row corresponds
    to one timestep and each column corresponds to one position along the line.

    Parameters
    ----------
    parent_folder
        Folder containing ``ecm_data_tX.vtk`` files.
    scalar_name
        Name of the point scalar array to extract.
    x1, y1, z1
        Coordinates of the first point.
    x2, y2, z2
        Coordinates of the second point.
    tol
        Geometric tolerance used to determine whether a mesh point belongs to the
        line segment.

    Returns
    -------
    timesteps : numpy.ndarray
        Array of shape ``(T,)`` with timestep numbers extracted from file names.
    distances : numpy.ndarray
        Array of shape ``(N,)`` with distances along the line.
    values_matrix : numpy.ndarray
        Array of shape ``(T, N)`` containing scalar values. Row ``i`` corresponds
        to timestep ``timesteps[i]``.

    Raises
    ------
    ValueError
        If different files produce different numbers of intersected points or
        inconsistent point locations along the line.
    """
    vtk_files = discover_ecm_vtk_files(parent_folder)

    timesteps: list[int] = []
    all_values: list[np.ndarray] = []
    reference_distances: np.ndarray | None = None

    for timestep, vtk_path in vtk_files:
        distances, values = extract_scalar_profile_along_line(
            filename=vtk_path,
            scalar_name=scalar_name,
            x1=x1,
            y1=y1,
            z1=z1,
            x2=x2,
            y2=y2,
            z2=z2,
            tol=tol,
        )

        if len(values) == 0:
            raise ValueError(
                f"No points intersect the requested line in file: {vtk_path}"
            )

        if reference_distances is None:
            reference_distances = distances
        else:
            if len(distances) != len(reference_distances):
                raise ValueError(
                    f"Inconsistent number of intersected points in file: {vtk_path}. "
                    f"Expected {len(reference_distances)}, found {len(distances)}."
                )
            if not np.allclose(distances, reference_distances, atol=tol, rtol=0.0):
                raise ValueError(
                    f"Inconsistent intersected point positions in file: {vtk_path}"
                )

        timesteps.append(timestep)
        all_values.append(values)

    values_matrix = np.vstack(all_values)

    return (
        np.asarray(timesteps, dtype=int),
        reference_distances.copy(),
        values_matrix,
    )


def plot_scalar_line_evolution_on_axis(
    ax: plt.Axes,
    distances: np.ndarray,
    values_matrix: np.ndarray,
    timesteps: np.ndarray,
    *,
    scalar_name: str,
    cmap: str = "viridis",
    linewidth: float = 1.5,
    alpha: float = 0.9,
    title: str | None = None,
    show_colorbar: bool = False,
    ylabel: str | None = None,
) -> plt.Axes:
    """
    Plot the evolution of a scalar profile along a line on a given axis.

    Each row of ``values_matrix`` is plotted as one curve. Curves are colored
    according to timestep.

    Parameters
    ----------
    ax
        Matplotlib axis where the curves are drawn.
    distances
        Array of shape ``(N,)`` with distances along the line.
    values_matrix
        Array of shape ``(T, N)`` where each row is one scalar profile.
    timesteps
        Array of shape ``(T,)`` used to color curves and optionally label the
        colorbar.
    scalar_name
        Name of the scalar variable, used for labels.
    cmap
        Matplotlib colormap name.
    linewidth
        Line width for each curve.
    alpha
        Line transparency.
    title
        Optional axis title.
    show_colorbar
        Whether to add a colorbar to this axis.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for plotting.
    """
    if values_matrix.ndim != 2:
        raise ValueError("values_matrix must be a 2D array of shape (T, N).")
    if len(timesteps) != values_matrix.shape[0]:
        raise ValueError("timesteps length must match number of rows in values_matrix.")
    if len(distances) != values_matrix.shape[1]:
        raise ValueError("distances length must match number of columns in values_matrix.")

    color_values = timesteps.astype(float)
    norm = plt.Normalize(vmin=np.min(color_values), vmax=np.max(color_values))
    cmap_obj = plt.get_cmap(cmap)

    for i in range(values_matrix.shape[0]):
        ax.plot(
            distances,
            values_matrix[i, :],
            color=cmap_obj(norm(color_values[i])),
            linewidth=linewidth,
            alpha=alpha,
        )

    if show_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax)
        cbar.set_label("Timestep")

    ax.set_xlabel(r"Dist.from boundary [$\mu$m]")
    ax.set_ylabel(ylabel if ylabel is not None else scalar_name)
    ax.set_title(title if title is not None else f"Evolution of {scalar_name}")
    ax.grid(True, alpha=0.3)

    return ax


def plot_scalar_line_evolution_comparison(
    parent_folder1: str | Path,
    parent_folder2: str | Path,
    scalar_name: str,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    *,
    tol: float = 1e-8,
    cmap: str = "viridis",
    linewidth: float = 1.5,
    alpha: float = 0.9,
    title1: str | None = None,
    title2: str | None = None,
    axes: np.ndarray | list[plt.Axes] | tuple[plt.Axes, plt.Axes] | None = None,
    ylabel: str | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build scalar line matrices for two folders and plot them side by side.

    If ``axes`` is not provided, a new 1x2 figure is created. If ``axes`` is
    provided, it must contain exactly two matplotlib axes.

    Parameters
    ----------
    parent_folder1, parent_folder2
        Folders containing ``ecm_data_tX.vtk`` files.
    scalar_name
        Name of the point scalar to extract.
    x1, y1, z1
        Coordinates of the first point.
    x2, y2, z2
        Coordinates of the second point.
    tol
        Geometric tolerance used to determine whether a mesh point belongs to the
        line segment.
    cmap
        Matplotlib colormap name.
    linewidth
        Line width for each curve.
    alpha
        Line transparency.
    title1, title2
        Optional titles for the left and right panels.
    axes
        Optional iterable with two axes.

    Returns
    -------
    axes : numpy.ndarray
        Array containing the two axes used for plotting.
    data1 : tuple
        ``(timesteps1, distances1, values_matrix1)``.
    data2 : tuple
        ``(timesteps2, distances2, values_matrix2)``.
    """
    data1 = build_scalar_line_matrix(
        parent_folder=parent_folder1,
        scalar_name=scalar_name,
        x1=x1,
        y1=y1,
        z1=z1,
        x2=x2,
        y2=y2,
        z2=z2,
        tol=tol,
    )
    data2 = build_scalar_line_matrix(
        parent_folder=parent_folder2,
        scalar_name=scalar_name,
        x1=x1,
        y1=y1,
        z1=z1,
        x2=x2,
        y2=y2,
        z2=z2,
        tol=tol,
    )

    timesteps1, distances1, values_matrix1 = data1
    timesteps2, distances2, values_matrix2 = data2

    if axes is None:
        _, axes_obj = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
        axes_arr = axes_obj.ravel()
    else:
        axes_arr = np.asarray(axes, dtype=object).ravel()
        if len(axes_arr) != 2:
            raise ValueError("axes must contain exactly two matplotlib axes.")

    plot_scalar_line_evolution_on_axis(
        ax=axes_arr[0],
        distances=distances1,
        values_matrix=values_matrix1,
        timesteps=timesteps1,
        scalar_name=scalar_name,
        cmap=cmap,
        linewidth=linewidth,
        alpha=alpha,
        title=title1 if title1 is not None else Path(parent_folder1).name,
        show_colorbar=False,
        ylabel=ylabel,
    )
    plot_scalar_line_evolution_on_axis(
        ax=axes_arr[1],
        distances=distances2,
        values_matrix=values_matrix2,
        timesteps=timesteps2,
        scalar_name=scalar_name,
        cmap=cmap,
        linewidth=linewidth,
        alpha=alpha,
        title=title2 if title2 is not None else Path(parent_folder2).name,
        show_colorbar=False,
        ylabel=ylabel,
    )

    return axes_arr, data1, data2


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract scalar profiles along a line from all ecm_data_tX.vtk files "
            "in two folders and plot their evolution side by side."
        )
    )
    parser.add_argument(
        "--parent-folder1",
        required=True,
        help="First folder containing ecm_data_tX.vtk files",
    )
    parser.add_argument(
        "--parent-folder2",
        required=True,
        help="Second folder containing ecm_data_tX.vtk files",
    )
    parser.add_argument(
        "--scalar-name",
        required=True,
        help="Name of the point scalar to extract",
    )
    parser.add_argument("--x1", type=float, required=True, help="First point x")
    parser.add_argument("--y1", type=float, required=True, help="First point y")
    parser.add_argument("--z1", type=float, required=True, help="First point z")
    parser.add_argument("--x2", type=float, required=True, help="Second point x")
    parser.add_argument("--y2", type=float, required=True, help="Second point y")
    parser.add_argument("--z2", type=float, required=True, help="Second point z")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Tolerance for point-line intersection test (default: 1e-8)",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap for time coloring (default: viridis)",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=1.5,
        help="Line width for curves (default: 1.5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Line transparency (default: 0.9)",
    )
    parser.add_argument(
        "--title1",
        default=None,
        help="Optional title for the first panel",
    )
    parser.add_argument(
        "--title2",
        default=None,
        help="Optional title for the second panel",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output image path, for example results/scalar_evolution_comparison.png",
    )
    parser.add_argument(
        "--save-npz1",
        default=None,
        help="Optional .npz path to save data from the first folder",
    )
    parser.add_argument(
        "--save-npz2",
        default=None,
        help="Optional .npz path to save data from the second folder",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    return parser.parse_args()


def main() -> None:
    """Run the scalar-line comparison workflow."""
    args = parse_args()

    print(f"Reading first folder: {args.parent_folder1}")
    print(f"Reading second folder: {args.parent_folder2}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    axes_arr, data1, data2 = plot_scalar_line_evolution_comparison(
        parent_folder1=args.parent_folder1,
        parent_folder2=args.parent_folder2,
        scalar_name=args.scalar_name,
        x1=args.x1,
        y1=args.y1,
        z1=args.z1,
        x2=args.x2,
        y2=args.y2,
        z2=args.z2,
        tol=args.tol,
        cmap=args.cmap,
        linewidth=args.linewidth,
        alpha=args.alpha,
        title1=args.title1,
        title2=args.title2,
        axes=axes.ravel(),
    )

    timesteps1, distances1, values_matrix1 = data1
    timesteps2, distances2, values_matrix2 = data2

    print(
        f"Folder 1: {len(timesteps1)} files, "
        f"{len(distances1)} points along line, matrix shape {values_matrix1.shape}"
    )
    print(
        f"Folder 2: {len(timesteps2)} files, "
        f"{len(distances2)} points along line, matrix shape {values_matrix2.shape}"
    )

    if args.save_npz1 is not None:
        save_path1 = Path(args.save_npz1)
        save_path1.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path1,
            timesteps=timesteps1,
            distances=distances1,
            values_matrix=values_matrix1,
        )
        print(f"Saved first dataset to: {save_path1}")

    if args.save_npz2 is not None:
        save_path2 = Path(args.save_npz2)
        save_path2.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path2,
            timesteps=timesteps2,
            distances=distances2,
            values_matrix=values_matrix2,
        )
        print(f"Saved second dataset to: {save_path2}")

    fig.tight_layout()

    if args.out is not None:
        outpath = Path(args.out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300)
        print(f"Saved figure to: {outpath}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()