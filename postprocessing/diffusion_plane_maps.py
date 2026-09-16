"""Axis-aligned ECM plane extraction and concentration-map rendering."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Plane:
    axis: str
    coordinate: float

    @property
    def label(self) -> str:
        return f"{self.axis}={self.coordinate:g}"

    @property
    def slug(self) -> str:
        return f"{self.axis}_{self.coordinate:g}".replace("-", "m").replace(".", "p").replace("+", "")


def resolve_planes(specifications: list[str]) -> list[Plane]:
    """Parse one requested coordinate plane per direction, in the given order."""
    planes = []
    for specification in specifications:
        spec = specification.lower().replace(" ", "")
        match = re.fullmatch(r"([xyz])=(.+)", spec)
        if match is None:
            raise ValueError(f"Invalid plane '{specification}'; use z=0 or x=-20")
        axis, coordinate = match.groups()
        try:
            value = float(coordinate)
        except ValueError as exc:
            raise ValueError(f"Invalid plane coordinate in '{specification}'") from exc
        if not np.isfinite(value):
            raise ValueError(f"Plane coordinates must be finite: '{specification}'")
        if any(plane.axis == axis for plane in planes):
            raise ValueError(f"Choose only one plane per direction; '{axis}' was specified more than once")
        planes.append(Plane(axis, value))
    if not planes:
        raise ValueError("No ECM grid planes are available")
    return list(dict.fromkeys(planes))


def extract_plane(frame: pd.DataFrame, plane: Plane, variables: list[str]) -> dict:
    """Return one 2D array per scalar, preserving each ECM point's value.

    Rows run along the second free axis and columns along the first (XY for a
    Z slice, XZ for Y, YZ for X). The closest saved plane is selected without
    blending adjacent planes; actual coordinates are returned for labelling.
    """
    levels = np.sort(frame[plane.axis].unique())
    if not len(levels):
        raise ValueError(f"Cannot extract {plane.label} from an empty ECM snapshot")
    if plane.coordinate < levels[0] - 1e-7 or plane.coordinate > levels[-1] + 1e-7:
        raise ValueError(f"Plane {plane.label} is outside the saved {plane.axis} range "
                         f"[{levels[0]:g}, {levels[-1]:g}]")
    actual = float(levels[np.argmin(np.abs(levels - plane.coordinate))])
    section = frame.loc[frame[plane.axis] == actual]
    horizontal, vertical = [axis for axis in AXES if axis != plane.axis]
    x = np.sort(section[horizontal].unique())
    y = np.sort(section[vertical].unique())
    if len(x) < 2 or len(y) < 2 or len(section) != len(x) * len(y):
        raise ValueError(f"Plane {plane.label} requires a complete axis-aligned ECM grid "
                         "with at least two points on each in-plane axis")
    if section.duplicated([horizontal, vertical]).any():
        raise ValueError(f"Duplicate ECM points on plane {plane.label}")
    for coordinates in (x, y):
        spacing = np.diff(coordinates)
        if not np.allclose(spacing, spacing[0], rtol=1e-5, atol=1e-7):
            raise ValueError("Pixel maps require uniformly spaced ECM points along each in-plane axis")
    arrays = {}
    for variable in variables:
        if variable not in section:
            raise ValueError(f"ECM scalar '{variable}' is missing from plane {plane.label}")
        arrays[variable] = section.pivot(index=vertical, columns=horizontal, values=variable).reindex(
            index=y, columns=x).to_numpy(dtype=float)
    # Pixel centers coincide with agent positions; boundaries extend half a grid interval.
    dx, dy = x[1] - x[0], y[1] - y[0]
    return {"actual": actual, "horizontal": horizontal, "vertical": vertical,
            "x": x, "y": y, "extent": (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2),
            "values": arrays}


def draw_map(ax, section, variable, limits, *, smooth=False, cmap="viridis"):
    """Nearest display keeps one matrix element per agent; bilinear smooths display only."""
    image = ax.imshow(np.ma.masked_invalid(section["values"][variable]), origin="lower",
                      extent=section["extent"], aspect="equal", cmap=cmap,
                      vmin=limits[0], vmax=limits[1],
                      interpolation="bilinear" if smooth else "nearest", interpolation_stage="data")
    ax.set_xlabel(section["horizontal"])
    ax.set_ylabel(section["vertical"])
    return image


def map_color_limits(summaries: pd.DataFrame, variables: list[str]) -> dict:
    """Use the same full-ECM range for a species across all planes and pages."""
    limits = {}
    for variable in variables:
        rows = summaries.loc[(summaries["dataset"] == "ecm") & (summaries["variable"] == variable)]
        low, high = rows["min"].min(), rows["max"].max()
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = 0.0, 1.0
        elif low == high:
            padding = max(abs(low) * 0.01, 1e-12)
            low, high = low - padding, high + padding
        limits[variable] = (float(low), float(high))
    return limits


def variable_label(variable: str) -> str:
    match = re.fullmatch(r"concentration_species_(\d+)", variable)
    return f"Species {match.group(1)}" if match else variable


def plot_map_page(samples, rows, limits, *, title, smooth, dt, plt):
    """Each row identifies a (plane, variable), and each column a saved step."""
    fig, axes = plt.subplots(len(rows), len(samples), squeeze=False,
                             figsize=(3.1 * len(samples) + 1, 3.1 * len(rows) + 0.5),
                             layout="constrained")
    artists = {}
    for i, (plane, variable) in enumerate(rows):
        for j, sample in enumerate(samples):
            section = sample["planes"][plane]
            ax = axes[i, j]
            ax.set_facecolor("#dddddd")
            artist = draw_map(ax, section, variable, limits[variable], smooth=smooth)
            artists[variable] = artist
            clock = f"Step {sample['step']}"
            if dt is not None:
                clock += f" | {sample['step'] * dt:g} s"
            if not np.isclose(section["actual"], plane.coordinate, rtol=0, atol=1e-7):
                clock += f"\nsampled {plane.axis}={section['actual']:g}"
            ax.set_title(clock, fontsize=10)
            ax.set_xlabel(f"{section['horizontal']} (length units)")
            if j == 0:
                ax.set_ylabel(f"{variable_label(variable)} | {plane.label}\n{section['vertical']} (length units)")
            else:
                ax.set_ylabel(section["vertical"])
    for variable, artist in artists.items():
        matching_axes = [axes[i, j] for i, (_, v) in enumerate(rows) if v == variable
                         for j in range(len(samples))]
        fig.colorbar(artist, ax=matching_axes, label=f"{variable_label(variable)} concentration", shrink=0.9)
    fig.suptitle(f"{title} | {'bilinear display' if smooth else 'one pixel per ECM agent'}", fontsize=12)
    return fig


def export_plane_maps(args, paths, first_snapshot, variables, summaries, dt, outdir, *, read_snapshot, saved_step):
    """Read at most one page of plane arrays at a time and export every selected step."""
    import json
    import matplotlib.pyplot as plt

    planes = resolve_planes(args.planes or ["x=0", "y=0", "z=0"])
    variables = [variable for variable in variables if variable in first_snapshot]
    limits = map_color_limits(summaries, variables)
    manifest = {"planes": [plane.label for plane in planes], "variables": variables,
                "steps": [saved_step(path) for path in paths], "time_step": dt,
                "layout": args.map_layout, "smooth": args.smooth,
                "columns_per_page": args.map_columns, "color_limits": limits,
                "color_limits_source": "ECM min/max across all selected saved steps",
                "pages": []}
    print(f"ECM maps: {', '.join(plane.label for plane in planes)}; "
          f"{len(variables)} species/scalars, {len(paths)} saved steps.")
    for start in range(0, len(paths), args.map_columns):
        samples = []
        for path in paths[start:start + args.map_columns]:
            frame = first_snapshot if path == paths[0] else read_snapshot(path, "ecm")
            try:
                sections = {plane: extract_plane(frame, plane, variables) for plane in planes}
            except ValueError as exc:
                raise ValueError(f"{path.name}: {exc}") from exc
            samples.append({"step": saved_step(path), "planes": sections})
        if args.map_layout == "by_species":
            groups = [(variable, [(plane, variable) for plane in planes], variable_label(variable))
                      for variable in variables]
        else:
            groups = [(plane.slug, [(plane, variable) for variable in variables], f"Plane {plane.label}")
                      for plane in planes]
        page_number = start // args.map_columns + 1
        for group, rows, title in groups:
            stem = f"diffusion_maps_{args.map_layout}_{group}_page{page_number:03d}"
            fig = plot_map_page(samples, rows, limits, title=f"{title} | page {page_number}",
                                smooth=args.smooth, dt=dt, plt=plt)
            files = []
            for extension in dict.fromkeys(args.formats):
                path = outdir / f"{stem}.{extension}"
                fig.savefig(path, dpi=180)
                files.append(path.name)
                print(f"Saved {path}", flush=True)
            manifest["pages"].append({"files": files,
                                      "steps": [sample["step"] for sample in samples],
                                      "sampled_planes": [{plane.label: sample["planes"][plane]["actual"]
                                                          for plane in planes} for sample in samples]})
            if not args.show:
                plt.close(fig)
    path = outdir / "diffusion_maps.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Map selection and color scales saved to {path}")
