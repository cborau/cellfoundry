
from __future__ import annotations

import os
import re
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Simple: match 't0001' anywhere in filename
_FRAME_RE = re.compile(r"t(\d+)", re.IGNORECASE)

def _extract_frame_int(path: str) -> int:
    base = os.path.basename(path)
    m = _FRAME_RE.search(base)
    if m:
        return int(m.group(1))
    # Fallback: last run of digits
    m2 = re.findall(r"(\d+)", base)
    if not m2:
        raise ValueError(f"Could not extract frame/time index from filename: {base}")
    return int(m2[-1])

def _read_n_numeric_tokens(lines: List[str], start_idx: int, n_tokens: int) -> Tuple[np.ndarray, int]:
    vals: List[float] = []
    idx = start_idx
    while idx < len(lines) and len(vals) < n_tokens:
        parts = lines[idx].strip().split()
        if parts:
            vals.extend(float(p) for p in parts)
        idx += 1
    if len(vals) < n_tokens:
        raise ValueError(f"Unexpected EOF while reading numeric tokens. Needed {n_tokens}, got {len(vals)}.")
    return np.array(vals[:n_tokens], dtype=np.float64), idx

def parse_legacy_vtk_ascii(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Parse legacy VTK ASCII file and return (points, scalars_dict) where:
      - points: (N,3) float64
      - scalars_dict: name -> (N,) or (N,k) float64
    Only reads POINTS and POINT_DATA SCALARS. Ignores VECTORS, CELL_DATA, etc.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    # POINTS
    n_points = None
    points = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.upper().startswith("POINTS"):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POINTS line in {path}: '{line}'")
            n_points = int(parts[1])
            flat, next_i = _read_n_numeric_tokens(lines, i + 1, 3 * n_points)
            points = flat.reshape((n_points, 3))
            i = next_i
            break
        i += 1
    if points is None:
        raise ValueError(f"Could not find POINTS section in {path}")

    # POINT_DATA
    scalars: Dict[str, np.ndarray] = {}
    point_data_n: Optional[int] = None
    while i < len(lines):
        line = lines[i].strip()
        if line.upper().startswith("POINT_DATA"):
            point_data_n = int(line.split()[1])
            i += 1
            break
        i += 1
    if point_data_n is None:
        return points, scalars

    # Parse POINT_DATA blocks
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        u = line.upper()
        if u.startswith("CELL_DATA"):
            break

        if u.startswith("SCALARS"):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed SCALARS line in {path}: '{line}'")
            name = parts[1]
            num_comp = int(parts[3]) if len(parts) >= 4 else 1

            # LOOKUP_TABLE
            i += 1
            if i >= len(lines) or not lines[i].strip().upper().startswith("LOOKUP_TABLE"):
                raise ValueError(f"Expected LOOKUP_TABLE after SCALARS '{name}' in {path}")
            i += 1

            flat, next_i = _read_n_numeric_tokens(lines, i, point_data_n * num_comp)
            arr = flat.reshape((point_data_n, num_comp))
            if num_comp == 1:
                arr = arr[:, 0]
            scalars[name] = arr
            i = next_i
            continue

        if u.startswith("VECTORS"):
            i += 1
            _, next_i = _read_n_numeric_tokens(lines, i, 3 * point_data_n)
            i = next_i
            continue

        i += 1

    # Safety align
    min_n = min(points.shape[0], point_data_n)
    points = points[:min_n, :]
    for k, v in list(scalars.items()):
        scalars[k] = v[:min_n] if hasattr(v, "shape") else v

    return points, scalars

def load_vtk_series(
    folder: str,
    pattern: str,
    *,
    time_step: Optional[float] = None,
    output_interval: Optional[int] = None,
    add_physical_time: bool = True,
) -> pd.DataFrame:
    """
    Load all matching VTK files into one wide DataFrame.
    Rows: (t, point_id). Columns: frame, t, point_id, x,y,z, <scalars...>, and optionally time.
    """
    folder = os.path.abspath(folder)
    paths = glob.glob(os.path.join(folder, pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in folder '{folder}'")
    paths_sorted = sorted(paths, key=_extract_frame_int)

    dfs = []
    for p in paths_sorted:
        frame = _extract_frame_int(p)
        pts, sc = parse_legacy_vtk_ascii(p)
        N = pts.shape[0]
        df = pd.DataFrame({
            "frame": frame,
            "t": frame,
            "point_id": np.arange(N, dtype=np.int64),
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
        })
        for name, arr in sc.items():
            df[name] = arr
        if add_physical_time and (time_step is not None) and (output_interval is not None):
            df["time"] = frame * (float(time_step) * float(output_interval))
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True, copy=False)
    out.sort_values(["t", "point_id"], inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out

def ts_by_point_id(df: pd.DataFrame, *, var: str, point_id: int, use_time_col: str = "t") -> pd.Series:
    if var not in df.columns:
        raise KeyError(f"Variable '{var}' not in DataFrame columns.")
    sub = df[df["point_id"] == int(point_id)].sort_values(use_time_col, kind="mergesort")
    return pd.Series(sub[var].to_numpy(), index=sub[use_time_col].to_numpy(), name=var)

def _nearest_point_id(points_xyz: np.ndarray, x: float, y: float, z: float) -> int:
    target = np.array([x, y, z], dtype=np.float64)
    d2 = np.sum((points_xyz - target[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))

def ts_nearest_point(
    df: pd.DataFrame,
    *,
    var: str,
    x: float,
    y: float,
    z: float,
    mode: str = "per_time",
    reference_time: Optional[float] = None,
    use_time_col: str = "t",
) -> pd.Series:
    """
    mode:
      - 'per_time': pick nearest point independently at each time (dynamic nearest).
      - 'fixed_id': pick nearest point at reference_time (or earliest), then track that point_id.
    """
    if var not in df.columns:
        raise KeyError(f"Variable '{var}' not in DataFrame columns.")
    if mode not in ("per_time", "fixed_id"):
        raise ValueError("mode must be 'per_time' or 'fixed_id'")

    times = np.sort(df[use_time_col].unique())

    if mode == "fixed_id":
        t0 = times[0] if reference_time is None else reference_time
        snap0 = df[df[use_time_col] == t0]
        if snap0.empty:
            raise ValueError(f"reference_time={t0} not present in df[{use_time_col!r}]")
        pid = _nearest_point_id(snap0[["x", "y", "z"]].to_numpy(dtype=np.float64), x, y, z)
        return ts_by_point_id(df, var=var, point_id=pid, use_time_col=use_time_col)

    out_t = []
    out_v = []
    for t in times:
        snap = df[df[use_time_col] == t]
        pid = _nearest_point_id(snap[["x", "y", "z"]].to_numpy(dtype=np.float64), x, y, z)
        v = snap.loc[snap["point_id"] == pid, var].iloc[0]
        out_t.append(t)
        out_v.append(v)
    return pd.Series(np.array(out_v), index=np.array(out_t), name=var)

def plot_vars_over_time(
    dfs,
    vars,
    *,
    point_id=None,
    nearest_xyz=None,
    nearest_mode="fixed_id",
    use_time_col="t",
    title=None,
    subplot_mode="by_dataset",
):
    """
    Plot multiple scalar variables over time.

    Parameters
    - dfs: a single DataFrame OR a dict like {"cells": df_cells, "ecm": df_ecm}
    - vars: list of variable names (strings), e.g. ["concentration_species_0", "concentration_species_1"]
    - point_id: int point_id to track (mutually exclusive with nearest_xyz)
    - nearest_xyz: tuple (x,y,z) to select nearest point (mutually exclusive with point_id)
    - nearest_mode: "per_time" or "fixed_id" (used only if nearest_xyz is provided)
    - use_time_col: "t" or "time" if you added physical time
    - title: optional figure title
    - subplot_mode: "by_dataset" (default) or "by_species"
        * "by_dataset": one subplot per dataset (e.g., cells/ecm), with X lines (vars)
        * "by_species": one subplot per variable, with lines per dataset (e.g., cells/ecm)
    """
    import matplotlib.pyplot as plt
    if isinstance(dfs, dict):
        df_items = list(dfs.items())
    else:
        df_items = [("data", dfs)]

    if (point_id is None) == (nearest_xyz is None):
        raise ValueError("Provide exactly one of point_id or nearest_xyz.")

    if subplot_mode not in ("by_dataset", "by_species"):
        raise ValueError("subplot_mode must be 'by_dataset' or 'by_species'")

    def _get_series(df, v):
        if point_id is not None:
            return ts_by_point_id(df, var=v, point_id=int(point_id), use_time_col=use_time_col)
        x, y, z = nearest_xyz
        return ts_nearest_point(
            df, var=v, x=float(x), y=float(y), z=float(z),
            mode=nearest_mode, use_time_col=use_time_col
        )

    if subplot_mode == "by_dataset":
        n = len(df_items)
        fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(8, 3.5 * n))
        if n == 1:
            axes = [axes]

        for ax, (name, df) in zip(axes, df_items):
            for v in vars:
                s = _get_series(df, v)
                ax.plot(s.index, s.values, label=v)

            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel(use_time_col)
    else:
        n = len(vars)
        fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(8, 3.5 * n))
        if n == 1:
            axes = [axes]

        for ax, v in zip(axes, vars):
            for name, df in df_items:
                s = _get_series(df, v)
                ax.plot(s.index, s.values, label=name)

            ax.set_ylabel(v)
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel(use_time_col)
    if title:
        fig.suptitle(title)
        fig.tight_layout()

    plt.show()