"""
Diagnostic calculations for the radial-glia rosette variant.

The script reads the variant __init__.py extracts PARAMS and configure_globals() assignments, inspects the relevant C++
files when present, and prints equilibrium thresholds plus scenario-specific
parameter suggestions.

Usage examples:
    # Run from cellfoundry with defaults:
    #   config         -> variants/radial_glia/__init__.py
    #   results-pickle -> result_files/output_data_0.pickle
    python variants/radial_glia/rg_rosette_diagnostics.py

    # Explicit config and observed sp2 stats:
    python variants/radial_glia/rg_rosette_diagnostics.py --config variants/radial_glia/__init__.py --observed-max-sp2 0.21 --observed-mean-sp2 0.12

    # Emphasize a scenario and override pickle path:
    python variants/radial_glia/rg_rosette_diagnostics.py --scenario no_rg --results-pickle result_files/output_data_0.pickle

    # Feed final rosette metrics directly (no CSV required):
    python variants/radial_glia/rg_rosette_diagnostics.py --observed-rg-fraction 0.092 --observed-n-alive-rg 46 --observed-largest-cluster-size 8 --observed-mean-rosette-maturity 0.4338 --observed-mean-apz 0.7013 --observed-rg-assembly-compactness 0.6606 --observed-mean-cluster-compactness 0.4856

    # Or load final rosette metrics from a CSV:
    python variants/radial_glia/rg_rosette_diagnostics.py --metrics-csv result_files/RG_ROSETTE_METRICS_OVER_TIME.csv

    # Override phenotype targets used by adaptive comments:
    python variants/radial_glia/rg_rosette_diagnostics.py --target-rg-fraction-min 0.15 --target-largest-cluster-size-min 12 --target-n-rg-clusters-max 5
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "__init__.py"
DEFAULT_RESULTS_PICKLE = PROJECT_ROOT / "result_files" / "output_data_0.pickle"

# Default phenotype targets used by adaptive comments/suggestions.
DEFAULT_TARGETS: Dict[str, float] = {
    "rg_fraction_min": 0.12,
    "largest_cluster_size_min": 12.0,
    "n_rg_clusters_max": 2.0,
    "mean_apz_min": 0.45,
    "mean_apz_max": 0.65,
    "mean_cluster_compactness_min": 0.35,
    "mean_cluster_compactness_max": 0.90,
    "rg_assembly_compactness_min": 0.40,
    "n_rg_type2_min": 5.0,
    "npc_near_rg_threshold_fraction_max": 0.30,
    "mean_rg_commit_level_min": 0.62,
    "rg_committed_fraction_min": 0.12,
    "mean_apz_cells_max": 0.60,
    "mean_rosette_maturity_cells_min": 0.25,
}
# -----------------------------------------------------------------------------
# Parsing utilities
# -----------------------------------------------------------------------------

def _safe_literal(node: ast.AST) -> Any:
    """Return a restricted Python literal, supporting list multiplication."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            key = _safe_literal(k)
            val = _safe_literal(v)
            out[key] = val
        return out
    if isinstance(node, ast.UnaryOp):
        val = _safe_literal(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
    if isinstance(node, ast.BinOp):
        left = _safe_literal(node.left)
        right = _safe_literal(node.right)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Div):
            return left / right
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def read_variant_config(config_path: Path) -> Dict[str, Any]:
    """Read PARAMS and g["..."] assignments from the radial-glia __init__.py."""
    source = config_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(config_path))

    params: Dict[str, Any] = {}
    globals_set: Dict[str, Any] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PARAMS":
                    value = _safe_literal(node.value)
                    if isinstance(value, dict):
                        params.update(value)

        if isinstance(node, ast.FunctionDef) and node.name == "configure_globals":
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                for target in sub.targets:
                    # Match g["KEY"] = literal
                    if isinstance(target, ast.Subscript):
                        is_g = isinstance(target.value, ast.Name) and target.value.id == "g"
                        key = None
                        if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                            key = target.slice.value
                        if is_g and key:
                            value = _safe_literal(sub.value)
                            if value is not None:
                                globals_set[key] = value

    merged = dict(params)
    merged.update(globals_set)
    return merged


def read_optional_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None


def read_rosette_metrics_final(path: Path) -> Dict[str, float]:
    """Read final-row values from a RG_ROSETTE_METRICS_OVER_TIME CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")

    last = rows[-1]
    out: Dict[str, float] = {}
    for k, v in last.items():
        fv = _safe_float(v)
        if fv is not None:
            out[k] = fv
    return out


def _as_records(obj: Any) -> list[Dict[str, Any]]:
    """Convert common tabular payloads (DataFrame/list/dict) to list[dict]."""
    if obj is None:
        return []

    if hasattr(obj, "to_dict"):
        try:
            records = obj.to_dict(orient="records")
            if isinstance(records, list):
                return [r for r in records if isinstance(r, dict)]
        except Exception:
            pass

    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]

    if isinstance(obj, dict):
        keys = list(obj.keys())
        if keys and all(isinstance(obj[k], list) for k in keys):
            n = min((len(obj[k]) for k in keys), default=0)
            rows = []
            for i in range(n):
                rows.append({k: obj[k][i] for k in keys})
            return rows

    return []


def summarize_rosette_metrics(obj: Any) -> Dict[str, float]:
    """Summarize rosette metrics from RG_ROSETTE_METRICS_OVER_TIME-like payload."""
    rows = _as_records(obj)
    if not rows:
        return {}

    def row_value(row: Dict[str, Any], key: str) -> Optional[float]:
        return _safe_float(row.get(key))

    first = rows[0]
    last = rows[-1]
    out: Dict[str, float] = {}

    for key in [
        "rg_fraction",
        "n_alive_rg",
        "n_rg_clusters",
        "largest_cluster_size",
        "mean_rosette_maturity",
        "mean_apz",
        "rg_assembly_compactness",
        "mean_cluster_compactness",
    ]:
        v = row_value(last, key)
        if v is not None:
            out[key] = v

    # Add simple trajectory deltas when possible.
    for key, delta_key in [
        ("n_alive_rg", "delta_n_alive_rg"),
        ("rg_fraction", "delta_rg_fraction"),
        ("largest_cluster_size", "delta_largest_cluster_size"),
        ("mean_cluster_compactness", "delta_mean_cluster_compactness"),
    ]:
        a = row_value(first, key)
        b = row_value(last, key)
        if a is not None and b is not None:
            out[delta_key] = b - a

    return out


def read_results_pickle(path: Path) -> Dict[str, Any]:
    """Load optional simulation pickle and summarize RG cell + rosette metrics."""
    if not path.exists():
        raise FileNotFoundError(f"Results pickle not found: {path}")

    # Ensure modules defined in the project root (e.g., helper_module.py)
    # are importable when unpickling from within variants/radial_glia.
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    with path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        return {}

    # Prefer current key, but tolerate legacy names.
    final_metrics_obj = payload.get("RG_FINAL_METRICS")
    if final_metrics_obj is None:
        final_metrics_obj = payload.get("RG_METRICS")

    rows = _as_records(final_metrics_obj)
    if not rows:
        return {}

    alive = [r for r in rows if int(_safe_float(r.get("dead")) or 0) == 0]
    use = alive if alive else rows
    n = len(use)
    if n == 0:
        return {}

    def mean_of(key: str, filt=None) -> Optional[float]:
        vals = []
        for r in use:
            if filt is not None and not filt(r):
                continue
            v = _safe_float(r.get(key))
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return sum(vals) / len(vals)

    def max_of(key: str, filt=None) -> Optional[float]:
        vals = []
        for r in use:
            if filt is not None and not filt(r):
                continue
            v = _safe_float(r.get(key))
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return max(vals)

    def count_where(pred) -> int:
        return sum(1 for r in use if pred(r))

    n_rg_type2 = count_where(lambda r: int(_safe_float(r.get("cell_type")) or -1) == 2)
    n_npc_type1 = count_where(lambda r: int(_safe_float(r.get("cell_type")) or -1) == 1)
    n_ipsc_type0 = count_where(lambda r: int(_safe_float(r.get("cell_type")) or -1) == 0)
    n_rg_committed = count_where(lambda r: int(_safe_float(r.get("rg_committed")) or 0) == 1)

    cell_summary = {
        "n_alive_rows": float(n),
        "n_rg_type2": float(n_rg_type2),
        "n_npc_type1": float(n_npc_type1),
        "n_ipsc_type0": float(n_ipsc_type0),
        "rg_committed_fraction": (n_rg_committed / n) if n > 0 else 0.0,
        "mean_rg_commit_level": mean_of("rg_commit_level"),
        "mean_epithelialization_level": mean_of("epithelialization_level"),
        "mean_rosette_maturity_cells": mean_of("rosette_maturity"),
        "mean_rg_neighbour_density": mean_of("rg_neighbour_density"),
        "mean_morphogen_local": mean_of("morphogen_local"),
        "max_morphogen_local": max_of("morphogen_local"),
        "mean_apz_cells": mean_of("apz"),
        "mean_apz_rg": mean_of("apz", filt=lambda r: int(_safe_float(r.get("cell_type")) or -1) == 2),
        "mean_apz_npc": mean_of("apz", filt=lambda r: int(_safe_float(r.get("cell_type")) or -1) == 1),
        "npc_near_rg_threshold_fraction": (
            count_where(lambda r: (int(_safe_float(r.get("cell_type")) or -1) == 1) and ((_safe_float(r.get("rg_commit_level")) or 0.0) >= 0.60))
            / max(n_npc_type1, 1)
        ),
    }

    rosette_summary = summarize_rosette_metrics(payload.get("RG_ROSETTE_METRICS_OVER_TIME"))
    return {
        "cell": cell_summary,
        "rosette": rosette_summary,
    }


# -----------------------------------------------------------------------------
# Model calculations
# -----------------------------------------------------------------------------

def arr_get(values: Any, index: int, default: float = 0.0) -> float:
    if isinstance(values, (list, tuple)) and len(values) > index:
        try:
            return float(values[index])
        except Exception:
            return default
    try:
        return float(values)
    except Exception:
        return default


def x_eq_from_drive(drive: float, decay: float, inhibit_rate: float = 0.0, delta: float = 0.0) -> float:
    denom = drive + decay + inhibit_rate * delta
    if denom <= 0.0:
        return 0.0
    return drive / denom


def required_drive_for_threshold(threshold: float, decay: float, inhibit_rate: float = 0.0, delta: float = 0.0) -> float:
    if threshold <= 0.0:
        return 0.0
    if threshold >= 1.0:
        return math.inf
    return (threshold / (1.0 - threshold)) * (decay + inhibit_rate * delta)


def required_sp2_for_threshold(
    threshold: float,
    k_auto: float,
    decay: float,
    inhibit_rate: float = 0.0,
    delta: float = 0.0,
    basal: float = 0.0,
) -> float:
    if k_auto <= 0.0:
        return math.inf
    required_drive = required_drive_for_threshold(threshold, decay, inhibit_rate, delta)
    return max(0.0, (required_drive - basal) / k_auto)


def time_to_threshold_hours(
    x0: float,
    threshold: float,
    drive: float,
    decay: float,
    inhibit_rate: float = 0.0,
    delta: float = 0.0,
) -> Optional[float]:
    """Time to reach threshold for dx/dt = drive*(1-x) - effective_decay*x."""
    effective_decay = decay + inhibit_rate * delta
    rate = drive + effective_decay
    if rate <= 0.0:
        return None
    xeq = drive / rate
    if x0 >= threshold:
        return 0.0
    if xeq <= threshold:
        return None
    ratio = (xeq - threshold) / (xeq - x0)
    if ratio <= 0.0:
        return None
    return -math.log(ratio) / rate / 3600.0


def fmt(x: Any, unit: str = "") -> str:
    if x is None:
        return "never"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isinf(xf):
        return "inf"
    suffix = f" {unit}" if unit else ""
    if abs(xf) >= 1e3 or (abs(xf) < 1e-3 and xf != 0.0):
        return f"{xf:.3e}{suffix}"
    return f"{xf:.4g}{suffix}"


def table(rows: Iterable[Iterable[Any]], headers: Iterable[str]) -> str:
    rows_s = [[str(v) for v in row] for row in rows]
    headers_s = [str(h) for h in headers]
    widths = [len(h) for h in headers_s]
    for row in rows_s:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))
    sep = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers_s)), sep]
    out.extend("  ".join(v.ljust(widths[i]) for i, v in enumerate(row)) for row in rows_s)
    return "\n".join(out)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def inspect_cpp(variant_dir: Path) -> Dict[str, bool]:
    diff_cpp = read_optional_text(variant_dir / "cell_rg_differentiation.cpp")
    pol_cpp = read_optional_text(variant_dir / "cell_rg_polarity_update.cpp")
    move_cpp = read_optional_text(variant_dir / "cell_move.cpp")
    return {
        "basal_only_iPSC": bool(re.search(r"agent_cell_type\s*==\s*0\s*\?\s*RG_COMMIT_RATE\s*:\s*0\.0f", diff_cpp)),
        "has_decay": "RG_COMMIT_DECAY_RATE" in diff_cpp and "rg_commit -= RG_COMMIT_DECAY_RATE" in diff_cpp,
        "has_lateral_inhibition": "RG_COMMIT_INHIBIT_RATE" in diff_cpp and "local_delta_signal" in diff_cpp,
        "positive_paracrine_removed": "RG_COMMIT_PARACRINE" not in diff_cpp,
        "sp2_from_ecm_macro": "C_SP_MACRO[2]" in diff_cpp,
        "ratchet_cell_type": "if (new_type < agent_cell_type)" in diff_cpp,
        "polarity_sp2_gate": "C0 > RG_POLARITY_SP2_THRESHOLD" in pol_cpp,
        "polarity_uses_spatial_messages": "flamegpu::MessageSpatial3D" in pol_cpp and "message_in(" in pol_cpp,
        "polarity_rg_only_centroid": "msg.getVariable<int>(\"cell_type\") != 2" in pol_cpp,
        "polarity_lumen_params_present": all(x in pol_cpp for x in ["RG_LUMEN_BIAS_STRENGTH", "RG_LUMEN_SEARCH_RADIUS", "RG_LUMEN_MIN_NEIGHBOURS"]),
        "rg_orientation_override": "if (agent_cell_type == 2)" in move_cpp and "agent_orx = FLAMEGPU->getVariable<float>(\"apx\")" in move_cpp,
        "apical_bias_uses_epi": "effective_bias = bias_strength * epith" in move_cpp,
    }


def print_diagnostics(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        resolved_cwd = (Path.cwd() / config_path).resolve()
        resolved_script = (SCRIPT_DIR / config_path).resolve()
        config_path = resolved_cwd if resolved_cwd.exists() else resolved_script
    else:
        config_path = config_path.resolve()
    variant_dir = config_path.parent
    p = read_variant_config(config_path)
    cpp = inspect_cpp(variant_dir)

    k_basal = float(p.get("RG_COMMIT_RATE", 5e-6))
    k_auto = float(p.get("RG_COMMIT_AUTOCRINE_RATE", 3e-5))
    k_decay = float(p.get("RG_COMMIT_DECAY_RATE", 1.5e-6))
    k_inhibit = float(p.get("RG_COMMIT_INHIBIT_RATE", 0.0))
    thr_npc = float(p.get("RG_COMMIT_THRESHOLD_NPC", 0.35))
    thr_rg = float(p.get("RG_COMMIT_THRESHOLD_RG", 0.70))
    noise = float(p.get("RG_COMMIT_NOISE", 0.0))
    dt = float(p.get("TIME_STEP", 60.0))
    steps = int(p.get("STEPS", 0))
    c_sat = arr_get(p.get("INIT_ECM_SAT_CONCENTRATION_VALS", [0, 0, 0]), 2)
    c_init_ecm = arr_get(p.get("INIT_ECM_CONCENTRATION_VALS", [0, 0, 0]), 2)
    prod_base = arr_get(p.get("INIT_CELL_PRODUCTION_RATES", [0, 0, 0]), 2)
    prod_mult_npc = arr_get(p.get("CELL_PRODUCTION_MULTIPLIER", [0, 0, 0]), 1)
    prod_mult_rg = arr_get(p.get("CELL_PRODUCTION_MULTIPLIER", [0, 0, 0]), 2)
    d_sp2 = arr_get(p.get("DIFFUSION_COEFF_MULTI", [0, 0, 0]), 2)
    lambda_sp2 = arr_get(p.get("ECM_DEGRADATION_RATE_MULTI", [0, 0, 0]), 2)
    pol_thr = float(p.get("RG_POLARITY_SP2_THRESHOLD", 0.0))
    k_epi = float(p.get("RG_EPITHELIAL_RATE", 0.0))
    apical_z = float(p.get("RG_INTRINSIC_APICAL_Z", 0.0))
    lumen_xy = float(p.get("RG_LUMEN_BIAS_STRENGTH", 0.0))
    lumen_radius = float(p.get("RG_LUMEN_SEARCH_RADIUS", 0.0))
    lumen_min_n = float(p.get("RG_LUMEN_MIN_NEIGHBOURS", 0.0))
    n_cells = int(p.get("N_CELLS", 0))
    cluster_radius = float(p.get("MONOLAYER_CLUSTER_RADIUS", 0.0))

    observed_max = args.observed_max_sp2
    observed_mean = args.observed_mean_sp2

    targets: Dict[str, float] = dict(DEFAULT_TARGETS)
    for k in list(DEFAULT_TARGETS.keys()):
        arg_name = f"target_{k}"
        v = getattr(args, arg_name, None)
        if v is not None:
            targets[k] = float(v)

    observed_metrics: Dict[str, float] = {}
    observed_metrics_source = "none"
    if args.metrics_csv is not None:
        metrics_path = Path(args.metrics_csv)
        if not metrics_path.is_absolute():
            metrics_path = (Path.cwd() / metrics_path).resolve()
        observed_metrics.update(read_rosette_metrics_final(metrics_path))
        observed_metrics_source = f"csv:{metrics_path}"

    # CLI overrides take precedence over CSV final-row values.
    for key, val in [
        ("rg_fraction", args.observed_rg_fraction),
        ("n_alive_rg", args.observed_n_alive_rg),
        ("n_rg_clusters", args.observed_n_rg_clusters),
        ("largest_cluster_size", args.observed_largest_cluster_size),
        ("mean_rosette_maturity", args.observed_mean_rosette_maturity),
        ("mean_apz", args.observed_mean_apz),
        ("rg_assembly_compactness", args.observed_rg_assembly_compactness),
        ("mean_cluster_compactness", args.observed_mean_cluster_compactness),
    ]:
        if val is not None:
            observed_metrics[key] = float(val)

    observed_cell_metrics: Dict[str, Any] = {}
    if args.results_pickle is not None:
        pickle_path = Path(args.results_pickle)
        if not pickle_path.is_absolute():
            pickle_path = (Path.cwd() / pickle_path).resolve()
        try:
            parsed = read_results_pickle(pickle_path)
            observed_cell_metrics = parsed.get("cell", {}) if isinstance(parsed, dict) else {}
            rosette_from_pickle = parsed.get("rosette", {}) if isinstance(parsed, dict) else {}
            if rosette_from_pickle:
                # Fill only missing keys so CSV/CLI can still override.
                for k, v in rosette_from_pickle.items():
                    if k not in observed_metrics:
                        observed_metrics[k] = v
                if observed_metrics_source == "none":
                    observed_metrics_source = f"pickle:{pickle_path}"
        except Exception as exc:
            print(f"Warning: could not parse results pickle ({pickle_path}): {exc}")
            print("Continuing without RG final cell metrics.")
            print()

    print("RG ROSETTE DIAGNOSTICS")
    print("=" * 78)
    print(f"Config: {config_path}")
    print()

    print("Extracted key parameters")
    print(table([
        ("RG_COMMIT_RATE", fmt(k_basal, "1/s")),
        ("RG_COMMIT_AUTOCRINE_RATE", fmt(k_auto, "1/(s*uM)")),
        ("RG_COMMIT_DECAY_RATE", fmt(k_decay, "1/s")),
        ("RG_COMMIT_INHIBIT_RATE", fmt(k_inhibit, "1/s")),
        ("RG_COMMIT_THRESHOLD_NPC", fmt(thr_npc)),
        ("RG_COMMIT_THRESHOLD_RG", fmt(thr_rg)),
        ("RG_COMMIT_NOISE", fmt(noise)),
        ("TIME_STEP", fmt(dt, "s")),
        ("duration", fmt(dt * steps / 3600.0, "h")),
        ("INIT_ECM_SAT_CONCENTRATION_VALS[2]", fmt(c_sat, "uM")),
        ("INIT_ECM_CONCENTRATION_VALS[2]", fmt(c_init_ecm, "uM")),
        ("INIT_CELL_PRODUCTION_RATES[2]", fmt(prod_base, "1/s")),
        ("CELL_PRODUCTION_MULTIPLIER[NPC, RG]", f"{prod_mult_npc:g}, {prod_mult_rg:g}"),
        ("D_sp2", fmt(d_sp2, "um^2/s")),
        ("lambda_sp2", fmt(lambda_sp2, "1/s")),
        ("RG_POLARITY_SP2_THRESHOLD", fmt(pol_thr, "uM")),
        ("RG_LUMEN_BIAS_STRENGTH", fmt(lumen_xy, "1/step")),
        ("RG_LUMEN_SEARCH_RADIUS", fmt(lumen_radius, "um")),
        ("RG_LUMEN_MIN_NEIGHBOURS", fmt(lumen_min_n)),
        ("N_CELLS, cluster_radius", f"{n_cells}, {cluster_radius:g} um"),
    ], headers=("parameter", "value")))
    print()

    print("Active phenotype targets")
    print(table([
        ("rg_fraction >=", fmt(targets["rg_fraction_min"])),
        ("largest_cluster_size >=", fmt(targets["largest_cluster_size_min"])),
        ("n_rg_clusters <=", fmt(targets["n_rg_clusters_max"])),
        ("mean_apz in", f"[{fmt(targets['mean_apz_min'])}, {fmt(targets['mean_apz_max'])}]"),
        ("mean_cluster_compactness in", f"[{fmt(targets['mean_cluster_compactness_min'])}, {fmt(targets['mean_cluster_compactness_max'])}]"),
        ("rg_assembly_compactness >=", fmt(targets["rg_assembly_compactness_min"])),
        ("n_rg_type2 >=", fmt(targets["n_rg_type2_min"])),
        ("npc_near_rg_threshold_fraction <=", fmt(targets["npc_near_rg_threshold_fraction_max"])),
        ("mean_rg_commit_level >=", fmt(targets["mean_rg_commit_level_min"])),
        ("rg_committed_fraction >=", fmt(targets["rg_committed_fraction_min"])),
        ("mean_apz_cells <=", fmt(targets["mean_apz_cells_max"])),
        ("mean_rosette_maturity_cells >=", fmt(targets["mean_rosette_maturity_cells_min"])),
    ], headers=("target", "value")))
    print()

    if observed_metrics:
        source_note = f" ({observed_metrics_source})" if observed_metrics_source != "none" else ""
        print(f"Observed rosette metrics (final frame{source_note})")
        rows = []
        for k in [
            "rg_fraction",
            "n_alive_rg",
            "n_rg_clusters",
            "largest_cluster_size",
            "mean_rosette_maturity",
            "mean_apz",
            "rg_assembly_compactness",
            "mean_cluster_compactness",
            "delta_n_alive_rg",
            "delta_rg_fraction",
            "delta_largest_cluster_size",
            "delta_mean_cluster_compactness",
        ]:
            if k in observed_metrics:
                rows.append((k, fmt(observed_metrics[k])))
        if rows:
            print(table(rows, headers=("metric", "value")))
            print()

    if observed_cell_metrics:
        print("Observed RG final cell metrics summary")
        rows = []
        for k in [
            "n_alive_rows",
            "n_ipsc_type0",
            "n_npc_type1",
            "n_rg_type2",
            "rg_committed_fraction",
            "mean_rg_commit_level",
            "mean_epithelialization_level",
            "mean_rosette_maturity_cells",
            "mean_rg_neighbour_density",
            "mean_morphogen_local",
            "max_morphogen_local",
            "mean_apz_cells",
            "mean_apz_npc",
            "mean_apz_rg",
            "npc_near_rg_threshold_fraction",
        ]:
            if k in observed_cell_metrics and observed_cell_metrics[k] is not None:
                rows.append((k, fmt(observed_cell_metrics[k])))
        if rows:
            print(table(rows, headers=("metric", "value")))
            print()

    print("C++ logic checks")
    for key, value in cpp.items():
        print(f"  {key:28s}: {'yes' if value else 'no'}")
    print()

    # Equilibrium thresholds.
    print("Equilibrium thresholds from the actual C++ ODE")
    print("Formula: x_eq = drive / (drive + decay + inhibit_rate * delta)")
    print("Important: basal drive is applied only while cell_type == 0 if basal_only_iPSC=yes.")
    print()

    basal_xeq_no_sp2 = x_eq_from_drive(k_basal, k_decay)
    t_to_npc_basal = time_to_threshold_hours(0.0, thr_npc, k_basal, k_decay)
    print(f"iPSC with no sp2: x_eq = {fmt(basal_xeq_no_sp2)}; time to NPC threshold = {fmt(t_to_npc_basal, 'h')}")
    if cpp["basal_only_iPSC"]:
        print("After NPC transition, basal is switched off, so further progression requires sp2.")
    print()

    rows = []
    for delta in [0.0, 0.05, 0.09, 0.12, 0.20, 0.30]:
        sp2_npc_from_npc = required_sp2_for_threshold(thr_npc, k_auto, k_decay, k_inhibit, delta, basal=0.0)
        sp2_rg_from_npc = required_sp2_for_threshold(thr_rg, k_auto, k_decay, k_inhibit, delta, basal=0.0)
        sp2_rg_from_ipsc = required_sp2_for_threshold(thr_rg, k_auto, k_decay, k_inhibit, delta, basal=k_basal)
        rows.append((fmt(delta), fmt(sp2_npc_from_npc, "uM"), fmt(sp2_rg_from_npc, "uM"), fmt(sp2_rg_from_ipsc, "uM")))
    print(table(rows, headers=("delta_signal", "NPC maintenance from NPC", "RG from NPC", "RG while basal active")))
    print()

    candidate_sp2 = [0.0, pol_thr, 0.15, 0.20, 0.21, 0.25, 0.30, 0.35, 0.40, c_sat]
    if observed_max is not None:
        candidate_sp2.append(observed_max)
    if observed_mean is not None:
        candidate_sp2.append(observed_mean)
    # Preserve order while removing near-duplicates.
    unique_sp2 = []
    for c in candidate_sp2:
        if c is None or c < 0:
            continue
        if all(abs(c - old) > 1e-9 for old in unique_sp2):
            unique_sp2.append(c)

    rows = []
    for c in unique_sp2:
        drive_npc = k_auto * c
        xeq_npc = x_eq_from_drive(drive_npc, k_decay)
        t_rg = time_to_threshold_hours(thr_npc, thr_rg, drive_npc, k_decay)
        label = ""
        if observed_max is not None and abs(c - observed_max) < 1e-9:
            label = "observed max"
        elif observed_mean is not None and abs(c - observed_mean) < 1e-9:
            label = "observed mean"
        elif abs(c - c_sat) < 1e-9:
            label = "C_sat"
        elif abs(c - pol_thr) < 1e-9:
            label = "polarity gate"
        rows.append((fmt(c, "uM"), label, fmt(xeq_npc), fmt(t_rg, "h")))
    print("NPC-to-RG behaviour if local sp2 were held constant")
    print(table(rows, headers=("local_sp2", "note", "x_eq for NPC/RG", "time 0.35 -> 0.70")))
    print()

    if d_sp2 > 0.0 and lambda_sp2 > 0.0:
        L = math.sqrt(d_sp2 / lambda_sp2)
        half_life = math.log(2.0) / lambda_sp2 / 3600.0
        print(f"sp2 diffusion length sqrt(D/lambda) = {fmt(L, 'um')}; degradation half-life = {fmt(half_life, 'h')}")
        print()

    # Primary diagnosis.
    threshold_no_inhib = required_sp2_for_threshold(thr_rg, k_auto, k_decay, k_inhibit, 0.0, basal=0.0)
    threshold_delta_012 = required_sp2_for_threshold(thr_rg, k_auto, k_decay, k_inhibit, 0.12, basal=0.0)
    print("Primary diagnosis")
    if cpp["basal_only_iPSC"]:
        print(
            f"  The first iPSC-to-NPC step can occur by basal drift, but NPC-to-RG requires "
            f"local_sp2 >= {fmt(threshold_no_inhib, 'uM')} even with no lateral inhibition."
        )
    else:
        print(
            f"  With basal active in all states, the no-inhibition RG threshold would be "
            f"{fmt(required_sp2_for_threshold(thr_rg, k_auto, k_decay, basal=k_basal), 'uM')}."
        )
    print(
        f"  With a mature inhibitory neighbourhood around delta=0.12, the RG threshold rises to "
        f"{fmt(threshold_delta_012, 'uM')}."
    )
    if observed_max is not None:
        xeq_obs = x_eq_from_drive(k_auto * observed_max, k_decay)
        if observed_max < threshold_no_inhib:
            print(
                f"  Observed max_sp2={fmt(observed_max, 'uM')} gives x_eq={fmt(xeq_obs)}, "
                "below the RG threshold. This explains no RG cells."
            )
        else:
            print(
                f"  Observed max_sp2={fmt(observed_max, 'uM')} is above the no-inhibition threshold. "
                "If no RG cells appear, inspect time-to-threshold, inhibition, and whether cells actually sample that voxel."
            )
    else:
        proxy_max = observed_cell_metrics.get("max_morphogen_local") if observed_cell_metrics else None
        proxy_mean = observed_cell_metrics.get("mean_morphogen_local") if observed_cell_metrics else None
        if proxy_max is not None:
            xeq_proxy = x_eq_from_drive(k_auto * proxy_max, k_decay)
            if proxy_max < threshold_no_inhib:
                print(
                    f"  No explicit observed_max_sp2 provided; using max_morphogen_local proxy={fmt(proxy_max, 'uM')} from RG_FINAL_METRICS. "
                    f"This gives x_eq={fmt(xeq_proxy)}, below RG threshold."
                )
            else:
                print(
                    f"  No explicit observed_max_sp2 provided; using max_morphogen_local proxy={fmt(proxy_max, 'uM')} from RG_FINAL_METRICS. "
                    f"This is above no-inhibition threshold (x_eq={fmt(xeq_proxy)})."
                )
        elif proxy_mean is not None:
            xeq_proxy = x_eq_from_drive(k_auto * proxy_mean, k_decay)
            print(
                f"  No explicit observed_max_sp2 provided; only mean_morphogen_local={fmt(proxy_mean, 'uM')} is available from RG_FINAL_METRICS "
                f"(x_eq at mean={fmt(xeq_proxy)})."
            )
        else:
            print(
                "  No observed sp2 metric is available from inputs/pickle; threshold diagnosis uses analytic bounds only. "
                "Provide --observed-max-sp2 to include direct field-peak validation."
            )
    print()

    # Required parameter values for observed maxima or useful reference targets.
    reference_peaks = []
    if observed_max is not None and observed_max > 0.0:
        reference_peaks.append(("observed max", observed_max))
    for c in [0.20, 0.21, 0.25, 0.30, c_sat]:
        if c > 0.0 and all(abs(c - old_c) > 1e-9 for _, old_c in reference_peaks):
            reference_peaks.append((f"peak {c:g} uM", c))

    rows = []
    for name, c in reference_peaks:
        k_auto_req = required_drive_for_threshold(thr_rg, k_decay) / c if c > 0.0 else math.inf
        decay_req = (1.0 - thr_rg) / thr_rg * (k_auto * c)
        thr_if_current = x_eq_from_drive(k_auto * c, k_decay)
        rows.append((name, fmt(c, "uM"), fmt(k_auto_req, "1/(s*uM)"), fmt(decay_req, "1/s"), fmt(thr_if_current)))
    print("Parameter targets to make a given sp2 peak just reach RG with no inhibition")
    print(table(rows, headers=("assumed peak", "sp2", "needed k_auto", "max decay at current k_auto", "current x_eq")))
    print()

    # Scenario suggestions.
    print("Scenario-specific suggestions")
    print("- no_rg:")
    print(f"  1) Lower the NPC-to-RG sp2 threshold. Current no-inhibition threshold is {fmt(threshold_no_inhib, 'uM')}; this is high relative to expected peaks around 0.20-0.25 uM.")
    print("  2) Prefer one clean fate-side change first: set RG_COMMIT_AUTOCRINE_RATE to about 3e-5 if max_sp2 is near 0.21 uM, or reduce RG_COMMIT_DECAY_RATE to about 1.5e-6.")
    print("  3) If max_sp2 is genuinely below 0.20 uM, increase sp2 availability instead: raise INIT_CELL_PRODUCTION_RATES[2], raise INIT_ECM_SAT_CONCENTRATION_VALS[2], reduce ECM_DEGRADATION_RATE_MULTI[2], or seed a tiny NPC fraction for debugging.")
    print("  4) Temporarily set RG_COMMIT_INHIBIT_RATE=0 only as a diagnostic. It should not block the first RG when there are no RG neighbours, but it can block expansion after the first RG appears.")
    print("- too_many_rosettes_or_all_RG:")
    print("  Reduce nucleation and spreading: lower RG_COMMIT_NOISE, lower RG_COMMIT_RATE, lower k_auto or sp2 production, increase RG_COMMIT_DECAY_RATE, increase RG_COMMIT_THRESHOLD_RG, or increase RG_COMMIT_INHIBIT_RATE after confirming apz rises.")
    print("- too_few_rosettes:")
    print("  Increase nucleation heterogeneity: increase RG_COMMIT_NOISE slightly, increase k_auto, increase NPC sp2 production, reduce decay, or lower RG_COMMIT_THRESHOLD_RG modestly. If only one central rosette forms, check whether sp2 diffusion/degradation is too local.")
    print("- rosettes_flat_after_RG_exists:")
    print("  Fate is no longer the bottleneck. Increase RG_EPITHELIAL_RATE or RG_APICAL_BIAS_RG, reduce RG_SUBSTRATE_K if z motion is over-constrained, and check that C0 exceeds RG_POLARITY_SP2_THRESHOLD so apz can align.")
    print("- cells_RGs_but_no_pattern:")
    print("  Strengthen patterning, not commitment: increase RG_COMMIT_INHIBIT_RATE, ensure RG_INTRINSIC_APICAL_Z drives |apz| upward, and keep the old positive paracrine term removed.")
    print()

    # Phenotype-aware suggestions from observed final metrics.
    if observed_metrics:
        print("Phenotype-aware suggestions from observed metrics")

        rg_fraction = observed_metrics.get("rg_fraction")
        n_alive_rg = observed_metrics.get("n_alive_rg")
        n_clusters = observed_metrics.get("n_rg_clusters")
        largest_cluster = observed_metrics.get("largest_cluster_size")
        mean_apz_obs = observed_metrics.get("mean_apz")
        compact_all = observed_metrics.get("rg_assembly_compactness")
        compact_cluster = observed_metrics.get("mean_cluster_compactness")

        if rg_fraction is not None and rg_fraction < targets["rg_fraction_min"]:
            print(
                f"  - Low RG abundance (rg_fraction={fmt(rg_fraction)} < target {fmt(targets['rg_fraction_min'])}): increase commitment drive first, "
                "e.g. raise RG_COMMIT_AUTOCRINE_RATE by +15-35% or reduce RG_COMMIT_DECAY_RATE by 15-30%."
            )

        if largest_cluster is not None and largest_cluster < targets["largest_cluster_size_min"]:
            print(
                f"  - Largest cluster is small (largest_cluster_size={fmt(largest_cluster)} < target {fmt(targets['largest_cluster_size_min'])}): strengthen aggregation and reduce over-fragmentation, "
                "e.g. increase RG_ADHESION_MATRIX[RG,RG] by +10-25% and/or reduce RG_COMMIT_INHIBIT_RATE by 10-20%."
            )

        if n_clusters is not None and n_clusters > targets["n_rg_clusters_max"]:
            print(
                f"  - Many RG clusters detected (n_rg_clusters={fmt(n_clusters)} > target {fmt(targets['n_rg_clusters_max'])}): dampen nucleation while improving cluster growth, "
                "e.g. reduce RG_COMMIT_NOISE by 10-25% and increase RG_LUMEN_SEARCH_RADIUS (about +5 to +10 um)."
            )

        if mean_apz_obs is not None and mean_apz_obs > targets["mean_apz_max"]:
            print(
                f"  - mean_apz is already high (mean_apz={fmt(mean_apz_obs)} > target max {fmt(targets['mean_apz_max'])}): polarity-Z is not the bottleneck. "
                "Prioritize fate/adhesion/lumen-XY parameters over increasing RG_INTRINSIC_APICAL_Z."
            )
        elif mean_apz_obs is not None and mean_apz_obs < targets["mean_apz_min"]:
            print(
                f"  - mean_apz is low (mean_apz={fmt(mean_apz_obs)} < target min {fmt(targets['mean_apz_min'])}): increase RG_INTRINSIC_APICAL_Z and/or lower RG_POLARITY_SP2_THRESHOLD slightly."
            )

        if compact_cluster is not None and compact_cluster < targets["mean_cluster_compactness_min"]:
            print(
                f"  - Cluster compactness is low (mean_cluster_compactness={fmt(compact_cluster)} < target min {fmt(targets['mean_cluster_compactness_min'])}): increase RG_LUMEN_BIAS_STRENGTH by +20-50% "
                "and consider lowering RG_LUMEN_MIN_NEIGHBOURS from 2 to 1 for earlier local centering."
            )
        elif compact_cluster is not None and compact_cluster > targets["mean_cluster_compactness_max"]:
            print(
                f"  - Cluster compactness is very high (mean_cluster_compactness={fmt(compact_cluster)} > target max {fmt(targets['mean_cluster_compactness_max'])}): avoid over-collapse by reducing RG_LUMEN_BIAS_STRENGTH 10-25%."
            )

        if (
            compact_all is not None
            and compact_all < targets["rg_assembly_compactness_min"]
            and compact_cluster is not None
            and compact_cluster >= targets["mean_cluster_compactness_min"]
        ):
            print(
                "  - Individual clusters are compact but global assembly is elongated: limit new distant nucleation "
                "(lower RG_COMMIT_NOISE or slightly raise RG_COMMIT_THRESHOLD_RG)."
            )

        if n_alive_rg is not None and n_cells > 0:
            rg_pct = 100.0 * n_alive_rg / n_cells
            print(f"  - Current RG count is {fmt(n_alive_rg)} / {n_cells} ({fmt(rg_pct, '%')}).")

        delta_rg = observed_metrics.get("delta_n_alive_rg")
        if delta_rg is not None:
            if delta_rg > 0:
                print(f"  - RG pool is still expanding over the run (delta_n_alive_rg={fmt(delta_rg)}).")
            elif delta_rg < 0:
                print(f"  - RG pool contracted over the run (delta_n_alive_rg={fmt(delta_rg)}); investigate stability/survival.")

        delta_comp = observed_metrics.get("delta_mean_cluster_compactness")
        if delta_comp is not None:
            if delta_comp > 0:
                print(f"  - Cluster compactness improved over time (delta_mean_cluster_compactness={fmt(delta_comp)}).")
            elif delta_comp < 0:
                print(f"  - Cluster compactness degraded over time (delta_mean_cluster_compactness={fmt(delta_comp)}).")

        print()

    if observed_cell_metrics:
        print("Cell-level suggestions from RG final cell metrics")
        n_rg_type2 = observed_cell_metrics.get("n_rg_type2")
        mean_commit = observed_cell_metrics.get("mean_rg_commit_level")
        near_thr = observed_cell_metrics.get("npc_near_rg_threshold_fraction")
        mean_apz_cells = observed_cell_metrics.get("mean_apz_cells")
        mean_rosette_cells = observed_cell_metrics.get("mean_rosette_maturity_cells")
        rg_comm_frac = observed_cell_metrics.get("rg_committed_fraction")

        if n_rg_type2 is not None and n_rg_type2 < targets["n_rg_type2_min"]:
            print(
                f"  - Very few type-2 RG cells in final snapshot (n_rg_type2={fmt(n_rg_type2)} < target {fmt(targets['n_rg_type2_min'])}): increase commitment progression "
                "(raise RG_COMMIT_AUTOCRINE_RATE by ~15-30% and/or reduce RG_COMMIT_DECAY_RATE by ~10-25%)."
            )

        if near_thr is not None and near_thr > targets["npc_near_rg_threshold_fraction_max"]:
            print(
                f"  - Many NPCs are near RG threshold (npc_near_rg_threshold_fraction={fmt(near_thr)} > target {fmt(targets['npc_near_rg_threshold_fraction_max'])}): system is threshold-limited. "
                "Try lowering RG_COMMIT_THRESHOLD_RG slightly (e.g. 0.70 -> 0.66-0.68) or modestly increasing k_auto."
            )

        if mean_commit is not None and mean_commit < targets["mean_rg_commit_level_min"]:
            print(
                f"  - Mean commit is still low (mean_rg_commit_level={fmt(mean_commit)} < target {fmt(targets['mean_rg_commit_level_min'])}): prioritize fate-side tuning before polarity tuning."
            )

        if rg_comm_frac is not None and rg_comm_frac < targets["rg_committed_fraction_min"]:
            print(
                f"  - Low rg_committed fraction (rg_committed_fraction={fmt(rg_comm_frac)} < target {fmt(targets['rg_committed_fraction_min'])}): avoid increasing inhibition now; first ensure cells cross commitment threshold consistently."
            )

        if mean_apz_cells is not None and mean_apz_cells > targets["mean_apz_cells_max"]:
            print(
                f"  - Mean apz across cells is already substantial (mean_apz_cells={fmt(mean_apz_cells)} > target {fmt(targets['mean_apz_cells_max'])}): do not increase RG_INTRINSIC_APICAL_Z as first intervention."
            )

        if mean_rosette_cells is not None and mean_rosette_cells < targets["mean_rosette_maturity_cells_min"]:
            print(
                f"  - Low per-cell rosette maturity (mean_rosette_maturity_cells={fmt(mean_rosette_cells)} < target {fmt(targets['mean_rosette_maturity_cells_min'])}): strengthen local centering by raising RG_LUMEN_BIAS_STRENGTH (~20-40%) "
                "or lowering RG_LUMEN_MIN_NEIGHBOURS (2 -> 1) for earlier cue activation."
            )

        print()

    if args.scenario != "all":
        print(f"Requested scenario: {args.scenario}")
        print("The scenario-specific section above includes the targeted actions.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose RG rosette commitment thresholds from variant files.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to variants/radial_glia/__init__.py")
    parser.add_argument("--observed-max-sp2", type=float, default=None, help="Observed maximum extracellular sp2 from a run/debug log [uM]")
    parser.add_argument("--observed-mean-sp2", type=float, default=None, help="Observed mean extracellular sp2 from a run/debug log [uM]")
    parser.add_argument(
        "--results-pickle",
        type=Path,
        default=DEFAULT_RESULTS_PICKLE,
        help="Results pickle containing RG_FINAL_METRICS (default: result_files/output_data_0.pickle)",
    )
    parser.add_argument("--metrics-csv", type=Path, default=None, help="Optional CSV file for RG_ROSETTE_METRICS_OVER_TIME; final row is used")
    parser.add_argument("--observed-rg-fraction", type=float, default=None, help="Final rg_fraction")
    parser.add_argument("--observed-n-alive-rg", type=float, default=None, help="Final n_alive_rg")
    parser.add_argument("--observed-n-rg-clusters", type=float, default=None, help="Final n_rg_clusters")
    parser.add_argument("--observed-largest-cluster-size", type=float, default=None, help="Final largest_cluster_size")
    parser.add_argument("--observed-mean-rosette-maturity", type=float, default=None, help="Final mean_rosette_maturity")
    parser.add_argument("--observed-mean-apz", type=float, default=None, help="Final mean_apz")
    parser.add_argument("--observed-rg-assembly-compactness", type=float, default=None, help="Final rg_assembly_compactness")
    parser.add_argument("--observed-mean-cluster-compactness", type=float, default=None, help="Final mean_cluster_compactness")
    parser.add_argument("--target-rg-fraction-min", type=float, default=None, help="Target minimum rg_fraction")
    parser.add_argument("--target-largest-cluster-size-min", type=float, default=None, help="Target minimum largest_cluster_size")
    parser.add_argument("--target-n-rg-clusters-max", type=float, default=None, help="Target maximum n_rg_clusters")
    parser.add_argument("--target-mean-apz-min", type=float, default=None, help="Target minimum mean_apz")
    parser.add_argument("--target-mean-apz-max", type=float, default=None, help="Target maximum mean_apz")
    parser.add_argument("--target-mean-cluster-compactness-min", type=float, default=None, help="Target minimum mean_cluster_compactness")
    parser.add_argument("--target-mean-cluster-compactness-max", type=float, default=None, help="Target maximum mean_cluster_compactness")
    parser.add_argument("--target-rg-assembly-compactness-min", type=float, default=None, help="Target minimum rg_assembly_compactness")
    parser.add_argument("--target-n-rg-type2-min", type=float, default=None, help="Target minimum n_rg_type2")
    parser.add_argument("--target-npc-near-rg-threshold-fraction-max", type=float, default=None, help="Target maximum npc_near_rg_threshold_fraction")
    parser.add_argument("--target-mean-rg-commit-level-min", type=float, default=None, help="Target minimum mean_rg_commit_level")
    parser.add_argument("--target-rg-committed-fraction-min", type=float, default=None, help="Target minimum rg_committed_fraction")
    parser.add_argument("--target-mean-apz-cells-max", type=float, default=None, help="Target maximum mean_apz_cells")
    parser.add_argument("--target-mean-rosette-maturity-cells-min", type=float, default=None, help="Target minimum mean_rosette_maturity_cells")
    parser.add_argument(
        "--scenario",
        choices=["all", "no_rg", "too_many_rosettes", "too_few_rosettes", "flat_rosettes", "no_pattern"],
        default="all",
        help="Scenario to emphasize in the final suggestions",
    )
    args = parser.parse_args()
    print_diagnostics(args)


if __name__ == "__main__":
    main()
