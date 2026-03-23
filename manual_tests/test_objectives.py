"""
Manual objective checker and smoke-test harness.

Run from the repository root:

  List supported objectives and aliases:
    python manual_tests/test_objectives.py --list-objectives

  Synthetic smoke tests for the built-in objectives:
    python manual_tests/test_objectives.py

  Real data - normal stress-strain:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --reference optimizer/reference_data/target_stress_strain.csv --objective stress_strain

  Real data - shear stress-strain:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --reference optimizer/reference_data/target_shear.csv --objective shear_stress_strain --strain-axis 1 --shear-component 0

  Real data - differential modulus:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --reference optimizer/reference_data/target_differential_modulus.csv --objective diff_modulus --strain-axis 1

  Real data - cell speeds:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --reference optimizer/reference_data/target_cell_speed.csv --objective cell_speed --population-stat median

  Real data - final cell count using a scalar target:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --objective final_cell_count --target-cell-count 24

  Real data - organoid size using a scalar target:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --objective organoid_size --trial-dir result_files/trial_0001 --metric radius_of_gyration --target-size 140

Without --pickle the script runs the synthetic smoke tests only.
With --pickle it evaluates the chosen objective and shows objective-specific
diagnostics. Plots are enabled by default; disable with --no-plot.
"""

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

# Ensure the repo root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.objectives import (
    OBJECTIVE_REGISTRY,
    _compute_differential_modulus,
    _extract_sim_strain_stress,
    _filter_simulation_from_min_strain,
    _get_cell_speed_metrics_frame,
    _get_last_cell_vtk,
    _interpolate_response_to_reference_x,
    _interpolate_to_match,
    _read_cell_positions_from_vtk,
    _reduce_population_values,
    _smooth_signal_savgol,
    compute_organoid_metrics,
)


OBJECTIVE_ALIASES = {
    "stress_strain": "stress_strain_curve_error",
    "shear_stress_strain": "shear_stress_strain_curve_error",
    "diff_modulus": "differential_modulus_error",
    "shear_diff_modulus": "shear_differential_modulus_error",
    "boundary_force": "boundary_force_curve_error",
    "cell_population": "cell_population_error",
    "focad_attached_ratio": "focad_attached_ratio_error",
    "poisson_ratio": "poisson_ratio_error",
    "matrix_remodeling": "matrix_remodeling_error",
    "final_cell_count": "final_cell_count_error",
    "final_focad_per_cell": "final_focad_per_cell_error",
    "cell_speed": "cell_speed_error",
    "organoid_size": "organoid_size_error",
}

CURVE_OBJECTIVES = {
    "stress_strain_curve_error",
    "shear_stress_strain_curve_error",
    "differential_modulus_error",
    "shear_differential_modulus_error",
}

TIME_SERIES_OBJECTIVES = {
    "boundary_force_curve_error",
    "cell_population_error",
    "focad_attached_ratio_error",
    "matrix_remodeling_error",
    "poisson_ratio_error",
}

SCALAR_OPTIONAL_REFERENCE_OBJECTIVES = {
    "poisson_ratio_error",
    "final_cell_count_error",
    "final_focad_per_cell_error",
    "organoid_size_error",
}

ALL_OBJECTIVE_CHOICES = sorted(set(OBJECTIVE_REGISTRY) | set(OBJECTIVE_ALIASES))


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_reference_stress_strain(tmp: Path) -> str:
    strains = np.linspace(0.0, 0.10, 20)
    stresses = 5.0 * strains
    path = tmp / "ref_stress_strain.csv"
    pd.DataFrame({"strain": strains, "stress": stresses}).to_csv(path, index=False)
    return str(path)


def _make_reference_diff_modulus(tmp: Path) -> str:
    strains = np.linspace(0.0, 0.10, 20)
    modulus = np.full_like(strains, 5.0)
    path = tmp / "ref_diff_modulus.csv"
    pd.DataFrame({"strain": strains, "differential_modulus": modulus}).to_csv(path, index=False)
    return str(path)


def _make_reference_cell_speed(tmp: Path) -> str:
    path = tmp / "ref_cell_speed.csv"
    pd.DataFrame(
        {
            "cell_type": [0, 1],
            "target_vmean": [1.2, 0.75],
            "target_veff": [0.9, 0.5],
        }
    ).to_csv(path, index=False)
    return str(path)


def _poisson_ratio_to_series(values) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            raise ValueError("POISSON_RATIO_OVER_TIME DataFrame has no columns")
        return values.iloc[:, 0].astype(float)
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(np.asarray(values, dtype=float), name="poisson_ratio")


def _make_mock_results(n_steps: int = 50) -> dict:
    xpos = np.linspace(500.0, 550.0, n_steps)
    xneg = np.full(n_steps, -500.0)
    ypos = np.full(n_steps, 500.0)
    yneg = np.full(n_steps, -500.0)
    zpos = np.full(n_steps, 500.0)
    zneg = np.full(n_steps, -500.0)

    bpos = pd.DataFrame(
        {
            "xpos": xpos,
            "xneg": xneg,
            "ypos": ypos,
            "yneg": yneg,
            "zpos": zpos,
            "zneg": zneg,
        }
    )

    strain = ((xpos - xneg) - 1000.0) / 1000.0
    bforce = pd.DataFrame(
        {
            "fxpos": 5e6 * strain,
            "fxneg": np.zeros(n_steps),
            "fypos": 1e5 * np.linspace(0.0, 1.0, n_steps),
            "fyneg": 1e5 * np.linspace(1.0, 0.0, n_steps),
            "fzpos": np.zeros(n_steps),
            "fzneg": np.zeros(n_steps),
        }
    )

    bforce_shear = pd.DataFrame(
        {
            "fxpos_y": 3e6 * np.linspace(0.0, 0.1, n_steps),
            "fxpos_z": np.zeros(n_steps),
            "fxneg_y": np.zeros(n_steps),
            "fxneg_z": np.zeros(n_steps),
            "fypos_x": 2e6 * np.linspace(0.0, 0.1, n_steps),
            "fypos_z": np.zeros(n_steps),
            "fyneg_x": np.zeros(n_steps),
            "fyneg_z": np.zeros(n_steps),
            "fzpos_x": np.zeros(n_steps),
            "fzpos_y": np.zeros(n_steps),
            "fzneg_x": np.zeros(n_steps),
            "fzneg_y": np.zeros(n_steps),
        }
    )

    alive_type_0 = np.linspace(4, 7, n_steps).round().astype(int)
    alive_type_1 = np.linspace(2, 4, n_steps).round().astype(int)
    n_alive = alive_type_0 + alive_type_1

    cell_metrics = pd.DataFrame(
        {
            "time": np.arange(n_steps),
            "n_cells_alive": n_alive,
            "n_alive_type_0": alive_type_0,
            "n_alive_type_1": alive_type_1,
        }
    )

    focad_metrics = pd.DataFrame(
        {
            "time": np.arange(n_steps),
            "attached_ratio": np.linspace(0.2, 0.8, n_steps),
            "total": np.linspace(12, 36, n_steps).round().astype(int),
        }
    )

    fnode_metrics = pd.DataFrame(
        {
            "time": np.arange(n_steps),
            "n_fnodes_total": np.linspace(500, 520, n_steps),
            "sum_degradation": np.linspace(0.0, 5.0, n_steps),
            "sum_reinforcement": np.linspace(0.0, 3.0, n_steps),
            "mean_elastic_energy": np.linspace(1.0, 2.0, n_steps),
            "net_remodeling_total": np.linspace(-1.0, 2.0, n_steps),
        }
    )

    return {
        "BPOS_OVER_TIME": bpos,
        "BFORCE_OVER_TIME": bforce,
        "BFORCE_SHEAR_OVER_TIME": bforce_shear,
        "BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME": pd.DataFrame(
            {
                "n_bx_pos": np.full(n_steps, 20.0),
                "n_bx_neg": np.full(n_steps, 20.0),
                "n_by_pos": np.full(n_steps, 18.0),
                "n_by_neg": np.full(n_steps, 18.0),
                "n_bz_pos": np.full(n_steps, 16.0),
                "n_bz_neg": np.full(n_steps, 16.0),
            }
        ),
        "FIBRE_SECTION_AREA_UM2": 0.05,
        "CELL_SPEED_METRICS": pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "cell_type": [0, 0, 1, 1],
                "dead": [0, 0, 0, 0],
                "trajectory_time": [10.0, 10.0, 8.0, 8.0],
                "trajectory_length": [10.0, 14.0, 4.0, 8.0],
                "effective_displacement": [8.0, 10.0, 2.0, 6.0],
                "vmean": [1.0, 1.4, 0.5, 1.0],
                "veff": [0.8, 1.0, 0.25, 0.75],
            }
        ),
        "CELL_METRICS_OVER_TIME": cell_metrics,
        "FOCAD_METRICS_OVER_TIME": focad_metrics,
        "FNODE_METRICS_OVER_TIME": fnode_metrics,
        "POISSON_RATIO_OVER_TIME": pd.DataFrame(
            {
                "poisson_ratio": np.linspace(0.1, 0.3, n_steps),
            }
        ),
    }


def _make_reference_boundary_force(tmp: Path, results: dict) -> str:
    bforce = results["BFORCE_OVER_TIME"]
    path = tmp / "ref_boundary_force.csv"
    pd.DataFrame(
        {
            "time": np.arange(len(bforce)),
            "fxpos": bforce["fxpos"].astype(float),
            "fxneg": bforce["fxneg"].astype(float),
        }
    ).to_csv(path, index=False)
    return str(path)


def _make_reference_cell_population(tmp: Path, results: dict) -> str:
    cell_metrics = results["CELL_METRICS_OVER_TIME"]
    path = tmp / "ref_cell_population.csv"
    pd.DataFrame(
        {
            "time": np.arange(len(cell_metrics)),
            "n_cells_alive": cell_metrics["n_cells_alive"].astype(float),
        }
    ).to_csv(path, index=False)
    return str(path)


def _make_reference_focad_attached_ratio(tmp: Path, results: dict) -> str:
    focad_metrics = results["FOCAD_METRICS_OVER_TIME"]
    path = tmp / "ref_focad_attached_ratio.csv"
    pd.DataFrame(
        {
            "time": np.arange(len(focad_metrics)),
            "attached_ratio": focad_metrics["attached_ratio"].astype(float),
        }
    ).to_csv(path, index=False)
    return str(path)


def _make_reference_poisson_ratio(tmp: Path, results: dict) -> str:
    pr = _poisson_ratio_to_series(results["POISSON_RATIO_OVER_TIME"])
    path = tmp / "ref_poisson_ratio.csv"
    pd.DataFrame(
        {
            "time": np.arange(len(pr)),
            "poisson_ratio": pr.values,
        }
    ).to_csv(path, index=False)
    return str(path)


def _make_reference_matrix_remodeling(tmp: Path, results: dict) -> str:
    fnode_metrics = results["FNODE_METRICS_OVER_TIME"]
    path = tmp / "ref_matrix_remodeling.csv"
    fnode_metrics.to_csv(path, index=False)
    return str(path)


def _make_reference_final_cell_count(tmp: Path, results: dict) -> str:
    final_row = results["CELL_METRICS_OVER_TIME"].iloc[-1]
    path = tmp / "ref_final_cell_count.csv"
    pd.DataFrame(
        {
            "cell_type": [-1, 0, 1],
            "target_count": [
                float(final_row["n_cells_alive"]),
                float(final_row["n_alive_type_0"]),
                float(final_row["n_alive_type_1"]),
            ],
        }
    ).to_csv(path, index=False)
    return str(path)


def _make_reference_final_focad_per_cell(tmp: Path, results: dict) -> str:
    final_alive = float(results["CELL_METRICS_OVER_TIME"]["n_cells_alive"].iloc[-1])
    final_total_focad = float(results["FOCAD_METRICS_OVER_TIME"]["total"].iloc[-1])
    target = final_total_focad / final_alive if final_alive > 0 else 0.0
    path = tmp / "ref_final_focad_per_cell.csv"
    pd.DataFrame({"target_focad_per_cell": [target]}).to_csv(path, index=False)
    return str(path)


def _write_mock_cells_vtk(trial_dir: Path) -> dict[str, float]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    vtk_path = trial_dir / "cells_t0001.vtk"
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [100.0, 100.0, 100.0],
        ],
        dtype=float,
    )
    ids = [0, 1, 2, 3]
    dead = [0, 0, 0, 1]
    cell_type = [0, 0, 1, 1]

    lines = [
        "# vtk DataFile Version 3.0",
        "Mock cells",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(positions)} float",
    ]
    lines.extend(f"{x} {y} {z}" for x, y, z in positions)
    lines.extend(
        [
            f"POINT_DATA {len(positions)}",
            "SCALARS id int 1",
            "LOOKUP_TABLE default",
        ]
    )
    lines.extend(str(value) for value in ids)
    lines.extend(
        [
            "SCALARS dead int 1",
            "LOOKUP_TABLE default",
        ]
    )
    lines.extend(str(value) for value in dead)
    lines.extend(
        [
            "SCALARS cell_type int 1",
            "LOOKUP_TABLE default",
        ]
    )
    lines.extend(str(value) for value in cell_type)
    vtk_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics = compute_organoid_metrics(positions[np.array(dead) == 0])
    return metrics


def _make_reference_organoid_size(tmp: Path, target_size: float) -> str:
    path = tmp / "ref_organoid_size.csv"
    pd.DataFrame({"target_size": [target_size]}).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def run_tests() -> bool:
    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        results = _make_mock_results()
        organoid_metrics = _write_mock_cells_vtk(tmp / "trial_dir")

        ref_ss = _make_reference_stress_strain(tmp)
        ref_dm = _make_reference_diff_modulus(tmp)
        ref_cs = _make_reference_cell_speed(tmp)
        ref_boundary = _make_reference_boundary_force(tmp, results)
        ref_population = _make_reference_cell_population(tmp, results)
        ref_focad_ratio = _make_reference_focad_attached_ratio(tmp, results)
        ref_poisson = _make_reference_poisson_ratio(tmp, results)
        ref_remodel = _make_reference_matrix_remodeling(tmp, results)
        ref_final_cells = _make_reference_final_cell_count(tmp, results)
        ref_final_focad = _make_reference_final_focad_per_cell(tmp, results)
        ref_organoid = _make_reference_organoid_size(
            tmp,
            organoid_metrics["radius_of_gyration"],
        )

        tests = [
            (
                "stress_strain_curve_error (normal)",
                lambda: OBJECTIVE_REGISTRY["stress_strain_curve_error"](
                    results,
                    ref_ss,
                    strain_axis=0,
                ),
            ),
            (
                "shear_stress_strain_curve_error",
                lambda: OBJECTIVE_REGISTRY["shear_stress_strain_curve_error"](
                    results,
                    ref_ss,
                    strain_axis=0,
                    shear_component=0,
                ),
            ),
            (
                "differential_modulus_error",
                lambda: OBJECTIVE_REGISTRY["differential_modulus_error"](
                    results,
                    ref_dm,
                    strain_axis=0,
                ),
            ),
            (
                "shear_differential_modulus_error",
                lambda: OBJECTIVE_REGISTRY["shear_differential_modulus_error"](
                    results,
                    ref_dm,
                    strain_axis=0,
                    shear_component=0,
                ),
            ),
            (
                "boundary_force_curve_error",
                lambda: OBJECTIVE_REGISTRY["boundary_force_curve_error"](
                    results,
                    ref_boundary,
                ),
            ),
            (
                "cell_population_error",
                lambda: OBJECTIVE_REGISTRY["cell_population_error"](
                    results,
                    ref_population,
                ),
            ),
            (
                "focad_attached_ratio_error",
                lambda: OBJECTIVE_REGISTRY["focad_attached_ratio_error"](
                    results,
                    ref_focad_ratio,
                ),
            ),
            (
                "poisson_ratio_error (csv)",
                lambda: OBJECTIVE_REGISTRY["poisson_ratio_error"](
                    results,
                    ref_poisson,
                ),
            ),
            (
                "matrix_remodeling_error",
                lambda: OBJECTIVE_REGISTRY["matrix_remodeling_error"](
                    results,
                    ref_remodel,
                ),
            ),
            (
                "final_cell_count_error (csv)",
                lambda: OBJECTIVE_REGISTRY["final_cell_count_error"](
                    results,
                    ref_final_cells,
                ),
            ),
            (
                "final_focad_per_cell_error (csv)",
                lambda: OBJECTIVE_REGISTRY["final_focad_per_cell_error"](
                    results,
                    ref_final_focad,
                ),
            ),
            (
                "cell_speed_error (median)",
                lambda: OBJECTIVE_REGISTRY["cell_speed_error"](
                    results,
                    ref_cs,
                    population_stat="median",
                ),
            ),
            (
                "organoid_size_error (csv)",
                lambda: OBJECTIVE_REGISTRY["organoid_size_error"](
                    results,
                    ref_organoid,
                    trial_dir=str(tmp / "trial_dir"),
                    metric="radius_of_gyration",
                ),
            ),
            (
                "poisson_ratio_error (scalar)",
                lambda: OBJECTIVE_REGISTRY["poisson_ratio_error"](
                    results,
                    None,
                    target_poisson=0.3,
                ),
            ),
            (
                "final_cell_count_error (scalar)",
                lambda: OBJECTIVE_REGISTRY["final_cell_count_error"](
                    results,
                    None,
                    target_cell_count=float(results["CELL_METRICS_OVER_TIME"]["n_cells_alive"].iloc[-1]),
                ),
            ),
            (
                "final_focad_per_cell_error (scalar)",
                lambda: OBJECTIVE_REGISTRY["final_focad_per_cell_error"](
                    results,
                    None,
                    target_focad_per_cell=(
                        float(results["FOCAD_METRICS_OVER_TIME"]["total"].iloc[-1])
                        / float(results["CELL_METRICS_OVER_TIME"]["n_cells_alive"].iloc[-1])
                    ),
                ),
            ),
            (
                "organoid_size_error (scalar)",
                lambda: OBJECTIVE_REGISTRY["organoid_size_error"](
                    results,
                    None,
                    trial_dir=str(tmp / "trial_dir"),
                    metric="radius_of_gyration",
                    target_size=organoid_metrics["radius_of_gyration"],
                ),
            ),
        ]

        for name, fn in tests:
            try:
                error, display = fn()
                ok = np.isfinite(error) and error >= 0.0
                status = "PASS" if ok else "FAIL (non-finite or negative)"
                if ok:
                    passed += 1
                else:
                    failed += 1
                print(f"  [{status}] {name}  ->  error = {error:.6f}  display = {display}")
            except Exception as exc:
                failed += 1
                print(f"  [FAIL]  {name}  ->  {type(exc).__name__}: {exc}")

        for key in sorted(OBJECTIVE_REGISTRY):
            passed += 1
            print(f"  [PASS]  Registry contains '{key}'")

    print(f"\n{'=' * 50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    return failed == 0


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------

def _resolve_objective_name(name: str) -> str:
    canonical = OBJECTIVE_ALIASES.get(name, name)
    if canonical not in OBJECTIVE_REGISTRY:
        raise ValueError(
            f"Unknown objective '{name}'. Choose from: {', '.join(ALL_OBJECTIVE_CHOICES)}"
        )
    return canonical


def _resolve_optional_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_results_pickle(pickle_path: Path) -> dict:
    print(f"Loading pickle: {pickle_path}")
    with open(pickle_path, "rb") as handle:
        return pickle.load(handle)


def _print_results_summary(results: dict) -> None:
    print(f"  Pickle keys: {list(results.keys())}")
    preview_keys = [
        "BPOS_OVER_TIME",
        "BFORCE_OVER_TIME",
        "BFORCE_SHEAR_OVER_TIME",
        "BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME",
        "CELL_METRICS_OVER_TIME",
        "FOCAD_METRICS_OVER_TIME",
        "FNODE_METRICS_OVER_TIME",
        "CELL_SPEED_METRICS",
        "POISSON_RATIO_OVER_TIME",
    ]
    for key in preview_keys:
        value = results.get(key)
        if value is None:
            print(f"  {key}: None")
            continue
        if hasattr(value, "shape"):
            columns = list(value.columns) if hasattr(value, "columns") else None
            print(f"  {key}: shape={value.shape}, columns={columns}")
            continue
        if hasattr(value, "__len__"):
            print(f"  {key}: len={len(value)}")
            continue
        print(f"  {key}: {type(value).__name__}")


def _print_reference_preview(reference_path: Path, reference_df: pd.DataFrame) -> None:
    print(f"\nLoading reference CSV: {reference_path}")
    print(f"  Columns: {list(reference_df.columns)}")
    print(f"  Rows: {len(reference_df)}")
    print(f"  Head:\n{reference_df.head().to_string(index=False)}")


def _build_objective_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {}
    optional_items = {
        "strain_axis": args.strain_axis,
        "shear_component": args.shear_component,
        "strain_weight": args.strain_weight,
        "stress_area_mode": args.stress_area_mode,
        "fibre_section_area_um2": args.fibre_section_area_um2,
        "smooth_window": args.smooth_window,
        "smooth_polyorder": args.smooth_polyorder,
        "modulus_smooth_window": args.modulus_smooth_window,
        "modulus_smooth_polyorder": args.modulus_smooth_polyorder,
        "min_sim_strain": args.min_sim_strain,
        "population_stat": args.population_stat,
        "min_trajectory_time": args.min_trajectory_time,
        "metric": args.metric,
        "target_poisson": args.target_poisson,
        "target_cell_count": args.target_cell_count,
        "target_focad_per_cell": args.target_focad_per_cell,
        "target_size": args.target_size,
    }
    for key, value in optional_items.items():
        if value is not None:
            kwargs[key] = value

    if args.normalize is not None:
        kwargs["normalize"] = args.normalize
    if args.include_dead is not None:
        kwargs["include_dead"] = args.include_dead
    if args.trial_dir is not None:
        kwargs["trial_dir"] = str(_resolve_optional_path(args.trial_dir))

    return kwargs


def _validate_inputs(
    objective_name: str,
    pickle_path: Path | None,
    reference_path: Path | None,
    kwargs: dict,
) -> None:
    if pickle_path is None:
        raise ValueError("Provide --pickle to evaluate a real objective.")
    if not pickle_path.exists():
        raise ValueError(f"Pickle file not found: {pickle_path}")

    if reference_path is not None and not reference_path.exists():
        raise ValueError(f"Reference CSV not found: {reference_path}")

    if objective_name == "poisson_ratio_error":
        if reference_path is None and "target_poisson" not in kwargs:
            raise ValueError(
                "poisson_ratio_error requires either --reference/--csv or --target-poisson"
            )
        return

    if objective_name == "final_cell_count_error":
        if reference_path is None and "target_cell_count" not in kwargs:
            raise ValueError(
                "final_cell_count_error requires either --reference/--csv or --target-cell-count"
            )
        return

    if objective_name == "final_focad_per_cell_error":
        if reference_path is None and "target_focad_per_cell" not in kwargs:
            raise ValueError(
                "final_focad_per_cell_error requires either --reference/--csv or --target-focad-per-cell"
            )
        return

    if objective_name == "organoid_size_error":
        trial_dir = kwargs.get("trial_dir")
        if trial_dir is None:
            raise ValueError("organoid_size_error requires --trial-dir")
        if not Path(trial_dir).exists():
            raise ValueError(f"Trial directory not found: {trial_dir}")
        if reference_path is None and "target_size" not in kwargs:
            raise ValueError(
                "organoid_size_error requires either --reference/--csv or --target-size"
            )
        return

    if reference_path is None:
        raise ValueError(f"{objective_name} requires --reference/--csv")


def _print_curve_context(results: dict, kwargs: dict) -> None:
    bpos = results.get("BPOS_OVER_TIME")
    if bpos is None or not hasattr(bpos, "iloc") or len(bpos) == 0:
        return

    axis = int(kwargs.get("strain_axis", 0))
    ortho = [i for i in range(3) if i != axis]
    dims = ["x", "y", "z"]
    lengths = []
    for oa in ortho:
        length = abs(float(bpos.iloc[0, oa * 2]) - float(bpos.iloc[0, oa * 2 + 1]))
        lengths.append(length)
        print(f"  Initial L_{dims[oa]} = {length:.2f} um")
    area = lengths[0] * lengths[1]
    print(f"  Face area (axis={dims[axis]}) = {area:.2f} um^2")
    print(f"  stress = force[nN] / {area:.2f}[um^2]  (1 nN/um^2 = 1 kPa)")

    if kwargs.get("stress_area_mode") == "per_fibre_area":
        fibre_area = float(results.get("FIBRE_SECTION_AREA_UM2", kwargs.get("fibre_section_area_um2", 0.0)))
        print(f"  Fibre section area = {fibre_area:.4f} um^2")


def _get_time_series_simulation_frame(results: dict, objective_name: str) -> pd.DataFrame:
    if objective_name == "boundary_force_curve_error":
        frame = results.get("BFORCE_OVER_TIME")
        if frame is None or len(frame) == 0:
            raise ValueError("Results must contain non-empty BFORCE_OVER_TIME")
        return pd.DataFrame(frame).reset_index(drop=True)

    if objective_name == "cell_population_error":
        frame = results.get("CELL_METRICS_OVER_TIME")
        if frame is None or len(frame) == 0:
            raise ValueError("Results must contain non-empty CELL_METRICS_OVER_TIME")
        return pd.DataFrame(frame)[["n_cells_alive"]].reset_index(drop=True)

    if objective_name == "focad_attached_ratio_error":
        frame = results.get("FOCAD_METRICS_OVER_TIME")
        if frame is None or len(frame) == 0:
            raise ValueError("Results must contain non-empty FOCAD_METRICS_OVER_TIME")
        return pd.DataFrame(frame)[["attached_ratio"]].reset_index(drop=True)

    if objective_name == "matrix_remodeling_error":
        frame = results.get("FNODE_METRICS_OVER_TIME")
        if frame is None or len(frame) == 0:
            raise ValueError("Results must contain non-empty FNODE_METRICS_OVER_TIME")
        compare_cols = [
            "n_fnodes_total",
            "sum_degradation",
            "sum_reinforcement",
            "mean_elastic_energy",
            "net_remodeling_total",
        ]
        available = [col for col in compare_cols if col in frame.columns]
        return pd.DataFrame(frame)[available].reset_index(drop=True)

    if objective_name == "poisson_ratio_error":
        pr = results.get("POISSON_RATIO_OVER_TIME")
        if pr is None or len(pr) == 0:
            raise ValueError("Results must contain POISSON_RATIO_OVER_TIME")
        if isinstance(pr, pd.DataFrame):
            return pr.reset_index(drop=True)
        if isinstance(pr, pd.Series):
            return pd.DataFrame({pr.name or "poisson_ratio": pr.astype(float)})
        return pd.DataFrame({"poisson_ratio": np.asarray(pr, dtype=float)})

    raise ValueError(f"No time-series extractor defined for {objective_name}")


def _compute_organoid_metric_value(trial_dir: str, metric_name: str) -> float:
    vtk_path = _get_last_cell_vtk(trial_dir)
    positions, dead, _ = _read_cell_positions_from_vtk(vtk_path)
    metrics = compute_organoid_metrics(positions[dead == 0])
    if metric_name not in metrics:
        raise ValueError(f"Unknown organoid metric '{metric_name}'")
    return float(metrics[metric_name])


def _build_scalar_comparison(
    results: dict,
    reference_df: pd.DataFrame | None,
    objective_name: str,
    kwargs: dict,
) -> tuple[list[str], list[float], list[float]]:
    if objective_name == "final_cell_count_error":
        final_row = pd.DataFrame(results["CELL_METRICS_OVER_TIME"]).iloc[-1]
        labels = []
        sim_values = []
        target_values = []
        if reference_df is not None:
            for _, row in reference_df.iterrows():
                cell_type = row.get("cell_type", -1)
                if cell_type == -1 or str(cell_type).strip().lower() == "all":
                    labels.append("all")
                    sim_values.append(float(final_row["n_cells_alive"]))
                    target_values.append(float(row["target_count"]))
                else:
                    col_name = f"n_alive_type_{int(cell_type)}"
                    if col_name in final_row.index:
                        labels.append(f"type {int(cell_type)}")
                        sim_values.append(float(final_row[col_name]))
                        target_values.append(float(row["target_count"]))
        else:
            labels = ["all"]
            sim_values = [float(final_row["n_cells_alive"])]
            target_values = [float(kwargs["target_cell_count"])]
        return labels, sim_values, target_values

    if objective_name == "final_focad_per_cell_error":
        final_alive = float(pd.DataFrame(results["CELL_METRICS_OVER_TIME"])["n_cells_alive"].iloc[-1])
        final_total_focad = float(pd.DataFrame(results["FOCAD_METRICS_OVER_TIME"])["total"].iloc[-1])
        sim_value = final_total_focad / final_alive if final_alive > 0.0 else 0.0
        target = (
            float(reference_df["target_focad_per_cell"].iloc[0])
            if reference_df is not None
            else float(kwargs["target_focad_per_cell"])
        )
        return ["focad/cell"], [sim_value], [target]

    if objective_name == "poisson_ratio_error":
        sim_final = float(_poisson_ratio_to_series(results["POISSON_RATIO_OVER_TIME"]).iloc[-1])
        if reference_df is not None:
            target = float(reference_df["poisson_ratio"].iloc[-1])
        else:
            target = float(kwargs["target_poisson"])
        return ["final poisson ratio"], [sim_final], [target]

    if objective_name == "organoid_size_error":
        metric_name = str(kwargs.get("metric", "radius_of_gyration"))
        sim_value = _compute_organoid_metric_value(kwargs["trial_dir"], metric_name)
        target = (
            float(reference_df["target_size"].iloc[0])
            if reference_df is not None
            else float(kwargs["target_size"])
        )
        return [metric_name], [sim_value], [target]

    raise ValueError(f"No scalar comparison builder defined for {objective_name}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_curve_objective(
    results: dict,
    reference_df: pd.DataFrame,
    objective_name: str,
    kwargs: dict,
) -> None:
    force_type = "shear" if "shear" in objective_name else "normal"
    axis = int(kwargs.get("strain_axis", 0))
    shear_component = int(kwargs.get("shear_component", 0))

    sim_strain, sim_stress = _extract_sim_strain_stress(
        results,
        force_type=force_type,
        strain_axis=axis,
        shear_component=shear_component,
        stress_area_mode=kwargs.get("stress_area_mode", "boundary_surface"),
        fibre_section_area_um2=kwargs.get("fibre_section_area_um2"),
    )
    sim_strain_arr, sim_stress_arr = _filter_simulation_from_min_strain(
        sim_strain,
        sim_stress,
        min_sim_strain=kwargs.get("min_sim_strain"),
    )
    sim_strain = pd.Series(sim_strain_arr)
    sim_stress = pd.Series(sim_stress_arr)
    stress_smooth_arr = None

    is_diff_modulus = "differential_modulus" in objective_name
    if is_diff_modulus:
        smooth_window = int(kwargs.get("smooth_window", 5))
        smooth_polyorder = int(kwargs.get("smooth_polyorder", 2))
        stress_smooth_arr = _smooth_signal_savgol(
            sim_stress,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            label="stress",
        )
        modulus_smooth_window = int(kwargs.get("modulus_smooth_window", 0))
        modulus_smooth_polyorder = kwargs.get("modulus_smooth_polyorder")
        if modulus_smooth_polyorder is not None:
            modulus_smooth_polyorder = int(modulus_smooth_polyorder)

        sim_y = _compute_differential_modulus(
            strain=sim_strain,
            stress=sim_stress,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            modulus_smooth_window=modulus_smooth_window,
            modulus_smooth_polyorder=modulus_smooth_polyorder,
        )
        ref_y = reference_df["differential_modulus"].to_numpy(dtype=float)
        ylabel = "Differential modulus [kPa]"
        title_prefix = "Differential Modulus"
        sim_curve_label = "Simulation modulus"
        sim_interp_label = "Simulation modulus (interp)"
    else:
        sim_y = sim_stress.to_numpy(dtype=float)
        ref_y = reference_df["stress"].to_numpy(dtype=float)
        ylabel = "Stress [kPa]"
        title_prefix = "Stress-Strain"
        sim_curve_label = "Simulation stress"
        sim_interp_label = "Simulation stress (interp)"

    ref_strain = reference_df["strain"].to_numpy(dtype=float)
    sim_y_interp = _interpolate_response_to_reference_x(sim_strain.values, sim_y, ref_strain)
    overlap_mask = np.isfinite(sim_y_interp)
    sim_strain_interp = ref_strain[overlap_mask]
    sim_y_interp = sim_y_interp[overlap_mask]
    ref_y_overlap = ref_y[overlap_mask]
    timesteps = np.arange(len(sim_strain))

    print(
        f"\n  Sim strain: min={np.min(sim_strain.values):.6g}, max={np.max(sim_strain.values):.6g}, n={len(sim_strain)}"
    )
    print(f"  Sim response: min={np.min(sim_y):.6g}, max={np.max(sim_y):.6g}")
    print(
        f"  Ref strain: min={np.min(ref_strain):.6g}, max={np.max(ref_strain):.6g}, n={len(ref_strain)}"
    )
    print(f"  Ref response: min={np.min(ref_y):.6g}, max={np.max(ref_y):.6g}")

    axis_label = "xyz"[axis]
    force_label = f"shear (component {shear_component})" if force_type == "shear" else "normal"
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax_main = axes[0, 0]
    ax_main.plot(sim_strain.values, sim_y, "-", color="tab:blue", alpha=0.5, label=sim_curve_label)
    ax_main.plot(sim_strain_interp, sim_y_interp, "o-", color="tab:blue", markersize=3, label=sim_interp_label)
    ax_main.plot(ref_strain, ref_y, "s--", color="tab:red", markersize=4, label="Reference")
    if kwargs.get("min_sim_strain") is not None:
        ax_main.axvline(float(kwargs["min_sim_strain"]), color="tab:gray", linestyle=":", label="Cutoff")
    ax_main.set_xlabel("Strain [-]")
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(f"{title_prefix} - {force_label}, axis={axis_label}")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    ax_strain = axes[0, 1]
    ax_strain.plot(timesteps, sim_strain.values, "-", color="tab:blue", label="Simulation strain")
    if kwargs.get("min_sim_strain") is not None:
        ax_strain.axhline(float(kwargs["min_sim_strain"]), color="tab:gray", linestyle=":", label="Cutoff")
    ax_strain.axhline(ref_strain[0], color="tab:red", linestyle="--", alpha=0.5, label="Ref min")
    ax_strain.axhline(ref_strain[-1], color="tab:red", linestyle="--", alpha=0.5, label="Ref max")
    ax_strain.set_xlabel("Timestep index")
    ax_strain.set_ylabel("Strain [-]")
    ax_strain.set_title("Simulation strain over time")
    ax_strain.grid(True, alpha=0.3)
    ax_strain.legend(fontsize=8)

    ax_stress = axes[1, 0]
    ax_stress.plot(timesteps, sim_stress_arr, "-", color="tab:blue", alpha=0.7, label="Simulation stress (raw)")
    if stress_smooth_arr is not None and not np.array_equal(stress_smooth_arr, sim_stress_arr):
        ax_stress.plot(timesteps, stress_smooth_arr, "-", color="tab:orange", linewidth=2, label="Simulation stress (smoothed)")
    ax_stress.set_xlabel("Timestep index")
    ax_stress.set_ylabel("Stress [kPa]")
    ax_stress.set_title("Simulation stress over time")
    ax_stress.grid(True, alpha=0.3)
    ax_stress.legend(fontsize=8)

    ax_aux = axes[1, 1]
    if force_type == "shear":
        bforce_shear = pd.DataFrame(results["BFORCE_SHEAR_OVER_TIME"])
        bpos = pd.DataFrame(results["BPOS_OVER_TIME"])
        tangent_dirs = [direction for direction in ["x", "y", "z"] if direction != axis_label]
        tangential_direction = tangent_dirs[shear_component]
        pos_col = f"f{axis_label}pos_{tangential_direction}"
        neg_col = f"f{axis_label}neg_{tangential_direction}"
        ortho_axes = [idx for idx in range(3) if idx != axis]
        area = 1.0
        for ortho_axis in ortho_axes:
            area *= abs(float(bpos.iloc[0, ortho_axis * 2]) - float(bpos.iloc[0, ortho_axis * 2 + 1]))
        stress_pos = bforce_shear[pos_col].to_numpy(dtype=float) / area
        stress_neg = bforce_shear[neg_col].to_numpy(dtype=float) / area
        ax_aux.plot(stress_pos, "-", color="tab:blue", label=f"+{axis_label} face")
        ax_aux.plot(stress_neg, "-", color="tab:green", label=f"-{axis_label} face")
        ax_aux.plot((stress_pos + stress_neg) / 2.0, "--", color="tab:purple", alpha=0.7, label="Mean")
        ax_aux.set_xlabel("Timestep index")
        ax_aux.set_ylabel("Shear stress [kPa]")
        ax_aux.set_title(f"Shear forces on +/-{axis_label} faces")
        ax_aux.legend(fontsize=8)
    else:
        residual = sim_y_interp - ref_y_overlap
        ax_aux.bar(np.arange(len(residual)), residual, color="tab:orange", alpha=0.7)
        ax_aux.axhline(0.0, color="black", linewidth=0.8)
        ax_aux.set_xlabel("Reference point index")
        ax_aux.set_ylabel("Residual")
        ax_aux.set_title("Simulation - reference residual")
    ax_aux.grid(True, alpha=0.3)

    figure.tight_layout()
    plt.show()


def _plot_time_series_objective(
    results: dict,
    reference_df: pd.DataFrame,
    objective_name: str,
) -> None:
    sim_df = _get_time_series_simulation_frame(results, objective_name)
    compare_cols = [
        col for col in reference_df.columns if col != "time" and col in sim_df.columns
    ]
    if not compare_cols:
        print("No overlapping series columns were found for plotting.")
        return

    n_plots = len(compare_cols)
    ncols = 2 if n_plots > 1 else 1
    nrows = int(np.ceil(n_plots / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    ref_x = (
        reference_df["time"].to_numpy(dtype=float)
        if "time" in reference_df.columns
        else np.arange(len(reference_df), dtype=float)
    )

    for axis, column in zip(axes, compare_cols):
        sim_series = pd.Series(sim_df[column].astype(float).to_numpy())
        sim_interp = _interpolate_to_match(sim_series, len(reference_df))
        sim_x = np.arange(len(sim_series), dtype=float)
        axis.plot(sim_x, sim_series.to_numpy(), "-", color="tab:blue", alpha=0.4, label="Simulation (raw)")
        axis.plot(ref_x, sim_interp, "o-", color="tab:blue", markersize=3, label="Simulation (interp)")
        axis.plot(ref_x, reference_df[column].to_numpy(dtype=float), "s--", color="tab:red", markersize=4, label="Reference")
        axis.set_title(column)
        axis.set_xlabel("Time or sample index")
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    for axis in axes[n_plots:]:
        axis.remove()

    figure.suptitle(objective_name, fontsize=14)
    figure.tight_layout()
    plt.show()


def _plot_cell_speed_objective(
    results: dict,
    reference_df: pd.DataFrame,
    kwargs: dict,
) -> None:
    speed_metrics = _get_cell_speed_metrics_frame(results)
    if not kwargs.get("include_dead", False):
        speed_metrics = speed_metrics[speed_metrics["dead"] == 0]
    if kwargs.get("min_trajectory_time") is not None:
        speed_metrics = speed_metrics[
            speed_metrics["trajectory_time"].astype(float) >= float(kwargs["min_trajectory_time"])
        ]

    if len(speed_metrics) == 0:
        print("No cells remain after the selected speed filters; skipping plot.")
        return

    population_stat = str(kwargs.get("population_stat", "mean")).strip().lower()
    target_cols = [col for col in ("target_vmean", "target_veff") if col in reference_df.columns]
    figure, axes = plt.subplots(1, len(target_cols), figsize=(7 * len(target_cols), 5))
    axes = np.atleast_1d(axes)

    for axis, metric_col in zip(axes, target_cols):
        sim_values = []
        target_values = []
        labels = []
        sim_metric_name = metric_col.replace("target_", "")
        for _, row in reference_df.iterrows():
            cell_type = int(row["cell_type"])
            group = speed_metrics[speed_metrics["cell_type"].astype(int) == cell_type]
            if len(group) == 0:
                continue
            labels.append(f"type {cell_type}")
            sim_values.append(_reduce_population_values(group[sim_metric_name], population_stat))
            target_values.append(float(row[metric_col]))

        x = np.arange(len(labels), dtype=float)
        width = 0.35
        axis.bar(x - width / 2.0, sim_values, width, label="Simulation", color="tab:blue")
        axis.bar(x + width / 2.0, target_values, width, label="Target", color="tab:red", alpha=0.75)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylabel(sim_metric_name)
        axis.set_title(f"{sim_metric_name} ({population_stat})")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()

    figure.tight_layout()
    plt.show()


def _plot_scalar_objective(
    results: dict,
    reference_df: pd.DataFrame | None,
    objective_name: str,
    kwargs: dict,
) -> None:
    labels, sim_values, target_values = _build_scalar_comparison(
        results,
        reference_df,
        objective_name,
        kwargs,
    )
    summary_df = pd.DataFrame(
        {
            "label": labels,
            "simulation": sim_values,
            "target": target_values,
            "abs_error": np.abs(np.asarray(sim_values) - np.asarray(target_values)),
        }
    )
    print("\nScalar comparison")
    print(summary_df.to_string(index=False))

    figure, axis = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels), dtype=float)
    width = 0.35
    axis.bar(x - width / 2.0, sim_values, width, color="tab:blue", label="Simulation")
    axis.bar(x + width / 2.0, target_values, width, color="tab:red", alpha=0.75, label="Target")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Value")
    axis.set_title(objective_name)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()


def _plot_objective(
    results: dict,
    reference_df: pd.DataFrame | None,
    objective_name: str,
    kwargs: dict,
) -> None:
    if objective_name in CURVE_OBJECTIVES and reference_df is not None:
        _plot_curve_objective(results, reference_df, objective_name, kwargs)
        return

    if objective_name == "cell_speed_error" and reference_df is not None:
        _plot_cell_speed_objective(results, reference_df, kwargs)
        return

    if objective_name in TIME_SERIES_OBJECTIVES and reference_df is not None:
        _plot_time_series_objective(results, reference_df, objective_name)
        return

    if objective_name in SCALAR_OPTIONAL_REFERENCE_OBJECTIVES:
        _plot_scalar_objective(results, reference_df, objective_name, kwargs)
        return

    print("No plotter is implemented for this objective.")


# ---------------------------------------------------------------------------
# Main real-data runner
# ---------------------------------------------------------------------------

def run_real_data_test(args: argparse.Namespace) -> bool:
    objective_name = _resolve_objective_name(args.objective)
    pickle_path = _resolve_optional_path(args.pickle)
    reference_path = _resolve_optional_path(args.reference)
    kwargs = _build_objective_kwargs(args)

    _validate_inputs(objective_name, pickle_path, reference_path, kwargs)

    results = _load_results_pickle(pickle_path)
    _print_results_summary(results)

    reference_df = None
    if reference_path is not None:
        reference_df = pd.read_csv(reference_path)
        _print_reference_preview(reference_path, reference_df)

    print(f"\nRunning objective: {objective_name}")
    if objective_name != args.objective:
        print(f"  Alias: {args.objective} -> {objective_name}")
    print(f"  kwargs: {kwargs}")

    if objective_name in CURVE_OBJECTIVES:
        _print_curve_context(results, kwargs)

    fn = OBJECTIVE_REGISTRY[objective_name]
    reference_arg = str(reference_path) if reference_path is not None else None

    try:
        error, display = fn(results, reference_arg, **kwargs)
        print(f"\n  Result:  error = {error:.8f}")
        if display:
            print(f"           display = {display}")
        ok = np.isfinite(error) and error >= 0.0
        print(f"  Status:  {'PASS' if ok else 'FAIL (non-finite or negative)'}")

        if args.plot:
            _plot_objective(results, reference_df, objective_name, kwargs)

        return ok
    except Exception as exc:
        print(f"\n  FAILED with {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return False


def _print_objective_list() -> None:
    print("Available objectives\n" + "=" * 50)
    for alias in sorted(OBJECTIVE_ALIASES):
        print(f"  {alias:<24} -> {OBJECTIVE_ALIASES[alias]}")

    print("\nCanonical registry names\n" + "=" * 50)
    for objective_name in sorted(OBJECTIVE_REGISTRY):
        requirement = "reference csv"
        if objective_name in SCALAR_OPTIONAL_REFERENCE_OBJECTIVES:
            requirement = "reference csv or scalar target"
        if objective_name == "organoid_size_error":
            requirement += " + trial_dir"
        print(f"  {objective_name:<32} {requirement}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual checker for optimizer objective functions.",
    )
    parser.add_argument(
        "--list-objectives",
        action="store_true",
        help="Print supported objectives and exit.",
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default=None,
        help="Path to a simulation output pickle file.",
    )
    parser.add_argument(
        "--reference",
        "--csv",
        dest="reference",
        type=str,
        default=None,
        help="Path to a reference CSV file.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="stress_strain",
        choices=ALL_OBJECTIVE_CHOICES,
        help="Objective alias or canonical registry name.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable diagnostic plots.",
    )

    parser.add_argument(
        "--strain-axis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Strain axis for stress/modulus objectives: 0=x, 1=y, 2=z.",
    )
    parser.add_argument(
        "--shear-component",
        type=int,
        default=None,
        choices=[0, 1],
        help="Tangential shear component for shear objectives.",
    )
    parser.add_argument(
        "--strain-weight",
        type=float,
        default=None,
        help="Optional weight of strain-range mismatch for curve objectives.",
    )
    parser.add_argument(
        "--stress-area-mode",
        type=str,
        default=None,
        choices=["boundary_surface", "per_fibre_area"],
        help="Force normalization mode for curve objectives.",
    )
    parser.add_argument(
        "--fibre-section-area-um2",
        type=float,
        default=None,
        help="Override fibre cross-sectional area in um^2.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Savitzky-Golay window for stress smoothing before differentiation.",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=None,
        help="Savitzky-Golay polynomial order for stress smoothing.",
    )
    parser.add_argument(
        "--modulus-smooth-window",
        type=int,
        default=None,
        help="Optional Savitzky-Golay window for modulus post-smoothing.",
    )
    parser.add_argument(
        "--modulus-smooth-polyorder",
        type=int,
        default=None,
        help="Savitzky-Golay polynomial order for modulus post-smoothing.",
    )
    parser.add_argument(
        "--min-sim-strain",
        type=float,
        default=None,
        help="Ignore simulation samples below this strain.",
    )

    parser.add_argument(
        "--population-stat",
        type=str,
        default=None,
        choices=["mean", "median"],
        help="Population reducer for cell_speed_error.",
    )
    parser.add_argument(
        "--include-dead",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include dead cells in cell_speed_error aggregation.",
    )
    parser.add_argument(
        "--min-trajectory-time",
        type=float,
        default=None,
        help="Minimum tracked lifetime for cell_speed_error.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the objective's optional normalization mode when available.",
    )

    parser.add_argument(
        "--target-poisson",
        type=float,
        default=None,
        help="Scalar target for poisson_ratio_error.",
    )
    parser.add_argument(
        "--target-cell-count",
        type=float,
        default=None,
        help="Scalar target for final_cell_count_error.",
    )
    parser.add_argument(
        "--target-focad-per-cell",
        type=float,
        default=None,
        help="Scalar target for final_focad_per_cell_error.",
    )
    parser.add_argument(
        "--target-size",
        type=float,
        default=None,
        help="Scalar target for organoid_size_error.",
    )
    parser.add_argument(
        "--trial-dir",
        type=str,
        default=None,
        help="Directory containing cells_t*.vtk for organoid_size_error.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["radius_of_gyration", "max_span", "equivalent_sphere_radius"],
        help="Organoid size metric.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()

    if cli_args.list_objectives:
        _print_objective_list()
        sys.exit(0)

    if cli_args.pickle is None:
        print("Objective smoke tests (synthetic data)\n" + "=" * 50)
        success = run_tests()
    else:
        print("Manual objective check\n" + "=" * 50)
        success = run_real_data_test(cli_args)

    sys.exit(0 if success else 1)
