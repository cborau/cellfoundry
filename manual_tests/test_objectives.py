"""
Quick smoke-test for the strain-stress and differential-modulus objectives.

Run from the repository root:

  Synthetic mock data (no files needed):
    python manual_tests/test_objectives.py

  Real data — normal stress-strain:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --csv optimizer/reference_data/target_stress_strain.csv --objective stress_strain

  Real data — shear stress-strain:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --csv optimizer/reference_data/target_shear.csv --objective shear_stress_strain --strain-axis 1 --shear-component 0

  Real data — differential modulus:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --csv optimizer/reference_data/target_differential_modulus.csv --objective diff_modulus

  Real data — shear differential modulus:
    python manual_tests/test_objectives.py --pickle result_files/output_data_0.pickle --csv optimizer/reference_data/target_differential_modulus.csv --objective shear_diff_modulus --strain-axis 0

Without --pickle / --csv the script runs the synthetic smoke tests only.
With --pickle and --csv it loads actual data and evaluates the chosen objective.
"""

import sys
import argparse
import pickle
import tempfile
from pathlib import Path

# Ensure the repo root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from optimizer.objectives import (
    stress_strain_curve_error,
    shear_stress_strain_curve_error,
    differential_modulus_error,
    shear_differential_modulus_error,
    _extract_sim_strain_stress,
    _interpolate_to_match,
    OBJECTIVE_REGISTRY,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_reference_stress_strain(tmp: Path) -> str:
    """Write a small reference strain-stress CSV and return its path."""
    strains = np.linspace(0, 0.10, 20)
    stresses = 5.0 * strains  # simple linear: K = 5
    df = pd.DataFrame({"strain": strains, "stress": stresses})
    p = tmp / "ref_stress_strain.csv"
    df.to_csv(p, index=False)
    return str(p)


def _make_reference_diff_modulus(tmp: Path) -> str:
    """Write a small reference differential modulus CSV and return its path."""
    strains = np.linspace(0, 0.10, 20)
    K = np.full_like(strains, 5.0)  # constant modulus = 5
    df = pd.DataFrame({"strain": strains, "differential_modulus": K})
    p = tmp / "ref_diff_modulus.csv"
    df.to_csv(p, index=False)
    return str(p)


def _make_mock_results(n_steps: int = 50) -> dict:
    """Build a mock results dict that mimics the pickle structure.

    Domain: 1000 × 1000 × 1000 µm  (boundaries at ±500 µm per axis).
    Face area (e.g. x-face) = Ly × Lz = 1000 × 1000 = 1e6 µm².
    Forces are in nN; stress = force / area → nN/µm² = kPa.
    """
    # Boundary positions: ±500 µm (matching default model.py BOUNDARY_COORDS)
    # +x face displaces outward by 10% → 500 → 550
    xpos = np.linspace(500.0, 550.0, n_steps)   # +x face moves outward
    xneg = np.full(n_steps, -500.0)
    ypos = np.full(n_steps, 500.0)
    yneg = np.full(n_steps, -500.0)
    zpos = np.full(n_steps, 500.0)
    zneg = np.full(n_steps, -500.0)

    bpos = pd.DataFrame({
        "xpos": xpos, "xneg": xneg,
        "ypos": ypos, "yneg": yneg,
        "zpos": zpos, "zneg": zneg,
    })

    # Face area for x-faces: Ly0 * Lz0 = 1000 * 1000 = 1e6 µm²
    # Force in nN; after area normalization stress = force / 1e6 kPa
    # For a target stress of ~5e-6 kPa per unit strain ≈ 5 nN/µm at 10% strain:
    bforce = pd.DataFrame({
        "fxpos": 5e6 * ((xpos - xneg) - 1000.0) / 1000.0,  # ~5 kPa after /area
        "fxneg": np.zeros(n_steps),
        "fypos": np.zeros(n_steps),
        "fyneg": np.zeros(n_steps),
        "fzpos": np.zeros(n_steps),
        "fzneg": np.zeros(n_steps),
    })

    # Shear forces: tangential (y-direction) force on +x face
    bforce_shear = pd.DataFrame({
        "fxpos_y": 3e6 * np.linspace(0, 0.1, n_steps),
        "fxpos_z": np.zeros(n_steps),
        "fxneg_y": np.zeros(n_steps),
        "fxneg_z": np.zeros(n_steps),
        "fypos_x": 2e6 * np.linspace(0, 0.1, n_steps),
        "fypos_z": np.zeros(n_steps),
        "fyneg_x": np.zeros(n_steps),
        "fyneg_z": np.zeros(n_steps),
        "fzpos_x": np.zeros(n_steps),
        "fzpos_y": np.zeros(n_steps),
        "fzneg_x": np.zeros(n_steps),
        "fzneg_y": np.zeros(n_steps),
    })

    return {
        "BPOS_OVER_TIME": bpos,
        "BFORCE_OVER_TIME": bforce,
        "BFORCE_SHEAR_OVER_TIME": bforce_shear,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_ss = _make_reference_stress_strain(tmp)
        ref_dm = _make_reference_diff_modulus(tmp)
        results = _make_mock_results()

        tests = [
            ("stress_strain_curve_error (normal)",
             lambda: stress_strain_curve_error(results, ref_ss, force_type="normal", strain_axis=0)),
            ("stress_strain_curve_error (shear)",
             lambda: shear_stress_strain_curve_error(results, ref_ss, strain_axis=0, shear_component=0)),
            ("differential_modulus_error (normal)",
             lambda: differential_modulus_error(results, ref_dm, force_type="normal", strain_axis=0)),
            ("differential_modulus_error (shear)",
             lambda: shear_differential_modulus_error(results, ref_dm, strain_axis=0, shear_component=0)),
            ("differential_modulus_error (no smoothing)",
             lambda: differential_modulus_error(results, ref_dm, force_type="normal", smooth_window=0)),
        ]

        for name, fn in tests:
            try:
                error, display = fn()
                ok = np.isfinite(error) and error >= 0
                status = "PASS" if ok else "FAIL (non-finite or negative)"
                if ok:
                    passed += 1
                else:
                    failed += 1
                print(f"  [{status}] {name}  ->  error = {error:.6f}  display = {display}")
            except Exception as e:
                failed += 1
                print(f"  [FAIL]  {name}  ->  {type(e).__name__}: {e}")

        # Verify registry has all expected entries
        expected_keys = [
            "stress_strain_curve_error",
            "shear_stress_strain_curve_error",
            "differential_modulus_error",
            "shear_differential_modulus_error",
        ]
        for key in expected_keys:
            if key in OBJECTIVE_REGISTRY:
                passed += 1
                print(f"  [PASS]  Registry contains '{key}'")
            else:
                failed += 1
                print(f"  [FAIL]  Registry missing '{key}'")

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


# ── Real-data test ────────────────────────────────────────────────────────────

OBJECTIVE_MAP = {
    "stress_strain":       stress_strain_curve_error,
    "shear_stress_strain": shear_stress_strain_curve_error,
    "diff_modulus":        differential_modulus_error,
    "shear_diff_modulus":  shear_differential_modulus_error,
}


def _plot_real_data(results: dict, ref_df: pd.DataFrame, obj_name: str, kwargs: dict):
    """Plot simulated vs reference curves for visual inspection.

    Layout (2×2):
        Top-left : stress (or K) vs strain — sim raw, sim interpolated, ref
        Top-right: strain vs timestep index  (raw sim time-series)
        Bot-left : stress (or K) vs timestep index  (raw sim time-series)
        Bot-right: residuals  (sim_interp − ref)
    """
    force_type = "shear" if "shear" in obj_name else "normal"
    axis = kwargs.get("strain_axis", 0)
    shear_comp = kwargs.get("shear_component", 0)

    sim_strain, sim_stress = _extract_sim_strain_stress(
        results, force_type=force_type, strain_axis=axis,
        shear_component=shear_comp,
    )

    is_diff_modulus = "diff_modulus" in obj_name

    if is_diff_modulus:
        stress_arr = sim_stress.values.astype(float)
        strain_arr = sim_strain.values.astype(float)
        d_stress = np.gradient(stress_arr)
        d_strain = np.gradient(strain_arr)
        safe = np.abs(d_strain) > 1e-15
        sim_K = np.zeros_like(stress_arr)
        sim_K[safe] = d_stress[safe] / d_strain[safe]

        smooth_window = int(kwargs.get("smooth_window", 5))
        smooth_polyorder = int(kwargs.get("smooth_polyorder", 2))
        if smooth_window > 2 and len(sim_K) >= smooth_window:
            from scipy.signal import savgol_filter
            if smooth_window % 2 == 0:
                smooth_window += 1
            sim_K = savgol_filter(sim_K, smooth_window, min(smooth_polyorder, smooth_window - 1))

        sim_y = sim_K
        sim_y_interp = _interpolate_to_match(pd.Series(sim_K), len(ref_df))
        ref_y = ref_df["differential_modulus"].values
        ylabel = "Diff. modulus K(ε) [kPa]"
        title_suffix = "Differential Modulus"
    else:
        sim_y = sim_stress.values
        sim_y_interp = _interpolate_to_match(sim_stress, len(ref_df))
        ref_y = ref_df["stress"].values
        ylabel = "Stress [kPa]"
        title_suffix = "Stress-Strain"

    sim_strain_arr = sim_strain.values
    sim_strain_interp = _interpolate_to_match(sim_strain, len(ref_df))
    ref_strain = ref_df["strain"].values
    timesteps = np.arange(len(sim_strain_arr))

    # Print diagnostic ranges
    print(f"\n  Sim  strain  : min={np.min(sim_strain_arr):.6g}, max={np.max(sim_strain_arr):.6g}, n={len(sim_strain_arr)}")
    print(f"  Sim  {ylabel.split('[')[0].strip()}: min={np.min(sim_y):.6g}, max={np.max(sim_y):.6g}")
    print(f"  Ref  strain  : min={np.min(ref_strain):.6g}, max={np.max(ref_strain):.6g}, n={len(ref_strain)}")
    print(f"  Ref  {ylabel.split('[')[0].strip()}: min={np.min(ref_y):.6g}, max={np.max(ref_y):.6g}")

    force_label = f"shear (comp {shear_comp})" if force_type == "shear" else "normal"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top-left: main curve (stress or K vs strain) ──────────────────────
    ax = axes[0, 0]
    ax.plot(sim_strain_arr, sim_y, "-", color="tab:blue", alpha=0.4, linewidth=1, label="Sim (raw)")
    ax.plot(sim_strain_interp, sim_y_interp, "o-", color="tab:blue", markersize=3, label="Sim (interp)")
    ax.plot(ref_strain, ref_y, "s--", color="tab:red", markersize=4, label="Reference")
    ax.set_xlabel("Strain [-]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_suffix} — {force_label}, axis={'xyz'[axis]}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Top-right: strain vs timestep ─────────────────────────────────────
    ax_ts = axes[0, 1]
    ax_ts.plot(timesteps, sim_strain_arr, "-", color="tab:blue", label="Sim strain")
    ax_ts.axhline(ref_strain[0], color="tab:red", ls="--", alpha=0.5, label=f"Ref strain min ({ref_strain[0]:.4g})")
    ax_ts.axhline(ref_strain[-1], color="tab:red", ls="--", alpha=0.5, label=f"Ref strain max ({ref_strain[-1]:.4g})")
    ax_ts.set_xlabel("Timestep index")
    ax_ts.set_ylabel("Strain [-]")
    ax_ts.set_title("Sim strain over time")
    ax_ts.legend(fontsize=8)
    ax_ts.grid(True, alpha=0.3)

    # ── Bot-left: stress (or K) vs timestep ───────────────────────────────
    ax_sf = axes[1, 0]
    ax_sf.plot(timesteps, sim_y, "-", color="tab:blue", label=f"Sim {ylabel.split('[')[0].strip()}")
    ax_sf.axhline(np.min(ref_y), color="tab:red", ls="--", alpha=0.5, label=f"Ref min ({np.min(ref_y):.4g})")
    ax_sf.axhline(np.max(ref_y), color="tab:red", ls="--", alpha=0.5, label=f"Ref max ({np.max(ref_y):.4g})")
    ax_sf.set_xlabel("Timestep index")
    ax_sf.set_ylabel(ylabel)
    ax_sf.set_title(f"Sim {ylabel.split('[')[0].strip()} over time")
    ax_sf.legend(fontsize=8)
    ax_sf.grid(True, alpha=0.3)

    # ── Bot-right: residuals ──────────────────────────────────────────────
    ax2 = axes[1, 1]
    residual = sim_y_interp - ref_y
    ax2.bar(np.arange(len(residual)), residual, color="tab:orange", alpha=0.7)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel("Reference point index")
    ax2.set_ylabel(f"Residual ({ylabel.split('[')[0].strip()})")
    ax2.set_title("Sim − Reference residuals")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_real_data_test(args):
    """Load a pickle + CSV and evaluate the chosen objective on real data."""
    pickle_path = Path(args.pickle)
    csv_path = Path(args.csv)

    # Resolve relative paths from project root so the script works from any CWD
    if not pickle_path.is_absolute():
        pickle_path = PROJECT_ROOT / pickle_path
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not pickle_path.exists():
        print(f"ERROR: pickle file not found: {pickle_path}")
        return False
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return False

    # Load pickle
    print(f"Loading pickle: {pickle_path}")
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    # Show available keys
    print(f"  Pickle keys: {list(results.keys())}")
    for key in ["BPOS_OVER_TIME", "BFORCE_OVER_TIME", "BFORCE_SHEAR_OVER_TIME"]:
        val = results.get(key)
        if val is not None and hasattr(val, "shape"):
            print(f"  {key}: shape={val.shape}, columns={list(val.columns)}")
        elif val is not None and hasattr(val, "__len__"):
            print(f"  {key}: len={len(val)}")
        else:
            print(f"  {key}: {type(val).__name__}")

    # Show boundary dimensions and face area (for stress = force / area)
    bpos = results.get("BPOS_OVER_TIME")
    if bpos is not None and hasattr(bpos, "iloc") and len(bpos) > 0:
        axis = args.strain_axis
        ortho = [i for i in range(3) if i != axis]
        dims_label = ["x", "y", "z"]
        L = []
        for oa in ortho:
            L_oa = abs(float(bpos.iloc[0, oa * 2]) - float(bpos.iloc[0, oa * 2 + 1]))
            L.append(L_oa)
            print(f"  Initial L_{dims_label[oa]} = {L_oa:.2f} µm")
        area = L[0] * L[1]
        print(f"  Face area (axis={dims_label[axis]}) = {area:.2f} µm²")
        print(f"  → stress = force[nN] / {area:.2f}[µm²]  (1 nN/µm² = 1 kPa)")

    # Load and preview CSV
    print(f"\nLoading reference CSV: {csv_path}")
    ref_df = pd.read_csv(csv_path)
    print(f"  Columns: {list(ref_df.columns)}")
    print(f"  Rows: {len(ref_df)}")
    print(f"  Head:\n{ref_df.head().to_string(index=False)}")

    # Pick objective function
    obj_name = args.objective
    if obj_name not in OBJECTIVE_MAP:
        print(f"ERROR: unknown objective '{obj_name}'. Choose from: {list(OBJECTIVE_MAP.keys())}")
        return False

    fn = OBJECTIVE_MAP[obj_name]

    # Build kwargs
    kwargs = {
        "strain_axis": args.strain_axis,
    }
    if "shear" in obj_name:
        kwargs["shear_component"] = args.shear_component
    if "diff_modulus" in obj_name:
        kwargs["smooth_window"] = args.smooth_window
        kwargs["smooth_polyorder"] = args.smooth_polyorder
    kwargs["strain_weight"] = args.strain_weight

    print(f"\nRunning: {obj_name}")
    print(f"  kwargs: {kwargs}")

    try:
        error, display = fn(results, str(csv_path), **kwargs)
        print(f"\n  Result:  error = {error:.8f}")
        if display:
            print(f"           display = {display}")
        ok = np.isfinite(error) and error >= 0
        print(f"  Status:  {'PASS' if ok else 'FAIL (non-finite or negative)'}")

        # ── Plot sim vs reference ─────────────────────────────────────
        _plot_real_data(results, ref_df, obj_name, kwargs)

        return ok
    except Exception as e:
        print(f"\n  FAILED with {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke-test objective functions (synthetic or real data)."
    )
    parser.add_argument(
        "--pickle", type=str, default=None,
        help="Path to a simulation output pickle file (e.g. result_files/output_data_0.pickle).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a reference CSV file (columns depend on the objective).",
    )
    parser.add_argument(
        "--objective", type=str, default="stress_strain",
        choices=list(OBJECTIVE_MAP.keys()),
        help="Which objective to evaluate (default: stress_strain).",
    )
    parser.add_argument(
        "--strain-axis", type=int, default=0, choices=[0, 1, 2],
        help="Strain axis: 0=x, 1=y, 2=z (default: 0).",
    )
    parser.add_argument(
        "--shear-component", type=int, default=0, choices=[0, 1],
        help="Shear tangential component: 0 or 1 (default: 0).",
    )
    parser.add_argument(
        "--strain-weight", type=float, default=0.1,
        help="Relative weight of strain MSE term (default: 0.1).",
    )
    parser.add_argument(
        "--smooth-window", type=int, default=5,
        help="Savitzky-Golay window for diff. modulus smoothing (default: 5).",
    )
    parser.add_argument(
        "--smooth-polyorder", type=int, default=2,
        help="Savitzky-Golay polynomial order (default: 2).",
    )
    args = parser.parse_args()

    if args.pickle and args.csv:
        # Real-data mode
        print("Real-data objective test\n" + "=" * 50)
        success = run_real_data_test(args)
    elif args.pickle or args.csv:
        print("ERROR: both --pickle and --csv must be provided for real-data mode.")
        success = False
    else:
        # Synthetic smoke-test mode
        print("Objective smoke tests (synthetic data)\n" + "=" * 50)
        success = run_tests()

    sys.exit(0 if success else 1)
