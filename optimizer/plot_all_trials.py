#!/usr/bin/env python3
"""
Plot stress vs time and differential modulus vs strain for all Optuna trial folders.

Expected structure
------------------
optimizer/
└── optuna_diff_modulus_results/
    ├── trial_00000/
    │   └── output_data_0.pickle
    ├── trial_00001/
    │   └── output_data_0.pickle
    └── ...

This script:
1. Finds all trial_* folders
2. Loads output_data_0.pickle from each one
3. Extracts strain and stress using helpers from objectives.py
4. Infers time using MODEL_CONFIG.TIME_STEP * MODEL_CONFIG.SAVE_EVERY_N_STEPS
5. Plots:
   - stress vs time
    - differential modulus vs strain

If time cannot be inferred from MODEL_CONFIG, it falls back to saved-frame index.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from objectives import _compute_differential_modulus, _extract_sim_strain_stress


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
    
def normalize_trial_token(token: str) -> str:
    """
    Convert user trial selector into canonical folder name.

    Accepted examples:
        "7" -> "trial_00007"
        "00007" -> "trial_00007"
        "trial_00007" -> "trial_00007"
    """
    token = str(token).strip()

    if token.startswith("trial_"):
        suffix = token[len("trial_"):]
        if suffix.isdigit():
            return f"trial_{int(suffix):05d}"
        return token

    if token.isdigit():
        return f"trial_{int(token):05d}"

    return token


def build_trial_filter_set(trial_tokens) -> set[str] | None:
    """
    Convert CLI tokens into a set of canonical trial folder names.
    """
    if not trial_tokens:
        return None
    return {normalize_trial_token(tok) for tok in trial_tokens}


def infer_time_from_results(results: dict, n_points: int) -> np.ndarray:
    """
    Infer physical time from MODEL_CONFIG if available.

    Uses:
        dt_saved = TIME_STEP * SAVE_EVERY_N_STEPS
        time = arange(n_points) * dt_saved

    Falls back to plain saved-frame index if config is missing.
    """
    model_config = results.get("MODEL_CONFIG", None)

    if model_config is not None:
        time_step = getattr(model_config, "TIME_STEP", None)
        save_every = getattr(model_config, "SAVE_EVERY_N_STEPS", None)

        if time_step is not None and save_every is not None:
            dt_saved = float(time_step) * float(save_every)
            return np.arange(n_points, dtype=float) * dt_saved

    return np.arange(n_points, dtype=float)


def smooth_signal(y: np.ndarray, window: int) -> np.ndarray:
    """
    Centered moving average smoothing.
    window <= 1 means no smoothing.
    """
    if window <= 1:
        return y.copy()

    if window % 2 == 0:
        window += 1

    if window > len(y):
        return y.copy()

    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def find_trial_pickles(
    root: Path,
    pickle_name: str = "output_data_0.pickle",
    only_trials=None,
    skip_trials=None,
) -> list[Path]:
    """
    Find pickle files inside all trial_* folders under root, with optional
    inclusion/exclusion filters.
    """
    only_set = build_trial_filter_set(only_trials)
    skip_set = build_trial_filter_set(skip_trials)

    trial_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("trial_")
    )

    filtered_dirs = []
    for p in trial_dirs:
        trial_name = p.name

        if only_set is not None and trial_name not in only_set:
            continue

        if skip_set is not None and trial_name in skip_set:
            continue

        filtered_dirs.append(p)

    return [p / pickle_name for p in filtered_dirs if (p / pickle_name).exists()]


def plot_all_trials(
    root: Path,
    force_type: str = "normal",
    strain_axis: int = 1,
    shear_component: int = 0,
    stress_area_mode: str = "boundary_surface",
    fibre_section_area_um2: float | None = None,
    smooth_window: int = 5,
    smooth_polyorder: int = 2,
    save_dir: Path | None = None,
    show: bool = True,
    only_trials=None,
    skip_trials=None,
) -> None:
    pickle_paths = find_trial_pickles(root, only_trials=only_trials, skip_trials=skip_trials)

    if not pickle_paths:
        raise FileNotFoundError(f"No output_data_0.pickle files found in {root}")

    all_data = []
    skipped_trials = []

    for pkl_path in pickle_paths:
        trial_name = pkl_path.parent.name
        results = load_pickle(pkl_path)
        try:
            sim_strain, sim_stress = _extract_sim_strain_stress(
                results=results,
                force_type=force_type,
                strain_axis=strain_axis,
                shear_component=shear_component,
                stress_area_mode=stress_area_mode,
                fibre_section_area_um2=fibre_section_area_um2,
            )

            if not isinstance(sim_strain, pd.Series):
                sim_strain = pd.Series(sim_strain)
            if not isinstance(sim_stress, pd.Series):
                sim_stress = pd.Series(sim_stress)

            strain = sim_strain.to_numpy(dtype=float)
            stress = sim_stress.to_numpy(dtype=float)
            time = infer_time_from_results(results, len(stress))
            stress_smooth = smooth_signal(stress, smooth_window)
            diff_modulus = _compute_differential_modulus(
                strain=sim_strain,
                stress=sim_stress,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
            )
        except ValueError as exc:
            skipped_trials.append((trial_name, str(exc)))
            print(f"[skip] {trial_name}: {exc}")
            continue

        all_data.append({
            "trial": trial_name,
            "time": time,
            "strain": strain,
            "stress": stress,
            "stress_smooth": stress_smooth,
            "diff_modulus": diff_modulus,
        })

    if not all_data:
        raise ValueError("All trial curves were invalid and were skipped.")

    if skipped_trials:
        print(f"Skipped {len(skipped_trials)} invalid trial(s).")

    # Combined stress plot
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for item in all_data:
        if smooth_window > 1:
            ax1.plot(item["time"], item["stress_smooth"], label=item["trial"])
        else:
            ax1.plot(item["time"], item["stress"], label=item["trial"])

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Stress [kPa]")
    ax1.set_title("Stress vs time for all trials")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)

    # Combined differential modulus plot
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    for item in all_data:
        ax2.plot(item["strain"], item["diff_modulus"], label=item["trial"])

    ax2.set_xlabel("Strain [-]")
    ax2.set_ylabel("Differential modulus d(stress)/d(strain) [kPa]")
    ax2.set_title("Differential modulus vs strain for all trials")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        fig1.savefig(save_dir / "all_trials_stress_vs_time.png", dpi=300, bbox_inches="tight")
        fig2.savefig(save_dir / "all_trials_diff_modulus_vs_strain.png", dpi=300, bbox_inches="tight")

        for item in all_data:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            axes[0].plot(item["time"], item["stress"], alpha=0.4, label="raw")
            if smooth_window > 1:
                axes[0].plot(item["time"], item["stress_smooth"], linewidth=2, label="smoothed")
            axes[0].set_ylabel("Stress [kPa]")
            axes[0].set_title(item["trial"])
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(item["strain"], item["diff_modulus"])
            axes[1].set_xlabel("Strain [-]")
            axes[1].set_ylabel("d(stress)/d(strain) [kPa]")
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(save_dir / f"{item['trial']}_stress_and_diff_modulus.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("optimizer") / "optuna_diff_modulus_results",
        help="Folder containing the trial_* folders",
    )
    parser.add_argument(
        "--force-type",
        type=str,
        default="normal",
        choices=["normal", "shear"],
    )
    parser.add_argument(
        "--strain-axis",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0=x, 1=y, 2=z. Default is 1 because the summary suggests loading in y.",
    )
    parser.add_argument(
        "--shear-component",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--stress-area-mode",
        type=str,
        default="boundary_surface",
        choices=["boundary_surface", "per_fibre_area"],
    )
    parser.add_argument(
        "--fibre-section-area-um2",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Savitzky-Golay window used to pre-smooth stress before computing differential modulus.",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order used for stress pre-smoothing.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory where PNGs will be saved",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show plots interactively",
    )
    parser.add_argument(
        "--only-trials",
        nargs="+",
        default=None,
        help=(
            "Only plot these trials. Accepts trial numbers or folder names, "
            "e.g. 0 3 7 or trial_00000 trial_00003"
        ),
    )
    parser.add_argument(
        "--skip-trials",
        nargs="+",
        default=None,
        help=(
            "Skip these trials. Accepts trial numbers or folder names, "
            "e.g. 12 19 or trial_00012"
        ),
    )

    args = parser.parse_args()

    plot_all_trials(
        root=args.root,
        force_type=args.force_type,
        strain_axis=args.strain_axis,
        shear_component=args.shear_component,
        stress_area_mode=args.stress_area_mode,
        fibre_section_area_um2=args.fibre_section_area_um2,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
        save_dir=args.save_dir,
        show=not args.no_show,
        only_trials=args.only_trials,
        skip_trials=args.skip_trials,
    )


if __name__ == "__main__":
    main()