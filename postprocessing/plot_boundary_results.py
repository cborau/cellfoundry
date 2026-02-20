#!/usr/bin/env python3
"""
Plot boundary results from pickle file saved during simulation.

Usage:
    python plot_boundary_results.py [pickle_file]

If no file is provided, defaults to result_files/output_data.pickle

Example:
    python plot_boundary_results.py result_files/output_data_0.pickle
    python plot_boundary_results.py  # Uses result_files/output_data.pickle
"""

import pickle
import sys
import pathlib
from helper_module import ModelParameterConfig


def load_results(pickle_file):
    """Load results from pickle file."""
    pickle_path = pathlib.Path(pickle_file)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def plot_results(data, show=True):
    """
    Plot results using ModelParameterConfig methods.
    
    Parameters
    ----------
    data : dict
        Dictionary containing BPOS_OVER_TIME, BFORCE_OVER_TIME, BFORCE_SHEAR_OVER_TIME,
        POISSON_RATIO_OVER_TIME, OSCILLATORY_STRAIN_OVER_TIME, and MODEL_CONFIG
    show : bool
        Whether to display plots interactively
    """
    MODEL_CONFIG = data.get('MODEL_CONFIG')
    if MODEL_CONFIG is None:
        raise ValueError("MODEL_CONFIG not found in pickle file")
    
    bpos_over_time = data.get('BPOS_OVER_TIME')
    bforce_over_time = data.get('BFORCE_OVER_TIME')
    bforce_shear_over_time = data.get('BFORCE_SHEAR_OVER_TIME')
    poisson_ratio_over_time = data.get('POISSON_RATIO_OVER_TIME')
    oscillatory_strain_over_time = data.get('OSCILLATORY_STRAIN_OVER_TIME')
    
    # Plot main figure with all boundary data
    print("Plotting boundary positions, forces, and shear forces...")
    MODEL_CONFIG.plot_all(
        bpos_over_time=bpos_over_time,
        bforce_over_time=bforce_over_time,
        bforce_shear_over_time=bforce_shear_over_time,
        poisson_ratio_over_time=poisson_ratio_over_time,
        show=show,
    )
    
    # Plot oscillatory shear scatter if available
    if (
        MODEL_CONFIG.OSCILLATORY_SHEAR_ASSAY
        and oscillatory_strain_over_time is not None
        and bforce_shear_over_time is not None
    ):
        print("Plotting oscillatory shear scatter plots...")
        MODEL_CONFIG.plot_oscillatory_shear_scatter(
            oscillatory_strain_over_time=oscillatory_strain_over_time,
            bforce_shear_over_time=bforce_shear_over_time,
            max_strain=MODEL_CONFIG.MAX_STRAIN,
            show=show,
        )


def print_summary(data):
    """Print summary of loaded data."""
    MODEL_CONFIG = data.get('MODEL_CONFIG')
    if MODEL_CONFIG is not None:
        print("\n" + "=" * 50)
        MODEL_CONFIG.print_summary()
        print("=" * 50)
        MODEL_CONFIG.print_boundary_config()
        print("=" * 50 + "\n")


def main():
    default_pickle = "result_files/output_data_0.pickle"
    
    if len(sys.argv) < 2:
        pickle_file = default_pickle
    else:
        pickle_file = sys.argv[1]
    
    try:
        data = load_results(pickle_file)
        print_summary(data)
        plot_results(data, show=True)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
