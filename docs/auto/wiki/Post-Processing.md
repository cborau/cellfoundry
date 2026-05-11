# Post Processing

The `postprocessing/` folder contains analysis and plotting utilities for simulation outputs in `result_files/` (CSV, pickle, and VTK-derived signals).

## Main Scripts

- [`report_focad.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/report_focad.py)
  - Loads simulation pickle or VTK data and exports focal adhesion metrics/polarity time series to CSV.
  - Builds summary tables (including last-20% statistics) and generates diagnostic plots.
- [`report_cell_population.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/report_cell_population.py)
  - Tracks per-step cell counts (total, alive/dead, proliferation proxy, death by cause) from pickle or VTK outputs.
  - Generates population time-series plots and CSV reports.
- [`report_matrix_remodeling.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/report_matrix_remodeling.py)
  - Tracks FNODE count, degradation, reinforcement, net remodeling, and elastic energy per step.
  - Produces time-series plots and CSV reports from pickle or VTK outputs.
- [`compare_linc_runs.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/compare_linc_runs.py)
  - Compares tagged LINC OFF/ON CSV outputs and produces comparison tables and figures.
- [`plot_migration_results.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/plot_migration_results.py)
  - Plots CELL migration metrics (speed, trajectories) from pickle files.
  - Supports optional target CSV for reference comparison lines.
- [`plot_migration_comparison.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/plot_migration_comparison.py)
  - Compares migration results across one or two conditions via violin plots and 3D trajectory plots.
  - Computes and plots directionality ratio curves per cell type.
- [`plot_migration_diff_profiles_comparison.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/plot_migration_diff_profiles_comparison.py)
  - Produces a fixed 2×4 figure comparing migration metrics, directionality, and diffusion-profile evolution for two conditions.
- [`plot_organoid_metrics.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/plot_organoid_metrics.py)
  - Plots organoid growth metrics over time (radius of gyration, sphericity, span, cell count per type).
  - Supports single or dual-condition comparison.
- [`plot_boundary_results.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/plot_boundary_results.py)
  - Loads boundary-related outputs from pickle and produces force/position/shear visualizations.
- [`plot_diffusion_results.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/plot_diffusion_results.py)
  - Demonstrates time-series plotting for concentration variables from VTK-derived datasets.

## Typical Outputs

- Time-series CSV files for metrics and polarity indicators.
- Summary CSV files for run-level comparison.
- PNG figures for trends, diagnostics, and side-by-side run comparisons.

## Recommended Usage

1. Run simulation and generate outputs in `result_files/`.
2. Use `report_focad.py` for baseline focal adhesion reports.
3. Use `report_cell_population.py` and `report_matrix_remodeling.py` for population and matrix diagnostics.
4. Use migration plotting scripts for speed and trajectory analysis.
5. Use `compare_linc_runs.py` for condition-to-condition analysis.
6. Use boundary/diffusion plotting scripts for targeted diagnostics.
