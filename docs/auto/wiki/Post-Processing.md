# Post Processing

The `postprocessing/` folder contains analysis and plotting utilities for simulation outputs in `result_files/` (CSV, pickle, and VTK-derived signals).

## Main Scripts

- [`focad_report.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/focad_report.py)
  - Loads simulation pickle data and exports focal adhesion metrics/polarity time series to CSV.
  - Builds summary tables (including last-20% statistics) and generates diagnostic plots.
- [`compare_linc_runs.py`](https://github.com/cborau/cellfoundry/blob/master/postprocessing/compare_linc_runs.py)
  - Compares tagged LINC OFF/ON CSV outputs and produces comparison tables and figures.
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
2. Use `focad_report.py` for baseline reports.
3. Use `compare_linc_runs.py` for condition-to-condition analysis.
4. Use boundary/diffusion plotting scripts for targeted diagnostics.
