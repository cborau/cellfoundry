# Mock Reference Data for CellFoundry Optimization
#
# These CSV files provide synthetic "target" data used by the objective
# functions in optimizer/objectives.py.  Replace them with real experimental
# or high-fidelity simulation data when running actual optimizations.
#
# Files:
#   target_cell_count.csv                 — final alive-cell count targets (scalar)
#                                           Used by: final_cell_count_error
#                                           cell_type -1 = all cells combined
#
#   target_focad_per_cell.csv             — target average focal adhesions per cell
#                                           Used by: final_focad_per_cell_error
#
#   target_cell_population_timeseries.csv — alive-cell population over time
#                                           Used by: cell_population_error
#
#   target_stress_strain.csv              — stress-strain curve
#                                           Columns: strain [-], stress [kPa]
#                                           Used by: stress_strain_curve_error,
#                                                    shear_stress_strain_curve_error
#                                           Stress = force[nN] / face area[µm²] = kPa
#
#   target_differential_modulus.csv       — differential modulus K(ε) = dσ/dε vs strain
#                                           Columns: strain [-], differential_modulus [kPa]
#                                           Used by: differential_modulus_error,
#                                                    shear_differential_modulus_error
#
#   target_focad_attached_ratio.csv       — FOCAD attached ratio over time
#                                           Used by: focad_attached_ratio_error
#
#   target_organoid_size.csv              — target organoid radius (Rg) in µm
#                                           Used by: organoid_size_error
#
#   target_rg_n_large_clusters.csv        — target number of large RG rosette clusters
#                                           Used by: rg_rosette_2d_error (metric: n_large_rg_clusters)
#
#   target_rg_rosette_maturity.csv        — target mean rosette maturity over time
#                                           Used by: rg_rosette_2d_error (metric: mean_rosette_maturity)
#
# ---------------------------------------------------------------------------
# Inline scalar targets (no CSV file needed)
# ---------------------------------------------------------------------------
# Any objective function can be given an inline numeric target directly in the
# YAML config instead of a reference CSV.  Set:
#
#     reference: null
#     kwargs:
#       metric: <column_name>
#       target_value: <float>
#
# The optimizer dispatch layer (optimize.py) intercepts target_value, writes a
# temporary single-row CSV  (<metric>\n<value>\n)  and passes it as
# reference_path to the objective function — transparently, with no per-function
# changes required.
#
# This is equivalent to creating a scalar CSV file on the fly.  Use it whenever
# the desired target is a simple number rather than an experimentally measured
# curve.  Examples:
#
#   # Push toward circular rosettes (PCA compactness ≥ 0.75)
#   - function: rg_rosette_2d_error
#     reference: null
#     kwargs:
#       metric: mean_cluster_compactness
#       target_value: 0.75
#
#   # Target exactly 2 large clusters at final time
#   - function: rg_rosette_2d_error
#     reference: null
#     kwargs:
#       metric: n_large_rg_clusters
#       target_value: 2
#
# Note: target_value always compares the FINAL simulation value (last row of
# RG_ROSETTE_METRICS_OVER_TIME).  Use a time-series CSV for trajectory targets.
