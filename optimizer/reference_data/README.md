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
