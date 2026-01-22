import matplotlib.pyplot as plt
from vtk_timeseries_loader import load_vtk_series, ts_by_point_id, ts_nearest_point, plot_vars_over_time

folder = "./result_files"

df_cells = load_vtk_series(folder, pattern="cells_t*.vtk")
df_ecm   = load_vtk_series(folder, pattern="ecm_data_t*.vtk")

"""
Notes on nearest-point tracking modes
------------------------------------

nearest_mode = "fixed_id"
- The nearest point to (x,y,z) is computed ONCE,
  at a reference time (by default the first available time).
- The corresponding point_id is then tracked across all times.
- Use this when you want to follow the SAME agent/voxel over time,
  even if it moves in space.

Typical use:
- Tracking the history of a specific cell
- Following one ECM voxel through time
- Lagrangian viewpoint (agent-based)

        Example:
            # Follow the same cell that was closest to (0,0,0) at t=0
            plot_vars_over_time(
                df_cells,
                ["concentration_species_0", "concentration_species_1"],
                nearest_xyz=(0.0, 0.0, 0.0),
                nearest_mode="fixed_id",
            )

nearest_mode = "per_time"
- At EACH time step, the nearest point to (x,y,z) is recomputed.
- The point_id may change from one time to the next.
- Use this when you want the value "at a fixed location in space".

Typical use:
- Sampling concentration at a fixed spatial probe
- Eulerian viewpoint (field-based)
- Post-processing similar to ParaView "Probe Location"

        Example:
            # Sample concentration at a fixed spatial point
            plot_vars_over_time(
                df_ecm,
                ["concentration_species_0", "concentration_species_1"],
                nearest_xyz=(0.5, 0.5, 0.5),
                nearest_mode="per_time",
            )

Important:
- If agents move (cells), "per_time" does NOT track the same cell.
- If points are on a fixed grid (ECM voxels), both modes may coincide.
"""


# # 1) Plot scalar over time for cell point_id = 0
# s = ts_by_point_id(df_cells, var="concentration_species_0", point_id=0)
# plt.figure()
# plt.plot(s.index, s.values)
# plt.xlabel("t")
# plt.ylabel("concentration_species_0")
# plt.show()

# # 2) Plot scalar over time for the cell closest to (x,y,z) at each time (dynamic nearest)
# s2 = ts_nearest_point(df_cells, var="concentration_species_0", x=0.0, y=0.0, z=0.0, mode="per_time")
# plt.figure()
# plt.plot(s2.index, s2.values)
# plt.xlabel("t")
# plt.ylabel("concentration_species_0")
# plt.show()

# # 3) Same, but choose nearest cell at the first time and track that same point_id over time
# s3 = ts_nearest_point(df_cells, var="concentration_species_1", x=0.0, y=0.0, z=0.0, mode="fixed_id")
# plt.figure()
# plt.plot(s3.index, s3.values)
# plt.xlabel("t")
# plt.ylabel("concentration_species_1")
# plt.show()

# plot_vars_over_time(
#     {"cells": df_cells, "ecm": df_ecm},
#     ["concentration_species_0", "concentration_species_1"],
#     point_id=0,
#     use_time_col="t",
#     title="Species over time at point_id=0",
# )

plot_vars_over_time(
    {"cells": df_cells, "ecm": df_ecm},
    [f"concentration_species_{i}" for i in range(2)],
    nearest_xyz=(0.0, 0.0, 0.0),
    nearest_mode="per_time",   # or "per_time"
    use_time_col="t",
    title="Nearest-point species traces",
    subplot_mode="by_species",
)

# plot_vars_over_time(
#      {"cells": df_cells, "ecm": df_ecm},
#      [f"concentration_species_{i}" for i in range(2)],
#      nearest_xyz=(0.0, 0.0, 0.0),
#      nearest_mode="per_time",
#      use_time_col="t",
#      title="Nearest-point species traces (by species)",
#      subplot_mode="by_dataset",
#  )

