import numpy as np
import matplotlib.pyplot as plt

# --------------------------- DATA ---------------------------
N_SPECIES = 2
DIFFUSION_COEFF_MULTI = [300.0, 300.0]
 
BOUNDARY_CONC_INIT_MULTI = [
    [50.0,50.0, 50.0, 50.0, 50.0, 50.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]
BOUNDARY_CONC_FIXED_MULTI = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]
 
CELL_RADIUS = 8.412
INIT_ECM_CONCENTRATION_VALS = [50.0, 0.0]
INIT_CELL_CONCENTRATION_VALS = [15.0, 0.0]
INIT_CELL_CONC_MASS_VALS = [
    x * (4/3 * 3.1415926 * CELL_RADIUS**3) for x in INIT_CELL_CONCENTRATION_VALS
]
INIT_ECM_SAT_CONCENTRATION_VALS = [0.0, 10.0]
 
INIT_CELL_CONSUMPTION_RATES = [0.001, 0.0]
INIT_CELL_PRODUCTION_RATES  = [0.0, 10.0]
INIT_CELL_REACTION_RATES    = [0.00018, 0.00018]
 
N = 21
BOUNDARY_COORDS = [100.0, -100.0, 100.0, -100.0, 100.0, -100.0]
TIME_STEP = 0.01
STEPS = 6000

L0_x = abs(BOUNDARY_COORDS[0] - BOUNDARY_COORDS[1])
L0_y = abs(BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3])
L0_z = abs(BOUNDARY_COORDS[4] - BOUNDARY_COORDS[5])

dx = L0_x / (N - 1)
dy = L0_y / (N - 1)
dz = L0_z / (N - 1)
ECM_VOXEL_VOLUME = dx * dy * dz

CELL_VOLUME = 4.0 / 3.0 * 3.1415926 * (CELL_RADIUS ** 3)


# --------------------------- Helpers ---------------------------
def apply_initial_boundary_conditions(C):
    """
    Apply BOUNDARY_CONC_INIT_MULTI to boundary voxels.
    Convention: [ +X, -X, +Y, -Y, +Z, -Z ]
    """
    for s in range(N_SPECIES):
        vals = BOUNDARY_CONC_INIT_MULTI[s]
        if vals[0] != -1.0: C[s, N-1, :, :] = vals[0]  # +X
        if vals[1] != -1.0: C[s, 0,   :, :] = vals[1]  # -X
        if vals[2] != -1.0: C[s, :, N-1, :] = vals[2]  # +Y
        if vals[3] != -1.0: C[s, :, 0,   :] = vals[3]  # -Y
        if vals[4] != -1.0: C[s, :, :, N-1] = vals[4]  # +Z
        if vals[5] != -1.0: C[s, :, :, 0  ] = vals[5]  # -Z


def enforce_fixed_boundary_conditions(C):
    """
    Apply BOUNDARY_CONC_FIXED_MULTI every step.
    If fixed value == -1.0: do nothing for that face.
    """
    for s in range(N_SPECIES):
        vals = BOUNDARY_CONC_FIXED_MULTI[s]
        if vals[0] != -1.0: C[s, N-1, :, :] = vals[0]
        if vals[1] != -1.0: C[s, 0,   :, :] = vals[1]
        if vals[2] != -1.0: C[s, :, N-1, :] = vals[2]
        if vals[3] != -1.0: C[s, :, 0,   :] = vals[3]
        if vals[4] != -1.0: C[s, :, :, N-1] = vals[4]
        if vals[5] != -1.0: C[s, :, :, 0  ] = vals[5]


def diffuse_3d_explicit_dirichlet(C):
    """
    Explicit 3D diffusion, update interior only, then enforce fixed boundaries.
    Assumes dx=dy=dz (true here).
    """
    Cn = C.copy()
    for s in range(N_SPECIES):
        D = DIFFUSION_COEFF_MULTI[s]
        r = D * TIME_STEP / (dx * dx)

        lap = (
            Cn[s, 2:, 1:-1, 1:-1] + Cn[s, :-2, 1:-1, 1:-1] +
            Cn[s, 1:-1, 2:, 1:-1] + Cn[s, 1:-1, :-2, 1:-1] +
            Cn[s, 1:-1, 1:-1, 2:] + Cn[s, 1:-1, 1:-1, :-2] -
            6.0 * Cn[s, 1:-1, 1:-1, 1:-1]
        )
        C[s, 1:-1, 1:-1, 1:-1] = Cn[s, 1:-1, 1:-1, 1:-1] + r * lap

    enforce_fixed_boundary_conditions(C)


def cell_ecm_exchange_and_metabolism(C_ecm_center, M_cell, C_cell):
    """
    Exchange part: backward Euler PhysiCell-style + conservative clamping
    Metabolism part: reaction inside the cell 
    """
    alpha = CELL_VOLUME / ECM_VOXEL_VOLUME

    # Exchange with voxel (per species)
    for i in range(N_SPECIES):
        C_ecm_old = C_ecm_center[i]
        C_sp_sat  = INIT_ECM_SAT_CONCENTRATION_VALS[i]
        M_cell_old = M_cell[i]

        c1 = TIME_STEP * alpha * INIT_CELL_PRODUCTION_RATES[i] * C_sp_sat
        c2 = 1.0 + TIME_STEP * alpha * (INIT_CELL_PRODUCTION_RATES[i] + INIT_CELL_CONSUMPTION_RATES[i])
        C_ecm_prop = (C_ecm_old + c1) / c2

        M_voxel_old = C_ecm_old * ECM_VOXEL_VOLUME
        M_voxel_prop = C_ecm_prop * ECM_VOXEL_VOLUME
        deltaM_voxel_prop = M_voxel_prop - M_voxel_old  # >0 secretion, <0 uptake

        # Clamp to avoid negative voxel or cell mass
        deltaM_voxel = deltaM_voxel_prop
        if deltaM_voxel_prop < 0.0:
            uptake = -deltaM_voxel_prop
            uptake_clamped = min(uptake, M_voxel_old)
            deltaM_voxel = -uptake_clamped
        elif deltaM_voxel_prop > 0.0:
            secretion = deltaM_voxel_prop
            secretion_clamped = min(secretion, M_cell_old)
            deltaM_voxel = secretion_clamped

        M_voxel_new = M_voxel_old + deltaM_voxel
        M_cell_new  = M_cell_old  - deltaM_voxel

        C_ecm_new  = M_voxel_new / ECM_VOXEL_VOLUME
        C_cell_new = M_cell_new  / CELL_VOLUME

        C_ecm_center[i] = C_ecm_new
        M_cell[i] = M_cell_new
        C_cell[i] = C_cell_new

    # Internal metabolism 
    C_cell[0] -= TIME_STEP * INIT_CELL_REACTION_RATES[0] * C_cell[0]  # consume species 0
    C_cell[1] += TIME_STEP * INIT_CELL_REACTION_RATES[1] * C_cell[0]  # produce species 1 from updated C_sp[0]

    # Mirror back to amounts
    for i in range(N_SPECIES):
        M_cell[i] = C_cell[i] * CELL_VOLUME

    return C_ecm_center, M_cell, C_cell


# --------------------------- INITIALIZE ---------------------------
C_ecm = np.zeros((N_SPECIES, N, N, N), dtype=np.float64)
for s in range(N_SPECIES):
    C_ecm[s, :, :, :] = INIT_ECM_CONCENTRATION_VALS[s]

apply_initial_boundary_conditions(C_ecm)
enforce_fixed_boundary_conditions(C_ecm)

C_cell = np.array(INIT_CELL_CONCENTRATION_VALS, dtype=np.float64)
M_cell = np.array(INIT_CELL_CONC_MASS_VALS, dtype=np.float64)

# Cell at (0,0,0) -> center voxel for symmetric boundaries
ci = cj = ck = (N - 1) // 2

# --------------------------- SIMULATE ---------------------------
times = np.arange(STEPS + 1) * TIME_STEP

cell_hist = np.zeros((STEPS + 1, N_SPECIES), dtype=np.float64)
ecm_hist  = np.zeros((STEPS + 1, N_SPECIES), dtype=np.float64)

cell_hist[0, :] = C_cell
ecm_hist[0, :]  = C_ecm[:, ci, cj, ck]

for t in range(1, STEPS + 1):
    # Cell <-> closest ECM voxel + internal metabolism
    C_center = C_ecm[:, ci, cj, ck].copy()
    C_center, M_cell, C_cell = cell_ecm_exchange_and_metabolism(C_center, M_cell, C_cell)
    C_ecm[:, ci, cj, ck] = C_center

    # Diffusion in ECM
    diffuse_3d_explicit_dirichlet(C_ecm)

    # Store time series
    cell_hist[t, :] = C_cell
    ecm_hist[t, :]  = C_ecm[:, ci, cj, ck]

# --------------------------- PLOT ---------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for s in range(N_SPECIES):
    ax = axes[s]
    ax.plot(times, cell_hist[:, s], label=f"Cell C_sp[{s}]")
    ax.plot(times, ecm_hist[:, s],  label=f"ECM (center voxel) C_sp[{s}]")
    ax.set_ylabel(f"Species {s} concentration")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

print("dx=dy=dz:", dx, "ECM_VOXEL_VOLUME:", ECM_VOXEL_VOLUME, "CELL_VOLUME:", CELL_VOLUME)
print("Final cell concentrations:", cell_hist[-1])
print("Final center ECM concentrations:", ecm_hist[-1])
