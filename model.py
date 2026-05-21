# +====================================================================+
# | Model: CELLFOUNDRY                                                 |
# | Last update: 26/02/2026 - 13:28:03                                 |
# +====================================================================+


# +====================================================================+
# | IMPORTS                                                            |
# +====================================================================+
import sys as _sys
import sys                                     # keep 'sys' available for existing usage
import pathlib
_ORIGINAL_ARGV = list(_sys.argv)          # snapshot BEFORE pyflamegpu touches sys.argv
from pyflamegpu import *
import time, math
from dataclasses import make_dataclass
import pandas as pd
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
import check_hard_coded_values
from helper_module import compute_expected_boundary_pos_from_corners, build_model_config_from_namespace, load_fibre_network, getRandomCoordsAroundPoint, compute_u_ref_from_anchor_pos, getRadialOrientations, build_save_data_context, save_data_to_file_step, print_fibre_calibration_summary, print_focad_birth_calibration_summary, apply_param_overrides, load_param_overrides_from_cli, loadCachedCellInitialization, generateCellInitializationData, recompute_derived_params, getRandomOrientationOnPlane, getCoordsOnPlane, getCellTypeList

# TODO LIST:
# A- Add cell guidance by fibre orientation (cells prefer to move along the main fibre orientation, which could be implemented by making them prefer to move towards areas where the fibre segments are more aligned in a certain direction)
# B- Add VESSEL agents 

start_time = time.time()

# +====================================================================+
# | GLOBAL SIMULATION PARAMETERS                                       |
# +====================================================================+
# Set whether to run single model or ensemble, agent population size, and simulation steps 
ENSEMBLE = False
ENSEMBLE_RUNS = 0
VISUALISATION = False  # Change to false if pyflamegpu has not been built with visualisation support
DEBUG_PRINTING = True
ABORT_ON_UNSTABLE_FNODE_MOVE = False  # If True, abort when any FNODE moves farther than one segment rest length in a single step.
PAUSE_EVERY_STEP = False  # If True, the visualization stops every step until P is pressed
SAVE_PICKLE = True  # If True, dumps model configuration into a pickle file for post-processing
SHOW_PLOTS = False  # Show plots at the end of the simulation
SAVE_DATA_TO_FILE = True  # If true, agent data is exported to .vtk file every SAVE_EVERY_N_STEPS steps
SAVE_EVERY_N_STEPS = 20 # Affects both the .vtk files and the Dataframes storing boundary data
DEBUG_PRINT_INTERVAL = 0  # [steps] Print live debug stats every N steps (0 = disabled). Only active when INCLUDE_RG_VARIABLES is True.

CURR_PATH = pathlib.Path(__file__).resolve().parent
RES_PATH = CURR_PATH / 'result_files'
RES_PATH.mkdir(parents=True, exist_ok=True)
EPSILON = 0.0000000001
CELL_INIT_CACHE_DIR = CURR_PATH

print("Executing in ", CURR_PATH)
# Minimum number of ECM agents per direction (x,y,z). 
# If domain is not cubical, N is asigned to the shorter dimension and more agents are added to the longer ones
# NOTE: ECM agents are always present (mandatory) eventhough they are only used when INCLUDE_DIFFUSION is True. If there is no diffusion, set N to a small value to reduce computational cost.
# ----------------------------------------------------------------------
N = 21

# Time simulation parameters
# ----------------------------------------------------------------------
TIME_STEP = 1.0 # s. WARNING: diffusion and cell migration events might need different scales
STEPS = 2

# +====================================================================+
# | BOUNDARY CONDITIONS                                                |
# +====================================================================+

# Boundary interactions and mechanical parameters
# ----------------------------------------------------------------------
ECM_K_ELAST = 0.2  # [nN/um]
ECM_D_DUMPING = 0.04  # [nN·s/um]
ECM_ETA = 0.15  # [nN·s/µm] Effective drag for overdamped FNODE motion (calibration parameter)

#BOUNDARY_COORDS = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]  # +X,-X,+Y,-Y,+Z,-Z
BOUNDARY_COORDS = [150.0, -150.0, 150.0, -150.0, 150.0, -150.0]  # microdevice dimensions in um
#BOUNDARY_COORDS = [coord / 1000.0 for coord in BOUNDARY_COORDS] # in mm
BOUNDARY_DISP_RATES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# perpendicular to each surface (+X,-X,+Y,-Y,+Z,-Z) [um/s]
BOUNDARY_DISP_RATES_PARALLEL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# parallel to each surface (+X_y,+X_z,-X_y,-X_z,+Y_x,+Y_z,-Y_x,-Y_z,+Z_x,+Z_y,-Z_x,-Z_y)[um/s]

POISSON_DIRS = [0, 1]  # 0: xdir, 1:ydir, 2:zdir. poisson_ratio ~= -incL(dir1)/incL(dir2) dir2 is the direction in which the load is applied
ALLOW_BOUNDARY_ELASTIC_MOVEMENT = [0, 0, 0, 0, 0, 0]  # [bool]
RELATIVE_BOUNDARY_STIFFNESS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
BOUNDARY_STIFFNESS_VALUE = 10.0  # nN/um
BOUNDARY_DUMPING_VALUE = 5.0
BOUNDARY_STIFFNESS = [BOUNDARY_STIFFNESS_VALUE * x for x in RELATIVE_BOUNDARY_STIFFNESS]
BOUNDARY_DUMPING = [BOUNDARY_DUMPING_VALUE * x for x in RELATIVE_BOUNDARY_STIFFNESS]
CLAMP_AGENT_TOUCHING_BOUNDARY = [1, 1, 1, 1, 1, 1]# +X,-X,+Y,-Y,+Z,-Z [bool] - shear assay
#CLAMP_AGENT_TOUCHING_BOUNDARY = [1, 1, 1, 1, 1, 1]# +X,-X,+Y,-Y,+Z,-Z [bool]
ALLOW_AGENT_SLIDING = [0, 0, 0, 0, 0, 0]# +X,-X,+Y,-Y,+Z,-Z [bool]

if any(rate != 0.0 for rate in BOUNDARY_DISP_RATES_PARALLEL) or any(rate != 0.0 for rate in BOUNDARY_DISP_RATES):
    MOVING_BOUNDARIES = True
else:   
    MOVING_BOUNDARIES = False

# Adjust number of agents if domain is not cubical
# ----------------------------------------------------------------------
# Calculate the differences between opposite pairs along each axis
diff_x = abs(BOUNDARY_COORDS[0] - BOUNDARY_COORDS[1])
diff_y = abs(BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3])
diff_z = abs(BOUNDARY_COORDS[4] - BOUNDARY_COORDS[5])

# Check if the differences are equal
if diff_x == diff_y == diff_z:
    ECM_AGENTS_PER_DIR = [N, N, N] # cubical domain
else:
    min_length = min(diff_x, diff_y, diff_z)
    dist_agents = min_length / (N - 1)
    ECM_AGENTS_PER_DIR = [int(diff_x / dist_agents) + 1, int(diff_y / dist_agents) + 1, int(diff_z / dist_agents) + 1]
    # Redefine BOUNDARY_COORDS due to rounding values
    diff_x = dist_agents * (ECM_AGENTS_PER_DIR[0] - 1)
    diff_y = dist_agents * (ECM_AGENTS_PER_DIR[1] - 1)
    diff_z = dist_agents * (ECM_AGENTS_PER_DIR[2] - 1)
    BOUNDARY_COORDS = [round(diff_x / 2, 2), -round(diff_x / 2, 2), round(diff_y / 2, 2), -round(diff_y / 2, 2), round(diff_z / 2, 2), -round(diff_z / 2, 2)] 
    
L0_x = abs(BOUNDARY_COORDS[0] - BOUNDARY_COORDS[1])
L0_y = abs(BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3])
L0_z = abs(BOUNDARY_COORDS[4] - BOUNDARY_COORDS[5])

ECM_POPULATION_SIZE = ECM_AGENTS_PER_DIR[0] * ECM_AGENTS_PER_DIR[1] * ECM_AGENTS_PER_DIR[2]
ECM_ECM_EQUILIBRIUM_DISTANCE = L0_x / (ECM_AGENTS_PER_DIR[0] - 1) # in units, all agents are evenly spaced
ECM_BOUNDARY_INTERACTION_RADIUS = 0.05
ECM_BOUNDARY_EQUILIBRIUM_DISTANCE = 0.0
ECM_VOXEL_VOLUME = (L0_x / (ECM_AGENTS_PER_DIR[0] - 1)) * (L0_y / (ECM_AGENTS_PER_DIR[1] - 1)) * (L0_z / (ECM_AGENTS_PER_DIR[2] - 1))
MAX_SEARCH_RADIUS_VASCULARIZATION = ECM_ECM_EQUILIBRIUM_DISTANCE  # this strongly affects the number of bins and therefore the memory allocated for simulations (more bins -> more memory -> faster (in theory))
MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION = ECM_ECM_EQUILIBRIUM_DISTANCE # this radius is used to find ECM agents
# NOTE: MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION is defined after CELL_RADIUS (see below)

OSCILLATORY_SHEAR_ASSAY = False  # if True, BOUNDARY_DISP_RATES_PARALLEL options are overrun but used to make the boundaries oscillate in their corresponding planes following a sin() function
MAX_STRAIN = 0.25  # maximum strain applied during oscillatory shear assay (used to compute OSCILLATORY_AMPLITUDE)
OSCILLATORY_AMPLITUDE = MAX_STRAIN * (BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3])  # range [0-1] * domain size in the direction of oscillation
OSCILLATORY_FREQ = 0.05  # strain oscillation frequency [s^-1]
OSCILLATORY_W = 2 * math.pi * OSCILLATORY_FREQ * TIME_STEP
# Compute expected boundary positions after motion, WARNING: make sure the direction matches with OSCILLATORY_AMPLITUDE definition
MAX_EXPECTED_BOUNDARY_POS_OSCILLATORY = 0.25 * (BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3]) + BOUNDARY_COORDS[2]  # max pos reached at sin()=1

# Parallel disp rate values are overrun in oscillatory assays
# ----------------------------------------------------------------------
if OSCILLATORY_SHEAR_ASSAY:
    for d in range(12):
        if abs(BOUNDARY_DISP_RATES_PARALLEL[d]) > 0.0:
            BOUNDARY_DISP_RATES_PARALLEL[d] = OSCILLATORY_AMPLITUDE * math.cos(
                OSCILLATORY_W * 0.0) * OSCILLATORY_W / TIME_STEP  # cos(w*t)*w is used because the slope of the sin(w*t) function is needed. Expressed in units/sec


# +====================================================================+
# | FIBRE NETWORK PARAMETERS                                           |
# +====================================================================+
INCLUDE_FIBRE_NETWORK = False
NETWORK_FILE = 'network_medium_density.pkl'  # path to the .pkl file with node_coords + connectivity
ALLOW_IRREGULAR_NETWORK = False  # default: False, meaning that all boundaries must have network nodes attached (e.g. a network going from -y to y and touching the other boundaries should have this variable set to True)

# Fitting parameters for the fiber strain-stiffening phenomena
# Ref: https://bio.physik.fau.de/publications/Steinwachs%20Nat%20Meth%202016.pdf
# ----------------------------------------------------------------------
BUCKLING_COEFF_D0 = 0.15
STRAIN_STIFFENING_COEFF_DS = 0.85
CRITICAL_STRAIN = 0.1
MAX_STRAIN_K_FACTOR = 12.0  # Cap on the strain-dependent stiffness multiplier (plateau / damage limit)

MAX_CONNECTIVITY = 8 # must match hard-coded C++ values
# NOTE: These are calibrated model parameters (effective segment-level mechanics), not universal material constants.
# They depend on collagen type/concentration, crosslinking, architecture and coarse-graining choices.
FIBRE_SEGMENT_K_ELAST = 0.08  # [nN/um] Effective fibre-segment stiffness (baseline for tuning)
FIBRE_SEGMENT_D_DUMPING = 0.0  # [nN*s/um] Effective fibre-segment damping (baseline for tuning)
FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = 5 # WARNING: must match the value used in network generation
FIBRE_SECTION_AREA_UM2 = 0.05  # [um^2] Approximate collagen-fibre cross-section used for effective stress normalization
FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS = 0.05
FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE = 0.0
MAX_SEARCH_RADIUS_FNODES = FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE / 10.0 # must me smaller than FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE
FIBRE_NODE_REPULSION_K = 0.2 * FIBRE_SEGMENT_K_ELAST  # [nN/um] Short-range FNODE-FNODE exclusion stiffness (kept below segment stiffness)
# WARNING: THESE VARIABLES SIZE DEPENDS ON N_CELL_TYPES (DEFINED BELOW IN THE CELL PARAMETERS SECTION)
# FNODE remodeling (degradation/deposition + birth/death)
INCLUDE_NETWORK_REMODELING = False
FNODE_DEGRADATION_RATE = [5.0e-4, 5.0e-4, 5.0e-4]  # [1/s] per-neighbor degradation contribution (per cell-type)
FNODE_DEPOSITION_RATE = [2.0e-4, 2.0e-4, 2.0e-4]  # [1/s] baseline repair/deposition (per cell-type)
FNODE_CELL_DEGRADATION_RADIUS = 0.75 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE  # [um] cutoff radius (scalar)
# CELL-driven FNODE birth — per-cell-type arrays
FNODE_BIRTH_K_0 = [2.0, 2.0, 2.0]  # [1/s] baseline probability rate for CELL-driven FNODE birth
FNODE_BIRTH_K_MAX = [2.0, 2.0, 2.0]  # [1/s] gated additive birth-rate gain
FNODE_BIRTH_SPECIES_INDEX = 0  # species index (global, same for all types)
FNODE_BIRTH_K_C = [5.0, 5.0, 5.0]  # concentration half-saturation for birth gate
FNODE_BIRTH_HILL_CONC = [2.0, 2.0, 2.0]
FNODE_BIRTH_K_SIGMA = [0.1, 0.1, 0.1]  # [kPa] stress half-saturation for birth gate
FNODE_BIRTH_HILL_SIGMA = [1.0, 1.0, 1.0]
FNODE_BIRTH_RADIUS = [0.5 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, 0.5 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, 0.5 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE]  # [um] newborn offset around CELL center
FNODE_BIRTH_LINK_MAX_DISTANCE = [2.0 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, 2.0 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, 2.0 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE]   # [um] parent FNODE search radius
FNODE_BIRTH_REFRACTORY = [20.0, 20.0, 20.0]  # [s]


# +====================================================================+
# | DIFFUSION PARAMETERS                                               |
# +====================================================================+
INCLUDE_DIFFUSION = False
N_SPECIES = 3
# Use check_hard_coded_values.py to automatically update all c++ files using N_SPECIES
# Use tools/resize_array_variables.py to automatically resize all per-species arrays.
DIFFUSION_COEFF_MULTI = [5.0, 5.0, 5.0]  # diffusion coefficient in [um^2/s] per specie
ECM_DEGRADATION_RATE_MULTI = [0.0, 0.0, 0.0]  # first-order ECM degradation [1/s] per species (0 = no degradation)
BOUNDARY_CONC_INIT_MULTI = [[2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # add as many lines as different species

BOUNDARY_CONC_FIXED_MULTI = [[2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # add as many lines as different species
HETEROGENEOUS_DIFFUSION = False  # if True, diffusion coefficient is multiplied by (1 - local ECM density) to simulate hindered diffusion through the ECM. WARNING: this is a very simple approximation of the phenomenon and highly depends on grid density (N). 
# +====================================================================+
# | CELL PARAMETERS                                                    |
# +====================================================================+
# --- Per-cell-type configuration ------------------------------------------
# N_CELL_TYPES controls the length of all per-type array variables below.
# WARNING: must match the hard-coded N_CELL_TYPES in every .cpp agent
# function that reads per-type environment arrays (same pattern as N_SPECIES).
# When a parameter below is given as a *scalar*, the helper function
# _broadcast_cell_type_params() will replicate it to N_CELL_TYPES copies
# so that every cell type shares the same default value.
# Use tools/resize_cell_types.py to automatically resize all per-type arrays and automatically update all c++ files using N_CELL_TYPES.
# --------------------------------------------------------------------------
N_CELL_TYPES = 3

INCLUDE_CELLS = True
INCLUDE_CELL_CELL_INTERACTION = True # If True, cells interact with each other through short-range repulsion and adhesion forces. 
INCLUDE_CELL_CYCLE = False # If True, cells go through a simplified cell cycle with G1, S, G2 and M phases, which can affect their behavior. Also includes birth/death dynamics (WARNING: USER-DEFINED in cell_cycle.cpp).
DEAD_CELLS_DISAPPEAR = False  # If True, dead CELL agents are removed; if False, they remain inert with dead=1.
PERIODIC_BOUNDARIES_FOR_CELLS = False
INCLUDE_CELL_FNODE_REPULSION = False
N_CELLS = 100
ORGANOID_ASSAY = False  # If True, cells are initialized in a small cluster in the center of the domain to simulate an organoid. If False, they are initialized randomly in the whole domain.
ORGANOID_INIT_RADIUS = 20.0  # [um] Radius of the initial cell cluster when ORGANOID_ASSAY is True. Cells are placed randomly within a sphere of this radius centered at the domain origin.
ORGANOID_ORIENTATION_NOISE = 0.0  # [rad] Std-dev of Gaussian angular noise added to the initial radially-outward cell orientations when ORGANOID_ASSAY is True. 0 = perfectly radial; ~0.3 = ~17 deg RMS jitter; increase towards pi for fully random.
MONOLAYER_ASSAY = True  # If True, cells are initialized in a monolayer at the bottom of the domain (e.g. near -Z boundary) to simulate a 2D culture. 
MONOLAYER_CELL_TYPE_RATIOS = [70, 20, 10]  # Relative proportions of each cell type in the MONOLAYER_ASSAY init. Must have length N_CELL_TYPES.
MONOLAYER_CLUSTER_RADIUS = None  # [µm] If set, cells are seeded inside a disc of this radius centred at (0,0) on the monolayer plane, rather than spread over the full domain. Useful when the ECM domain is large but cells must start close enough to interact mechanically.
MONOLAYER_Z = None  # [µm] z-coordinate of the monolayer plane for MONOLAYER_ASSAY. If None, defaults to coord_boundary_z_neg (the bottom boundary). Set to 0.0 to place cells in the centre of the domain, away from absorbing boundaries.


# Per-cell-type mechanical & morphological properties
# Each is a list of length N_CELL_TYPES.  A scalar is broadcast to all types.
CELL_K_ELAST = [2.0, 2.0, 2.0]  # [nN/um]
CELL_D_DUMPING = [0.4, 0.4, 0.4]  # [nN·s/um]
CELL_RADIUS = [5.0, 5.0, 5.0] # [um]
CELL_NUCLEUS_RADIUS = [r / 2 for r in CELL_RADIUS] # [um]
CELL_SPEED_REF = [0.00041817020062396415, 0.0006199050301202626, 0.0004034913399763545] # [um/s] Another option is to define it according to grid distance ECM_ECM_EQUILIBRIUM_DISTANCE / TIME_STEP / X. WARNING: if cell speed is too high, consider increasing N or reducing TIME_STEP.
BROWNIAN_MOTION_STRENGTH_FACTOR = [0.001, 0.001, 0.001]
BROWNIAN_MOTION_STRENGTH = [s * f for s, f in zip(CELL_SPEED_REF, BROWNIAN_MOTION_STRENGTH_FACTOR)]  # [um/s] # [um/s] Strength of random movement added to cell velocity.
ROTATIONAL_DIFFUSION_RATE = [0.001, 0.001, 0.001]  # [rad^2/s] Rotational diffusion coefficient per cell type. Controls how fast cell orientation decorrelates (persistence time ~ 1/(2*D_rot)). Set > 0 for tortuous random-walk trajectories. e.g. D_rot = 0.001 gives a persistence time of ~500s. 
CELL_CELL_REPULSION_K = [2.0 * k for k in CELL_K_ELAST]  # [nN/um] contact exclusion stiffness
CELL_CELL_ADHESION_K = [0.2 * k for k in CELL_K_ELAST]  # [nN/um] weak cohesion in near-contact shell
CELL_CELL_ADHESION_RANGE = [1.0 * r for r in CELL_RADIUS]  # [um] adhesive shell thickness outside contact
# Search radius for the cell-cell spatial message.
# Must cover the farthest distance at which two cells can interact:
#   contact distance (r1 + r2) + adhesion shell (max of CELL_CELL_ADHESION_RANGE)
# Using 3 * max(CELL_RADIUS) provides ~ 2*(R + 0.5*R) with a small margin.
MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION = 3.0 * max(CELL_RADIUS)  # [um]
CELL_CELL_DV_MAX = [0.5 * s for s in CELL_SPEED_REF]  # [um/s] cap for cell-cell interaction velocity contribution
CELL_FNODE_REPULSION_K = [0.5 * k for k in CELL_K_ELAST]  # [nN/um] exclusion stiffness around fibre nodes
CELL_FNODE_EXCLUSION_DISTANCE = list(CELL_RADIUS)  # [um] minimum distance from cell center to fibre nodes
CELL_FNODE_DV_MAX = [0.5 * s for s in CELL_SPEED_REF]  # [um/s] cap for cell-fnode interaction velocity contribution
print(f'Initial cell speed reference (per type): {CELL_SPEED_REF} um/s')   
print(f'Initial Brownian motion strength (per type): {BROWNIAN_MOTION_STRENGTH} um/s')
print(f'Rotational diffusion rate (per type): {ROTATIONAL_DIFFUSION_RATE} rad^2/s')

# LUMEN parameters — only active when INCLUDE_CELLS and (ORGANOID_ASSAY or MONOLAYER_ASSAY) are both True.
INCLUDE_LUMEN = False  # If True, LUMEN agents are present (secreted by cells in the apical direction).
LUMEN_RADIUS = 3.0  # [um] Radius of a single LUMEN droplet.
LUMEN_ETA = 0.15  # [nN·s/um] Overdamped drag coefficient for LUMEN agents.
LUMEN_K_LUMEN_LUMEN_REPULSION = 4.0  # [nN/um] Stiffness of volume-exclusion repulsion between LUMEN agents.
LUMEN_K_LUMEN_LUMEN_ADHESION = 0.8  # [nN/um] Cohesive surface-tension adhesion strength between LUMEN agents.
LUMEN_LUMEN_ADHESION_RANGE = 1.5 * LUMEN_RADIUS  # [um] Thickness of the adhesive shell outside contact distance.
LUMEN_K_LUMEN_CELL_REPULSION = 3.0  # [nN/um] Stiffness of repulsion between LUMEN and CELL agents.
LUMEN_LUMEN_CELL_DV_MAX = 0.5 * max(CELL_SPEED_REF)  # [um/s] Velocity cap for LUMEN-CELL interaction contributions.
MAX_SEARCH_RADIUS_LUMEN_LUMEN_INTERACTION = 3.0 * LUMEN_RADIUS  # [um] Spatial message search radius for LUMEN-LUMEN interactions.
MAX_SEARCH_RADIUS_LUMEN_CELL_INTERACTION = 2.0 * (max(CELL_RADIUS) + LUMEN_RADIUS)  # [um] Spatial message search radius for LUMEN-CELL interactions.
LUMEN_SECRETION_RATE = 5e-4  # [1/s] Probability rate of secreting a LUMEN droplet per cell per second.
LUMEN_SECRETION_COOLDOWN = 100.0  # [s] Refractory time after a CELL secretes a LUMEN droplet before it can secrete another.
LUMEN_DIFFUSION_COEFF_MULTI = [5.0, 5.0, 5.0]  # [um^2/s] Diffusion coefficients inside LUMEN voxels (overrides fibre-adjusted D_sp).

# +====================================================================+
# | VASCULARIZATION PARAMETERS                                         |
# +====================================================================+
INCLUDE_VASCULARIZATION = False  # If True, VASC agents representing a vascular network are included.
VASC_NETWORK_FILE = 'vascular_network.pickle'  # Path (relative to CURR_PATH) to the vascular network pickle file.
INIT_VASCULARIZATION_CONCENTRATION_VALS = [2.5, 2.5, 2.5]  # Initial concentration of each species in VASC agents.
MAX_VASC_CONNECTIVITY = 2  # Maximum number of parents or children per VASC node (fixed array size). Defined during vascular network generation (tools/generate_vascular_network.py).
INCLUDE_VASCULAR_CELL_RECRUITMENT = False  # If True, cells can be recruited from the vascular network into the simulation based on proximity and chemical cues.

# Per-cell-type cell cycle timing
debug_acc = 1.0 # [s] If > 1.0, accelerates the cell cycle for faster testing (all durations are divided by this value). Set to 1.0 for realistic timing.
_g1 = 10.0 * 3600 / debug_acc
_s  = 8.0 * 3600 / debug_acc
_g2 = 4.0 * 3600 / debug_acc
_m  = 2.0 * 3600 / debug_acc
CYCLE_PHASE_G1_DURATION = [_g1, _g1, _g1]  # [s]
CYCLE_PHASE_S_DURATION  = [_s, _s, _s]     # [s]
CYCLE_PHASE_G2_DURATION = [_g2, _g2, _g2]  # [s]
CYCLE_PHASE_M_DURATION  = [_m, _m, _m]     # [s]
CYCLE_PHASE_G1_START = [0.0, 0.0, 0.0]  # [s]
CYCLE_PHASE_S_START  = [CYCLE_PHASE_G1_DURATION[0], CYCLE_PHASE_G1_DURATION[1], CYCLE_PHASE_G1_DURATION[2]]
CYCLE_PHASE_G2_START = [CYCLE_PHASE_G1_DURATION[0] + CYCLE_PHASE_S_DURATION[0], CYCLE_PHASE_G1_DURATION[1] + CYCLE_PHASE_S_DURATION[1], CYCLE_PHASE_G1_DURATION[2] + CYCLE_PHASE_S_DURATION[2]]
CYCLE_PHASE_M_START  = [CYCLE_PHASE_G1_DURATION[0] + CYCLE_PHASE_S_DURATION[0] + CYCLE_PHASE_G2_DURATION[0], CYCLE_PHASE_G1_DURATION[1] + CYCLE_PHASE_S_DURATION[1] + CYCLE_PHASE_G2_DURATION[1], CYCLE_PHASE_G1_DURATION[2] + CYCLE_PHASE_S_DURATION[2] + CYCLE_PHASE_G2_DURATION[2]]
CELL_CYCLE_DURATION  = [CYCLE_PHASE_G1_DURATION[0] + CYCLE_PHASE_S_DURATION[0] + CYCLE_PHASE_G2_DURATION[0] + CYCLE_PHASE_M_DURATION[0], CYCLE_PHASE_G1_DURATION[1] + CYCLE_PHASE_S_DURATION[1] + CYCLE_PHASE_G2_DURATION[1] + CYCLE_PHASE_M_DURATION[1], CYCLE_PHASE_G1_DURATION[2] + CYCLE_PHASE_S_DURATION[2] + CYCLE_PHASE_G2_DURATION[2] + CYCLE_PHASE_M_DURATION[2]]  # typically 24h

# Per-cell-type cycle multipliers 
DIVISION_RATE_MULTIPLIER = [1.0, 1.15, 0.85]  # [-] scales division probability per cell type
DAMAGE_ACCUMULATION_MULTIPLIER = [1.0, 0.85, 1.25]  # [-] scales damage accrual per cell type
DAMAGE_REPAIR_MULTIPLIER = [1.0, 1.10, 0.85]  # [-] scales damage repair per cell type
DAMAGE_DEATH_THRESHOLD = [1.0, 1.0, 0.80]  # [-] damage threshold for death per cell type

# Per-cell-type species multipliers 
# These multiply the base per-species rates (INIT_CELL_*_RATES) for each cell type.
CELL_CONSUMPTION_MULTIPLIER = [1.0, 1.0, 1.0]  # [-] per-type scaling of consumption rates
CELL_PRODUCTION_MULTIPLIER = [1.0, 1.0, 1.0]   # [-] per-type scaling of production rates
CELL_REACTION_MULTIPLIER = [1.0, 1.0, 1.0]      # [-] per-type scaling of reaction rates
CELL_INIT_CONCENTRATION_MULTIPLIER = [1.0, 1.0, 1.0]  # [-] per-type scaling of initial species concentrations

INIT_ECM_CONCENTRATION_VALS = [2.5, 2.5, 2.5]  # initial concentration of each species on the ECM agents
INIT_CELL_CONCENTRATION_VALS = [2.5, 2.5, 2.5]  # initial concentration of each species on the CELL agents
# Reference concentrations (actual per-agent mass is computed at init using per-type volume & conc multiplier).
INIT_CELL_CONC_MASS_VALS = [x for x in INIT_CELL_CONCENTRATION_VALS]
INIT_ECM_SAT_CONCENTRATION_VALS = [0.0, 0.0, 0.0]  # initial saturation concentration of each species on the ECM agents
INIT_CELL_CONSUMPTION_RATES = [0.0, 0.0, 0.0]  # base consumption rate of each species by the CELL agents 
INIT_CELL_PRODUCTION_RATES = [0.0, 0.0, 0.0]  # base production rate of each species by the CELL agents 
DE_NOVO_PRODUCTION = [0, 0, 0]  # [bool per species] 1 = cells synthesise this species de-novo (no intracellular-pool
                                # depletion on secretion; PhysiCell-style source term). 0 = mass-conservative:
                                # the cell can only secrete what it already holds in its intracellular pool.
INIT_CELL_REACTION_RATES = [0.0, 0.0, 0.0]  # base metabolic reaction rates of each species by the CELL agents 

# Species index mapping for death pathways
OXYGEN_SPECIES_INDEX = 0    # index into C_sp[] used as oxygen proxy in cell_cycle
NUTRIENT_SPECIES_INDEX = 1  # index into C_sp[] used as nutrient proxy in cell_cycle

# Per-cell-type damage and death pathway controls
# Note: cell stress variables (sig_xx, sig_eig_1, etc.) are in [nN/um^2], numerically equivalent to [kPa].
CELL_HYPOXIA_THRESHOLD = [0.03, 0.03, 0.03]          # [concentration units of C_sp[OXYGEN_SPECIES_INDEX]] chronic hypoxia damage threshold
CELL_NUTRIENT_THRESHOLD = [0.03, 0.03, 0.03]         # [concentration units of C_sp[NUTRIENT_SPECIES_INDEX]] chronic nutrient-deprivation threshold
CELL_STRESS_THRESHOLD = [8.0, 8.0, 8.0]              # [kPa = nN/um^2] chronic mechanical-overstress damage threshold
CELL_HYPOXIA_DAMAGE_RATE = [2.0e-4, 2.0e-4, 2.0e-4]  # [1/s] damage accumulation rate scaling under hypoxia (hour-scale)
CELL_NUTRIENT_DAMAGE_RATE = [1.5e-4, 1.5e-4, 1.5e-4] # [1/s] damage accumulation rate scaling under nutrient deprivation (hour-scale)
CELL_STRESS_DAMAGE_RATE = [1.0e-4, 1.0e-4, 1.0e-4]   # [1/s] damage accumulation rate scaling under mechanical overstress (hour-scale)
CELL_BASAL_DAMAGE_REPAIR_RATE = [5.0e-5, 5.0e-5, 5.0e-5] # [1/s] baseline damage repair rate
CELL_ACUTE_HYPOXIA_THRESHOLD = [0.005, 0.005, 0.005]  # [concentration units of C_sp[OXYGEN_SPECIES_INDEX]] immediate death threshold
CELL_ACUTE_NUTRIENT_THRESHOLD = [0.005, 0.005, 0.005] # [concentration units of C_sp[NUTRIENT_SPECIES_INDEX]] immediate death threshold
CELL_ACUTE_STRESS_THRESHOLD = [25.0, 25.0, 25.0]      # [kPa = nN/um^2] immediate mechanical-failure threshold

# Estimate maximum CELL population for bucket bounds and id allocation.
# Assumes worst-case synchronized proliferative expansion with the shortest cycle period across all cell types.
_sim_time_s = STEPS * TIME_STEP
print(f"Total simulation time: {_sim_time_s/3600:.2f} hours")
_min_cycle_dur = min(CELL_CYCLE_DURATION) if isinstance(CELL_CYCLE_DURATION, list) else CELL_CYCLE_DURATION
print(f"Shortest cell cycle duration across types: {_min_cycle_dur/3600:.2f} hours")
if INCLUDE_CELLS and INCLUDE_CELL_CYCLE and _min_cycle_dur > 0.0:
    _doublings = _sim_time_s / _min_cycle_dur
    MAX_EXPECTED_N_CELLS = max(N_CELLS, int(math.ceil(N_CELLS * (2.0 ** _doublings) * 2.0))) # * 2.0 is a safety factor. 
    print(f"Estimated maximum cell population at the end of the simulation: {MAX_EXPECTED_N_CELLS} (doublings: {_doublings:.2f})")
else:
    MAX_EXPECTED_N_CELLS = N_CELLS + 1 # add 1 as bucket messages requires min <> max bounds.


# +====================================================================+
# | FOCAL ADHESION PARAMETERS  (units: um, s, nN)                      |
# +====================================================================+
INCLUDE_FOCAL_ADHESIONS = False
INIT_N_FOCAD_PER_CELL = 10 # initial number of focal adhesions per cell. 
N_ANCHOR_POINTS = 50 # number of anchor points to which focal adhesions can attach on the nucleus surface. Their positions change with nucleus deformation
MAX_SEARCH_RADIUS_FOCAD = 3.0 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE  # TEMP(debug attach): increased to strongly favor FA-node encounters. Reasonable baseline: 1.0 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE
MAX_FOCAD_ARM_LENGTH = 4 * max(CELL_RADIUS)  # maximum length of the focal adhesion "arm". Uses max radius across cell types. WARNING: make sure this value is consistent with CELL_RADIUS and MAX_SEARCH_RADIUS_FOCAD to avoid unrealistic behavior.
# WARNING: rate values below assume global timestep ~ 1.0 s
FOCAD_REST_LENGTH_0 = min(r - r/2 for r in CELL_RADIUS) # [um] Reference rest length (shortest across types); per-agent value uses actual cell-type radii at init.
FOCAD_MIN_REST_LENGTH = FOCAD_REST_LENGTH_0 / 10.0 # [um] Minimum rest length to prevent collapse. 
FOCAD_K_FA = [10.0, 10.0, 10.0] # [nN/um] Adhesion stiffness (effective spring constant). Typical range: ~0.1–10 nN/um; 
FOCAD_F_MAX= [0.0, 0.0, 0.0] # [nN] Maximum force per adhesion. 0 means "no cap" 
FOCAD_V_C = [0.2, 0.2, 0.2] # [um/s] Contractile shortening speed of L(t) (actomyosin-driven).
FOCAD_K_ON = [5.0, 5.0, 5.0] # [1/s] TEMP(debug attach): high binding rate. Reasonable baseline: 0.01 [1/s]
FOCAD_K_OFF_0 = [0.0002, 0.0002, 0.0002] # [1/s] TEMP(debug attach): low baseline detachment. Reasonable baseline: 0.003 [1/s]
FOCAD_F_C = [5.0, 5.0, 5.0] # [nN] Force scale controlling force sensitivity in koff(F).
# Example (simple slip): koff(F)=K_OFF_0*exp(|F|/F_C) => faster turnover under high force.
USE_CATCH_BOND = True  # If True, use a two-pathway catch+slip off-rate instead of pure slip-bond.
CATCH_BOND_CATCH_SCALE = [4.0, 4.0, 4.0]  # Multiplier of K_OFF_0 for catch branch.
CATCH_BOND_SLIP_SCALE = [0.2, 0.2, 0.2]  # Multiplier of K_OFF_0 for slip branch.
CATCH_BOND_F_CATCH = [2.0, 2.0, 2.0]  # [nN] Force scale for catch branch.
CATCH_BOND_F_SLIP = [4.0, 4.0, 4.0]  # [nN] Force scale for slip branch.
# Suggested starting point when USE_CATCH_BOND=True (to avoid over-stabilization):
#   FOCAD_K_ON ~ 0.02-0.1 [1/s], FOCAD_K_OFF_0 ~ 0.001-0.01 [1/s], FOCAD_K_REINF <= 0.001 [1/s].
FOCAD_K_REINF = [0.001, 0.001, 0.001] # [1/s] Reinforcement rate for adhesion strengthening.
FOCAD_F_REINF = [1.0, 1.0, 1.0] # [nN] Force scale for reinforcement saturation: g(F)=F/(F+F_REINF).
FOCAD_K_FA_MAX = [50.0, 50.0, 50.0] # [nN/um] Upper bound for reinforced adhesion stiffness.
FOCAD_K_FA_DECAY = [0.0, 0.0, 0.0] # [1/s] Optional decay towards baseline. 0 disables.
FOCAD_POLARITY_KON_FRONT_GAIN = [2.0, 2.0, 2.0]  # [-] Frontness gain for k_on.
FOCAD_POLARITY_KOFF_FRONT_REDUCTION = [0.5, 0.5, 0.5]  # [-] Fractional reduction of k_off_0 at front.
FOCAD_POLARITY_KOFF_REAR_GAIN = [1.0, 1.0, 1.0]  # [-] Fractional increase of k_off_0 at rear.
FOCAD_F_MATURE = [1.0, 1.0, 1.0]  # [nN] force threshold to transition nascent->mature
FOCAD_T_NASCENT_MAX = [120.0, 120.0, 120.0]  # [s] max nascent lifetime before disassembly
FOCAD_T_DETACHED_GRACE = [30.0, 30.0, 30.0]  # [s] detached grace before disassembly
FOCAD_T_DISASSEMBLY = [20.0, 20.0, 20.0]  # [s] disassembling state before deletion
ENABLE_FOCAD_BIRTH = True
FOCAD_BIRTH_SPECIES_INDEX = 0  # species index in CELL C_sp controlling biochemical gate (global)
FOCAD_BIRTH_N_MIN = [1.0, 1.0, 1.0]  # minimum target adhesions per cell (float for array registration; cast to int in C++)
FOCAD_BIRTH_N_MAX = [float(3 * INIT_N_FOCAD_PER_CELL)] * N_CELL_TYPES  # hard cap / maximum target adhesions per cell
FOCAD_BIRTH_K_0 = [0.001, 0.001, 0.001]  # [1/s] baseline birth rate
FOCAD_BIRTH_K_MAX = [0.03, 0.03, 0.03]  # [1/s] max stress/biochemical-driven birth gain
FOCAD_BIRTH_K_SIGMA = [0.1, 0.1, 0.1]  # [kPa] stress half-saturation for birth gate
FOCAD_BIRTH_HILL_SIGMA = [1.0, 1.0, 1.0]  # Hill exponent for stress gate
FOCAD_BIRTH_K_C = [5.0, 5.0, 5.0]  # concentration half-saturation for birth gate
FOCAD_BIRTH_HILL_CONC = [2.0, 2.0, 2.0]  # Hill exponent for concentration gate
FOCAD_BIRTH_REFRACTORY = [20.0, 20.0, 20.0]  # [s] minimum time between consecutive births per cell
# +====================================================================+
# | LINC coupling between cell nucleus and FOCAD                       |
# +====================================================================+
INCLUDE_LINC_COUPLING = False
LINC_K_ELAST = [10.0, 10.0, 10.0] # [nN/um] Effective LINC stiffness in series with FOCAD stiffness.
LINC_D_DUMPING = [0.0, 0.0, 0.0] # [nN·s/um] Optional damping along FOCAD-LINC axis.
LINC_REST_LENGTH = [0.0, 0.0, 0.0] # [um] Rest length of virtual LINC segment.

# +====================================================================+
# | NUCLEAR MECHANICS  (ONLY USED IF FOCAL ADHESIONS ARE INCLUDED)     |
# +====================================================================+
# Elasticity (small-strain linear)
NUCLEUS_E = [2.0, 2.0, 2.0]               # [nN/µm² = kPa] Young’s modulus of the nucleus.
NUCLEUS_NU = [0.48, 0.48, 0.48]             # [-] Poisson ratio. Nearly incompressible nucleus. WARNING: must be < 0.5.
# Viscoelastic relaxation
NUCLEUS_TAU = [0.2, 0.2, 0.2]            # [s] Relaxation time.
NUCLEUS_EPS_CLAMP = [0.30, 0.30, 0.30]      # [-] Clamp for each strain component.
# +====================================================================+
# | CHEMOTAXIS                                                         |
# +====================================================================+
INCLUDE_CHEMOTAXIS = False
CHEMOTAXIS_SENSITIVITY = [1.0, 0.0, 0.0] # [-1.0 to +1.0] Chemotactic sensitivity for each species. Positive: attraction, Negative: repulsion towards higher concentrations.
CHEMOTAXIS_ONLY_DIR = True # if True, chemotaxis only affects cell orientation, not speed. If False, chemotaxis affects both orientation and speed (e.g. by making cells move faster when they are oriented towards higher concentration gradient)
CHEMOTAXIS_CHI = [0.1, 0.1, 0.10] # [um^2/s] Chemotactic coefficient (χ) per cell-type. Typical range: 0.1–10 µm²/s.
# +====================================================================+
# | CHEMOKINESIS                                                         |
# +====================================================================+
INCLUDE_CHEMOKINESIS = False
CHEMOKINESIS_SENSITIVITY = [-100.0, 0.0, 0.0] # [-1.0 to +1.0] Chemokinesis sensitivity for each species. Positive: speed increases with higher concentrations, Negative: speed decreases with higher concentrations.
CHEMOKINESIS_ALPHA = [0.5, 0.5, 0.5] # [-] Baseline speed is multiplied by (1 + alpha * f(C_sp)) 
CHEMOKINESIS_K = [2.0, 2.0, 2.0] # [concentration units] Chemokinesis half-saturation constant for concentration-dependent speed modulation.
CHEMOKINESIS_HILL_N = [2.0, 2.0, 2.0] # Hill coefficient for chemokinesis response curve.
CHEMOKINESIS_ADAPT_TAU = [60.0, 60.0, 60.0] # [s] Time constant for chemokinesis adaptation (how quickly cells adjust their internal state to changes in chemoattractant concentration).
CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER = [1.0, 1.0, 1.0] # Multiplier for chemokinesis signal saturation level per cell-type.
CHEMOKINESIS_SIGNAL_SAT = [20.0, 20.0, 20.0] # [concentration units] Saturation level of the chemokinesis signal for each species. Can be tuned together with CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER to adjust per-species per cell-type sensitivity.

# +====================================================================+
# | CELL MIGRATION RELATED PARAMETERS                                  |
# +====================================================================+
INCLUDE_DUROTAXIS = False   # if True, cells prefer to move towards stiffer regions, which is implemented by making them prefer to move in the direction of maximum stress/strain. 
DUROTAXIS_ONLY_DIR = True  # if True, stress/strain direction changes movement vector (keeps speed), False: changes speed too
FOCAD_MOBILITY_MU  = [1e-4, 1e-4, 1e-4]   # Mobility scaling for stress contribution
INCLUDE_ORIENTATION_ALIGN = True  # True: enable gradual alignment to principal direction
ORIENTATION_ALIGN_RATE  = [1.0, 1.0, 1.0]  # Alignment rate [1/time]
ORIENTATION_ALIGN_USE_STRESS = True  # True: align to stress eigvec1, False: align to strain eigvec1
DUROTAXIS_BLEND_BETA = [0.5, 0.5, 0.5]   # 0: traction only, 1: principal direction only
DUROTAXIS_USE_STRESS = True   # True: use stress eigenpair, False: use strain eigenpair


# +===================================================================+
# | VARIANT MODULE LOADING                                             |
# +====================================================================+
# Load a named variant from the variants/ folder via --variant <name>.
# e.g.  python model.py --variant organoid
#
# Priority order (highest wins):
#   1. --overrides <json>   (CLI / optimizer)
#   2. variant.PARAMS        (variant module)
#   3. model.py defaults
#
# Variant FILES and the configure_layers / configure_globals hooks are
# applied later in this file at the appropriate execution points.
_ACTIVE_VARIANT = None
_VARIANT_NAME = None
for _vi, _varg in enumerate(_ORIGINAL_ARGV):
    if _varg == "--variant" and _vi + 1 < len(_ORIGINAL_ARGV):
        _VARIANT_NAME = _ORIGINAL_ARGV[_vi + 1]
        break
if _VARIANT_NAME:
    import importlib.util as _iutil
    # Resolve variant path: prefer package layout variants/<name>/__init__.py,
    # fall back to legacy flat layout variants/<name>.py.
    _variant_path_pkg  = CURR_PATH / "variants" / _VARIANT_NAME / "__init__.py"
    _variant_path_flat = CURR_PATH / "variants" / f"{_VARIANT_NAME}.py"
    if _variant_path_pkg.exists():
        _variant_path = _variant_path_pkg
    elif _variant_path_flat.exists():
        _variant_path = _variant_path_flat
    else:
        _available_flat = [p.stem for p in (CURR_PATH / "variants").glob("*.py") if p.stem != "__init__"]
        _available_pkg  = [p.name for p in (CURR_PATH / "variants").iterdir()
                           if p.is_dir() and (p / "__init__.py").exists()]
        _available = sorted(set(_available_flat + _available_pkg))
        raise FileNotFoundError(
            f"Variant '{_VARIANT_NAME}' not found.\n"
            f"Searched: {_variant_path_pkg}\n"
            f"       and {_variant_path_flat}\n"
            f"Available variants: {_available}"
        )
    _vspec = _iutil.spec_from_file_location(f"variants.{_VARIANT_NAME}", str(_variant_path))
    _ACTIVE_VARIANT = _iutil.module_from_spec(_vspec)
    _vspec.loader.exec_module(_ACTIVE_VARIANT)
    _variant_params = getattr(_ACTIVE_VARIANT, "PARAMS", {})
    if _variant_params:
        print(f"[VARIANT] Applying {len(_variant_params)} parameter(s) from variant '{_VARIANT_NAME}'")
        apply_param_overrides(globals(), _variant_params)
    print(f"[VARIANT] Loaded variant '{_VARIANT_NAME}' from {_variant_path}")

# Variant-gated feature flags.
# Variants listed here activate extra per-cell variables that would waste GPU memory and VTK bandwidth in unrelated assays.
VARIANTS_WITH_RG_VARIABLES = ["radial_glia"]  # add variant names here to activate RG variables
INCLUDE_RG_VARIABLES = (_VARIANT_NAME in VARIANTS_WITH_RG_VARIABLES) if _VARIANT_NAME else False


# +====================================================================+
# | PARAMETER OVERRIDES (for optimization / batch runs)                |
# +====================================================================+
# Load overrides from --overrides <file.json> CLI argument (if any).
# JSON overrides are applied AFTER variant PARAMS so they always win.
# e.g. python model.py --overrides ./optimizer/optuna_results/best_params.json
# This must run AFTER all defaults above so that derived values can be
# recomputed from the (possibly overridden) base parameters.
_PARAM_OVERRIDES, _RESULT_DIR_OVERRIDE = load_param_overrides_from_cli(_ORIGINAL_ARGV)
print(f"[DIAG] _ORIGINAL_ARGV = {_ORIGINAL_ARGV}")
print(f"[DIAG] _PARAM_OVERRIDES keys = {list(_PARAM_OVERRIDES.keys()) if _PARAM_OVERRIDES else '(none)'}")
print(f"[DIAG] _RESULT_DIR_OVERRIDE = {_RESULT_DIR_OVERRIDE}")
if _PARAM_OVERRIDES:
    print(f"Applying {len(_PARAM_OVERRIDES)} parameter override(s): {list(_PARAM_OVERRIDES.keys())}")
    apply_param_overrides(globals(), _PARAM_OVERRIDES)
if _RESULT_DIR_OVERRIDE:
    RES_PATH = pathlib.Path(_RESULT_DIR_OVERRIDE)
    RES_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Result directory overridden to: {RES_PATH}")

# When running inside an Optuna trial, suppress verbose summaries.
_OPTUNA_QUIET = bool(_PARAM_OVERRIDES)
print(f"[DIAG] After overrides: STEPS={STEPS}, SAVE_PICKLE={SAVE_PICKLE}, RES_PATH={RES_PATH}, _OPTUNA_QUIET={_OPTUNA_QUIET}")

# +====================================================================+
# | OTHER DERIVED PARAMETERS AND MODEL CHECKS                          |
# +====================================================================+
if not OSCILLATORY_SHEAR_ASSAY:
    MIN_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, moved_corners = compute_expected_boundary_pos_from_corners(
        BOUNDARY_COORDS,
        BOUNDARY_DISP_RATES,
        BOUNDARY_DISP_RATES_PARALLEL,
        STEPS,
        TIME_STEP,
    )
else:
    MIN_EXPECTED_BOUNDARY_POS = -MAX_EXPECTED_BOUNDARY_POS_OSCILLATORY
    MAX_EXPECTED_BOUNDARY_POS = MAX_EXPECTED_BOUNDARY_POS_OSCILLATORY

# Dataframe initialization data storage
# ----------------------------------------------------------------------
BPOS = make_dataclass("BPOS", [("xpos", float), ("xneg", float), ("ypos", float), ("yneg", float), ("zpos", float),
                               ("zneg", float)])
# Use a dataframe to store boundary positions over time
BPOS_OVER_TIME = pd.DataFrame([BPOS(BOUNDARY_COORDS[0], BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[3],
                                    BOUNDARY_COORDS[4], BOUNDARY_COORDS[5])])
OSOT = make_dataclass("OSOT", [("strain", float)])
OSCILLATORY_STRAIN_OVER_TIME = pd.DataFrame([OSOT(0)])
CELL_SPEED_METRICS = pd.DataFrame()
RG_METRICS = pd.DataFrame()       # per-cell RG snapshot at final step (radial_glia variant only)
RG_ROSETTE_METRICS_OVER_TIME = pd.DataFrame()  # rosette cluster metrics over time (radial_glia variant only)
ORGANOID_METRICS_OVER_TIME = pd.DataFrame()

# Checking for incompatible conditions
# ----------------------------------------------------------------------
critical_error = False
try:
    hard_coded_check_args = [
        "--model-file", str(CURR_PATH / "model.py"),
        "--scan-root", str(CURR_PATH),
    ]
    if _OPTUNA_QUIET:
        hard_coded_check_args.append("--fail-on-mismatch")

    hard_coded_check_exit_code = check_hard_coded_values.main(hard_coded_check_args)
    if hard_coded_check_exit_code != 0:
        print("ERROR: hard-coded value consistency check found mismatches or failed")
        critical_error = True
except Exception as e:
    print(f"WARNING: failed to execute hard-coded value consistency check: {e}\nSkipping this check. If execution fails later due to hard-coded value mismatches, please run the check separately and fix the issues")

msg_poisson = "WARNING: poisson ratio directions are not well defined or might not make sense due to boundary conditions \n"
if (BOUNDARY_DISP_RATES[0] != 0.0 or BOUNDARY_DISP_RATES[1] != 0.0) and POISSON_DIRS[1] != 0:
    print(msg_poisson)
if (BOUNDARY_DISP_RATES[2] != 0.0 or BOUNDARY_DISP_RATES[3] != 0.0) and POISSON_DIRS[1] != 1:
    print(msg_poisson)
if (BOUNDARY_DISP_RATES[4] != 0.0 or BOUNDARY_DISP_RATES[5] != 0.0) and POISSON_DIRS[1] != 2:
    print(msg_poisson)

msg_incompatible_conditions = "ERROR: CLAMP_AGENT_TOUCHING_BOUNDARY condition is incompatible with ALLOW_BOUNDARY_ELASTIC_MOVEMENT in position [{}]"
for i in range(6):
    if CLAMP_AGENT_TOUCHING_BOUNDARY[i] > 0 and ALLOW_BOUNDARY_ELASTIC_MOVEMENT[i] > 0:
        print(msg_incompatible_conditions.format(i))
        critical_error = True


if INCLUDE_FIBRE_NETWORK:
    nodes, connectivity, n_fib, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, fibre_critical_error = load_fibre_network(
        file_name=NETWORK_FILE,
        boundary_coords=BOUNDARY_COORDS,
        epsilon=EPSILON,
        fibre_segment_equilibrium_distance=FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE,
        allow_warning_on_mismatch=ALLOW_IRREGULAR_NETWORK
    )
    if fibre_critical_error:
        critical_error = True
    if nodes is not None and connectivity is not None:
        N_NODES = nodes.shape[0]
        NODE_COORDS = nodes
        INITIAL_NETWORK_CONNECTIVITY = connectivity
        AVG_NETWORK_VOXEL_DENSITY = math.ceil((N_NODES / (L0_x * L0_y * L0_z)) * ECM_VOXEL_VOLUME) # average number of fibre nodes per voxel, used to adjust the heterogeneous diffusion effect
        print(f'Average network voxel density (number of fibre nodes per voxel): {AVG_NETWORK_VOXEL_DENSITY}')
    if nodes is not None and connectivity is not None:
        N_FIBRES = n_fib
    else:
        N_FIBRES = None
    # FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE may have been updated by load_fibre_network
    # (matched to the network file's EDGE_LENGTH). Recompute derived params.
    recompute_derived_params(globals(), pinned=set(_PARAM_OVERRIDES.keys()) if _PARAM_OVERRIDES else None)
else: 
    N_NODES = None
    NODE_COORDS = None
    INITIAL_NETWORK_CONNECTIVITY = None
    AVG_NETWORK_VOXEL_DENSITY = None

if INCLUDE_VASCULARIZATION:
    _vasc_network_path = CURR_PATH / VASC_NETWORK_FILE
    with open(str(_vasc_network_path), 'rb') as _vf:
        _vasc_network = pickle.load(_vf)
    VASC_NODES = _vasc_network['nodes']
    N_VASC_NODES = len(VASC_NODES)
    print(f'Loaded vascular network: {N_VASC_NODES} nodes from {_vasc_network_path}')
    # --- DEBUG: spatial coverage sanity check ---
    if VASC_NODES and DEBUG_PRINTING:
        _vxs = [n['x'] for n in VASC_NODES]
        _vys = [n['y'] for n in VASC_NODES]
        _vzs = [n['z'] for n in VASC_NODES]
        print(f'[DEBUG VASC] bounding box: x=[{min(_vxs):.2f}, {max(_vxs):.2f}]  '
              f'y=[{min(_vys):.2f}, {max(_vys):.2f}]  '
              f'z=[{min(_vzs):.2f}, {max(_vzs):.2f}]')
    # Compute the global ID base for VASC agents (they come after ECM agents in ID space)
    _vasc_id_base = 8  # boundary corners 1-8
    if INCLUDE_FIBRE_NETWORK and N_NODES is not None:
        _vasc_id_base += N_NODES
    if INCLUDE_CELLS:
        _vasc_id_base += N_CELLS
        if INCLUDE_FOCAL_ADHESIONS:
            _vasc_id_base += N_CELLS * INIT_N_FOCAD_PER_CELL
    _vasc_id_base += ECM_POPULATION_SIZE
else:
    VASC_NODES = None
    N_VASC_NODES = 0

UNSTABLE_DIFFUSION = False
# Check diffusion parameters
if INCLUDE_DIFFUSION:
    if (len(DIFFUSION_COEFF_MULTI) != N_SPECIES) or (len(BOUNDARY_CONC_INIT_MULTI) != N_SPECIES) or (
            len(BOUNDARY_CONC_FIXED_MULTI) != N_SPECIES):
        print('ERROR: you must define a diffusion coefficient and the boundary conditions for each species simulated')
        critical_error = True
    # Check diffusion values for numerical stability
    dx = L0_x / (ECM_AGENTS_PER_DIR[0] - 1)
    for i in range(N_SPECIES):
        Fi_x = 3 * (DIFFUSION_COEFF_MULTI[i] * TIME_STEP / (dx * dx))  # this value should be < 0.5
        # print('Fi_x value: {0} for species {1}'.format(Fi_x, i + 1))
        if Fi_x > 0.5:
            print(
                f'WARNING: diffusion problem is ill conditioned (Fi_x {Fi_x} should be < 0.5), check parameters and consider decreasing time step\nSemi-implicit diffusion will be used instead')
            UNSTABLE_DIFFUSION = True
    dy = L0_y / (ECM_AGENTS_PER_DIR[1] - 1)
    for i in range(N_SPECIES):
        Fi_y = 3 * (DIFFUSION_COEFF_MULTI[i] * TIME_STEP / (dy * dy))  # this value should be < 0.5
        # print('Fi_y value: {0} for species {1}'.format(Fi_y, i + 1))
        if Fi_y > 0.5:
            print(
                f'WARNING: diffusion problem is ill conditioned (Fi_y {Fi_y} should be < 0.5), check parameters and consider decreasing time step\nSemi-implicit diffusion will be used instead')
            UNSTABLE_DIFFUSION = True
    dz = L0_z / (ECM_AGENTS_PER_DIR[2] - 1)
    for i in range(N_SPECIES):
        Fi_z = 3 * (DIFFUSION_COEFF_MULTI[i] * TIME_STEP / (dz * dz))  # this value should be < 0.5
        # print('Fi_z value: {0} for species {1}'.format(Fi_z, i + 1))
        if Fi_z > 0.5:
            print(
                f'WARNING: diffusion problem is ill conditioned (Fi_z {Fi_z} should be < 0.5), check parameters and consider decreasing time step\nSemi-implicit diffusion will be used instead')
            UNSTABLE_DIFFUSION = True
    if not INCLUDE_FIBRE_NETWORK and HETEROGENEOUS_DIFFUSION:
        print(f'WARNING: HETEROGENEOUS_DIFFUSION is set to True but no fibre network is included, default D values ({DIFFUSION_COEFF_MULTI}) will be used instead')
        HETEROGENEOUS_DIFFUSION = False

if INCLUDE_CELLS:
    _max_cell_radius = max(CELL_RADIUS) if isinstance(CELL_RADIUS, list) else CELL_RADIUS
    if INCLUDE_LUMEN and not ORGANOID_ASSAY and not MONOLAYER_ASSAY:
        print('ERROR: INCLUDE_LUMEN requires either ORGANOID_ASSAY or MONOLAYER_ASSAY to be True')
        critical_error = True
    if ORGANOID_ASSAY and MONOLAYER_ASSAY:
        print('ERROR: ORGANOID_ASSAY and MONOLAYER_ASSAY cannot both be True')
        critical_error = True
    if INCLUDE_FOCAL_ADHESIONS and not INCLUDE_FIBRE_NETWORK: 
        print('ERROR: focal adhesions cannot be included if there is no fibre network to interact with')
        critical_error = True
    if PERIODIC_BOUNDARIES_FOR_CELLS and INCLUDE_FOCAL_ADHESIONS:
        print('ERROR: PERIODIC_BOUNDARIES_FOR_CELLS and INCLUDE_FOCAL_ADHESIONS cannot both be True. Periodic wrapping would break focal adhesion connections to the fibre network.')
        critical_error = True
    if INCLUDE_FOCAL_ADHESIONS and MAX_FOCAD_ARM_LENGTH < _max_cell_radius:
        print('ERROR: MAX_FOCAD_ARM_LENGTH: {0} must be bigger than max(CELL_RADIUS): {1}, as focal adhesions are initiated at the cell surface and should be able to grow away'.format(MAX_FOCAD_ARM_LENGTH, _max_cell_radius))
    if INCLUDE_FOCAL_ADHESIONS and any(nmax < INIT_N_FOCAD_PER_CELL for nmax in FOCAD_BIRTH_N_MAX):
        print('ERROR: FOCAD_BIRTH_N_MAX: {0} must be >= INIT_N_FOCAD_PER_CELL: {1}'.format(FOCAD_BIRTH_N_MAX, INIT_N_FOCAD_PER_CELL))
        critical_error = True
    if INCLUDE_FOCAL_ADHESIONS and any(nmax < nmin for nmax, nmin in zip(FOCAD_BIRTH_N_MAX, FOCAD_BIRTH_N_MIN)):
        print('ERROR: FOCAD_BIRTH_N_MAX: {0} must be >= FOCAD_BIRTH_N_MIN: {1}'.format(FOCAD_BIRTH_N_MAX, FOCAD_BIRTH_N_MIN))
        critical_error = True
elif INCLUDE_FOCAL_ADHESIONS:
    print('ERROR: focal adhesions cannot be included if there are no cells to form them (INCLUDE_CELLS is set to False)')
    critical_error= True
elif INCLUDE_CELL_CYCLE:
    print('ERROR: cell cycle cannot be included if there are no cells (INCLUDE_CELLS is set to False)')
    critical_error= True

if INCLUDE_FIBRE_NETWORK and not _OPTUNA_QUIET:
    print_fibre_calibration_summary(
        fibre_segment_k_elast=FIBRE_SEGMENT_K_ELAST,
        fibre_segment_d_dumping=FIBRE_SEGMENT_D_DUMPING,
        fibre_segment_equilibrium_distance=FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE,
        dt = TIME_STEP,
    )

if INCLUDE_CELLS and INCLUDE_FOCAL_ADHESIONS and ENABLE_FOCAD_BIRTH and not _OPTUNA_QUIET:
    for _ct_i in range(N_CELL_TYPES):
        print(f"\n  [cell type {_ct_i}]")
        print_focad_birth_calibration_summary(
            dt=TIME_STEP,
            init_n_focad_per_cell=INIT_N_FOCAD_PER_CELL,
            n_min=FOCAD_BIRTH_N_MIN[_ct_i],
            n_max=FOCAD_BIRTH_N_MAX[_ct_i],
            k0=FOCAD_BIRTH_K_0[_ct_i],
            kmax=FOCAD_BIRTH_K_MAX[_ct_i],
            refractory_s=FOCAD_BIRTH_REFRACTORY[_ct_i],
            k_sigma=FOCAD_BIRTH_K_SIGMA[_ct_i],
            hill_sigma=FOCAD_BIRTH_HILL_SIGMA[_ct_i],
            k_c=FOCAD_BIRTH_K_C[_ct_i],
            hill_conc=FOCAD_BIRTH_HILL_CONC[_ct_i],
            species_index=FOCAD_BIRTH_SPECIES_INDEX,
        )


if critical_error:
    quit()

MODEL_CONFIG = build_model_config_from_namespace(globals())
if not _OPTUNA_QUIET:
    MODEL_CONFIG.print_configuration_summary(
        n_nodes=locals().get('N_NODES'),
        n_fibres=locals().get('N_FIBRES'),
    )
# +====================================================================+
# | FLAMEGPU2 IMPLEMENTATION                                           |
# +====================================================================+


# ++==================================================================++
# ++ Files                                                             |
# ++==================================================================++
"""
AGENT Files
"""
# Files containing agent functions for agents, which outputs publicly visible properties to a message list

# Agent function files
"""
  ECM
"""
ecm_grid_location_data_file = "ecm_grid_location_data.cpp"
ecm_ecm_interaction_file = "ecm_ecm_interaction.cpp"
ecm_boundary_concentration_conditions_file = "ecm_boundary_concentration_conditions.cpp"
ecm_move_file = "ecm_move.cpp"
ecm_Csp_update_file = "ecm_Csp_update.cpp"
ecm_Dsp_update_file = "ecm_Dsp_update.cpp"
ecm_Dsp_lumen_update_file = "ecm_Dsp_lumen_update.cpp"

"""
  CELL
"""
cell_spatial_location_data_file = "cell_spatial_location_data.cpp"
cell_ecm_interaction_metabolism_file = "cell_ecm_interaction_metabolism.cpp"
cell_move_file = "cell_move.cpp"
cell_cell_interaction_file = "cell_cell_interaction.cpp"
cell_fnode_repulsion_file = "cell_fnode_repulsion.cpp"
cell_fnode_remodel_file = "cell_fnode_remodel.cpp"
cell_bucket_location_data_file = "cell_bucket_location_data.cpp"
cell_focad_update_file = "cell_focad_update.cpp"
cell_stress_state_update_file = "cell_stress_state_update.cpp"
cell_cycle_file = "cell_cycle.cpp"
cell_maxid_update_file = "cell_MaxID_update.cpp"
cell_lumen_interaction_file = "cell_lumen_interaction.cpp"
cell_lumen_secretion_file = "cell_lumen_secretion.cpp"

"""
  FOCAD
"""
focad_bucket_location_data_file = "focad_bucket_location_data.cpp"
focad_spatial_location_data_file = "focad_spatial_location_data.cpp"
focad_anchor_update_file = "focad_anchor_update.cpp"
focad_post_cycle_update_file = "focad_post_cycle_update.cpp"
focad_fnode_interaction_file = "focad_fnode_interaction.cpp"
focad_move_file = "focad_move.cpp"

"""
  LUMEN
"""
lumen_spatial_location_data_file = "lumen_spatial_location_data.cpp"
lumen_lumen_interaction_file = "lumen_lumen_interaction.cpp"
lumen_cell_interaction_file = "lumen_cell_interaction.cpp"
lumen_move_file = "lumen_move.cpp"

"""
  VASC
"""
vasc_bucket_location_data_file = "vasc_bucket_location_data.cpp"
vasc_spatial_location_data_file = "vasc_spatial_location_data.cpp"
vasc_Csp_update_file = "vasc_Csp_update.cpp"
ecm_vasc_Csp_update_file = "ecm_vasc_Csp_update.cpp"
vasc_move_file = "vasc_move.cpp"
vasc_ecm_cell_spawn_file = "vasc_ecm_cell_spawn.cpp"



"""
  BCORNER  
"""
bcorner_output_location_data_file = "bcorner_output_location_data.cpp"
bcorner_move_file = "bcorner_move.cpp"

"""
  FIBRE NODES
"""
fnode_spatial_location_data_file = "fnode_spatial_location_data.cpp"
fnode_bucket_location_data_file = "fnode_bucket_location_data.cpp"
fnode_boundary_interaction_file = "fnode_boundary_interaction.cpp"
fnode_update_links_file = "fnode_update_links.cpp"
fnode_fnode_spatial_interaction_file = "fnode_fnode_spatial_interaction.cpp"
fnode_fnode_bucket_interaction_file = "fnode_fnode_bucket_interaction.cpp"
fnode_remodel_file = "fnode_remodel.cpp"
fnode_apply_remodel_updates_file = "fnode_apply_remodel_updates.cpp"
fnode_move_file = "fnode_move.cpp"
fnode_bucket_location_data_postmove_file = "fnode_bucket_location_data_postmove.cpp"
fnode_focad_interaction_file = "fnode_focad_interaction.cpp"
fnode_cell_repulsion_file = "fnode_cell_repulsion.cpp"

# Apply variant FILES overrides — redirects *_file variables to variant-
# specific .cpp paths.  Paths are relative to the project root (CURR_PATH).
if _ACTIVE_VARIANT is not None:
    for _vfkey, _vfpath in getattr(_ACTIVE_VARIANT, "FILES", {}).items():
        if _vfkey in globals():
            print(f"[VARIANT] Redirecting {_vfkey} -> {_vfpath}")
            globals()[_vfkey] = _vfpath
        else:
            print(f"[VARIANT] WARNING: FILES key '{_vfkey}' not found in model.py globals, ignoring")
    # configure_globals hook: may inject new global flags before model build.
    if hasattr(_ACTIVE_VARIANT, "configure_globals"):
        print(f"[VARIANT] Calling configure_globals for variant '{_VARIANT_NAME}'")
        _ACTIVE_VARIANT.configure_globals(globals())


model = pyflamegpu.ModelDescription("cellfoundry")

# ++==================================================================++
# ++ Globals                                                           |
# ++==================================================================++
"""
  GLOBAL SETTINGS
"""
env = model.Environment()
# Starting ID to generate agent populations
env.newPropertyUInt("CURRENT_ID", 0)
# Number of steps to simulate
env.newPropertyUInt("STEPS", STEPS)
# Time increment 
env.newPropertyFloat("TIME_STEP", TIME_STEP)
# Number of agents in the ECM grid per direction
env.newPropertyArrayUInt("ECM_AGENTS_PER_DIR", ECM_AGENTS_PER_DIR)
# Diffusion coefficient
env.newPropertyUInt("INCLUDE_DIFFUSION", INCLUDE_DIFFUSION)
env.newPropertyUInt("HETEROGENEOUS_DIFFUSION", HETEROGENEOUS_DIFFUSION)
env.newPropertyUInt("UNSTABLE_DIFFUSION", UNSTABLE_DIFFUSION)
if INCLUDE_FIBRE_NETWORK:
    env.newPropertyUInt("AVG_NETWORK_VOXEL_DENSITY", AVG_NETWORK_VOXEL_DENSITY)
env.newPropertyArrayFloat("DIFFUSION_COEFF_MULTI", DIFFUSION_COEFF_MULTI)
env.newPropertyArrayFloat("ECM_DEGRADATION_RATE_MULTI", ECM_DEGRADATION_RATE_MULTI)
env.newPropertyFloat("ECM_VOXEL_VOLUME", ECM_VOXEL_VOLUME)

# ------------------------------------------------------
# BOUNDARY BEHAVIOUR 
# ------------------------------------------------------
# Boundaries position
bcs = [BOUNDARY_COORDS[0], BOUNDARY_COORDS[1], 
      BOUNDARY_COORDS[2], BOUNDARY_COORDS[3], 
      BOUNDARY_COORDS[4], BOUNDARY_COORDS[5]]  # +X,-X,+Y,-Y,+Z,-Z
env.newPropertyArrayFloat("COORDS_BOUNDARIES", bcs)
env.newPropertyArrayFloat("INIT_COORDS_BOUNDARIES",
                          bcs)  # this is used to compute elastic forces with respect to initial position

# Boundaries displacement rate (units/time). 
# e.g. DISP_BOUNDARY_X_POS = 0.1 means that this boundary moves 0.1 units per time towards +X
env.newPropertyArrayFloat("DISP_RATES_BOUNDARIES", BOUNDARY_DISP_RATES)
env.newPropertyArrayFloat("DISP_RATES_BOUNDARIES_PARALLEL", BOUNDARY_DISP_RATES_PARALLEL)

# Boundary-Agent behaviour
env.newPropertyArrayUInt("CLAMP_AGENT_TOUCHING_BOUNDARY", CLAMP_AGENT_TOUCHING_BOUNDARY)
env.newPropertyArrayUInt("ALLOW_BOUNDARY_ELASTIC_MOVEMENT", ALLOW_BOUNDARY_ELASTIC_MOVEMENT)
env.newPropertyArrayFloat("BOUNDARY_STIFFNESS", BOUNDARY_STIFFNESS)
env.newPropertyArrayFloat("BOUNDARY_DUMPING", BOUNDARY_DUMPING)
env.newPropertyArrayUInt("ALLOW_AGENT_SLIDING", ALLOW_AGENT_SLIDING)
env.newPropertyFloat("ECM_BOUNDARY_INTERACTION_RADIUS", ECM_BOUNDARY_INTERACTION_RADIUS)
env.newPropertyFloat("ECM_BOUNDARY_EQUILIBRIUM_DISTANCE", ECM_BOUNDARY_EQUILIBRIUM_DISTANCE)
env.newPropertyFloat("FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS", FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS)
env.newPropertyFloat("FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE", FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE)
env.newPropertyFloat("FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE",FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE)
# Model macro/globals
env.newMacroPropertyFloat("C_SP_MACRO", N_SPECIES, ECM_POPULATION_SIZE)
env.newMacroPropertyFloat("BOUNDARY_CONC_INIT_MULTI", N_SPECIES,
                          6)  # a 2D matrix with the 6 boundary conditions (columns) for each species (rows)
env.newMacroPropertyFloat("BOUNDARY_CONC_FIXED_MULTI", N_SPECIES,
                          6)  # a 2D matrix with the 6 boundary conditions (columns) for each species (rows)
env.newMacroPropertyInt("MACRO_MAX_GLOBAL_CELL_ID", 1)  # shared current max CELL id across all proliferating cells
env.newMacroPropertyInt("MACRO_MAX_GLOBAL_FNODE_ID", 1)  # shared current max FNODE id across all remodeling cells
if INCLUDE_CELLS and (ORGANOID_ASSAY or MONOLAYER_ASSAY) and INCLUDE_LUMEN:
    env.newMacroPropertyInt("MACRO_MAX_GLOBAL_LUMEN_ID", 1)  # shared current max LUMEN id across all secreted lumen droplets
env.newPropertyUInt("ECM_POPULATION_SIZE", ECM_POPULATION_SIZE)

# Fibre network parameters
env.newPropertyUInt("INCLUDE_FIBRE_NETWORK", INCLUDE_FIBRE_NETWORK)
env.newPropertyFloat("MAX_SEARCH_RADIUS_FNODES",MAX_SEARCH_RADIUS_FNODES)
env.newPropertyFloat("FIBRE_SEGMENT_K_ELAST",FIBRE_SEGMENT_K_ELAST)
env.newPropertyFloat("FIBRE_SEGMENT_D_DUMPING",FIBRE_SEGMENT_D_DUMPING)
env.newPropertyFloat("FIBRE_NODE_REPULSION_K", FIBRE_NODE_REPULSION_K)
env.newPropertyUInt("INCLUDE_NETWORK_REMODELING", INCLUDE_NETWORK_REMODELING)
env.newPropertyArrayFloat("FNODE_DEGRADATION_RATE", FNODE_DEGRADATION_RATE)
env.newPropertyArrayFloat("FNODE_DEPOSITION_RATE", FNODE_DEPOSITION_RATE)
env.newPropertyFloat("FNODE_CELL_DEGRADATION_RADIUS", FNODE_CELL_DEGRADATION_RADIUS)
env.newPropertyArrayFloat("FNODE_BIRTH_K_0", FNODE_BIRTH_K_0)
env.newPropertyArrayFloat("FNODE_BIRTH_K_MAX", FNODE_BIRTH_K_MAX)
env.newPropertyUInt("FNODE_BIRTH_SPECIES_INDEX", FNODE_BIRTH_SPECIES_INDEX)
env.newPropertyArrayFloat("FNODE_BIRTH_K_C", FNODE_BIRTH_K_C)
env.newPropertyArrayFloat("FNODE_BIRTH_HILL_CONC", FNODE_BIRTH_HILL_CONC)
env.newPropertyArrayFloat("FNODE_BIRTH_K_SIGMA", FNODE_BIRTH_K_SIGMA)
env.newPropertyArrayFloat("FNODE_BIRTH_HILL_SIGMA", FNODE_BIRTH_HILL_SIGMA)
env.newPropertyArrayFloat("FNODE_BIRTH_RADIUS", FNODE_BIRTH_RADIUS)
env.newPropertyArrayFloat("FNODE_BIRTH_LINK_MAX_DISTANCE", FNODE_BIRTH_LINK_MAX_DISTANCE)
env.newPropertyArrayFloat("FNODE_BIRTH_REFRACTORY", FNODE_BIRTH_REFRACTORY)

# Cell properties — per-cell-type arrays (length N_CELL_TYPES)
env.newPropertyUInt("N_CELL_TYPES", N_CELL_TYPES)
env.newPropertyUInt("INCLUDE_CELL_CELL_INTERACTION", INCLUDE_CELL_CELL_INTERACTION)
env.newPropertyUInt("INCLUDE_CELL_FNODE_REPULSION", INCLUDE_CELL_FNODE_REPULSION)
env.newPropertyUInt("DEAD_CELLS_DISAPPEAR", DEAD_CELLS_DISAPPEAR)
env.newPropertyUInt("PERIODIC_BOUNDARIES_FOR_CELLS", PERIODIC_BOUNDARIES_FOR_CELLS)
env.newPropertyUInt("MONOLAYER_ASSAY", MONOLAYER_ASSAY)
env.newPropertyFloat("MONOLAYER_Z_PLANE", float(MONOLAYER_Z) if MONOLAYER_Z is not None else float(BOUNDARY_COORDS[5]))
env.newPropertyUInt("N_CELLS", N_CELLS)
env.newPropertyArrayFloat("CELL_K_ELAST", CELL_K_ELAST)
env.newPropertyArrayFloat("CELL_D_DUMPING", CELL_D_DUMPING)
env.newPropertyArrayFloat("CELL_RADIUS", CELL_RADIUS)
env.newPropertyArrayFloat("CELL_NUCLEUS_RADIUS", CELL_NUCLEUS_RADIUS)
env.newPropertyArrayFloat("CELL_SPEED_REF", CELL_SPEED_REF)
env.newPropertyArrayFloat("BROWNIAN_MOTION_STRENGTH", BROWNIAN_MOTION_STRENGTH)
env.newPropertyArrayFloat("ROTATIONAL_DIFFUSION_RATE", ROTATIONAL_DIFFUSION_RATE)
env.newPropertyFloat("MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION", MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION)
env.newPropertyFloat("MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION", MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION)
env.newPropertyArrayFloat("CELL_CELL_REPULSION_K", CELL_CELL_REPULSION_K)
env.newPropertyArrayFloat("CELL_CELL_ADHESION_K", CELL_CELL_ADHESION_K)
env.newPropertyArrayFloat("CELL_CELL_ADHESION_RANGE", CELL_CELL_ADHESION_RANGE)
env.newPropertyArrayFloat("CELL_CELL_DV_MAX", CELL_CELL_DV_MAX)
env.newPropertyArrayFloat("CELL_FNODE_REPULSION_K", CELL_FNODE_REPULSION_K)
env.newPropertyArrayFloat("CELL_FNODE_EXCLUSION_DISTANCE", CELL_FNODE_EXCLUSION_DISTANCE)
env.newPropertyArrayFloat("CELL_FNODE_DV_MAX", CELL_FNODE_DV_MAX)
env.newPropertyArrayFloat("CELL_CYCLE_DURATION", CELL_CYCLE_DURATION)
env.newPropertyArrayFloat("CYCLE_PHASE_G1_DURATION", CYCLE_PHASE_G1_DURATION)
env.newPropertyArrayFloat("CYCLE_PHASE_S_DURATION", CYCLE_PHASE_S_DURATION)
env.newPropertyArrayFloat("CYCLE_PHASE_G2_DURATION", CYCLE_PHASE_G2_DURATION)
env.newPropertyArrayFloat("CYCLE_PHASE_M_DURATION", CYCLE_PHASE_M_DURATION)
env.newPropertyArrayFloat("CYCLE_PHASE_G1_START", CYCLE_PHASE_G1_START)
env.newPropertyArrayFloat("CYCLE_PHASE_S_START", CYCLE_PHASE_S_START)
env.newPropertyArrayFloat("CYCLE_PHASE_G2_START", CYCLE_PHASE_G2_START)
env.newPropertyArrayFloat("CYCLE_PHASE_M_START", CYCLE_PHASE_M_START)
env.newPropertyArrayFloat("DIVISION_RATE_MULTIPLIER", DIVISION_RATE_MULTIPLIER)
env.newPropertyArrayFloat("DAMAGE_ACCUMULATION_MULTIPLIER", DAMAGE_ACCUMULATION_MULTIPLIER)
env.newPropertyArrayFloat("DAMAGE_REPAIR_MULTIPLIER", DAMAGE_REPAIR_MULTIPLIER)
env.newPropertyArrayFloat("DAMAGE_DEATH_THRESHOLD", DAMAGE_DEATH_THRESHOLD)
env.newPropertyArrayFloat("CELL_CONSUMPTION_MULTIPLIER", CELL_CONSUMPTION_MULTIPLIER)
env.newPropertyArrayFloat("CELL_PRODUCTION_MULTIPLIER", CELL_PRODUCTION_MULTIPLIER)
env.newPropertyArrayFloat("INIT_CELL_PRODUCTION_RATES", INIT_CELL_PRODUCTION_RATES)  # base rates as env property for variant access in C++
env.newPropertyArrayUInt("DE_NOVO_PRODUCTION", DE_NOVO_PRODUCTION)  # 1 = de-novo secretion (no pool depletion), 0 = mass-conserved
env.newPropertyArrayFloat("CELL_REACTION_MULTIPLIER", CELL_REACTION_MULTIPLIER)
# Species index mapping for death pathways
env.newPropertyUInt("OXYGEN_SPECIES_INDEX", OXYGEN_SPECIES_INDEX)
env.newPropertyUInt("NUTRIENT_SPECIES_INDEX", NUTRIENT_SPECIES_INDEX)
# Per-cell-type damage/death thresholds
env.newPropertyArrayFloat("CELL_HYPOXIA_THRESHOLD", CELL_HYPOXIA_THRESHOLD)
env.newPropertyArrayFloat("CELL_NUTRIENT_THRESHOLD", CELL_NUTRIENT_THRESHOLD)
env.newPropertyArrayFloat("CELL_STRESS_THRESHOLD", CELL_STRESS_THRESHOLD)
env.newPropertyArrayFloat("CELL_HYPOXIA_DAMAGE_RATE", CELL_HYPOXIA_DAMAGE_RATE)
env.newPropertyArrayFloat("CELL_NUTRIENT_DAMAGE_RATE", CELL_NUTRIENT_DAMAGE_RATE)
env.newPropertyArrayFloat("CELL_STRESS_DAMAGE_RATE", CELL_STRESS_DAMAGE_RATE)
env.newPropertyArrayFloat("CELL_BASAL_DAMAGE_REPAIR_RATE", CELL_BASAL_DAMAGE_REPAIR_RATE)
env.newPropertyArrayFloat("CELL_ACUTE_HYPOXIA_THRESHOLD", CELL_ACUTE_HYPOXIA_THRESHOLD)
env.newPropertyArrayFloat("CELL_ACUTE_NUTRIENT_THRESHOLD", CELL_ACUTE_NUTRIENT_THRESHOLD)
env.newPropertyArrayFloat("CELL_ACUTE_STRESS_THRESHOLD", CELL_ACUTE_STRESS_THRESHOLD)

# Focal adhesion properties
env.newPropertyUInt("INCLUDE_FOCAL_ADHESIONS", INCLUDE_FOCAL_ADHESIONS)
env.newPropertyUInt("INIT_N_FOCAD_PER_CELL", INIT_N_FOCAD_PER_CELL)
env.newPropertyUInt("N_ANCHOR_POINTS", N_ANCHOR_POINTS)
env.newPropertyFloat("MAX_SEARCH_RADIUS_FOCAD", MAX_SEARCH_RADIUS_FOCAD)
env.newPropertyFloat("MAX_FOCAD_ARM_LENGTH", MAX_FOCAD_ARM_LENGTH)
env.newPropertyFloat("FOCAD_REST_LENGTH_0", FOCAD_REST_LENGTH_0)
env.newPropertyFloat("FOCAD_MIN_REST_LENGTH", FOCAD_MIN_REST_LENGTH)
env.newPropertyArrayFloat("FOCAD_K_FA", FOCAD_K_FA)
env.newPropertyArrayFloat("FOCAD_F_MAX", FOCAD_F_MAX)
env.newPropertyArrayFloat("FOCAD_V_C", FOCAD_V_C)
env.newPropertyArrayFloat("FOCAD_K_ON", FOCAD_K_ON)
env.newPropertyArrayFloat("FOCAD_K_OFF_0", FOCAD_K_OFF_0)
env.newPropertyArrayFloat("FOCAD_F_C", FOCAD_F_C)
env.newPropertyUInt("USE_CATCH_BOND", USE_CATCH_BOND)
env.newPropertyArrayFloat("CATCH_BOND_CATCH_SCALE", CATCH_BOND_CATCH_SCALE)
env.newPropertyArrayFloat("CATCH_BOND_SLIP_SCALE", CATCH_BOND_SLIP_SCALE)
env.newPropertyArrayFloat("CATCH_BOND_F_CATCH", CATCH_BOND_F_CATCH)
env.newPropertyArrayFloat("CATCH_BOND_F_SLIP", CATCH_BOND_F_SLIP)
env.newPropertyArrayFloat("FOCAD_K_REINF", FOCAD_K_REINF)
env.newPropertyArrayFloat("FOCAD_F_REINF", FOCAD_F_REINF)
env.newPropertyArrayFloat("FOCAD_K_FA_MAX", FOCAD_K_FA_MAX)
env.newPropertyArrayFloat("FOCAD_K_FA_DECAY", FOCAD_K_FA_DECAY)
env.newPropertyArrayFloat("FOCAD_POLARITY_KON_FRONT_GAIN", FOCAD_POLARITY_KON_FRONT_GAIN)
env.newPropertyArrayFloat("FOCAD_POLARITY_KOFF_FRONT_REDUCTION", FOCAD_POLARITY_KOFF_FRONT_REDUCTION)
env.newPropertyArrayFloat("FOCAD_POLARITY_KOFF_REAR_GAIN", FOCAD_POLARITY_KOFF_REAR_GAIN)
env.newPropertyArrayFloat("FOCAD_F_MATURE", FOCAD_F_MATURE)
env.newPropertyArrayFloat("FOCAD_T_NASCENT_MAX", FOCAD_T_NASCENT_MAX)
env.newPropertyArrayFloat("FOCAD_T_DETACHED_GRACE", FOCAD_T_DETACHED_GRACE)
env.newPropertyArrayFloat("FOCAD_T_DISASSEMBLY", FOCAD_T_DISASSEMBLY)
env.newPropertyUInt("ENABLE_FOCAD_BIRTH", ENABLE_FOCAD_BIRTH)
env.newPropertyUInt("FOCAD_BIRTH_SPECIES_INDEX", FOCAD_BIRTH_SPECIES_INDEX)
env.newPropertyArrayFloat("FOCAD_BIRTH_N_MIN", FOCAD_BIRTH_N_MIN)
env.newPropertyArrayFloat("FOCAD_BIRTH_N_MAX", FOCAD_BIRTH_N_MAX)
env.newPropertyArrayFloat("FOCAD_BIRTH_K_0", FOCAD_BIRTH_K_0)
env.newPropertyArrayFloat("FOCAD_BIRTH_K_MAX", FOCAD_BIRTH_K_MAX)
env.newPropertyArrayFloat("FOCAD_BIRTH_K_SIGMA", FOCAD_BIRTH_K_SIGMA)
env.newPropertyArrayFloat("FOCAD_BIRTH_HILL_SIGMA", FOCAD_BIRTH_HILL_SIGMA)
env.newPropertyArrayFloat("FOCAD_BIRTH_K_C", FOCAD_BIRTH_K_C)
env.newPropertyArrayFloat("FOCAD_BIRTH_HILL_CONC", FOCAD_BIRTH_HILL_CONC)
env.newPropertyArrayFloat("FOCAD_BIRTH_REFRACTORY", FOCAD_BIRTH_REFRACTORY)
env.newPropertyUInt("INCLUDE_LINC_COUPLING", INCLUDE_LINC_COUPLING)
env.newPropertyArrayFloat("LINC_K_ELAST", LINC_K_ELAST)
env.newPropertyArrayFloat("LINC_D_DUMPING", LINC_D_DUMPING)
env.newPropertyArrayFloat("LINC_REST_LENGTH", LINC_REST_LENGTH)

# Nucleus mechanical properties
env.newPropertyArrayFloat("NUCLEUS_E", NUCLEUS_E)
env.newPropertyArrayFloat("NUCLEUS_NU", NUCLEUS_NU)
env.newPropertyArrayFloat("NUCLEUS_TAU", NUCLEUS_TAU)
env.newPropertyArrayFloat("NUCLEUS_EPS_CLAMP", NUCLEUS_EPS_CLAMP)

# Chemotaxis properties
env.newPropertyUInt("INCLUDE_CHEMOTAXIS", INCLUDE_CHEMOTAXIS)
env.newPropertyArrayFloat("CHEMOTAXIS_CHI", CHEMOTAXIS_CHI)
env.newPropertyUInt("CHEMOTAXIS_ONLY_DIR", CHEMOTAXIS_ONLY_DIR)
env.newPropertyArrayFloat("CHEMOTAXIS_SENSITIVITY", CHEMOTAXIS_SENSITIVITY)

# Chemokinesis properties
env.newPropertyUInt("INCLUDE_CHEMOKINESIS", INCLUDE_CHEMOKINESIS)
env.newPropertyArrayFloat("CHEMOKINESIS_SENSITIVITY", CHEMOKINESIS_SENSITIVITY)
env.newPropertyArrayFloat("CHEMOKINESIS_ALPHA", CHEMOKINESIS_ALPHA)
env.newPropertyArrayFloat("CHEMOKINESIS_K", CHEMOKINESIS_K)
env.newPropertyArrayFloat("CHEMOKINESIS_HILL_N", CHEMOKINESIS_HILL_N)
env.newPropertyArrayFloat("CHEMOKINESIS_ADAPT_TAU", CHEMOKINESIS_ADAPT_TAU)
env.newPropertyArrayFloat("CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER", CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER)
env.newPropertyArrayFloat("CHEMOKINESIS_SIGNAL_SAT", CHEMOKINESIS_SIGNAL_SAT)

# Cell migration (durotaxis/orientation alignment) properties
env.newPropertyUInt("INCLUDE_DUROTAXIS", INCLUDE_DUROTAXIS)
env.newPropertyUInt("DUROTAXIS_ONLY_DIR", DUROTAXIS_ONLY_DIR)
env.newPropertyArrayFloat("FOCAD_MOBILITY_MU", FOCAD_MOBILITY_MU)
env.newPropertyUInt("INCLUDE_ORIENTATION_ALIGN", INCLUDE_ORIENTATION_ALIGN)
env.newPropertyArrayFloat("ORIENTATION_ALIGN_RATE", ORIENTATION_ALIGN_RATE)
env.newPropertyUInt("ORIENTATION_ALIGN_USE_STRESS", ORIENTATION_ALIGN_USE_STRESS)
env.newPropertyArrayFloat("DUROTAXIS_BLEND_BETA", DUROTAXIS_BLEND_BETA)
env.newPropertyUInt("DUROTAXIS_USE_STRESS", DUROTAXIS_USE_STRESS)


# ECM BEHAVIOUR 
# ------------------------------------------------------
# Equilibrium radius at which elastic force is 0.  TODO: add ECM_FIBRE elements
# If ECM_ECM_INTERACTION_RADIUS > ECM_ECM_EQUILIBRIUM_DISTANCE: both repulsion/atraction can occur
# If ECM_ECM_INTERACTION_RADIUS <= ECM_ECM_EQUILIBRIUM_DISTANCE: only repulsion can occur
env.newPropertyFloat("ECM_ECM_EQUILIBRIUM_DISTANCE", ECM_ECM_EQUILIBRIUM_DISTANCE)
# Mechanical parameters
env.newPropertyFloat("ECM_K_ELAST", ECM_K_ELAST)  # initial K_ELAST for agents
env.newPropertyFloat("ECM_D_DUMPING", ECM_D_DUMPING)
env.newPropertyFloat("ECM_ETA", ECM_ETA)
env.newPropertyFloat("BUCKLING_COEFF_D0", BUCKLING_COEFF_D0)
env.newPropertyFloat("STRAIN_STIFFENING_COEFF_DS", STRAIN_STIFFENING_COEFF_DS)
env.newPropertyFloat("CRITICAL_STRAIN", CRITICAL_STRAIN)
env.newPropertyFloat("MAX_STRAIN_K_FACTOR", MAX_STRAIN_K_FACTOR)

# Other globals
env.newPropertyFloat("PI", 3.1415)
env.newPropertyUInt("DEBUG_PRINTING", DEBUG_PRINTING)
env.newPropertyUInt("DEBUG_DIFFUSION", False)
env.newPropertyFloat("EPSILON", EPSILON)
env.newPropertyUInt("MOVING_BOUNDARIES", MOVING_BOUNDARIES)
env.newPropertyUInt("ABORT_ON_UNSTABLE_FNODE_MOVE", ABORT_ON_UNSTABLE_FNODE_MOVE)

# LUMEN agent properties (only registered when LUMEN is active)
if INCLUDE_CELLS and (ORGANOID_ASSAY or MONOLAYER_ASSAY) and INCLUDE_LUMEN:
    env.newPropertyFloat("LUMEN_RADIUS", LUMEN_RADIUS)
    env.newPropertyFloat("LUMEN_ETA", LUMEN_ETA)
    env.newPropertyFloat("LUMEN_K_LUMEN_LUMEN_REPULSION", LUMEN_K_LUMEN_LUMEN_REPULSION)
    env.newPropertyFloat("LUMEN_K_LUMEN_LUMEN_ADHESION", LUMEN_K_LUMEN_LUMEN_ADHESION)
    env.newPropertyFloat("LUMEN_LUMEN_ADHESION_RANGE", LUMEN_LUMEN_ADHESION_RANGE)
    env.newPropertyFloat("LUMEN_K_LUMEN_CELL_REPULSION", LUMEN_K_LUMEN_CELL_REPULSION)
    env.newPropertyFloat("LUMEN_LUMEN_CELL_DV_MAX", LUMEN_LUMEN_CELL_DV_MAX)
    env.newPropertyFloat("MAX_SEARCH_RADIUS_LUMEN_LUMEN_INTERACTION", MAX_SEARCH_RADIUS_LUMEN_LUMEN_INTERACTION)
    env.newPropertyFloat("MAX_SEARCH_RADIUS_LUMEN_CELL_INTERACTION", MAX_SEARCH_RADIUS_LUMEN_CELL_INTERACTION)
    env.newPropertyFloat("LUMEN_SECRETION_RATE", LUMEN_SECRETION_RATE)
    env.newPropertyFloat("LUMEN_SECRETION_COOLDOWN", LUMEN_SECRETION_COOLDOWN)
    env.newPropertyArrayFloat("LUMEN_DIFFUSION_COEFF_MULTI", LUMEN_DIFFUSION_COEFF_MULTI)

# VASC agent properties
if INCLUDE_VASCULARIZATION:
    env.newPropertyUInt("INCLUDE_VASCULARIZATION", 1)
    env.newPropertyFloat("MAX_SEARCH_RADIUS_VASCULARIZATION", MAX_SEARCH_RADIUS_VASCULARIZATION)
    env.newPropertyArrayFloat("INIT_VASCULARIZATION_CONCENTRATION_VALS", INIT_VASCULARIZATION_CONCENTRATION_VALS)
    env.newPropertyUInt("N_VASC_NODES", N_VASC_NODES)
else:
    env.newPropertyUInt("INCLUDE_VASCULARIZATION", 0)

# ++==================================================================++
# ++ Messages                                                          |
# ++==================================================================++
"""
  LOCATION MESSAGES
"""
BCORNER_location_message = model.newMessageSpatial3D("bcorner_location_message")
# Set the range and bounds.
BCORNER_location_message.setRadius(MAX_EXPECTED_BOUNDARY_POS - MIN_EXPECTED_BOUNDARY_POS)  # corners are not actually interacting with anything
BCORNER_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS)
BCORNER_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS)
# A message to hold the location of an agent. WARNING: spatial3D messages already define x,y,z variables internally.
BCORNER_location_message.newVariableInt("id")

if INCLUDE_FIBRE_NETWORK:
    FNODE_spatial_location_message = model.newMessageSpatial3D("fnode_spatial_location_message")
    # If heterogeneous diffusion is included, the search/broadcast radius for fibre nodes must be at least equal to the equilibrium distance to make sure that ECM nodes can find all the fibre nodes when looking for neighbours. 
    # WARNING: increasing this radius will increase the number of messages of fnode_fnode_spatial_interaction and therefore the computational cost of the simulation, so it should be kept as low as possible while making sure that fibre nodes are found by ECM nodes.
    fnode_spatial_radius = MAX_SEARCH_RADIUS_FNODES
    if (MAX_SEARCH_RADIUS_FNODES < ECM_ECM_EQUILIBRIUM_DISTANCE) and INCLUDE_DIFFUSION and HETEROGENEOUS_DIFFUSION:
        fnode_spatial_radius = ECM_ECM_EQUILIBRIUM_DISTANCE
    _max_excl = max(CELL_FNODE_EXCLUSION_DISTANCE) if isinstance(CELL_FNODE_EXCLUSION_DISTANCE, list) else CELL_FNODE_EXCLUSION_DISTANCE
    if INCLUDE_CELLS and INCLUDE_CELL_FNODE_REPULSION and (fnode_spatial_radius < _max_excl):
        fnode_spatial_radius = _max_excl
    _max_birth_reach = max(a + b for a, b in zip(FNODE_BIRTH_LINK_MAX_DISTANCE, FNODE_BIRTH_RADIUS))
    if INCLUDE_CELLS and INCLUDE_NETWORK_REMODELING and (fnode_spatial_radius < _max_birth_reach):
        fnode_spatial_radius = _max_birth_reach
    FNODE_spatial_location_message.setRadius(fnode_spatial_radius)
    FNODE_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS,MIN_EXPECTED_BOUNDARY_POS)
    FNODE_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS,MAX_EXPECTED_BOUNDARY_POS)
    FNODE_spatial_location_message.newVariableInt("id") # as an edge can have multiple inner agents, this stores the position within the edge
    FNODE_spatial_location_message.newVariableUInt8("connectivity_count")
    FNODE_spatial_location_message.newVariableInt("closest_fnode_id")
    FNODE_spatial_location_message.newVariableInt("second_closest_fnode_id")
    FNODE_spatial_location_message.newVariableInt("marked_for_removal")

    FNODE_bucket_location_message = model.newMessageBucket("fnode_bucket_location_message")
    # Set the range and bounds.
    # setBounds(min, max) where min and max are the min and max ids of the message buckets. This is independent of the number of agents (there can be more agents than buckets and vice versa).
    # Here, we assign one bucket per fibre node so that each fibre node can be found in its own bucket when searching for neighbours.
    max_expected_fnodes = N_NODES + 1
    if INCLUDE_CELLS and INCLUDE_NETWORK_REMODELING:
        max_expected_fnodes += MAX_EXPECTED_N_CELLS * STEPS
    FNODE_bucket_location_message.setBounds(8 + 1, 8 + max_expected_fnodes) # +8 because domain corners have idx from 1 to 8. WARNING: make sure to initialize fibre nodes starting from index 9

    FNODE_bucket_location_message.newVariableInt("id")
    FNODE_bucket_location_message.newVariableFloat("x")
    FNODE_bucket_location_message.newVariableFloat("y")
    FNODE_bucket_location_message.newVariableFloat("z")
    FNODE_bucket_location_message.newVariableFloat("vx")
    FNODE_bucket_location_message.newVariableFloat("vy")
    FNODE_bucket_location_message.newVariableFloat("vz")
    FNODE_bucket_location_message.newVariableFloat("k_elast")
    FNODE_bucket_location_message.newVariableFloat("d_dumping")
    FNODE_bucket_location_message.newVariableFloat("degradation")
    FNODE_bucket_location_message.newVariableFloat("reinforcement")
    FNODE_bucket_location_message.newVariableInt("marked_for_removal")
    FNODE_bucket_location_message.newVariableArrayFloat("equilibrium_distance", MAX_CONNECTIVITY) # each segment can have a different equilibrium distance depending on the rest length assigned during network generation
    FNODE_bucket_location_message.newVariableArrayInt("linked_nodes", MAX_CONNECTIVITY) # store the index of the linked nodes, which is a proxy for the bucket id

    # Lightweight post-move bucket used only by focad_move (L8) so that
    # attached FOCADs read the FNODE position after fnode_move instead of
    # the stale L1 pre-move data.
    FNODE_bucket_location_message_postmove = model.newMessageBucket("fnode_bucket_location_message_postmove")
    FNODE_bucket_location_message_postmove.setBounds(8 + 1, 8 + max_expected_fnodes)
    FNODE_bucket_location_message_postmove.newVariableInt("id")
    FNODE_bucket_location_message_postmove.newVariableFloat("x")
    FNODE_bucket_location_message_postmove.newVariableFloat("y")
    FNODE_bucket_location_message_postmove.newVariableFloat("z")
    FNODE_bucket_location_message_postmove.newVariableFloat("vx")
    FNODE_bucket_location_message_postmove.newVariableFloat("vy")
    FNODE_bucket_location_message_postmove.newVariableFloat("vz")


if INCLUDE_CELLS and (ORGANOID_ASSAY or MONOLAYER_ASSAY) and INCLUDE_LUMEN:
    # LUMEN spatial location message — broadcast radius must cover the larger of lumen-lumen and lumen-cell search radii
    lumen_spatial_radius = max(MAX_SEARCH_RADIUS_LUMEN_LUMEN_INTERACTION, MAX_SEARCH_RADIUS_LUMEN_CELL_INTERACTION)
    # Also ensure ECM can find lumen for diffusion override
    lumen_spatial_radius = max(lumen_spatial_radius, ECM_ECM_EQUILIBRIUM_DISTANCE)
    LUMEN_spatial_location_message = model.newMessageSpatial3D("lumen_spatial_location_message")
    LUMEN_spatial_location_message.setRadius(lumen_spatial_radius)
    LUMEN_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS)
    LUMEN_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS)
    LUMEN_spatial_location_message.newVariableInt("id")
    LUMEN_spatial_location_message.newVariableFloat("radius")


ECM_grid_location_message = model.newMessageArray3D("ecm_grid_location_message")
ECM_grid_location_message.setDimensions(ECM_AGENTS_PER_DIR[0], ECM_AGENTS_PER_DIR[1], ECM_AGENTS_PER_DIR[2])
ECM_grid_location_message.newVariableInt("id")
ECM_grid_location_message.newVariableFloat("x")
ECM_grid_location_message.newVariableFloat("y")
ECM_grid_location_message.newVariableFloat("z")
ECM_grid_location_message.newVariableInt("grid_lin_id")
ECM_grid_location_message.newVariableUInt8("grid_i")
ECM_grid_location_message.newVariableUInt8("grid_j")
ECM_grid_location_message.newVariableUInt8("grid_k")
ECM_grid_location_message.newVariableArrayFloat("D_sp", N_SPECIES)  # diffusion coefficient of each species at the agent location (used for heterogeneous diffusion)
ECM_grid_location_message.newVariableArrayFloat("C_sp", N_SPECIES)  
ECM_grid_location_message.newVariableArrayFloat("C_sp_sat", N_SPECIES) 
ECM_grid_location_message.newVariableFloat("k_elast")
ECM_grid_location_message.newVariableFloat("d_dumping")
ECM_grid_location_message.newVariableFloat("vx")
ECM_grid_location_message.newVariableFloat("vy")
ECM_grid_location_message.newVariableFloat("vz")
ECM_grid_location_message.newVariableFloat("fx")
ECM_grid_location_message.newVariableFloat("fy")
ECM_grid_location_message.newVariableFloat("fz")
ECM_grid_location_message.newVariableUInt8("clamped_bx_pos")
ECM_grid_location_message.newVariableUInt8("clamped_bx_neg")
ECM_grid_location_message.newVariableUInt8("clamped_by_pos")
ECM_grid_location_message.newVariableUInt8("clamped_by_neg")
ECM_grid_location_message.newVariableUInt8("clamped_bz_pos")
ECM_grid_location_message.newVariableUInt8("clamped_bz_neg")

if INCLUDE_CELLS:
    # If message type is MessageSpatial3D, variables x, y, z are included internally.
    CELL_spatial_location_message = model.newMessageSpatial3D("cell_spatial_location_message")
    CELL_spatial_location_message.setRadius(MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION)
    CELL_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS)
    CELL_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS)
    CELL_spatial_location_message.newVariableInt("id")
    CELL_spatial_location_message.newVariableFloat("vx")
    CELL_spatial_location_message.newVariableFloat("vy")
    CELL_spatial_location_message.newVariableFloat("vz")
    CELL_spatial_location_message.newVariableFloat("orx")
    CELL_spatial_location_message.newVariableFloat("ory")
    CELL_spatial_location_message.newVariableFloat("orz")
    CELL_spatial_location_message.newVariableFloat("alignment")
    CELL_spatial_location_message.newVariableArrayFloat("k_consumption", N_SPECIES) 
    CELL_spatial_location_message.newVariableArrayFloat("k_production", N_SPECIES) 
    CELL_spatial_location_message.newVariableArrayFloat("k_reaction", N_SPECIES) 
    CELL_spatial_location_message.newVariableArrayFloat("C_sp", N_SPECIES) 
    CELL_spatial_location_message.newVariableArrayFloat("M_sp", N_SPECIES)
    CELL_spatial_location_message.newVariableFloat("radius")
    CELL_spatial_location_message.newVariableFloat("cycle_phase")
    CELL_spatial_location_message.newVariableFloat("clock")
    CELL_spatial_location_message.newVariableInt("completed_cycles")
    CELL_spatial_location_message.newVariableInt("dead")
    CELL_spatial_location_message.newVariableInt("dead_by")
    CELL_spatial_location_message.newVariableInt("cell_type")
    # Radial-glia variant message variables — registered only when INCLUDE_RG_VARIABLES is True
    if INCLUDE_RG_VARIABLES:
        CELL_spatial_location_message.newVariableFloat("rg_commit_level")
        CELL_spatial_location_message.newVariableFloat("epithelialization_level")
        CELL_spatial_location_message.newVariableFloat("apx")
        CELL_spatial_location_message.newVariableFloat("apy")
        CELL_spatial_location_message.newVariableFloat("apz")
        
    # Set the range and bounds.
    if INCLUDE_FOCAL_ADHESIONS:
        CELL_bucket_location_message = model.newMessageBucket("cell_bucket_location_message")
        cell_bucket_min = 8 + N_NODES + 1
        cell_bucket_max = 8 + N_NODES + MAX_EXPECTED_N_CELLS
        if cell_bucket_max <= cell_bucket_min:
            cell_bucket_max = cell_bucket_min + 1 # to avoid compilation errors in case there is only 1 cell
        # +8 because domain corners have idx from 1 to 8, +N_NODES because fibre nodes have idx from 9 to 8+N_NODES. WARNING: make sure to initialize cell agents starting from index 8+N_NODES
        CELL_bucket_location_message.setBounds(cell_bucket_min, cell_bucket_max)
        CELL_bucket_location_message.newVariableInt("id")
        CELL_bucket_location_message.newVariableFloat("x")
        CELL_bucket_location_message.newVariableFloat("y")
        CELL_bucket_location_message.newVariableFloat("z")
        CELL_bucket_location_message.newVariableFloat("orx")
        CELL_bucket_location_message.newVariableFloat("ory")
        CELL_bucket_location_message.newVariableFloat("orz")
        CELL_bucket_location_message.newVariableFloat("nucleus_radius")
        CELL_bucket_location_message.newVariableFloat("eps_xx")
        CELL_bucket_location_message.newVariableFloat("eps_yy")
        CELL_bucket_location_message.newVariableFloat("eps_zz")
        CELL_bucket_location_message.newVariableFloat("eps_xy")
        CELL_bucket_location_message.newVariableFloat("eps_xz")
        CELL_bucket_location_message.newVariableFloat("eps_yz")
        CELL_bucket_location_message.newVariableInt("dead")
        CELL_bucket_location_message.newVariableInt("just_divided")
        CELL_bucket_location_message.newVariableInt("daughter_id")
        CELL_bucket_location_message.newVariableInt("marked_for_removal")
        CELL_bucket_location_message.newVariableArrayFloat("u_ref_x_i", N_ANCHOR_POINTS)
        CELL_bucket_location_message.newVariableArrayFloat("u_ref_y_i", N_ANCHOR_POINTS)
        CELL_bucket_location_message.newVariableArrayFloat("u_ref_z_i", N_ANCHOR_POINTS)
        CELL_bucket_location_message.newVariableArrayFloat("x_i", N_ANCHOR_POINTS)
        CELL_bucket_location_message.newVariableArrayFloat("y_i", N_ANCHOR_POINTS)
        CELL_bucket_location_message.newVariableArrayFloat("z_i", N_ANCHOR_POINTS)
        
        FOCAD_bucket_location_message = model.newMessageBucket("focad_bucket_location_message")
        FOCAD_bucket_location_message.setBounds(cell_bucket_min, cell_bucket_max) # WARNING: the key in the bucket list is the cell_id, not the focad id
        FOCAD_bucket_location_message.newVariableInt("id")
        FOCAD_bucket_location_message.newVariableInt("cell_id")
        FOCAD_bucket_location_message.newVariableInt("fnode_id")
        FOCAD_bucket_location_message.newVariableFloat("x")
        FOCAD_bucket_location_message.newVariableFloat("y")
        FOCAD_bucket_location_message.newVariableFloat("z")
        FOCAD_bucket_location_message.newVariableFloat("vx")
        FOCAD_bucket_location_message.newVariableFloat("vy")
        FOCAD_bucket_location_message.newVariableFloat("vz")
        FOCAD_bucket_location_message.newVariableFloat("fx")
        FOCAD_bucket_location_message.newVariableFloat("fy")
        FOCAD_bucket_location_message.newVariableFloat("fz")
        FOCAD_bucket_location_message.newVariableInt("anchor_id") # to identify which anchor point of the cell this focal adhesion corresponds to
        FOCAD_bucket_location_message.newVariableFloat("x_i")
        FOCAD_bucket_location_message.newVariableFloat("y_i")
        FOCAD_bucket_location_message.newVariableFloat("z_i")
        FOCAD_bucket_location_message.newVariableFloat("x_c")
        FOCAD_bucket_location_message.newVariableFloat("y_c")
        FOCAD_bucket_location_message.newVariableFloat("z_c")
        FOCAD_bucket_location_message.newVariableFloat("rest_length_0")
        FOCAD_bucket_location_message.newVariableFloat("rest_length")
        FOCAD_bucket_location_message.newVariableFloat("k_fa")
        FOCAD_bucket_location_message.newVariableFloat("f_max")
        FOCAD_bucket_location_message.newVariableInt("attached")
        FOCAD_bucket_location_message.newVariableUInt8("active")
        FOCAD_bucket_location_message.newVariableFloat("v_c")
        FOCAD_bucket_location_message.newVariableUInt8("fa_state")
        FOCAD_bucket_location_message.newVariableFloat("age")
        FOCAD_bucket_location_message.newVariableFloat("detached_age")
        FOCAD_bucket_location_message.newVariableFloat("k_on")
        FOCAD_bucket_location_message.newVariableFloat("k_off_0")
        FOCAD_bucket_location_message.newVariableFloat("f_c")
        FOCAD_bucket_location_message.newVariableFloat("k_reinf")

        
        FOCAD_spatial_location_message = model.newMessageSpatial3D("focad_spatial_location_message")
        FOCAD_spatial_location_message.setRadius(MAX_SEARCH_RADIUS_FOCAD)
        FOCAD_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS)
        FOCAD_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS)
        FOCAD_spatial_location_message.newVariableInt("id")
        FOCAD_spatial_location_message.newVariableFloat("fx")
        FOCAD_spatial_location_message.newVariableFloat("fy")
        FOCAD_spatial_location_message.newVariableFloat("fz")
        FOCAD_spatial_location_message.newVariableInt("fnode_id")
        FOCAD_spatial_location_message.newVariableInt("attached")
        FOCAD_spatial_location_message.newVariableUInt8("active")

if INCLUDE_VASCULARIZATION:
    VASC_bucket_location_message = model.newMessageBucket("vasc_bucket_location_message")
    # Compute bucket ID bounds: VASC agents occupy the ID range immediately after ECM agents
    VASC_BUCKET_MIN = _vasc_id_base + 1
    VASC_BUCKET_MAX = _vasc_id_base + N_VASC_NODES + 1
    if VASC_BUCKET_MAX <= VASC_BUCKET_MIN:  # guard against empty range (0 VASC nodes)
        VASC_BUCKET_MAX = VASC_BUCKET_MIN + 1
    VASC_bucket_location_message.setBounds(VASC_BUCKET_MIN, VASC_BUCKET_MAX)
    VASC_bucket_location_message.newVariableInt("id")
    VASC_bucket_location_message.newVariableFloat("x")
    VASC_bucket_location_message.newVariableFloat("y")
    VASC_bucket_location_message.newVariableFloat("z")
    VASC_bucket_location_message.newVariableArrayInt("parent_ids", MAX_VASC_CONNECTIVITY)
    VASC_bucket_location_message.newVariableArrayInt("children_ids", MAX_VASC_CONNECTIVITY)
    VASC_bucket_location_message.newVariableArrayFloat("C_sp", N_SPECIES)
    VASC_bucket_location_message.newVariableInt("dead")

    VASC_spatial_location_message = model.newMessageSpatial3D("vasc_spatial_location_message")
    VASC_spatial_location_message.setRadius(MAX_SEARCH_RADIUS_VASCULARIZATION)
    VASC_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS)
    VASC_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS)
    VASC_spatial_location_message.newVariableInt("id")
    VASC_spatial_location_message.newVariableInt("dead")
    VASC_spatial_location_message.newVariableArrayFloat("C_sp", N_SPECIES)

# ++==================================================================++
# ++ Agents                                                            |
# ++==================================================================++
"""
  AGENTS
"""

"""
  BCORNER agent
"""
BCORNER_agent = model.newAgent("BCORNER") # boundary corner agent to track boundary positions
BCORNER_agent.newVariableInt("id") # unique boundary-corner id
BCORNER_agent.newVariableFloat("x") # boundary corner position [um]
BCORNER_agent.newVariableFloat("y")
BCORNER_agent.newVariableFloat("z")

BCORNER_agent.newRTCFunctionFile("bcorner_output_location_data", bcorner_output_location_data_file).setMessageOutput("bcorner_location_message")
if MOVING_BOUNDARIES:
    BCORNER_agent.newRTCFunctionFile("bcorner_move", bcorner_move_file)

"""
  FIBRE NODE agent
"""
if INCLUDE_FIBRE_NETWORK:
    FNODE_agent = model.newAgent("FNODE")
    FNODE_agent.newVariableInt("id") # unique fibre-node id
    FNODE_agent.newVariableFloat("x") # fibre-node position [um]
    FNODE_agent.newVariableFloat("y")
    FNODE_agent.newVariableFloat("z")
    FNODE_agent.newVariableFloat("vx", 0.0) # fibre-node velocity [um/s]
    FNODE_agent.newVariableFloat("vy", 0.0)
    FNODE_agent.newVariableFloat("vz", 0.0)
    FNODE_agent.newVariableFloat("fx", 0.0) # net force on the fibre node [nN]
    FNODE_agent.newVariableFloat("fy", 0.0)
    FNODE_agent.newVariableFloat("fz", 0.0)
    FNODE_agent.newVariableFloat("k_elast") # effective segment stiffness [nN/um]
    FNODE_agent.newVariableFloat("d_dumping") # effective segment damping [nN*s/um]
    FNODE_agent.newVariableArrayFloat("equilibrium_distance", MAX_CONNECTIVITY) # each segment can have a different equilibrium distance depending on the rest length assigned during network generation
    FNODE_agent.newVariableFloat("boundary_fx")  # boundary_f[A]: normal force coming from boundary [A] when elastic boundaries option is selected.
    FNODE_agent.newVariableFloat("boundary_fy")
    FNODE_agent.newVariableFloat("boundary_fz")
    FNODE_agent.newVariableFloat("f_bx_pos")  # f_b[A]_[B]: normal force transmitted to the boundary [A]_[B] when agent is clamped
    FNODE_agent.newVariableFloat("f_bx_neg")
    FNODE_agent.newVariableFloat("f_by_pos")
    FNODE_agent.newVariableFloat("f_by_neg")
    FNODE_agent.newVariableFloat("f_bz_pos")
    FNODE_agent.newVariableFloat("f_bz_neg")
    FNODE_agent.newVariableFloat("f_bx_pos_y")  # f_b[A]_[B]_[C]: shear force transmitted to the boundary [A]_[B] in the direction [C] when agent is clamped
    FNODE_agent.newVariableFloat("f_bx_pos_z")
    FNODE_agent.newVariableFloat("f_bx_neg_y")
    FNODE_agent.newVariableFloat("f_bx_neg_z")
    FNODE_agent.newVariableFloat("f_by_pos_x")
    FNODE_agent.newVariableFloat("f_by_pos_z")
    FNODE_agent.newVariableFloat("f_by_neg_x")
    FNODE_agent.newVariableFloat("f_by_neg_z")
    FNODE_agent.newVariableFloat("f_bz_pos_x")
    FNODE_agent.newVariableFloat("f_bz_pos_y")
    FNODE_agent.newVariableFloat("f_bz_neg_x")
    FNODE_agent.newVariableFloat("f_bz_neg_y")
    FNODE_agent.newVariableFloat("f_extension") # tensile load carried by connected segments [nN]
    FNODE_agent.newVariableFloat("f_compression") # compressive load carried by connected segments [nN]
    FNODE_agent.newVariableFloat("elastic_energy") # stored elastic energy [nN*um]
    FNODE_agent.newVariableUInt8("connectivity_count", 0) # number of linked neighbour nodes
    FNODE_agent.newVariableFloat("degradation", 0.0) # accumulated degradation state [-]
    FNODE_agent.newVariableFloat("reinforcement", 0.0) # accumulated reinforcement state [-]
    FNODE_agent.newVariableInt("secreted", 0) # 1 if this node was newly secreted by a cell
    FNODE_agent.newVariableInt("marked_for_removal", 0) # 1 if the node should be deleted
    FNODE_agent.newVariableInt("closest_fnode_id", -1) # id of the nearest neighbouring FNODE
    FNODE_agent.newVariableInt("second_closest_fnode_id", -1) # id of the second-nearest neighbouring FNODE
    FNODE_agent.newVariableArrayFloat("linked_nodes", MAX_CONNECTIVITY) # ids of connected neighbour FNODEs
    FNODE_agent.newVariableUInt8("clamped_bx_pos") # boundary clamp flags for each face (1 = clamped)
    FNODE_agent.newVariableUInt8("clamped_bx_neg")
    FNODE_agent.newVariableUInt8("clamped_by_pos")
    FNODE_agent.newVariableUInt8("clamped_by_neg")
    FNODE_agent.newVariableUInt8("clamped_bz_pos")
    FNODE_agent.newVariableUInt8("clamped_bz_neg")
    FNODE_agent.newVariableUInt8("unstable_move", 0)
    FNODE_agent.newVariableInt("focad_attached", 0)
    FNODE_agent.newVariableInt("focad_id", -1)

    FNODE_agent.newRTCFunctionFile("fnode_spatial_location_data", fnode_spatial_location_data_file).setMessageOutput("fnode_spatial_location_message")
    FNODE_agent.newRTCFunctionFile("fnode_bucket_location_data", fnode_bucket_location_data_file).setMessageOutput("fnode_bucket_location_message")
    FNODE_agent.newRTCFunctionFile("fnode_bucket_location_data_postmove", fnode_bucket_location_data_postmove_file).setMessageOutput("fnode_bucket_location_message_postmove")
    FNODE_agent.newRTCFunctionFile("fnode_boundary_interaction", fnode_boundary_interaction_file)
    FNODE_agent.newRTCFunctionFile("fnode_update_links", fnode_update_links_file).setMessageInput("fnode_bucket_location_message")
    FNODE_agent.newRTCFunctionFile("fnode_fnode_spatial_interaction", fnode_fnode_spatial_interaction_file).setMessageInput("fnode_spatial_location_message")
    FNODE_agent.newRTCFunctionFile("fnode_fnode_bucket_interaction", fnode_fnode_bucket_interaction_file).setMessageInput("fnode_bucket_location_message")
    if INCLUDE_CELLS and INCLUDE_NETWORK_REMODELING:
        FNODE_agent.newRTCFunctionFile("fnode_remodel", fnode_remodel_file).setMessageInput("cell_spatial_location_message")
        fna = FNODE_agent.newRTCFunctionFile("fnode_apply_remodel_updates", fnode_apply_remodel_updates_file)
        fna.setMessageInput("fnode_spatial_location_message")
        fna.setAllowAgentDeath(True)
    FNODE_agent.newRTCFunctionFile("fnode_move", fnode_move_file)
    if INCLUDE_FOCAL_ADHESIONS:
        FNODE_agent.newRTCFunctionFile("fnode_focad_interaction", fnode_focad_interaction_file).setMessageInput("focad_spatial_location_message")
    if INCLUDE_CELLS and INCLUDE_CELL_FNODE_REPULSION:
        FNODE_agent.newRTCFunctionFile("fnode_cell_repulsion", fnode_cell_repulsion_file).setMessageInput("cell_spatial_location_message")


"""
  ECM agent
"""
ECM_agent = model.newAgent("ECM")
ECM_agent.newVariableInt("id", 0) # unique ECM-agent id
ECM_agent.newVariableFloat("x", 0.0) # ECM grid-point spatial position [um]
ECM_agent.newVariableFloat("y", 0.0)
ECM_agent.newVariableFloat("z", 0.0)
ECM_agent.newVariableInt("grid_lin_id", 0) # linear index in the 3D grid that maps to i,j,k positions
ECM_agent.newVariableUInt8("grid_i", 0) # grid index, (i,j,k) maps to (x,y,z)
ECM_agent.newVariableUInt8("grid_j", 0)
ECM_agent.newVariableUInt8("grid_k", 0)
ECM_agent.newVariableArrayFloat("D_sp", N_SPECIES) # diffusion coefficient of each species at the agent location (used for heterogeneous diffusion)
ECM_agent.newVariableArrayFloat("C_sp", N_SPECIES) # species concentrations at this ECM node
ECM_agent.newVariableArrayFloat("C_sp_sat", N_SPECIES) # saturation concentrations for each species
ECM_agent.newVariableFloat("k_elast") # ECM spring stiffness [nN/um] (used only for smooth grid adapation if boundaries move)
ECM_agent.newVariableFloat("d_dumping") # ECM damping coefficient [nN*s/um] (used only for smooth grid adapation if boundaries move)
ECM_agent.newVariableFloat("vx") # ECM grid-point velocity [um/s] (used only for smooth grid adapation if boundaries move)
ECM_agent.newVariableFloat("vy")
ECM_agent.newVariableFloat("vz")
ECM_agent.newVariableFloat("fx") # net force on the ECM grid-point [nN] (used only for smooth grid adapation if boundaries move)
ECM_agent.newVariableFloat("fy")
ECM_agent.newVariableFloat("fz")
ECM_agent.newVariableUInt8("clamped_bx_pos") # boundary clamp flags for each face (1 = clamped) (used only for smooth grid adapation if boundaries move)
ECM_agent.newVariableUInt8("clamped_bx_neg")
ECM_agent.newVariableUInt8("clamped_by_pos")
ECM_agent.newVariableUInt8("clamped_by_neg")
ECM_agent.newVariableUInt8("clamped_bz_pos")
ECM_agent.newVariableUInt8("clamped_bz_neg")
ECM_agent.newRTCFunctionFile("ecm_grid_location_data", ecm_grid_location_data_file).setMessageOutput("ecm_grid_location_message")
ECM_agent.newRTCFunctionFile("ecm_ecm_interaction", ecm_ecm_interaction_file).setMessageInput("ecm_grid_location_message")
ECM_agent.newRTCFunctionFile("ecm_boundary_concentration_conditions", ecm_boundary_concentration_conditions_file)
ECM_agent.newRTCFunctionFile("ecm_Csp_update", ecm_Csp_update_file)
if HETEROGENEOUS_DIFFUSION and INCLUDE_FIBRE_NETWORK:
    ECM_agent.newRTCFunctionFile("ecm_Dsp_update", ecm_Dsp_update_file).setMessageInput("fnode_spatial_location_message")
if INCLUDE_CELLS and (ORGANOID_ASSAY or MONOLAYER_ASSAY) and INCLUDE_LUMEN and HETEROGENEOUS_DIFFUSION:
    ECM_agent.newRTCFunctionFile("ecm_Dsp_lumen_update", ecm_Dsp_lumen_update_file).setMessageInput("lumen_spatial_location_message")
if MOVING_BOUNDARIES:
    ECM_agent.newRTCFunctionFile("ecm_move", ecm_move_file)

"""
  CELL agent
"""
if INCLUDE_CELLS:
    cell_focad_update_fn = None
    CELL_agent = model.newAgent("CELL")
    CELL_agent.newVariableInt("id", 0) # unique cell id
    CELL_agent.newVariableFloat("x", 0.0) # cell-center position [um]
    CELL_agent.newVariableFloat("y", 0.0)
    CELL_agent.newVariableFloat("z", 0.0)
    CELL_agent.newVariableFloat("vx", 0.0) # cell velocity [um/s]
    CELL_agent.newVariableFloat("vy", 0.0)
    CELL_agent.newVariableFloat("vz", 0.0)
    CELL_agent.newVariableFloat("trajectory_length", 0.0) # cumulative path length since birth/latest division [um]
    CELL_agent.newVariableFloat("trajectory_time", 0.0) # elapsed tracked lifetime since birth/latest division [s]
    CELL_agent.newVariableFloat("birth_x", 0.0) # reference position for effective speed [um]
    CELL_agent.newVariableFloat("birth_y", 0.0)
    CELL_agent.newVariableFloat("birth_z", 0.0)
    CELL_agent.newVariableFloat("orx") # cell polarity/orientation unit vector
    CELL_agent.newVariableFloat("ory")
    CELL_agent.newVariableFloat("orz")
    CELL_agent.newVariableInt("cell_type", 0) # to represent different phenotypes (e.g.  different cell lines). The specific meaning of the values assigned to this variable is up to the user and is not defined by the model.
    CELL_agent.newVariableFloat("k_elast") # cell stiffness [nN/um] per-type; set during init (unused in the current implementation)
    CELL_agent.newVariableFloat("d_dumping") # cell damping coefficient [nN*s/um] per-type; set during init (unused in the current implementation)
    CELL_agent.newVariableFloat("alignment", 0.0) # alignment score with the local fibre field [-] (unused in the current implementation)
    CELL_agent.newVariableArrayFloat("k_consumption", N_SPECIES) # per-species consumption rate constants
    CELL_agent.newVariableArrayFloat("k_production", N_SPECIES) # per-species production rate constants
    CELL_agent.newVariableArrayFloat("k_reaction", N_SPECIES) # per-species reaction rate constants
    CELL_agent.newVariableArrayFloat("C_sp", N_SPECIES) # per-species cell-associated concentration state
    CELL_agent.newVariableArrayFloat("M_sp", N_SPECIES) # per-species cell-associated mass state
    CELL_agent.newVariableFloat("speed_ref")   # per-type; set during init
    CELL_agent.newVariableFloat("radius")        # per-type; set during init
    CELL_agent.newVariableFloat("nucleus_radius") # per-type; set during init
    CELL_agent.newVariableFloat("cc_dvx", 0.0)  # [um/s] velocity contribution from cell_cell_interaction
    CELL_agent.newVariableFloat("cc_dvy", 0.0)  
    CELL_agent.newVariableFloat("cc_dvz", 0.0)  
    CELL_agent.newVariableFloat("cf_dvx", 0.0)  # [um/s] velocity contribution from cell_fnode_repulsion
    CELL_agent.newVariableFloat("cf_dvy", 0.0)  
    CELL_agent.newVariableFloat("cf_dvz", 0.0)  
    CELL_agent.newVariableFloat("cl_dvx", 0.0)  # [um/s] velocity contribution from cell_lumen_interaction
    CELL_agent.newVariableFloat("cl_dvy", 0.0)  
    CELL_agent.newVariableFloat("cl_dvz", 0.0)  
    # Contact-derived stresslet accumulators [nN·um] — summed each step by interaction functions
    CELL_agent.newVariableFloat("cc_S_xx", 0.0)  # cell-cell stresslet components
    CELL_agent.newVariableFloat("cc_S_yy", 0.0)
    CELL_agent.newVariableFloat("cc_S_zz", 0.0)
    CELL_agent.newVariableFloat("cc_S_xy", 0.0)
    CELL_agent.newVariableFloat("cc_S_xz", 0.0)
    CELL_agent.newVariableFloat("cc_S_yz", 0.0)
    CELL_agent.newVariableFloat("cf_S_xx", 0.0)  # cell-fnode stresslet components
    CELL_agent.newVariableFloat("cf_S_yy", 0.0)
    CELL_agent.newVariableFloat("cf_S_zz", 0.0)
    CELL_agent.newVariableFloat("cf_S_xy", 0.0)
    CELL_agent.newVariableFloat("cf_S_xz", 0.0)
    CELL_agent.newVariableFloat("cf_S_yz", 0.0)
    CELL_agent.newVariableFloat("cl_S_xx", 0.0)  # cell-lumen stresslet components
    CELL_agent.newVariableFloat("cl_S_yy", 0.0)
    CELL_agent.newVariableFloat("cl_S_zz", 0.0)
    CELL_agent.newVariableFloat("cl_S_xy", 0.0)
    CELL_agent.newVariableFloat("cl_S_xz", 0.0)
    CELL_agent.newVariableFloat("cl_S_yz", 0.0)
    CELL_agent.newVariableFloat("focad_S_xx", 0.0)  # focal-adhesion stresslet components
    CELL_agent.newVariableFloat("focad_S_yy", 0.0)
    CELL_agent.newVariableFloat("focad_S_zz", 0.0)
    CELL_agent.newVariableFloat("focad_S_xy", 0.0)
    CELL_agent.newVariableFloat("focad_S_xz", 0.0)
    CELL_agent.newVariableFloat("focad_S_yz", 0.0)
    CELL_agent.newVariableFloat("lumen_secretion_cooldown", 0.0)  # [s] cooldown timer before cell can secrete another lumen droplet
    CELL_agent.newVariableInt("cycle_phase", 1) # [1:G1] [2:S] [3:G2] [4:M]
    CELL_agent.newVariableFloat("clock", 0.0) # internal clock of the cell to switch phases
    CELL_agent.newVariableInt("completed_cycles", 0) # number of completed cell cycles
    CELL_agent.newVariableInt("max_global_cell_id", 0) # cached global max CELL id (to atomically track newly created cells)
    CELL_agent.newVariableFloat("damage", 0.0) # accumulated damage score in [0,1], where 1 is lethal threshold
    CELL_agent.newVariableInt("dead", 0) # 0: alive, 1: dead (dead cells can be kept as debris or removed using the DEAD_CELLS_DISAPPEAR flag)
    CELL_agent.newVariableInt("dead_by", -1) # -1:none, 0:hypoxia, 1:starvation, 2:mechanical, 3:cumulative_damage
    CELL_agent.newVariableInt("mother_id", -1) # id of the parent cell if this cell is a daughter
    CELL_agent.newVariableInt("daughter_id", -1) # id of the daughter created in the latest division
    CELL_agent.newVariableInt("just_divided", 0) # 1 during the step immediately after division
    CELL_agent.newVariableInt("marked_for_removal", 0) # 1 if the cell should be removed
    CELL_agent.newVariableFloat("fnode_birth_cooldown", 0.0) # refractory time before creating another FNODE [s]
    CELL_agent.newVariableFloat("focad_birth_cooldown", 0.0) # refractory time before creating another FOCAD [s]
    CELL_agent.newVariableArrayFloat("chemokinesis_promotive_adapt_state", N_SPECIES) # memory of past chemokine exposure to diminish migration promoting signaling if concentration is constant
    CELL_agent.newVariableArrayFloat("chemokinesis_inhibitory_adapt_state", N_SPECIES) # memory of past chemokine exposure to diminish migration inhibiting signaling if concentration is constant
    CELL_agent.newRTCFunctionFile("cell_spatial_location_data", cell_spatial_location_data_file).setMessageOutput("cell_spatial_location_message")
    if INCLUDE_CELL_CELL_INTERACTION:
        CELL_agent.newRTCFunctionFile("cell_cell_interaction", cell_cell_interaction_file).setMessageInput("cell_spatial_location_message")
    if INCLUDE_FIBRE_NETWORK and INCLUDE_CELL_FNODE_REPULSION:
        CELL_agent.newRTCFunctionFile("cell_fnode_repulsion", cell_fnode_repulsion_file).setMessageInput("fnode_spatial_location_message")
    if INCLUDE_FIBRE_NETWORK and INCLUDE_NETWORK_REMODELING:
        cfr = CELL_agent.newRTCFunctionFile("cell_fnode_remodel", cell_fnode_remodel_file)
        cfr.setMessageInput("fnode_spatial_location_message")
        cfr.setAgentOutput(FNODE_agent)
    CELL_agent.newRTCFunctionFile("cell_ecm_interaction_metabolism", cell_ecm_interaction_metabolism_file).setMessageInput("ecm_grid_location_message")
    CELL_agent.newRTCFunctionFile("cell_move", cell_move_file)
    if (ORGANOID_ASSAY or MONOLAYER_ASSAY) and INCLUDE_LUMEN:
        CELL_agent.newRTCFunctionFile("cell_lumen_interaction", cell_lumen_interaction_file).setMessageInput("lumen_spatial_location_message")
        cls_fn = CELL_agent.newRTCFunctionFile("cell_lumen_secretion", cell_lumen_secretion_file)
        # agent_out for lumen is set after LUMEN_agent is defined (below)
    CELL_agent.newVariableArrayFloat("x_i", N_ANCHOR_POINTS) # focal-adhesion anchor point positions on the cell nucleus surface. Unused if INCLUDE_FOCAL_ADHESIONS is False
    CELL_agent.newVariableArrayFloat("y_i", N_ANCHOR_POINTS) 
    CELL_agent.newVariableArrayFloat("z_i", N_ANCHOR_POINTS)
    CELL_agent.newVariableArrayFloat("u_ref_x_i", N_ANCHOR_POINTS) # unit direction vector from the cell center to the anchor point in the reference configuration (used for elastic force calculation). Unused if INCLUDE_FOCAL_ADHESIONS is False
    CELL_agent.newVariableArrayFloat("u_ref_y_i", N_ANCHOR_POINTS)
    CELL_agent.newVariableArrayFloat("u_ref_z_i", N_ANCHOR_POINTS)
    CELL_agent.newVariableFloat("eps_xx", 0.0) # strain tensor
    CELL_agent.newVariableFloat("eps_yy", 0.0)
    CELL_agent.newVariableFloat("eps_zz", 0.0)
    CELL_agent.newVariableFloat("eps_xy", 0.0)
    CELL_agent.newVariableFloat("eps_xz", 0.0)
    CELL_agent.newVariableFloat("eps_yz", 0.0)
    CELL_agent.newVariableFloat("sig_xx", 0.0) # stress tensor [kPa]
    CELL_agent.newVariableFloat("sig_yy", 0.0)
    CELL_agent.newVariableFloat("sig_zz", 0.0)
    CELL_agent.newVariableFloat("sig_xy", 0.0)
    CELL_agent.newVariableFloat("sig_xz", 0.0)
    CELL_agent.newVariableFloat("sig_yz", 0.0)
    CELL_agent.newVariableFloat("sig_eig_1", 0.0) # principal stresses (eigen values) [kPa]
    CELL_agent.newVariableFloat("sig_eig_2", 0.0) 
    CELL_agent.newVariableFloat("sig_eig_3", 0.0) 
    CELL_agent.newVariableFloat("sig_eigvec1_x", 0.0) # first principal-stress direction
    CELL_agent.newVariableFloat("sig_eigvec1_y", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec1_z", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec2_x", 0.0) # second principal-stress direction
    CELL_agent.newVariableFloat("sig_eigvec2_y", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec2_z", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec3_x", 0.0) # third principal-stress direction
    CELL_agent.newVariableFloat("sig_eigvec3_y", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec3_z", 0.0)
    CELL_agent.newVariableFloat("eps_eig_1", 0.0) # principal strains  (eigen values)[-]
    CELL_agent.newVariableFloat("eps_eig_2", 0.0) 
    CELL_agent.newVariableFloat("eps_eig_3", 0.0) 
    CELL_agent.newVariableFloat("eps_eigvec1_x", 0.0) # first principal-strain direction
    CELL_agent.newVariableFloat("eps_eigvec1_y", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec1_z", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec2_x", 0.0) # second principal-strain direction
    CELL_agent.newVariableFloat("eps_eigvec2_y", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec2_z", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec3_x", 0.0) # third principal-strain direction
    CELL_agent.newVariableFloat("eps_eigvec3_y", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec3_z", 0.0)
    # Radial-glia variant variables — registered only when INCLUDE_RG_VARIABLES is True
    if INCLUDE_RG_VARIABLES:
        CELL_agent.newVariableFloat("rg_commit_level",         0.0)  # [-] logistic commit state (0=iPSC, 1=RG)
        CELL_agent.newVariableFloat("epithelialization_level", 0.0)  # [-] junction coverage (0=unpolarised, 1=epithelial)
        CELL_agent.newVariableFloat("rosette_maturity",        0.0)  # [-] rosette formation index
        CELL_agent.newVariableFloat("apx", 0.0)  # apical polarity vector x
        CELL_agent.newVariableFloat("apy", 0.0)  # apical polarity vector y
        CELL_agent.newVariableFloat("apz", 0.0)  # apical polarity vector z
        CELL_agent.newVariableFloat("rg_neighbour_density", 0.0)  # normalised local RG-cell count
        CELL_agent.newVariableFloat("morphogen_local",      0.0)  # cached ECM morphogen concentration sample (sp2) at cell location
        CELL_agent.newVariableInt("rg_committed", 0)              # 0/1 irreversible commit flag
        CELL_agent.newVariableFloat("substrate_anchor_x", 0.0)   # xy substrate anchor position for bond-spring (NPC/RG)
        CELL_agent.newVariableFloat("substrate_anchor_y", 0.0)
    if INCLUDE_FOCAL_ADHESIONS:  
        CELL_agent.newRTCFunctionFile("cell_bucket_location_data", cell_bucket_location_data_file).setMessageOutput("cell_bucket_location_message")
        cell_focad_update_fn = CELL_agent.newRTCFunctionFile("cell_focad_update", cell_focad_update_file)
        cell_focad_update_fn.setMessageInput("focad_bucket_location_message")
    CELL_agent.newRTCFunctionFile("cell_stress_state_update", cell_stress_state_update_file)
    if INCLUDE_CELL_CYCLE:
        CELL_agent.newRTCFunctionFile("cell_MaxID_update", cell_maxid_update_file)
        ccf = CELL_agent.newRTCFunctionFile("cell_cycle", cell_cycle_file)
        ccf.setAgentOutput(CELL_agent)
        ccf.setAllowAgentDeath(True) 
        
"""
  FOCAD agent
"""
if INCLUDE_FOCAL_ADHESIONS:
    FOCAD_agent = model.newAgent("FOCAD")
    FOCAD_agent.newVariableInt("id", 0) # unique focal-adhesion id
    FOCAD_agent.newVariableInt("cell_id") # id of the owner cell
    FOCAD_agent.newVariableInt("cell_type", 0)  # cell type of the owner cell (for per-type property lookups)
    FOCAD_agent.newVariableInt("fnode_id") # id of the interacting fibre node if attached (-1 if not)
    FOCAD_agent.newVariableFloat("x", 0.0) # focal-adhesion position [um]
    FOCAD_agent.newVariableFloat("y", 0.0)
    FOCAD_agent.newVariableFloat("z", 0.0)
    FOCAD_agent.newVariableFloat("vx", 0.0) # focal-adhesion velocity [um/s]
    FOCAD_agent.newVariableFloat("vy", 0.0)
    FOCAD_agent.newVariableFloat("vz", 0.0)
    FOCAD_agent.newVariableFloat("fx", 0.0) # net force on the focal adhesion [nN]
    FOCAD_agent.newVariableFloat("fy", 0.0)
    FOCAD_agent.newVariableFloat("fz", 0.0)
    FOCAD_agent.newVariableInt("anchor_id",-1) # index of the associated cell anchor point
    FOCAD_agent.newVariableFloat("x_i", 0.0) # anchor-point position on the cell surface [um]
    FOCAD_agent.newVariableFloat("y_i", 0.0)
    FOCAD_agent.newVariableFloat("z_i", 0.0)
    FOCAD_agent.newVariableFloat("x_c", 0.0) # owner-cell center position [um]
    FOCAD_agent.newVariableFloat("y_c", 0.0)
    FOCAD_agent.newVariableFloat("z_c", 0.0)
    FOCAD_agent.newVariableFloat("orx", 1.0) # owner-cell orientation unit vector
    FOCAD_agent.newVariableFloat("ory", 0.0)
    FOCAD_agent.newVariableFloat("orz", 0.0)
    FOCAD_agent.newVariableFloat("rest_length_0") # rest length at adhesion birth [um]
    FOCAD_agent.newVariableFloat("rest_length") # current effective rest length [um]
    FOCAD_agent.newVariableFloat("k_fa") # focal-adhesion stiffness [nN/um]
    FOCAD_agent.newVariableFloat("f_max") # maximum sustainable traction force [nN]
    FOCAD_agent.newVariableInt("attached") # 1 if attached to a fibre node
    FOCAD_agent.newVariableUInt8("active") # 1 if the adhesion is active in the simulation
    FOCAD_agent.newVariableFloat("v_c") # rest-length shortening speed [um/s]
    FOCAD_agent.newVariableUInt8("fa_state") # adhesion state code: 1 nascent, 2 mature, 3 disassembling
    FOCAD_agent.newVariableFloat("age") # adhesion lifetime [s]
    FOCAD_agent.newVariableFloat("detached_age", 0.0) # time spent detached [s]
    FOCAD_agent.newVariableFloat("k_on") # attachment rate [1/s]
    FOCAD_agent.newVariableFloat("k_off_0") # baseline detachment rate [1/s]
    FOCAD_agent.newVariableFloat("f_c") # force scale in the slip-bond law [nN]
    FOCAD_agent.newVariableFloat("k_reinf") # force-dependent reinforcement rate [1/s]
    FOCAD_agent.newVariableFloat("f_mag", 0.0)  # |F_FA| traction magnitude [nN] at current step
    FOCAD_agent.newVariableInt("is_front", 0)  # 1 if adhesion is classified in the cell front hemisphere, else 0
    FOCAD_agent.newVariableInt("is_rear", 0)  # 1 if adhesion is classified in the cell rear hemisphere, else 0
    FOCAD_agent.newVariableInt("attached_front", 0)  # 1 if attached and in front
    FOCAD_agent.newVariableInt("attached_rear", 0)  # 1 if attached and in rear
    FOCAD_agent.newVariableFloat("frontness_front", 0.0)  # frontness score used for front-biased kinetics (front branch). Polarity score p in [-1,1] from orientation vs anchor direction (cell center towards anchor)
    FOCAD_agent.newVariableFloat("frontness_rear", 0.0)  # rearness score used for rear-biased kinetics
    FOCAD_agent.newVariableFloat("k_on_eff_front", 0.0)  # effective attachment rate used for front-side update [1/s]
    FOCAD_agent.newVariableFloat("k_on_eff_rear", 0.0)  # effective attachment rate used for rear-side update [1/s]
    FOCAD_agent.newVariableFloat("k_off_0_eff_front", 0.0)  # effective baseline detachment rate at front [1/s]
    FOCAD_agent.newVariableFloat("k_off_0_eff_rear", 0.0)  # effective baseline detachment rate at rear [1/s]
    FOCAD_agent.newVariableFloat("linc_prev_total_length", 0.0)  # previous-step LINC internal length state for Kelvin-Voigt-in-series solve [um]


    FOCAD_agent.newRTCFunctionFile("focad_bucket_location_data", focad_bucket_location_data_file).setMessageOutput("focad_bucket_location_message")
    FOCAD_agent.newRTCFunctionFile("focad_spatial_location_data", focad_spatial_location_data_file).setMessageOutput("focad_spatial_location_message")
    FOCAD_agent.newRTCFunctionFile("focad_anchor_update", focad_anchor_update_file).setMessageInput("cell_bucket_location_message")
    fpcu = FOCAD_agent.newRTCFunctionFile("focad_post_cycle_update", focad_post_cycle_update_file)
    fpcu.setMessageInput("cell_bucket_location_message")
    fpcu.setAllowAgentDeath(True)
    faf = FOCAD_agent.newRTCFunctionFile("focad_fnode_interaction", focad_fnode_interaction_file)
    faf.setMessageInput("fnode_spatial_location_message")
    faf.setAllowAgentDeath(True) # WARNING: if this flag is not set, the function will not be able to actually kill the agent (eventhough the function returns flamegpu::DEAD), which will cause errors in the logic of the model.
    FOCAD_agent.newRTCFunctionFile("focad_move", focad_move_file).setMessageInput("fnode_bucket_location_message_postmove")        
    # Now that the agent exists, set it as the output agent of the cell_focad_update function, which allows focal adhesion agents to be created and destroyed by that function based on the state of the cell and the local environment.
    cell_focad_update_fn.setAgentOutput(FOCAD_agent)


"""
  LUMEN agent
"""
if INCLUDE_CELLS and (ORGANOID_ASSAY or MONOLAYER_ASSAY) and INCLUDE_LUMEN:
    LUMEN_agent = model.newAgent("LUMEN")
    LUMEN_agent.newVariableInt("id", 0)       # unique LUMEN agent id
    LUMEN_agent.newVariableFloat("x", 0.0)   # LUMEN position [um]
    LUMEN_agent.newVariableFloat("y", 0.0)
    LUMEN_agent.newVariableFloat("z", 0.0)
    LUMEN_agent.newVariableFloat("vx", 0.0)  # LUMEN velocity [um/s]
    LUMEN_agent.newVariableFloat("vy", 0.0)
    LUMEN_agent.newVariableFloat("vz", 0.0)
    LUMEN_agent.newVariableFloat("radius", LUMEN_RADIUS)  # droplet radius [um]
    LUMEN_agent.newVariableFloat("ll_dvx", 0.0)  # [um/s] velocity contribution from lumen_lumen_interaction
    LUMEN_agent.newVariableFloat("ll_dvy", 0.0)
    LUMEN_agent.newVariableFloat("ll_dvz", 0.0)
    LUMEN_agent.newVariableFloat("lc_dvx", 0.0)  # [um/s] velocity contribution from lumen_cell_interaction
    LUMEN_agent.newVariableFloat("lc_dvy", 0.0)
    LUMEN_agent.newVariableFloat("lc_dvz", 0.0)

    LUMEN_agent.newRTCFunctionFile("lumen_spatial_location_data", lumen_spatial_location_data_file).setMessageOutput("lumen_spatial_location_message")
    LUMEN_agent.newRTCFunctionFile("lumen_lumen_interaction", lumen_lumen_interaction_file).setMessageInput("lumen_spatial_location_message")
    LUMEN_agent.newRTCFunctionFile("lumen_cell_interaction", lumen_cell_interaction_file).setMessageInput("cell_spatial_location_message")
    LUMEN_agent.newRTCFunctionFile("lumen_move", lumen_move_file)

    # Wire the cell_lumen_secretion output agent now that LUMEN_agent exists
    cls_fn.setAgentOutput(LUMEN_agent)

"""
  VASC agent
"""
if INCLUDE_VASCULARIZATION:
    VASC_agent = model.newAgent("VASC")
    VASC_agent.newVariableInt("id")                               # unique VASC node id
    VASC_agent.newVariableFloat("x", 0.0)                        # VASC position [um]
    VASC_agent.newVariableFloat("y", 0.0)
    VASC_agent.newVariableFloat("z", 0.0)
    VASC_agent.newVariableFloat("vx", 0.0)                       # VASC velocity [um/s] (used only when MOVING_BOUNDARIES)
    VASC_agent.newVariableFloat("vy", 0.0)
    VASC_agent.newVariableFloat("vz", 0.0)
    VASC_agent.newVariableArrayInt("parent_ids", MAX_VASC_CONNECTIVITY)  # global ids of parents (-1 = empty slot; source nodes have all entries == -1)
    VASC_agent.newVariableArrayInt("children_ids", MAX_VASC_CONNECTIVITY)  # global ids of children (-1 = empty slot)
    VASC_agent.newVariableArrayFloat("C_sp", N_SPECIES)          # per-species concentration
    VASC_agent.newVariableInt("dead", 0)                         # 0 = alive, 1 = dead

    VASC_agent.newRTCFunctionFile("vasc_bucket_location_data", vasc_bucket_location_data_file).setMessageOutput("vasc_bucket_location_message")
    VASC_agent.newRTCFunctionFile("vasc_Csp_update", vasc_Csp_update_file).setMessageInput("vasc_bucket_location_message")
    VASC_agent.newRTCFunctionFile("vasc_spatial_location_data", vasc_spatial_location_data_file).setMessageOutput("vasc_spatial_location_message")
    if MOVING_BOUNDARIES:
        VASC_agent.newRTCFunctionFile("vasc_move", vasc_move_file).setMessageInput("ecm_grid_location_message")
    if INCLUDE_CELLS and INCLUDE_VASCULAR_CELL_RECRUITMENT:
        vasc_ecm_cell_spawn_fn = VASC_agent.newRTCFunctionFile("vasc_ecm_cell_spawn", vasc_ecm_cell_spawn_file)
        vasc_ecm_cell_spawn_fn.setAgentOutput(CELL_agent)
        vasc_ecm_cell_spawn_fn.setAllowAgentDeath(False)

    # Add the ECM→VASC concentration update to the ECM agent
    ECM_agent.newRTCFunctionFile("ecm_vasc_Csp_update", ecm_vasc_Csp_update_file).setMessageInput("vasc_spatial_location_message")

# Agent population initialization 
# ----------------------------------------------------------------------    
# IMPORTANT NOTE: agents must be initialized in the following order to make sure that their ids are consistent with the assumptions made in the RTC functions and bucket message bounds and the vtk files writing:
# 1) Boundary corners (idx 1 to 8)
# 2) Fibre nodes (idx 9 to 8+N_NODES) if INCLUDE_FIBRE_NETWORK is True
# 3) Cell agents (idx 8+N_NODES+1 to 8+N_NODES+N_CELLS)
# 4) Focal adhesions (idx 8+N_NODES+N_CELLS+1 to 8+N_NODES+N_CELLS+(INIT_N_FOCAD_PER_CELL*N_CELLS)) if INCLUDE_FOCAL_ADHESIONS is True.
# 5) ECM agents (idx starting from 8+N_NODES+N_CELLS+(INIT_N_FOCAD_PER_CELL*N_CELLS)+1)
class initAgentPopulations(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        global INCLUDE_CELLS, N_CELLS, INIT_CELL_CONCENTRATION_VALS, INIT_CELL_REACTION_RATES
        global INIT_CELL_CONC_MASS_VALS, INIT_CELL_CONSUMPTION_RATES, INIT_CELL_PRODUCTION_RATES, DE_NOVO_PRODUCTION
        global INCLUDE_FOCAL_ADHESIONS, N_ANCHOR_POINTS, INIT_N_FOCAD_PER_CELL, CELL_RADIUS, CELL_NUCLEUS_RADIUS
        global FOCAD_REST_LENGTH_0, FOCAD_K_FA, FOCAD_F_MAX, FOCAD_K_ON, FOCAD_K_OFF_0, FOCAD_F_C, FOCAD_K_REINF
        global INCLUDE_DIFFUSION, N_SPECIES, DIFFUSION_COEFF_MULTI
        global INIT_ECM_CONCENTRATION_VALS, INIT_ECM_SAT_CONCENTRATION_VALS
        global INCLUDE_FIBRE_NETWORK, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, MAX_CONNECTIVITY
        global INCLUDE_VASCULARIZATION, VASC_NODES, N_VASC_NODES, INIT_VASCULARIZATION_CONCENTRATION_VALS, MAX_VASC_CONNECTIVITY
        # BOUNDARY CORNERS
        current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
        coord_boundary = FLAMEGPU.environment.getPropertyArrayFloat("COORDS_BOUNDARIES")
        coord_boundary_x_pos = coord_boundary[0]
        coord_boundary_x_neg = coord_boundary[1]
        coord_boundary_y_pos = coord_boundary[2]
        coord_boundary_y_neg = coord_boundary[3]
        coord_boundary_z_pos = coord_boundary[4]
        coord_boundary_z_neg = coord_boundary[5]
        # Monolayer z-plane: use explicit MONOLAYER_Z override if set, else default to z-neg boundary.
        _monolayer_z = MONOLAYER_Z if (MONOLAYER_Z is not None) else coord_boundary_z_neg
        print("--- Initializing CORNERS (8)")
        print("  |-> current_id:", current_id)

        for i in range(1, 9):
            instance = FLAMEGPU.agent("BCORNER").newAgent()
            instance.setVariableInt("id", current_id + i)
            if i == 1:
                # +x,+y,+z
                instance.setVariableFloat("x", coord_boundary_x_pos)
                instance.setVariableFloat("y", coord_boundary_y_pos)
                instance.setVariableFloat("z", coord_boundary_z_pos)
            elif i == 2:
                # -x,+y,+z
                instance.setVariableFloat("x", coord_boundary_x_neg)
                instance.setVariableFloat("y", coord_boundary_y_pos)
                instance.setVariableFloat("z", coord_boundary_z_pos)
            elif i == 3:
                # -x,-y,+z
                instance.setVariableFloat("x", coord_boundary_x_neg)
                instance.setVariableFloat("y", coord_boundary_y_neg)
                instance.setVariableFloat("z", coord_boundary_z_pos)
            elif i == 4:
                # +x,-y,+z
                instance.setVariableFloat("x", coord_boundary_x_pos)
                instance.setVariableFloat("y", coord_boundary_y_neg)
                instance.setVariableFloat("z", coord_boundary_z_pos)
            elif i == 5:
                # +x,+y,-z
                instance.setVariableFloat("x", coord_boundary_x_pos)
                instance.setVariableFloat("y", coord_boundary_y_pos)
                instance.setVariableFloat("z", coord_boundary_z_neg)
            elif i == 6:
                # -x,+y,-z
                instance.setVariableFloat("x", coord_boundary_x_neg)
                instance.setVariableFloat("y", coord_boundary_y_pos)
                instance.setVariableFloat("z", coord_boundary_z_neg)
            elif i == 7:
                # -x,-y,-z
                instance.setVariableFloat("x", coord_boundary_x_neg)
                instance.setVariableFloat("y", coord_boundary_y_neg)
                instance.setVariableFloat("z", coord_boundary_z_neg)
            elif i == 8:
                # +x,-y,-z
                instance.setVariableFloat("x", coord_boundary_x_pos)
                instance.setVariableFloat("y", coord_boundary_y_neg)
                instance.setVariableFloat("z", coord_boundary_z_neg)
            else:
                sys.exit("Bad initialization of boundary corners!")

        FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", 8)

        # FIBRE NODES
        if INCLUDE_FIBRE_NETWORK:
            k_elast = FLAMEGPU.environment.getPropertyFloat("FIBRE_SEGMENT_K_ELAST")
            d_dumping = FLAMEGPU.environment.getPropertyFloat("FIBRE_SEGMENT_D_DUMPING")
            current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
            current_id += 1
            print(f"--- Initializing FIBRE NODES ({N_NODES})")
            print("  |-> current_id:", current_id)   
            count = -1
            offset = current_id
            for fn in range(N_NODES):
                x = NODE_COORDS[fn, 0]
                y = NODE_COORDS[fn, 1]
                z = NODE_COORDS[fn, 2]
                linked_nodes = np.array(INITIAL_NETWORK_CONNECTIVITY.get(fn, []))   
                # Add the offset to all values above -1
                linked_nodes = np.where(linked_nodes > -1, linked_nodes + offset, linked_nodes) 

                count += 1
                instance = FLAMEGPU.agent("FNODE").newAgent()
                instance.setVariableInt("id", current_id + count)
                instance.setVariableFloat("x", x)
                instance.setVariableFloat("y", y)
                instance.setVariableFloat("z", z)            
                instance.setVariableFloat("vy", 0.0)
                instance.setVariableFloat("vz", 0.0)
                instance.setVariableFloat("vx", 0.0)
                instance.setVariableFloat("fx", 0.0)
                instance.setVariableFloat("fy", 0.0)
                instance.setVariableFloat("fz", 0.0)
                instance.setVariableFloat("k_elast", k_elast)
                instance.setVariableFloat("d_dumping", d_dumping)
                instance.setVariableArrayFloat("equilibrium_distance", [FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE] * MAX_CONNECTIVITY) 
                instance.setVariableFloat("boundary_fx", 0.0)
                instance.setVariableFloat("boundary_fy", 0.0)
                instance.setVariableFloat("boundary_fz", 0.0)
                instance.setVariableFloat("f_bx_pos", 0.0)
                instance.setVariableFloat("f_bx_neg", 0.0)
                instance.setVariableFloat("f_by_pos", 0.0)
                instance.setVariableFloat("f_by_neg", 0.0)
                instance.setVariableFloat("f_bz_pos", 0.0)
                instance.setVariableFloat("f_bz_neg", 0.0)
                instance.setVariableFloat("f_bx_pos_y", 0.0)
                instance.setVariableFloat("f_bx_pos_z", 0.0)
                instance.setVariableFloat("f_bx_neg_y", 0.0)
                instance.setVariableFloat("f_bx_neg_z", 0.0)
                instance.setVariableFloat("f_by_pos_x", 0.0)
                instance.setVariableFloat("f_by_pos_z", 0.0)
                instance.setVariableFloat("f_by_neg_x", 0.0)
                instance.setVariableFloat("f_by_neg_z", 0.0)
                instance.setVariableFloat("f_bz_pos_x", 0.0)
                instance.setVariableFloat("f_bz_pos_y", 0.0)
                instance.setVariableFloat("f_bz_neg_x", 0.0)
                instance.setVariableFloat("f_bz_neg_y", 0.0)
                instance.setVariableFloat("f_extension", 0.0)
                instance.setVariableFloat("f_compression", 0.0)
                instance.setVariableFloat("elastic_energy", 0.0)
                instance.setVariableUInt8("connectivity_count", int(np.sum(linked_nodes > -1)))
                instance.setVariableFloat("degradation", 0.0)
                instance.setVariableFloat("reinforcement", 0.0)
                instance.setVariableInt("secreted", 0)
                instance.setVariableInt("marked_for_removal", 0)
                instance.setVariableInt("closest_fnode_id", -1)
                instance.setVariableInt("second_closest_fnode_id", -1)
                instance.setVariableUInt8("clamped_bx_pos", 0)
                instance.setVariableUInt8("clamped_bx_neg", 0)
                instance.setVariableUInt8("clamped_by_pos", 0)
                instance.setVariableUInt8("clamped_by_neg", 0)
                instance.setVariableUInt8("clamped_bz_pos", 0)
                instance.setVariableUInt8("clamped_bz_neg", 0)
                instance.setVariableInt("focad_id", -1) # id of the attached focal adhesion if attached (-1 if none attached)
                instance.setVariableInt("focad_attached", 0) # 1 if a focal adhesion is attached to this fibre node, else 0
                instance.setVariableArrayFloat("linked_nodes", linked_nodes.tolist())            


            FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id + count)
            max_global_fnode_id_macro = FLAMEGPU.environment.getMacroPropertyInt("MACRO_MAX_GLOBAL_FNODE_ID")
            max_global_fnode_id_macro[0] = current_id + count

            
        # CELLS
        if INCLUDE_CELLS:
            current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
            current_id += 1
            print(f"--- Initializing CELLS ({N_CELLS})")
            print("  |-> current_id:", current_id)
            count = -1
            if N_CELLS == 1: # DEBUGGING. FIX CELL POSITION TO 0,0,0
                cell_pos = np.array([[0.0, 0.0, 0.0]], dtype=float) # for testing with 1 cell. 
                cell_orientations = np.array([[1.0, 0.0, 0.0]], dtype=float) 
                cell_types = [0]
                print(f"  |-> Single cell initialized at origin with orientation along +x axis")
            elif ORGANOID_ASSAY:
                cell_pos = getRandomCoordsAroundPoint(N_CELLS, 0.0, 0.0, 0.0, ORGANOID_INIT_RADIUS)
                cell_orientations = getRadialOrientations(
                    cell_pos, center=(0.0, 0.0, 0.0), noise_sigma=ORGANOID_ORIENTATION_NOISE
                )
                cell_types = getCellTypeList(N_CELLS, N_CELL_TYPES, [1, 1, 1], shuffle=False)
                print(f"  |-> Organoid assay: cells clustered in sphere of radius {ORGANOID_INIT_RADIUS} um at origin")
                print(f"  |-> Cell orientations: radially outward (noise_sigma={ORGANOID_ORIENTATION_NOISE:.3f} rad)")
            elif MONOLAYER_ASSAY:
                cell_pos = getCoordsOnPlane("z", _monolayer_z, N_CELLS, coord_boundary,
                                            min_dist=CELL_RADIUS[0], mode="random",
                                            cluster_radius=MONOLAYER_CLUSTER_RADIUS)
                cell_orientations = getRandomOrientationOnPlane("z",N_CELLS)
                cell_types = getCellTypeList(N_CELLS, N_CELL_TYPES, MONOLAYER_CELL_TYPE_RATIOS, shuffle=True)
                if MONOLAYER_CLUSTER_RADIUS is not None:
                    print(f"  |-> Monolayer assay: cells seeded in a cluster of radius {MONOLAYER_CLUSTER_RADIUS} um at z={_monolayer_z} um")
                else:
                    print(f"  |-> Monolayer assay: cells randomly distributed in a monolayer at z={_monolayer_z} um")
                print(f"  |-> Cell orientations: random")
            else:
                cached_cell_init = loadCachedCellInitialization(N_CELLS, coord_boundary, CELL_INIT_CACHE_DIR, atol=EPSILON)
                if cached_cell_init is not None:
                    cell_pos, cell_orientations, cache_path, _cache_data = cached_cell_init
                    print(f"  |-> Loaded cached cell positions/orientations from {cache_path}")
                else:
                    init_gen_start = time.perf_counter()
                    cell_pos, cell_orientations = generateCellInitializationData(N_CELLS, coord_boundary)
                    total_gen_time = time.perf_counter() - init_gen_start
                    print(
                        f"  |-> Generated cell positions/orientations on the fly "
                        f"(total: {total_gen_time:.3f}s)"
                    )
                cell_types = getCellTypeList(N_CELLS, N_CELL_TYPES, [1, 1, 1], shuffle=False)
            
            cell_id_list = []
            cell_progress_interval = max(1, N_CELLS // 100)
            for i in range(N_CELLS):
                count += 1
                cell_type_i = cell_types[i]  # assign cell types based on the generated list
                cell_id_list.append(current_id + count) # store the cell ids in a list to be used for focal adhesion initialization if INCLUDE_FOCAL_ADHESIONS is True
                instance = FLAMEGPU.agent("CELL").newAgent()
                instance.setVariableInt("id", current_id + count)
                instance.setVariableFloat("x", cell_pos[i, 0])
                instance.setVariableFloat("y", cell_pos[i, 1])
                instance.setVariableFloat("z", cell_pos[i, 2])
                instance.setVariableFloat("vx", 0.0)
                instance.setVariableFloat("vy", 0.0)
                instance.setVariableFloat("vz", 0.0)
                instance.setVariableFloat("trajectory_length", 0.0)
                instance.setVariableFloat("trajectory_time", 0.0)
                instance.setVariableFloat("birth_x", cell_pos[i, 0])
                instance.setVariableFloat("birth_y", cell_pos[i, 1])
                instance.setVariableFloat("birth_z", cell_pos[i, 2])
                instance.setVariableFloat("orx", cell_orientations[i, 0])
                instance.setVariableFloat("ory", cell_orientations[i, 1])
                instance.setVariableFloat("orz", cell_orientations[i, 2])
                instance.setVariableFloat("alignment", 0.0)
                instance.setVariableFloat("k_elast", CELL_K_ELAST[cell_type_i])
                instance.setVariableFloat("d_dumping", CELL_D_DUMPING[cell_type_i])
                _cell_vol_i = (4.0/3.0) * 3.1415926 * CELL_RADIUS[cell_type_i]**3
                _conc_mult_i = CELL_INIT_CONCENTRATION_MULTIPLIER[cell_type_i]
                _C_sp_i = [c * _conc_mult_i for c in INIT_CELL_CONCENTRATION_VALS]
                _M_sp_i = [c * _cell_vol_i for c in _C_sp_i]
                instance.setVariableArrayFloat("C_sp", _C_sp_i)
                instance.setVariableArrayFloat("M_sp", _M_sp_i)                
                instance.setVariableArrayFloat("k_consumption", [r * CELL_CONSUMPTION_MULTIPLIER[cell_type_i] for r in INIT_CELL_CONSUMPTION_RATES])
                instance.setVariableArrayFloat("k_production", [r * CELL_PRODUCTION_MULTIPLIER[cell_type_i] for r in INIT_CELL_PRODUCTION_RATES])
                instance.setVariableArrayFloat("k_reaction", [r * CELL_REACTION_MULTIPLIER[cell_type_i] for r in INIT_CELL_REACTION_RATES])
                instance.setVariableFloat("radius", CELL_RADIUS[cell_type_i])
                instance.setVariableFloat("nucleus_radius", CELL_NUCLEUS_RADIUS[cell_type_i])
                instance.setVariableFloat("cc_dvx", 0.0)
                instance.setVariableFloat("cc_dvy", 0.0)
                instance.setVariableFloat("cc_dvz", 0.0)
                instance.setVariableFloat("cf_dvx", 0.0)
                instance.setVariableFloat("cf_dvy", 0.0)
                instance.setVariableFloat("cf_dvz", 0.0)
                instance.setVariableFloat("speed_ref", CELL_SPEED_REF[cell_type_i])
                instance.setVariableInt("cell_type", cell_type_i)
                cycle_phase = random.randint(1, 4) # [1:G1] [2:S] [3:G2] [4:M]
                instance.setVariableInt("cycle_phase", cycle_phase)
                cycle_clock = 0.0
                if cycle_phase == 1:
                    cycle_clock = CYCLE_PHASE_G1_START[cell_type_i] \
                    + np.random.uniform(0.0, 1.0) * CYCLE_PHASE_G1_DURATION[cell_type_i]                
                elif cycle_phase == 2:
                    cycle_clock = CYCLE_PHASE_S_START[cell_type_i] \
                    + np.random.uniform(0.0, 1.0) * CYCLE_PHASE_S_DURATION[cell_type_i]                    
                elif cycle_phase == 3:
                    cycle_clock = CYCLE_PHASE_G2_START[cell_type_i] \
                    + np.random.uniform(0.0, 1.0) * CYCLE_PHASE_G2_DURATION[cell_type_i]                    
                elif cycle_phase == 4:
                    cycle_clock = CYCLE_PHASE_M_START[cell_type_i] \
                    + np.random.uniform(0.0, 1.0) * CYCLE_PHASE_M_DURATION[cell_type_i]                    
                instance.setVariableFloat("clock", cycle_clock)
                instance.setVariableInt("completed_cycles",0)
                instance.setVariableInt("max_global_cell_id", current_id + N_CELLS - 1)
                instance.setVariableFloat("damage", 0.0)
                instance.setVariableInt("dead", 0)
                instance.setVariableInt("dead_by", -1)
                instance.setVariableInt("mother_id", -1)
                instance.setVariableInt("daughter_id", -1)
                instance.setVariableInt("just_divided", 0)
                instance.setVariableInt("marked_for_removal", 0)
                instance.setVariableFloat("fnode_birth_cooldown", 0.0)
                instance.setVariableFloat("focad_birth_cooldown", 0.0)
                instance.setVariableArrayFloat("chemokinesis_promotive_adapt_state", [r * CELL_INIT_CONCENTRATION_MULTIPLIER[cell_type_i] for r in INIT_CELL_CONCENTRATION_VALS])
                instance.setVariableArrayFloat("chemokinesis_inhibitory_adapt_state", [r * CELL_INIT_CONCENTRATION_MULTIPLIER[cell_type_i] for r in INIT_CELL_CONCENTRATION_VALS])
                if INCLUDE_RG_VARIABLES:
                    _ap_angle = np.random.uniform(0.0, 2.0 * np.pi)  # random in-plane apical direction (no established z-polarity)
                    instance.setVariableFloat("apx", float(np.cos(_ap_angle)))
                    instance.setVariableFloat("apy", float(np.sin(_ap_angle)))
                    instance.setVariableFloat("apz", 0.0)
                    instance.setVariableFloat("rg_commit_level",         0.0)  # start at zero; sp2 gradient determines spatial nucleation site
                    instance.setVariableFloat("epithelialization_level", 0.0)
                    instance.setVariableFloat("rosette_maturity",        0.0)
                    instance.setVariableFloat("rg_neighbour_density",    0.0)
                    instance.setVariableFloat("morphogen_local",           0.0)
                    instance.setVariableInt("rg_committed", 0)
                    instance.setVariableFloat("substrate_anchor_x", float(cell_pos[i, 0]))
                    instance.setVariableFloat("substrate_anchor_y", float(cell_pos[i, 1]))

                anchor_pos = getRandomCoordsAroundPoint(N_ANCHOR_POINTS, cell_pos[i, 0], cell_pos[i, 1], cell_pos[i, 2], CELL_NUCLEUS_RADIUS[cell_type_i], on_surface=True)
                instance.setVariableArrayFloat("x_i", anchor_pos[:, 0].tolist())
                instance.setVariableArrayFloat("y_i", anchor_pos[:, 1].tolist())
                instance.setVariableArrayFloat("z_i", anchor_pos[:, 2].tolist())
                instance.setVariableFloat("eps_xx", 0.0)
                instance.setVariableFloat("eps_yy", 0.0)
                instance.setVariableFloat("eps_zz", 0.0)
                instance.setVariableFloat("eps_xy", 0.0)
                instance.setVariableFloat("eps_xz", 0.0)
                instance.setVariableFloat("eps_yz", 0.0)
                instance.setVariableFloat("sig_xx", 0.0)
                instance.setVariableFloat("sig_yy", 0.0)
                instance.setVariableFloat("sig_zz", 0.0)
                instance.setVariableFloat("sig_xy", 0.0)
                instance.setVariableFloat("sig_xz", 0.0)   
                instance.setVariableFloat("sig_yz", 0.0)  
                instance.setVariableFloat("sig_eig_1", 0.0)
                instance.setVariableFloat("sig_eig_2", 0.0)
                instance.setVariableFloat("sig_eig_3", 0.0)
                instance.setVariableFloat("sig_eigvec1_x", 0.0)
                instance.setVariableFloat("sig_eigvec1_y", 0.0)
                instance.setVariableFloat("sig_eigvec1_z", 0.0)
                instance.setVariableFloat("sig_eigvec2_x", 0.0)
                instance.setVariableFloat("sig_eigvec2_y", 0.0)
                instance.setVariableFloat("sig_eigvec2_z", 0.0)
                instance.setVariableFloat("sig_eigvec3_x", 0.0)
                instance.setVariableFloat("sig_eigvec3_y", 0.0)
                instance.setVariableFloat("sig_eigvec3_z", 0.0)
                instance.setVariableFloat("eps_eig_1", 0.0)
                instance.setVariableFloat("eps_eig_2", 0.0)
                instance.setVariableFloat("eps_eig_3", 0.0)
                instance.setVariableFloat("eps_eigvec1_x", 0.0)
                instance.setVariableFloat("eps_eigvec1_y", 0.0)
                instance.setVariableFloat("eps_eigvec1_z", 0.0)
                instance.setVariableFloat("eps_eigvec2_x", 0.0)
                instance.setVariableFloat("eps_eigvec2_y", 0.0)
                instance.setVariableFloat("eps_eigvec2_z", 0.0)
                instance.setVariableFloat("eps_eigvec3_x", 0.0)
                instance.setVariableFloat("eps_eigvec3_y", 0.0)
                instance.setVariableFloat("eps_eigvec3_z", 0.0)
                u_ref = compute_u_ref_from_anchor_pos(anchor_pos, cell_pos[i, :])
                instance.setVariableArrayFloat("u_ref_x_i", u_ref[:, 0].tolist())
                instance.setVariableArrayFloat("u_ref_y_i", u_ref[:, 1].tolist())
                instance.setVariableArrayFloat("u_ref_z_i", u_ref[:, 2].tolist())
                if N_CELLS >= 100000 and ((i + 1) % cell_progress_interval == 0 or (i + 1) == N_CELLS):
                    print(f"  |-> Cells initialized: {i + 1}/{N_CELLS}")


            FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id + count)
            max_global_cell_id_macro = FLAMEGPU.environment.getMacroPropertyInt("MACRO_MAX_GLOBAL_CELL_ID")
            max_global_cell_id_macro[0] = current_id + count
            
        if INCLUDE_FOCAL_ADHESIONS:
            current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
            current_id += 1
            print(f"--- Initializing FOCAL ADHESIONS ({N_CELLS * INIT_N_FOCAD_PER_CELL})")
            print("  |-> current_id:", current_id)
            count = -1
            focad_progress_interval = max(1, N_CELLS // 100)
            for i in range(N_CELLS):
                cell_type_i = i % N_CELL_TYPES
                focad_pos = getRandomCoordsAroundPoint(INIT_N_FOCAD_PER_CELL, cell_pos[i, 0], cell_pos[i, 1], cell_pos[i, 2], CELL_RADIUS[cell_type_i], on_surface=True)
                for j in range(INIT_N_FOCAD_PER_CELL):
                    count += 1
                    instance = FLAMEGPU.agent("FOCAD").newAgent()
                    instance.setVariableInt("id", current_id + count)
                    instance.setVariableInt("fnode_id", -1) # initialized as not attached to any fibre node
                    instance.setVariableInt("cell_id", cell_id_list[i])
                    instance.setVariableInt("cell_type", cell_type_i)  # inherit cell type from parent cell
                    instance.setVariableFloat("x", focad_pos[j, 0])
                    instance.setVariableFloat("y", focad_pos[j, 1])
                    instance.setVariableFloat("z", focad_pos[j, 2])                    
                    instance.setVariableFloat("vx", 0.0)
                    instance.setVariableFloat("vy", 0.0)
                    instance.setVariableFloat("vz", 0.0)
                    instance.setVariableFloat("fx", 0.0)
                    instance.setVariableFloat("fy", 0.0)
                    instance.setVariableFloat("fz", 0.0)
                    instance.setVariableInt("anchor_id", -1) # initialized as not attached to any anchor point
                    focad_dir = focad_pos[j, :] - cell_pos[i, :]
                    anchor_pos = cell_pos[i, :] + (focad_dir / np.linalg.norm(focad_dir)) * CELL_NUCLEUS_RADIUS[cell_type_i]
                    instance.setVariableFloat("x_i", anchor_pos[0])
                    instance.setVariableFloat("y_i", anchor_pos[1])
                    instance.setVariableFloat("z_i", anchor_pos[2])
                    instance.setVariableFloat("x_c", cell_pos[i, 0])
                    instance.setVariableFloat("y_c", cell_pos[i, 1])
                    instance.setVariableFloat("z_c", cell_pos[i, 2])
                    instance.setVariableFloat("orx", cell_orientations[i, 0])
                    instance.setVariableFloat("ory", cell_orientations[i, 1])
                    instance.setVariableFloat("orz", cell_orientations[i, 2])
                    _focad_rl0_i = CELL_RADIUS[cell_type_i] - CELL_NUCLEUS_RADIUS[cell_type_i]
                    instance.setVariableFloat("rest_length_0", _focad_rl0_i)
                    instance.setVariableFloat("rest_length", _focad_rl0_i)  # initialized at rest length for this cell type
                    instance.setVariableFloat("k_fa", FOCAD_K_FA[cell_type_i])
                    instance.setVariableFloat("f_max", FOCAD_F_MAX[cell_type_i]) # WARNING: 0 means "no cap" 
                    instance.setVariableInt("attached", 0) # initialized as not attached
                    instance.setVariableUInt8("active", 1) # initialized as active (can form new attachments)
                    instance.setVariableFloat("v_c", FOCAD_V_C[cell_type_i])
                    instance.setVariableUInt8("fa_state", 1) # [1: nascent] [2: mature] [3: disassembling]
                    instance.setVariableFloat("age", 0.0)
                    instance.setVariableFloat("detached_age", 0.0)
                    instance.setVariableFloat("k_on", FOCAD_K_ON[cell_type_i])
                    instance.setVariableFloat("k_off_0", FOCAD_K_OFF_0[cell_type_i])
                    instance.setVariableFloat("f_c", FOCAD_F_C[cell_type_i])
                    instance.setVariableFloat("k_reinf", FOCAD_K_REINF[cell_type_i])
                    instance.setVariableFloat("f_mag", 0.0)
                    instance.setVariableInt("is_front", 0)
                    instance.setVariableInt("is_rear", 0)
                    instance.setVariableInt("attached_front", 0)
                    instance.setVariableInt("attached_rear", 0)
                    instance.setVariableFloat("frontness_front", 0.0)
                    instance.setVariableFloat("frontness_rear", 0.0)
                    instance.setVariableFloat("k_on_eff_front", 0.0)
                    instance.setVariableFloat("k_on_eff_rear", 0.0)
                    instance.setVariableFloat("k_off_0_eff_front", 0.0)
                    instance.setVariableFloat("k_off_0_eff_rear", 0.0)
                    instance.setVariableFloat("linc_prev_total_length", 0.0)
                if N_CELLS >= 100000 and ((i + 1) % focad_progress_interval == 0 or (i + 1) == N_CELLS):
                    print(f"  |-> Cells with focal adhesions initialized: {i + 1}/{N_CELLS}")
            
            FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id + count)

        # ECM
        k_elast = FLAMEGPU.environment.getPropertyFloat("ECM_K_ELAST")
        d_dumping = FLAMEGPU.environment.getPropertyFloat("ECM_D_DUMPING")
        current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
        current_id += 1        
        agents_per_dir = FLAMEGPU.environment.getPropertyArrayUInt("ECM_AGENTS_PER_DIR")
        print(f"--- Initializing ECM (agents per dir:{agents_per_dir})")
        print("  |-> current_id:", current_id)
        offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # +X,-X,+Y,-Y,+Z,-Z
        coords_x = np.linspace(coord_boundary[1] + offset[1], coord_boundary[0] - offset[0], agents_per_dir[0])
        coords_y = np.linspace(coord_boundary[3] + offset[3], coord_boundary[2] - offset[2], agents_per_dir[1])
        coords_z = np.linspace(coord_boundary[5] + offset[5], coord_boundary[4] - offset[4], agents_per_dir[2])

        count = -1 # this is the general counter for all agents created
        grid_lin_count = -1  # this is the linear counter for grid positions
        i = -1
        j = -1
        k = -1

        for x in coords_x:
            i += 1
            j = -1
            for y in coords_y:
                j += 1
                k = -1
                for z in coords_z:
                    k += 1
                    count += 1
                    grid_lin_count += 1
                    instance = FLAMEGPU.agent("ECM").newAgent()
                    instance.setVariableInt("id", current_id + count)
                    instance.setVariableInt("grid_lin_id", grid_lin_count)
                    instance.setVariableFloat("x", x)
                    instance.setVariableFloat("y", y)
                    instance.setVariableFloat("z", z)
                    instance.setVariableFloat("vx", 0.0)
                    instance.setVariableFloat("vy", 0.0)
                    instance.setVariableFloat("vz", 0.0)
                    instance.setVariableFloat("fx", 0.0)
                    instance.setVariableFloat("fy", 0.0)
                    instance.setVariableFloat("fz", 0.0)
                    instance.setVariableFloat("k_elast", k_elast)
                    instance.setVariableFloat("d_dumping", d_dumping)
                    instance.setVariableArrayFloat("D_sp", DIFFUSION_COEFF_MULTI)
                    instance.setVariableArrayFloat("C_sp", INIT_ECM_CONCENTRATION_VALS)
                    instance.setVariableArrayFloat("C_sp_sat", INIT_ECM_SAT_CONCENTRATION_VALS)
                    instance.setVariableUInt8("clamped_bx_pos", 0)
                    instance.setVariableUInt8("clamped_bx_neg", 0)
                    instance.setVariableUInt8("clamped_by_pos", 0)
                    instance.setVariableUInt8("clamped_by_neg", 0)
                    instance.setVariableUInt8("clamped_bz_pos", 0)
                    instance.setVariableUInt8("clamped_bz_neg", 0)
                    instance.setVariableUInt8("grid_i", i)
                    instance.setVariableUInt8("grid_j", j)
                    instance.setVariableUInt8("grid_k", k)

        FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id + count)
        
        if INCLUDE_VASCULARIZATION:
            current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
            current_id += 1
            print(f"--- Initializing VASC ({N_VASC_NODES})")
            print("  |-> current_id:", current_id)
            _init_conc = list(INIT_VASCULARIZATION_CONCENTRATION_VALS)
            count = -1
            for vn in VASC_NODES:
                count += 1
                instance = FLAMEGPU.agent("VASC").newAgent()
                instance.setVariableInt("id", current_id + count)
                instance.setVariableFloat("x", float(vn["x"]))
                instance.setVariableFloat("y", float(vn["y"]))
                instance.setVariableFloat("z", float(vn["z"]))
                instance.setVariableFloat("vx", 0.0)
                instance.setVariableFloat("vy", 0.0)
                instance.setVariableFloat("vz", 0.0)
                # Offset parent_ids from local 0-indexed to global ID space and pad to MAX_VASC_CONNECTIVITY
                _parent_ids_local = list(vn["parent_ids"])
                _parent_ids_global = [(current_id + p) if p >= 0 else p for p in _parent_ids_local]
                while len(_parent_ids_global) < MAX_VASC_CONNECTIVITY:
                    _parent_ids_global.append(-1)
                instance.setVariableArrayInt("parent_ids", _parent_ids_global[:MAX_VASC_CONNECTIVITY])
                # Offset children_ids and pad to MAX_VASC_CONNECTIVITY
                _children_raw = list(vn.get("children_ids", []))[:MAX_VASC_CONNECTIVITY]
                _children_global = [(current_id + c) if c >= 0 else -1 for c in _children_raw]
                while len(_children_global) < MAX_VASC_CONNECTIVITY:
                    _children_global.append(-1)
                instance.setVariableArrayInt("children_ids", _children_global)
                instance.setVariableArrayFloat("C_sp", _init_conc)
                instance.setVariableInt("dead", 0)
            FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id + count)

        if INCLUDE_CELLS and ORGANOID_ASSAY and INCLUDE_LUMEN:
            lumen_id_macro = FLAMEGPU.environment.getMacroPropertyInt("MACRO_MAX_GLOBAL_LUMEN_ID")
            # Lumen IDs must start after the last used ID (which may include VASC if active)
            lumen_id_macro[0] = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
        
        
        return


# Add function callback to INIT functions for population generation
initialAgentPopulation = initAgentPopulations()
model.addInitFunction(initialAgentPopulation)
# WARNING: MacroProperties have getters but no setters, meaning they are automatically updated here
def resetMacroProperties(self, FLAMEGPU):
    global BOUNDARY_CONC_INIT_MULTI, BOUNDARY_CONC_FIXED_MULTI
    bcim = FLAMEGPU.environment.getMacroPropertyFloat("BOUNDARY_CONC_INIT_MULTI")
    bcfm = FLAMEGPU.environment.getMacroPropertyFloat("BOUNDARY_CONC_FIXED_MULTI")
    for i in range(len(BOUNDARY_CONC_INIT_MULTI)):
        for j in range(len(BOUNDARY_CONC_INIT_MULTI[i])):
            bcim[i][j] = BOUNDARY_CONC_INIT_MULTI[i][j]
    for i in range(len(BOUNDARY_CONC_FIXED_MULTI)):
        for j in range(len(BOUNDARY_CONC_FIXED_MULTI[i])):
            bcfm[i][j] = BOUNDARY_CONC_FIXED_MULTI[i][j]
    print("Reseting MacroProperties")
    print(BOUNDARY_CONC_INIT_MULTI)
    print(BOUNDARY_CONC_FIXED_MULTI)
    return
# Initialize the MacroProperties
class initMacroProperties(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        global INIT_ECM_CONCENTRATION_VALS, ECM_POPULATION_SIZE, N_SPECIES
        resetMacroProperties(self, FLAMEGPU)
        c_sp_macro = FLAMEGPU.environment.getMacroPropertyFloat("C_SP_MACRO")
        for i in range(ECM_POPULATION_SIZE):
            for j in range(N_SPECIES):
                c_sp_macro[j][i] = INIT_ECM_CONCENTRATION_VALS[j]

        return

initialMacroProperties = initMacroProperties()
model.addInitFunction(initialMacroProperties)

# ++==================================================================++
# ++ Step functions                                                    |
# ++==================================================================++
"""
  STEP FUNCTIONS
"""
# pyflamegpu requires step functions to be a class which extends the StepFunction base class.
# This class must extend the handle function
class MoveBoundaries(pyflamegpu.HostFunction):
    """
     pyflamegpu requires step functions to be a class which extends the StepFunction base class.
     This class must extend the handle function
     """

    # Define Python class 'constructor'
    def __init__(self):
        super().__init__()
        self.apply_parallel_disp = list()
        for d in range(12):
            if abs(BOUNDARY_DISP_RATES_PARALLEL[d]) > 0.0:
                self.apply_parallel_disp.append(True)
            else:
                self.apply_parallel_disp.append(False)

    # Override C++ method: virtual void run(FLAMEGPU_HOST_API*)
    def run(self, FLAMEGPU):
        stepCounter = FLAMEGPU.getStepCounter() + 1
        global BOUNDARY_DISP_RATES, ALLOW_BOUNDARY_ELASTIC_MOVEMENT, BOUNDARY_STIFFNESS, BOUNDARY_DUMPING, BPOS_OVER_TIME
        global CLAMP_AGENT_TOUCHING_BOUNDARY, OSCILLATORY_SHEAR_ASSAY, OSCILLATORY_AMPLITUDE, OSCILLATORY_W, OSCILLATORY_STRAIN_OVER_TIME
        global DEBUG_PRINTING, PAUSE_EVERY_STEP, TIME_STEP

        boundaries_moved = False
        if PAUSE_EVERY_STEP:
            input()  # pause everystep
    
        coord_boundary = list(FLAMEGPU.environment.getPropertyArrayFloat("COORDS_BOUNDARIES"))
        if OSCILLATORY_SHEAR_ASSAY:
            if stepCounter % SAVE_EVERY_N_STEPS == 0 or stepCounter == 1:
                new_val = pd.DataFrame([OSOT(OSCILLATORY_AMPLITUDE * math.sin(OSCILLATORY_W * stepCounter))])
                # OSCILLATORY_STRAIN_OVER_TIME = OSCILLATORY_STRAIN_OVER_TIME.append(new_val, ignore_index=True) #TODO: FIX?
                OSCILLATORY_STRAIN_OVER_TIME = pd.concat([OSCILLATORY_STRAIN_OVER_TIME, new_val], ignore_index=True)
            for d in range(12):
                if self.apply_parallel_disp[d]:
                    BOUNDARY_DISP_RATES_PARALLEL[d] = OSCILLATORY_AMPLITUDE * math.cos(
                        OSCILLATORY_W * stepCounter) * OSCILLATORY_W / TIME_STEP  # cos(w*t)*t is used because the slope of the sin(w*t) function is needed

            FLAMEGPU.environment.setPropertyArrayFloat("DISP_RATES_BOUNDARIES_PARALLEL", BOUNDARY_DISP_RATES_PARALLEL)

        if any(catb < 1 for catb in CLAMP_AGENT_TOUCHING_BOUNDARY) or any(
                abem > 0 for abem in ALLOW_BOUNDARY_ELASTIC_MOVEMENT):
            boundaries_moved = True
            agent = FLAMEGPU.agent("ECM")
            minmax_positions = list()
            minmax_positions.append(agent.maxFloat("x"))
            minmax_positions.append(agent.minFloat("x"))
            minmax_positions.append(agent.maxFloat("y"))
            minmax_positions.append(agent.minFloat("y"))
            minmax_positions.append(agent.maxFloat("z"))
            minmax_positions.append(agent.minFloat("z"))
            boundary_equil_distances = list()
            boundary_equil_distances.append(ECM_BOUNDARY_EQUILIBRIUM_DISTANCE)
            boundary_equil_distances.append(-ECM_BOUNDARY_EQUILIBRIUM_DISTANCE)
            boundary_equil_distances.append(ECM_BOUNDARY_EQUILIBRIUM_DISTANCE)
            boundary_equil_distances.append(-ECM_BOUNDARY_EQUILIBRIUM_DISTANCE)
            boundary_equil_distances.append(ECM_BOUNDARY_EQUILIBRIUM_DISTANCE)
            boundary_equil_distances.append(-ECM_BOUNDARY_EQUILIBRIUM_DISTANCE)
            for i in range(6):
                if CLAMP_AGENT_TOUCHING_BOUNDARY[i] < 1:
                    if ALLOW_BOUNDARY_ELASTIC_MOVEMENT[i] > 0:
                        coord_boundary[i] = minmax_positions[i] + boundary_equil_distances[i]
                    else:
                        coord_boundary[i] = minmax_positions[i]

            bcs = [coord_boundary[0], coord_boundary[1], coord_boundary[2], coord_boundary[3], coord_boundary[4],
                   coord_boundary[5]]  # +X,-X,+Y,-Y,+Z,-Z
            FLAMEGPU.environment.setPropertyArrayFloat("COORDS_BOUNDARIES", bcs)

            if stepCounter % SAVE_EVERY_N_STEPS == 0 or stepCounter == 1:
                print("====== MOVING FREE BOUNDARIES  ======")
                print("New boundary positions [+X,-X,+Y,-Y,+Z,-Z]: ", coord_boundary)
                print("=====================================")

        if any(dr > 0.0 or dr < 0.0 for dr in BOUNDARY_DISP_RATES):
            boundaries_moved = True
            for i in range(6):
                coord_boundary[i] += (BOUNDARY_DISP_RATES[i] * TIME_STEP)

            bcs = [coord_boundary[0], coord_boundary[1], coord_boundary[2], coord_boundary[3], coord_boundary[4],
                   coord_boundary[5]]  # +X,-X,+Y,-Y,+Z,-Z
            FLAMEGPU.environment.setPropertyArrayFloat("COORDS_BOUNDARIES", bcs)
            if stepCounter % SAVE_EVERY_N_STEPS == 0 or stepCounter == 1:
                print("====== MOVING BOUNDARIES DUE TO CONDITIONS ======")
                print("New boundary positions [+X,-X,+Y,-Y,+Z,-Z]: ", coord_boundary)
                print("=================================================")

        # if any(abem > 0 for abem in ALLOW_BOUNDARY_ELASTIC_MOVEMENT):
        #   boundaries_moved = True
        #   print ("====== MOVING BOUNDARIES DUE TO FORCES ======")
        #   agent = FLAMEGPU.agent("ECM")
        #   sum_bx_pos = agent.sumFloat("f_bx_pos")
        #   sum_bx_neg = agent.sumFloat("f_bx_neg")
        #   sum_by_pos = agent.sumFloat("f_by_pos")
        #   sum_by_neg = agent.sumFloat("f_by_neg")
        #   sum_bz_pos = agent.sumFloat("f_bz_pos")
        #   sum_bz_neg = agent.sumFloat("f_bz_neg")
        #   print ("Total forces [+X,-X,+Y,-Y,+Z,-Z]: ", sum_bx_pos, sum_bx_neg, sum_by_pos, sum_by_neg, sum_bz_pos, sum_bz_neg)
        #   boundary_forces = [sum_bx_pos, sum_bx_neg, sum_by_pos, sum_by_neg, sum_bz_pos, sum_bz_neg]
        #   for i in range(6):
        #       if BOUNDARY_DISP_RATES[i] < EPSILON and BOUNDARY_DISP_RATES[i] > -EPSILON and ALLOW_BOUNDARY_ELASTIC_MOVEMENT[i]:
        #           #u = boundary_forces[i] / BOUNDARY_STIFFNESS[i]
        #           u = (boundary_forces[i] * TIME_STEP)/ (BOUNDARY_STIFFNESS[i] * TIME_STEP + BOUNDARY_DUMPING[i])
        #           print ("Displacement for boundary {} = {}".format(i,u))
        #           coord_boundary[i] += u

        #   bcs = [coord_boundary[0], coord_boundary[1], coord_boundary[2], coord_boundary[3], coord_boundary[4], coord_boundary[5]]  #+X,-X,+Y,-Y,+Z,-Z
        #   FLAMEGPU.environment.setPropertyArrayFloat("COORDS_BOUNDARIES", bcs)
        #   print ("New boundary positions [+X,-X,+Y,-Y,+Z,-Z]: ", coord_boundary)
        #   print ("=================================================")

        if boundaries_moved:
            if stepCounter % SAVE_EVERY_N_STEPS == 0 or stepCounter == 1:
                new_pos = pd.DataFrame([BPOS(coord_boundary[0], coord_boundary[1], coord_boundary[2],
                                             coord_boundary[3], coord_boundary[4], coord_boundary[5])])
                # BPOS_OVER_TIME = BPOS_OVER_TIME.append(new_pos, ignore_index=True)
                BPOS_OVER_TIME = pd.concat([BPOS_OVER_TIME, new_pos], ignore_index=True)

        # print ("End of step: ", stepCounter)

# VTK extra-field lists — populated by variant-gated flags; empty for base runs
CELL_VTK_EXTRA_SCALARS = []   # list of (vtk_name, agent_variable_name, dtype_str)
CELL_VTK_EXTRA_VECTORS = []   # list of (vtk_name, vx_var, vy_var, vz_var)
if INCLUDE_RG_VARIABLES:
    CELL_VTK_EXTRA_SCALARS = [
        ("rg_commit_level",         "rg_commit_level",         "float"),
        ("epithelialization_level", "epithelialization_level", "float"),
        ("rosette_maturity",        "rosette_maturity",        "float"),
        ("rg_neighbour_density",    "rg_neighbour_density",    "float"),
        ("morphogen_local",        "morphogen_local",        "float"),
        ("rg_committed",            "rg_committed",            "int"),
    ]
    CELL_VTK_EXTRA_VECTORS = [
        ("apical_vector", "apx", "apy", "apz"),
    ]
# TODO: add extra fields for future variants.

class SaveDataToFile(pyflamegpu.HostFunction):
    def __init__(self):
        global ECM_AGENTS_PER_DIR, INCLUDE_FIBRE_NETWORK, N_NODES, INCLUDE_NETWORK_REMODELING
        super().__init__()
        self.save_context = build_save_data_context(
            ecm_agents_per_dir=ECM_AGENTS_PER_DIR,
            include_fibre_network=INCLUDE_FIBRE_NETWORK,
            n_nodes=N_NODES,
        )
        
    def run(self, FLAMEGPU):
        global SAVE_DATA_TO_FILE, SAVE_EVERY_N_STEPS, N_SPECIES
        global RES_PATH
        global INCLUDE_FIBRE_NETWORK, HETEROGENEOUS_DIFFUSION, INITIAL_NETWORK_CONNECTIVITY, N_NODES, INCLUDE_CELLS, ECM_POPULATION_SIZE
        global INCLUDE_FOCAL_ADHESIONS, INCLUDE_CELL_CELL_INTERACTION, INCLUDE_CELL_FNODE_REPULSION
        global INCLUDE_VASCULARIZATION, N_VASC_NODES
        save_data_to_file_step(
            FLAMEGPU=FLAMEGPU,
            save_context=self.save_context,
            config={
                "SAVE_DATA_TO_FILE": SAVE_DATA_TO_FILE,
                "SAVE_EVERY_N_STEPS": SAVE_EVERY_N_STEPS,
                "N_SPECIES": N_SPECIES,
                "RES_PATH": RES_PATH,
                "INCLUDE_FIBRE_NETWORK": INCLUDE_FIBRE_NETWORK,
                "HETEROGENEOUS_DIFFUSION": HETEROGENEOUS_DIFFUSION,
                "INITIAL_NETWORK_CONNECTIVITY": INITIAL_NETWORK_CONNECTIVITY,
                "N_NODES": N_NODES,
                "INCLUDE_CELLS": INCLUDE_CELLS,
                "ECM_POPULATION_SIZE": ECM_POPULATION_SIZE,
                "INCLUDE_FOCAL_ADHESIONS": INCLUDE_FOCAL_ADHESIONS,
                "INCLUDE_CELL_CELL_INTERACTION": INCLUDE_CELL_CELL_INTERACTION,
                "INCLUDE_CELL_FNODE_REPULSION": INCLUDE_CELL_FNODE_REPULSION,
                "INCLUDE_NETWORK_REMODELING": INCLUDE_NETWORK_REMODELING,
                "INCLUDE_LUMEN": INCLUDE_LUMEN,
                "INCLUDE_VASCULARIZATION": INCLUDE_VASCULARIZATION,
                "N_VASC_NODES": N_VASC_NODES,
                "pyflamegpu": pyflamegpu,
                "CELL_VTK_EXTRA_SCALARS": CELL_VTK_EXTRA_SCALARS,
                "CELL_VTK_EXTRA_VECTORS": CELL_VTK_EXTRA_VECTORS,
            },
        )


class CollectCellMetrics(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        global INCLUDE_CELLS, STEPS, CELL_SPEED_METRICS, ORGANOID_METRICS_OVER_TIME
        global ORGANOID_ASSAY, SAVE_EVERY_N_STEPS, N_CELL_TYPES
        global INCLUDE_RG_VARIABLES, RG_METRICS, RG_ROSETTE_METRICS_OVER_TIME, CELL_RADIUS

        if not INCLUDE_CELLS:
            return

        step = FLAMEGPU.getStepCounter() + 1
        is_final = (step == STEPS)

        # --- Organoid metrics (every SAVE_EVERY_N_STEPS and at final step) ---
        if ORGANOID_ASSAY and (step % SAVE_EVERY_N_STEPS == 0 or step == 1 or is_final):
            cell_agent = FLAMEGPU.agent("CELL")
            positions = []
            alive_per_type = [0] * N_CELL_TYPES
            for ai in cell_agent.getPopulationData():
                if int(ai.getVariableInt("dead")) == 0:
                    positions.append([
                        float(ai.getVariableFloat("x")),
                        float(ai.getVariableFloat("y")),
                        float(ai.getVariableFloat("z")),
                    ])
                    ct = int(ai.getVariableInt("cell_type"))
                    if 0 <= ct < N_CELL_TYPES:
                        alive_per_type[ct] += 1
            pos_arr = np.array(positions) if len(positions) > 0 else np.empty((0, 3))
            n_alive = len(pos_arr)
            if n_alive > 0:
                centroid = pos_arr.mean(axis=0)
                displacements = pos_arr - centroid
                sq_dists = np.sum(displacements ** 2, axis=1)
                rg = float(np.sqrt(np.mean(sq_dists)))
                equivalent_r = rg * np.sqrt(5.0 / 3.0)
                if n_alive > 1:
                    from scipy.spatial.distance import pdist
                    max_span = float(pdist(pos_arr).max())
                else:
                    max_span = 0.0
                # Sphericity from eigenvalues of covariance matrix
                if n_alive >= 4:
                    eigvals = np.linalg.eigvalsh(np.cov(displacements.T))
                    eigvals = np.maximum(eigvals, 0.0)
                    sphericity = float(eigvals.min() / eigvals.max()) if eigvals.max() > 0 else 1.0
                else:
                    sphericity = 1.0
                # Mean nearest-neighbour distance
                if n_alive >= 2:
                    from scipy.spatial import KDTree
                    tree = KDTree(pos_arr)
                    dd, _ = tree.query(pos_arr, k=2)
                    mean_nn_dist = float(dd[:, 1].mean())
                else:
                    mean_nn_dist = 0.0
            else:
                centroid = np.array([0.0, 0.0, 0.0])
                rg = 0.0
                equivalent_r = 0.0
                max_span = 0.0
                sphericity = 1.0
                mean_nn_dist = 0.0

            row = pd.DataFrame([{
                "step": step,
                "time": step * FLAMEGPU.environment.getPropertyFloat("TIME_STEP"),
                "n_alive": n_alive,
                "n_alive_type": ",".join(str(c) for c in alive_per_type),
                "radius_of_gyration": rg,
                "equivalent_sphere_radius": equivalent_r,
                "max_span": max_span,
                "sphericity": sphericity,
                "mean_nn_distance": mean_nn_dist,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
            }])
            if len(ORGANOID_METRICS_OVER_TIME) == 0:
                ORGANOID_METRICS_OVER_TIME = row
            else:
                ORGANOID_METRICS_OVER_TIME = pd.concat(
                    [ORGANOID_METRICS_OVER_TIME, row], ignore_index=True
                )

        # --- RG rosette metrics over time (radial_glia variant only) ---
        if INCLUDE_RG_VARIABLES and (step == 1 or step % SAVE_EVERY_N_STEPS == 0 or is_final):
            from sklearn.cluster import DBSCAN as _DBSCAN
            cell_agent = FLAMEGPU.agent("CELL")
            rg_positions = []
            rg_maturities = []
            rg_apz_vals = []
            n_alive_total = 0
            n_alive_rg = 0
            for ai in cell_agent.getPopulationData():
                if int(ai.getVariableInt("dead")) != 0:
                    continue
                n_alive_total += 1
                ct = int(ai.getVariableInt("cell_type"))
                if ct == 2:  # RG
                    n_alive_rg += 1
                    rg_positions.append([
                        float(ai.getVariableFloat("x")),
                        float(ai.getVariableFloat("y")),
                        float(ai.getVariableFloat("z")),
                    ])
                    rg_maturities.append(float(ai.getVariableFloat("rosette_maturity")))
                    rg_apz_vals.append(abs(float(ai.getVariableFloat("apz"))))

            rg_fraction = n_alive_rg / n_alive_total if n_alive_total > 0 else 0.0
            mean_rosette_maturity_val = float(np.mean(rg_maturities)) if rg_maturities else 0.0
            mean_apz_val = float(np.mean(rg_apz_vals)) if rg_apz_vals else 0.0

            pos_arr = np.array(rg_positions) if rg_positions else np.zeros((0, 3))
            labels = np.full(n_alive_rg, -1, dtype=int)
            n_rg_clusters = 0
            mean_cluster_size_val = 0.0
            largest_cluster_size = 0
            if n_alive_rg >= 2:
                eps_cluster = 3.0 * float(CELL_RADIUS[2])  # 3 × RG cell radius ≈ 15 µm
                labels = _DBSCAN(eps=eps_cluster, min_samples=2).fit_predict(pos_arr)
                clustered = labels[labels >= 0]
                if len(clustered) > 0:
                    n_rg_clusters = int(np.unique(clustered).shape[0])
                    cluster_sizes = np.bincount(clustered)
                    mean_cluster_size_val = float(cluster_sizes.mean())
                    largest_cluster_size = int(cluster_sizes.max())
            elif n_alive_rg == 1:
                n_rg_clusters = 1
                mean_cluster_size_val = 1.0
                largest_cluster_size = 1

            # --- Compactness metrics (PCA eigenvalue ratio; 0=linear, 1=circular) ---
            # rg_assembly_compactness: shape of the entire RG assembly in XY
            rg_assembly_compactness_val = 0.0
            if n_alive_rg >= 3:
                cov_all = np.cov(pos_arr[:, :2].T)
                ev_all = np.linalg.eigvalsh(cov_all)  # ascending order
                if ev_all[-1] > 1e-12:
                    rg_assembly_compactness_val = float(ev_all[0] / ev_all[-1])

            # mean_cluster_compactness: per-cluster PCA compactness, weighted by size
            mean_cluster_compactness_val = 0.0
            if n_rg_clusters > 0 and n_alive_rg >= 3:
                weighted_sum = 0.0
                total_weight = 0
                for _lid in range(n_rg_clusters):
                    _mask = (labels == _lid)
                    _sz = int(_mask.sum())
                    if _sz < 3:
                        continue
                    _cxy = pos_arr[np.where(_mask)[0], :2]
                    _cov = np.cov(_cxy.T)
                    _ev = np.linalg.eigvalsh(_cov)
                    if _ev[-1] > 1e-12:
                        weighted_sum += (_ev[0] / _ev[-1]) * _sz
                        total_weight += _sz
                if total_weight > 0:
                    mean_cluster_compactness_val = weighted_sum / total_weight

            time_val = step * FLAMEGPU.environment.getPropertyFloat("TIME_STEP")
            rg_row = pd.DataFrame([{
                "step": step,
                "time": time_val,
                "n_alive_total": n_alive_total,
                "n_alive_rg": n_alive_rg,
                "rg_fraction": rg_fraction,
                "n_rg_clusters": n_rg_clusters,
                "mean_cluster_size": mean_cluster_size_val,
                "largest_cluster_size": largest_cluster_size,
                "mean_rosette_maturity": mean_rosette_maturity_val,
                "mean_apz": mean_apz_val,
                "rg_assembly_compactness": rg_assembly_compactness_val,
                "mean_cluster_compactness": mean_cluster_compactness_val,
            }])
            if len(RG_ROSETTE_METRICS_OVER_TIME) == 0:
                RG_ROSETTE_METRICS_OVER_TIME = rg_row
            else:
                RG_ROSETTE_METRICS_OVER_TIME = pd.concat(
                    [RG_ROSETTE_METRICS_OVER_TIME, rg_row], ignore_index=True
                )

        # --- Speed metrics (final step only) ---
        if not is_final:
            return

        rows = []
        cell_agent = FLAMEGPU.agent("CELL")
        cell_agent.sortInt("id", pyflamegpu.HostAgentAPI.Asc)
        for ai in cell_agent.getPopulationData():
            cell_x = float(ai.getVariableFloat("x"))
            cell_y = float(ai.getVariableFloat("y"))
            cell_z = float(ai.getVariableFloat("z"))
            birth_x = float(ai.getVariableFloat("birth_x"))
            birth_y = float(ai.getVariableFloat("birth_y"))
            birth_z = float(ai.getVariableFloat("birth_z"))
            tracked_time = float(ai.getVariableFloat("trajectory_time"))
            tracked_length = float(ai.getVariableFloat("trajectory_length"))
            displacement = math.sqrt(
                (cell_x - birth_x) ** 2
                + (cell_y - birth_y) ** 2
                + (cell_z - birth_z) ** 2
            )
            if tracked_time > 1e-12:
                vmean = tracked_length / tracked_time
                veff = displacement / tracked_time
            else:
                vmean = 0.0
                veff = 0.0
            rows.append({
                "id": int(ai.getVariableInt("id")),
                "cell_type": int(ai.getVariableInt("cell_type")),
                "dead": int(ai.getVariableInt("dead")),
                "mother_id": int(ai.getVariableInt("mother_id")),
                "trajectory_time": tracked_time,
                "trajectory_length": tracked_length,
                "effective_displacement": displacement,
                "vmean": vmean,
                "veff": veff,
            })
        CELL_SPEED_METRICS = pd.DataFrame(rows)

        # --- RG final snapshot (radial_glia variant only, final step only) ---
        if INCLUDE_RG_VARIABLES and is_final:
            rg_rows = []
            cell_agent = FLAMEGPU.agent("CELL")
            for ai in cell_agent.getPopulationData():
                rg_rows.append({
                    "id":                    int(ai.getVariableInt("id")),
                    "cell_type":             int(ai.getVariableInt("cell_type")),
                    "dead":                  int(ai.getVariableInt("dead")),
                    "mother_id":             int(ai.getVariableInt("mother_id")),
                    "rg_commit_level":       float(ai.getVariableFloat("rg_commit_level")),
                    "epithelialization_level": float(ai.getVariableFloat("epithelialization_level")),
                    "rosette_maturity":      float(ai.getVariableFloat("rosette_maturity")),
                    "rg_neighbour_density":  float(ai.getVariableFloat("rg_neighbour_density")),
                    "morphogen_local":       float(ai.getVariableFloat("morphogen_local")),
                    "rg_committed":          int(ai.getVariableInt("rg_committed")),
                    "apx":                   float(ai.getVariableFloat("apx")),
                    "apy":                   float(ai.getVariableFloat("apy")),
                    "apz":                   float(ai.getVariableFloat("apz")),
                })
            RG_METRICS = pd.DataFrame(rg_rows)


class CheckFNODEStability(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        global INCLUDE_FIBRE_NETWORK, ABORT_ON_UNSTABLE_FNODE_MOVE, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE

        if not INCLUDE_FIBRE_NETWORK or not ABORT_ON_UNSTABLE_FNODE_MOVE:
            return

        unstable_moves = FLAMEGPU.agent("FNODE").sumUInt8("unstable_move")
        if unstable_moves > 0:
            stepCounter = FLAMEGPU.getStepCounter() + 1
            raise RuntimeError(
                "Unstable FNODE motion detected at step "
                f"{stepCounter}: {unstable_moves} node(s) exceeded "
                f"FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE:{FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE} in a single step."
            )




class ReportFAMetrics(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        global INCLUDE_FOCAL_ADHESIONS, SAVE_EVERY_N_STEPS
        if not INCLUDE_FOCAL_ADHESIONS:
            return

        stepCounter = FLAMEGPU.getStepCounter() + 1
        if not (stepCounter % SAVE_EVERY_N_STEPS == 0 or stepCounter == 1):
            return

        focad_agent = FLAMEGPU.agent("FOCAD")
        n_focad = focad_agent.count()
        if n_focad <= 0:
            print(f"FA metrics (step {stepCounter:04d}) -> no FOCAD agents")
            return

        attached_count = focad_agent.sumInt("attached")
        total_force_mag = focad_agent.sumFloat("f_mag")
        attached_ratio = attached_count / float(n_focad)
        mean_force_mag = total_force_mag / float(n_focad)

        # print(
        #     f"FA metrics (step {stepCounter:04d}) -> attached={int(attached_count)}/{n_focad} "
        #     f"(ratio={attached_ratio:.3f}), mean|F|={mean_force_mag:.4f} nN"
        # )


                          

class UpdateBoundaryConcentrationMulti(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        global BOUNDARY_CONC_INIT_MULTI, BOUNDARY_CONC_FIXED_MULTI
        stepCounter = FLAMEGPU.getStepCounter() + 1
        if stepCounter == 2:  # after first step BOUNDARY_CONC_INIT_MULTI is removed (set to -1.0) and BOUNDARY_CONC_FIXED_MULTI prevails
            print("====== CONCENTRATION MULTI BOUNDARY CONDITIONS SET  ======")
            print("Initial concentration boundary conditions [+X,-X,+Y,-Y,+Z,-Z]: ", BOUNDARY_CONC_INIT_MULTI)
            print("Fixed concentration boundary conditions [+X,-X,+Y,-Y,+Z,-Z]: ", BOUNDARY_CONC_FIXED_MULTI)
            for i in range(len(BOUNDARY_CONC_INIT_MULTI)):
                for j in range(len(BOUNDARY_CONC_INIT_MULTI[i])):
                    BOUNDARY_CONC_INIT_MULTI[i][j] = -1.0
            resetMacroProperties(self, FLAMEGPU)
            
class UpdateAgentCount(pyflamegpu.HostFunction): # if cells proliferate, N_CELLS must be updated
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.environment.setPropertyUInt("N_CELLS", FLAMEGPU.agent("CELL").count())


class PrintDebugStats(pyflamegpu.HostFunction):
    """Print per-step summary of cell-type counts and RG differentiation state.

    Activated only when INCLUDE_RG_VARIABLES is True and DEBUG_PRINT_INTERVAL > 0.
    Runs every DEBUG_PRINT_INTERVAL steps (and at step 1) to give a live
    progress indicator without cluttering the output every step.
    """
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        global INCLUDE_RG_VARIABLES, DEBUG_PRINT_INTERVAL, N_CELL_TYPES

        if not INCLUDE_RG_VARIABLES or DEBUG_PRINT_INTERVAL <= 0:
            return

        step = FLAMEGPU.getStepCounter() + 1
        if step != 1 and step % DEBUG_PRINT_INTERVAL != 0:
            return

        time_h = step * FLAMEGPU.environment.getPropertyFloat("TIME_STEP") / 3600.0
        type_counts = [0] * N_CELL_TYPES
        commit_sum = 0.0
        commit_max = 0.0
        morph_sum  = 0.0
        morph_max  = 0.0
        n_alive = 0

        for ai in FLAMEGPU.agent("CELL").getPopulationData():
            if int(ai.getVariableInt("dead")) == 1:
                continue
            ct = int(ai.getVariableInt("cell_type"))
            if 0 <= ct < N_CELL_TYPES:
                type_counts[ct] += 1
            commit = float(ai.getVariableFloat("rg_commit_level"))
            morph  = float(ai.getVariableFloat("morphogen_local"))
            commit_sum += commit
            morph_sum  += morph
            if commit > commit_max:
                commit_max = commit
            if morph > morph_max:
                morph_max = morph
            n_alive += 1

        mean_commit = commit_sum / n_alive if n_alive > 0 else 0.0
        mean_morph  = morph_sum  / n_alive if n_alive > 0 else 0.0
        _names = ["iPSC", "NPC", "RG"]
        type_str = "  ".join(
            f"{(_names[i] if i < len(_names) else f'type{i}')}={type_counts[i]}"
            for i in range(N_CELL_TYPES)
        )
        print(
            f"[DBG t={time_h:6.2f}h step={step:5d}]  {type_str}"
            f"  | rg_commit mean={mean_commit:.4f} max={commit_max:.4f}"
            f"  | morphogen mean={mean_morph:.3e} max={morph_max:.3e}"
        )


if INCLUDE_DIFFUSION:
    ubcm = UpdateBoundaryConcentrationMulti()
    model.addStepFunction(ubcm)

if INCLUDE_CELLS:
    if INCLUDE_CELL_CYCLE:
        uac = UpdateAgentCount()
        model.addStepFunction(uac)

if MOVING_BOUNDARIES:
    mb = MoveBoundaries()
    model.addStepFunction(mb)

if INCLUDE_FIBRE_NETWORK:
    cfs = CheckFNODEStability()
    model.addStepFunction(cfs)

sdf = SaveDataToFile()
# SaveDataToFile host function; behavior is controlled by SAVE_DATA_TO_FILE flag.
model.addStepFunction(sdf)

if INCLUDE_RG_VARIABLES and DEBUG_PRINT_INTERVAL > 0:
    pds = PrintDebugStats()
    model.addStepFunction(pds)

if SAVE_PICKLE and INCLUDE_CELLS:
    csm = CollectCellMetrics()
    model.addStepFunction(csm)

if INCLUDE_FOCAL_ADHESIONS:
    fam = ReportFAMetrics()
    model.addStepFunction(fam)


"""
  END OF STEP FUNCTIONS
"""
# ++==================================================================++
# ++ Layers                                                            |
# ++==================================================================++
"""
  Control flow

  _build_default_layers() contains the full default L0-L8 layer sequence.
  It is called directly when no variant is loaded, or when a variant does
  not define configure_layers().

  Variants that need to insert, reorder, or remove layers define their own
  configure_layers(model, g) and are responsible for the full sequence.
  They may call g['_build_default_layers']() to include the defaults and
  then add variant-specific layers around or after that call — but note
  that once a layer is registered it cannot be repositioned, so true
  insertion between default layers requires the variant to replicate the
  relevant portion of the block manually.  See Tutorial-Model-Variants.md.
"""

def _build_default_layers():
    # L0: VASC concentration update — runs BEFORE L1 so the ECM grid message
    # (broadcast in L1) already reflects the VASC-imposed concentration floor.
    if INCLUDE_VASCULARIZATION:
        model.newLayer("L0_VASC_Bucket_Locations").addAgentFunction("VASC", "vasc_bucket_location_data")
        model.newLayer("L0_VASC_Csp_Update").addAgentFunction("VASC", "vasc_Csp_update")
        model.newLayer("L0_VASC_Spatial_Locations").addAgentFunction("VASC", "vasc_spatial_location_data")
        model.newLayer("L0_ECM_VASC_Csp_Update").addAgentFunction("ECM", "ecm_vasc_Csp_update")
        if INCLUDE_CELLS and INCLUDE_VASCULAR_CELL_RECRUITMENT:
            model.newLayer("L0_VASC_Cell_Spawn").addAgentFunction("VASC", "vasc_ecm_cell_spawn")

    # L1: Agent_Locations
    model.newLayer("L1_Agent_Locations").addAgentFunction("BCORNER", "bcorner_output_location_data")
    if INCLUDE_DIFFUSION:
        model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_grid_location_data")
    if INCLUDE_CELLS:
        model.Layer("L1_Agent_Locations").addAgentFunction("CELL", "cell_spatial_location_data")
        if INCLUDE_FOCAL_ADHESIONS:
            model.newLayer("L1_CELL_Locations_2").addAgentFunction("CELL", "cell_bucket_location_data")  # these functions share data of the same agent, so must be in separate layers
    if INCLUDE_CELLS and ORGANOID_ASSAY and INCLUDE_LUMEN:
        model.newLayer("L1_LUMEN_Locations").addAgentFunction("LUMEN", "lumen_spatial_location_data")
    if INCLUDE_FIBRE_NETWORK:
        model.newLayer("L1_FNODE_Locations_1").addAgentFunction("FNODE", "fnode_spatial_location_data")
        # These functions share data of the same agent, so must be in separate layers
        model.newLayer("L1_FNODE_Locations_2").addAgentFunction("FNODE", "fnode_bucket_location_data")

    # L2: Boundary_Interactions
    if INCLUDE_DIFFUSION:
        model.newLayer("L2_ECM_Boundary_Interactions").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
    if INCLUDE_FIBRE_NETWORK:
        model.newLayer("L2_FNODE_Boundary_Interactions").addAgentFunction("FNODE", "fnode_boundary_interaction")
    if INCLUDE_FIBRE_NETWORK and INCLUDE_CELLS and INCLUDE_NETWORK_REMODELING:
        model.newLayer("L2_CELL_FNODE_Remodel").addAgentFunction("CELL", "cell_fnode_remodel")
        model.newLayer("L2_FNODE_Remodel").addAgentFunction("FNODE", "fnode_remodel")
        model.newLayer("L2_FNODE_Remodel_Apply").addAgentFunction("FNODE", "fnode_apply_remodel_updates")
        model.newLayer("L2_FNODE_Update_Links").addAgentFunction("FNODE", "fnode_update_links")

    # L3: Metabolism & Cell Cycle
    if INCLUDE_CELLS and INCLUDE_DIFFUSION:
        model.newLayer("L3_Metabolism").addAgentFunction("CELL", "cell_ecm_interaction_metabolism")
    if INCLUDE_CELLS and INCLUDE_CELL_CYCLE:
        model.newLayer("L3_Cell_MaxID_Update").addAgentFunction("CELL", "cell_MaxID_update")
        model.newLayer("L3_Cell_Cycle").addAgentFunction("CELL", "cell_cycle")
        if INCLUDE_FOCAL_ADHESIONS:
            model.newLayer("L3_Cell_Bucket_PostCycle").addAgentFunction("CELL", "cell_bucket_location_data")
            model.newLayer("L3_FOCAD_PostCycle_Update").addAgentFunction("FOCAD", "focad_post_cycle_update")
    if INCLUDE_DIFFUSION:
        # L4_ECM_Csp_Update
        model.newLayer("L4_ECM_Csp_Update").addAgentFunction("ECM", "ecm_Csp_update")
        if HETEROGENEOUS_DIFFUSION and INCLUDE_FIBRE_NETWORK:
            model.newLayer("L4_ECM_Dsp_Update").addAgentFunction("ECM", "ecm_Dsp_update")
        if INCLUDE_CELLS and ORGANOID_ASSAY and INCLUDE_LUMEN:
            model.newLayer("L4_ECM_Dsp_Lumen_Update").addAgentFunction("ECM", "ecm_Dsp_lumen_update")
        # L5_Diffusion
        model.newLayer("L5_Diffusion").addAgentFunction("ECM", "ecm_ecm_interaction")
        # L6_Diffusion_Boundary (called twice to ensure concentration at boundaries is properly shown visually)
        model.newLayer("L6_Diffusion_Boundary").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
    if INCLUDE_FIBRE_NETWORK:
        # L7_Fibre_Network Mechanical interactions
        model.newLayer("L7_FNODE_Repulsion").addAgentFunction("FNODE", "fnode_fnode_spatial_interaction")
        model.newLayer("L7_FNODE_Network_Mechanics").addAgentFunction("FNODE", "fnode_fnode_bucket_interaction")
        if INCLUDE_FOCAL_ADHESIONS:
            model.newLayer("L7_FOCAD_Mechanics").addAgentFunction("FOCAD", "focad_fnode_interaction")
            # These FOCAD location functions are placed here because they require updated force information to be broadcasted to  FNODE and CELL update functions
            model.newLayer("L7_FOCAD_Locations_1").addAgentFunction("FOCAD", "focad_spatial_location_data")
            model.newLayer("L7_FOCAD_Locations_2").addAgentFunction("FOCAD", "focad_bucket_location_data")
            model.newLayer("L7_FNODE_Force_Update").addAgentFunction("FNODE", "fnode_focad_interaction")
            model.newLayer("L7_CELL_Stress_Update").addAgentFunction("CELL", "cell_focad_update")

    if INCLUDE_CELLS and INCLUDE_CELL_CELL_INTERACTION:
        model.newLayer("L7_CELL_CELL_Interaction").addAgentFunction("CELL", "cell_cell_interaction")
    if INCLUDE_CELLS and INCLUDE_FIBRE_NETWORK and INCLUDE_CELL_FNODE_REPULSION:
        model.newLayer("L7_CELL_FNODE_Repulsion").addAgentFunction("CELL", "cell_fnode_repulsion")
        model.newLayer("L7_FNODE_CELL_Repulsion").addAgentFunction("FNODE", "fnode_cell_repulsion")
    if INCLUDE_CELLS and ORGANOID_ASSAY and INCLUDE_LUMEN:
        model.newLayer("L7_LUMEN_LUMEN_Interaction").addAgentFunction("LUMEN", "lumen_lumen_interaction")
        model.newLayer("L7_LUMEN_CELL_Interaction").addAgentFunction("LUMEN", "lumen_cell_interaction")
        model.newLayer("L7_CELL_LUMEN_Interaction").addAgentFunction("CELL", "cell_lumen_interaction")
    # Unified nucleus stress finalization
    if INCLUDE_CELLS:
        model.newLayer("L7_CELL_Stress_State_Update").addAgentFunction("CELL", "cell_stress_state_update")

    # L8_Agent_Movement
    if INCLUDE_CELLS:
        model.newLayer("L8_CELL_Movement").addAgentFunction("CELL", "cell_move")
        if INCLUDE_FOCAL_ADHESIONS:
            # Re-broadcast CELL bucket after movement so FOCAD anchor update
            # reads post-move anchor positions instead of stale L1 data.
            model.newLayer("L8_CELL_Bucket_Post_Move").addAgentFunction("CELL", "cell_bucket_location_data")
    if INCLUDE_CELLS and ORGANOID_ASSAY and INCLUDE_LUMEN:
        model.newLayer("L8_LUMEN_Movement").addAgentFunction("LUMEN", "lumen_move")
        model.newLayer("L8_CELL_LUMEN_Secretion").addAgentFunction("CELL", "cell_lumen_secretion")
    if INCLUDE_FIBRE_NETWORK:
        model.newLayer("L8_FNODE_Movement").addAgentFunction("FNODE", "fnode_move")
        if INCLUDE_FOCAL_ADHESIONS:
            # Broadcast FNODE post-move positions into dedicated message list
            # so focad_move reads current-step coordinates.
            model.newLayer("L8_FNODE_Locations_Post_Move").addAgentFunction("FNODE", "fnode_bucket_location_data_postmove")
            # Sync FOCAD anchor (x_i) with post-move cell position before focad_move,
            # so the ori vector (x_i - x) is consistent at step end.
            model.newLayer("L8_FOCAD_Anchor_Post_Move").addAgentFunction("FOCAD", "focad_anchor_update")
            model.newLayer("L8_FOCAD_Movement").addAgentFunction("FOCAD", "focad_move")
    # If boundaries are not moving, the ECM grid does not need to be updated
    if MOVING_BOUNDARIES:
        model.newLayer("L8_BCORNER_Movement").addAgentFunction("BCORNER", "bcorner_move")
        model.newLayer("L8_ECM_Movement").addAgentFunction("ECM", "ecm_move")
        if INCLUDE_VASCULARIZATION:
            model.newLayer("L8_VASC_Movement").addAgentFunction("VASC", "vasc_move")


# If the active variant defines configure_layers(), hand full control of the
# layer sequence to it.  The variant is responsible for calling
# g['_build_default_layers']() if it wants the default L0-L8 layers.
# If no variant is loaded, or the variant has no configure_layers(), run the
# default sequence directly.
if _ACTIVE_VARIANT is not None and hasattr(_ACTIVE_VARIANT, "configure_layers"):
    print(f"[VARIANT] Calling configure_layers for variant '{_VARIANT_NAME}'")
    _ACTIVE_VARIANT.configure_layers(model, globals())
else:
    _build_default_layers()

# ++==================================================================++
# ++ Logging                                                           |
# ++==================================================================++
"""
  Logging
"""

# Create and configure logging details 
logging_config = pyflamegpu.LoggingConfig(model)

logging_config.logEnvironment("COORDS_BOUNDARIES")
if INCLUDE_FIBRE_NETWORK:
    fnode_agent_log = logging_config.agent("FNODE")
    fnode_agent_log.logCount()
    fnode_agent_log.logSumFloat("f_bx_pos")
    fnode_agent_log.logSumFloat("f_bx_neg")
    fnode_agent_log.logSumFloat("f_by_pos")
    fnode_agent_log.logSumFloat("f_by_neg")
    fnode_agent_log.logSumFloat("f_bz_pos")
    fnode_agent_log.logSumFloat("f_bz_neg")

    fnode_agent_log.logSumFloat("f_bx_pos_y")
    fnode_agent_log.logSumFloat("f_bx_pos_z")
    fnode_agent_log.logSumFloat("f_bx_neg_y")
    fnode_agent_log.logSumFloat("f_bx_neg_z")
    fnode_agent_log.logSumFloat("f_by_pos_x")
    fnode_agent_log.logSumFloat("f_by_pos_z")
    fnode_agent_log.logSumFloat("f_by_neg_x")
    fnode_agent_log.logSumFloat("f_by_neg_z")
    fnode_agent_log.logSumFloat("f_bz_pos_x")
    fnode_agent_log.logSumFloat("f_bz_pos_y")
    fnode_agent_log.logSumFloat("f_bz_neg_x")
    fnode_agent_log.logSumFloat("f_bz_neg_y")

    fnode_agent_log.logMeanFloat("f_bx_pos")
    fnode_agent_log.logMeanFloat("f_bx_neg")
    fnode_agent_log.logMeanFloat("f_by_pos")
    fnode_agent_log.logMeanFloat("f_by_neg")
    fnode_agent_log.logMeanFloat("f_bz_pos")
    fnode_agent_log.logMeanFloat("f_bz_neg")
    fnode_agent_log.logStandardDevFloat("f_bx_pos")
    fnode_agent_log.logStandardDevFloat("f_bx_neg")
    fnode_agent_log.logStandardDevFloat("f_by_pos")
    fnode_agent_log.logStandardDevFloat("f_by_neg")
    fnode_agent_log.logStandardDevFloat("f_bz_pos")
    fnode_agent_log.logStandardDevFloat("f_bz_neg")
    
    fnode_agent_log.logSumUInt8("clamped_bx_pos")
    fnode_agent_log.logSumUInt8("clamped_bx_neg")
    fnode_agent_log.logSumUInt8("clamped_by_pos")
    fnode_agent_log.logSumUInt8("clamped_by_neg")
    fnode_agent_log.logSumUInt8("clamped_bz_pos")
    fnode_agent_log.logSumUInt8("clamped_bz_neg")

    fnode_agent_log.logSumFloat("degradation")
    fnode_agent_log.logSumFloat("reinforcement")
    fnode_agent_log.logSumFloat("elastic_energy")
    fnode_agent_log.logSumInt("secreted")

if INCLUDE_FOCAL_ADHESIONS:
    focad_agent_log = logging_config.agent("FOCAD")
    focad_agent_log.logCount()
    focad_agent_log.logSumInt("attached")
    focad_agent_log.logSumInt("is_front")
    focad_agent_log.logSumInt("is_rear")
    focad_agent_log.logSumInt("attached_front")
    focad_agent_log.logSumInt("attached_rear")
    focad_agent_log.logSumFloat("f_mag")
    focad_agent_log.logSumFloat("frontness_front")
    focad_agent_log.logSumFloat("frontness_rear")
    focad_agent_log.logSumFloat("k_on_eff_front")
    focad_agent_log.logSumFloat("k_on_eff_rear")
    focad_agent_log.logSumFloat("k_off_0_eff_front")
    focad_agent_log.logSumFloat("k_off_0_eff_rear")

if INCLUDE_CELLS:
    cell_agent_log = logging_config.agent("CELL")
    cell_agent_log.logCount()
    cell_agent_log.logSumInt("dead")

step_log = pyflamegpu.StepLoggingConfig(logging_config)
step_log.setFrequency(1) # if 1, data will be logged every step

# ++==================================================================++
# ++ Model runner                                                      |
# ++==================================================================++
"""
  Create Model Runner
"""
if ENSEMBLE:

    """
    Create Control Run Plan
    """
    # Create a control run plan, this will define the common properties across all plans
    # https://docs.flamegpu.com/guide/running-multiple-simulations/index.html#creating-a-runplanvector
    run_control = pyflamegpu.RunPlan(model)

    # Ensure that repeated runs use the same Random values within the RunPlans
    run_control.setRandomPropertySeed(34523) # This method only exists at the vector level, if you're not using setPropertyRandom(), it woud have no effect.
    # All runs have the same steps
    run_control.setSteps(STEPS)
    run_control.setPropertyUInt("STEPS", STEPS)

    # Create the first dimension of the parameter sweep
    ensemble_runs = pyflamegpu.RunPlanVector(model, 0)
	# Example: varying 3 model variables to check model sensitivity
    #for VARIABLE_1_value in np.linspace(?, ?, ?): # min, max, number of divisions
    #    for VARIABLE_2_value in np.linspace(?, ?, ?):
    #        for VARIABLE_3_value in np.linspace(?, ?, ?):
    #            run_control.setPropertyFloat("VARIABLE_1", VARIABLE_1_value)
    #            run_control.setPropertyFloat("VARIABLE_2", VARIABLE_2_value)
    #            run_control.setPropertyFloat("VARIABLE_3", VARIABLE_3_value)
    #            ensemble_runs += run_control
    #            dir_name = f"VARIABLE_1_{VARIABLE_1_value:.3f}_VARIABLE_2_{VARIABLE_2_value:.3f}_VARIABLE_3_{VARIABLE_3_value:.3f}" # Create directory names using the parameter values
    #            full_path = RES_PATH / dir_name # Combine the base directory with the current directory name
    #            full_path.mkdir(parents=True, exist_ok=True)

    # Create a CUDAEnsemble to execute the RunPlanVector
    ensemble = pyflamegpu.CUDAEnsemble(model)

    # Override config defaults
    ensemble.Config().out_directory = RES_PATH.as_posix()
    ensemble.Config().out_format = "json"
    ensemble.Config().concurrent_runs = 1  # This is concurrent runs per device, higher values may improve performance for "small" models
    ensemble.Config().timing = False
    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast  # Kills the ensemble as soon as the first error is detected

    # Pass any logging configs to the CUDAEnsemble
    # https://docs.flamegpu.com/guide/running-multiple-simulations/index.html#creating-a-logging-configuration
    ensemble.setStepLog(step_log)
    ensemble.setExitLog(logging_config)

else:
    simulation = pyflamegpu.CUDASimulation(model)
    simulation.SimulationConfig().steps = STEPS
    simulation.setStepLog(step_log)
    simulation.setExitLog(logging_config)

# ++==================================================================++
# ++ Visualization                                                     |
# ++==================================================================++
"""
  Create Visualisation
"""
if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
    vis = simulation.getVisualisation()
    # Configure vis
    domain_width = MAX_EXPECTED_BOUNDARY_POS - MIN_EXPECTED_BOUNDARY_POS
    INIT_CAM = MAX_EXPECTED_BOUNDARY_POS * 4.5 # A value of the position of the domain by the end of the simulation, multiplied by 5, looks nice
    vis.setInitialCameraLocation(0.0, 0.0, INIT_CAM)
    vis.setCameraSpeed(0.002 * domain_width) # values <<1 (e.g. 0.002) work fine
    if DEBUG_PRINTING:
        vis.setSimulationSpeed(1)
    vis.setBeginPaused(True)

    CELL_vis_agent = vis.addAgent("CELL")
    # Position vars are named x, y, z so they are used by default
    CELL_vis_agent.setModel(pyflamegpu.ICOSPHERE)
    CELL_vis_agent.setModelScale(0.03 * domain_width) # values <<1 (e.g. 0.03) work fine
    CELL_vis_agent.setColor(pyflamegpu.Color("#00aaff"))

    ECM_vis_agent = vis.addAgent("ECM")
    # Position vars are named x, y, z so they are used by default
    ECM_vis_agent.setModel(pyflamegpu.CUBE)
    ECM_vis_agent.setModelScale(0.025 * domain_width) # values <<1 (e.g. 0.03) work fine
    ECM_vis_agent.setColor(pyflamegpu.Color("#ffaa00"))

    BCORNER_vis_agent = vis.addAgent("BCORNER")
    BCORNER_vis_agent.setModel(pyflamegpu.CUBE)
    BCORNER_vis_agent.setModelScale(0.025 * domain_width)
    BCORNER_vis_agent.setColor(pyflamegpu.RED)

    coord_boundary = list(env.getPropertyArrayFloat("COORDS_BOUNDARIES"))
    pen = vis.newLineSketch(1, 1, 1, 0.8)
    pen.addVertex(coord_boundary[0], coord_boundary[2], coord_boundary[4])
    pen.addVertex(coord_boundary[0], coord_boundary[2], coord_boundary[5])
    pen.addVertex(coord_boundary[0], coord_boundary[3], coord_boundary[4])
    pen.addVertex(coord_boundary[0], coord_boundary[3], coord_boundary[5])
    pen.addVertex(coord_boundary[1], coord_boundary[2], coord_boundary[4])
    pen.addVertex(coord_boundary[1], coord_boundary[2], coord_boundary[5])
    pen.addVertex(coord_boundary[1], coord_boundary[3], coord_boundary[4])
    pen.addVertex(coord_boundary[1], coord_boundary[3], coord_boundary[5])

    pen.addVertex(coord_boundary[0], coord_boundary[2], coord_boundary[4])
    pen.addVertex(coord_boundary[0], coord_boundary[3], coord_boundary[4])
    pen.addVertex(coord_boundary[0], coord_boundary[2], coord_boundary[5])
    pen.addVertex(coord_boundary[0], coord_boundary[3], coord_boundary[5])
    pen.addVertex(coord_boundary[1], coord_boundary[2], coord_boundary[4])
    pen.addVertex(coord_boundary[1], coord_boundary[3], coord_boundary[4])
    pen.addVertex(coord_boundary[1], coord_boundary[2], coord_boundary[5])
    pen.addVertex(coord_boundary[1], coord_boundary[3], coord_boundary[5])

    pen.addVertex(coord_boundary[0], coord_boundary[2], coord_boundary[4])
    pen.addVertex(coord_boundary[1], coord_boundary[2], coord_boundary[4])
    pen.addVertex(coord_boundary[0], coord_boundary[3], coord_boundary[4])
    pen.addVertex(coord_boundary[1], coord_boundary[3], coord_boundary[4])
    pen.addVertex(coord_boundary[0], coord_boundary[2], coord_boundary[5])
    pen.addVertex(coord_boundary[1], coord_boundary[2], coord_boundary[5])
    pen.addVertex(coord_boundary[0], coord_boundary[3], coord_boundary[5])
    pen.addVertex(coord_boundary[1], coord_boundary[3], coord_boundary[5])

    vis.activate()

# ++==================================================================++
# ++ Execution                                                         |
# ++==================================================================++
"""
  Execution
"""
print(f"[DIAG] About to simulate: ENSEMBLE={ENSEMBLE}, STEPS={STEPS}")
_sim_start_time = time.time()
PYTHON_SETUP_TIME = _sim_start_time - start_time
print(f"--- PYTHON SETUP TIME: {PYTHON_SETUP_TIME:.6f} seconds ---")
if ENSEMBLE:
    # Execute the ensemble using the specified RunPlans
    errs = ensemble.simulate(ensemble_runs)
    _sim_end_time = time.time()
    # Ensemble: fall back to wall-clock timing (per-run internal timers
    # are not directly accessible from CUDAEnsemble).
    RTC_TIME = 0.0
    INIT_FUNCTIONS_TIME = 0.0
    EXIT_FUNCTIONS_TIME = 0.0
    SIMULATION_TIME = _sim_end_time - _sim_start_time
    INIT_TIME = PYTHON_SETUP_TIME
else:
    simulation.simulate()
    _sim_end_time = time.time()
    # Use CUDASimulation's internal high-resolution timers for an accurate
    # breakdown (see CUDASimulation.h for the C++ API).
    RTC_TIME = simulation.getElapsedTimeRTCInitialisation()
    INIT_FUNCTIONS_TIME = simulation.getElapsedTimeInitFunctions()
    EXIT_FUNCTIONS_TIME = simulation.getElapsedTimeExitFunctions()
    _step_times = simulation.getElapsedTimeSteps()
    SIMULATION_TIME = sum(_step_times) if _step_times else 0.0
    INIT_TIME = PYTHON_SETUP_TIME + RTC_TIME + INIT_FUNCTIONS_TIME
    print(f"--- Internal timers: RTC={RTC_TIME:.4f}s, "
          f"InitFunctions={INIT_FUNCTIONS_TIME:.4f}s, "
          f"Stepping={SIMULATION_TIME:.4f}s ({len(_step_times)} steps), "
          f"ExitFunctions={EXIT_FUNCTIONS_TIME:.4f}s ---")
print("[DIAG] simulation.simulate() completed")


if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
    vis.join() # join the visualisation thread and stops the visualisation closing after the simulation finishes

EXECUTION_TIME = time.time() - start_time
print(f"[BENCHMARK] EXECUTION_TIME={EXECUTION_TIME:.6f} STEPS={STEPS} TIME_PER_STEP={SIMULATION_TIME/max(STEPS,1):.6f} INIT_TIME={INIT_TIME:.6f} SIMULATION_TIME={SIMULATION_TIME:.6f} RTC_TIME={RTC_TIME:.6f} INIT_FUNCTIONS_TIME={INIT_FUNCTIONS_TIME:.6f} EXIT_FUNCTIONS_TIME={EXIT_FUNCTIONS_TIME:.6f}")

incL_dir1 = (BPOS_OVER_TIME.iloc[:, POISSON_DIRS[0] * 2] - BPOS_OVER_TIME.iloc[:, POISSON_DIRS[0] * 2 + 1]) - (
        BPOS_OVER_TIME.iloc[0, POISSON_DIRS[0] * 2] - BPOS_OVER_TIME.iloc[0, POISSON_DIRS[0] * 2 + 1])
incL_dir2 = (BPOS_OVER_TIME.iloc[:, POISSON_DIRS[1] * 2] - BPOS_OVER_TIME.iloc[:, POISSON_DIRS[1] * 2 + 1]) - (
        BPOS_OVER_TIME.iloc[0, POISSON_DIRS[1] * 2] - BPOS_OVER_TIME.iloc[0, POISSON_DIRS[1] * 2 + 1])

POISSON_RATIO_OVER_TIME = -1 * incL_dir1 / incL_dir2


def manageLogs(steps, is_ensemble, idx):
    global SAVE_EVERY_N_STEPS, SAVE_PICKLE, SHOW_PLOTS, RES_PATH, MODEL_CONFIG, EXECUTION_TIME, STEPS
    global BPOS_OVER_TIME, BFORCE_OVER_TIME, BFORCE_SHEAR_OVER_TIME, POISSON_RATIO_OVER_TIME, OSCILLATORY_STRAIN_OVER_TIME
    global CELL_SPEED_METRICS, ORGANOID_METRICS_OVER_TIME, RG_METRICS, RG_ROSETTE_METRICS_OVER_TIME, INCLUDE_RG_VARIABLES
    global INCLUDE_FIBRE_NETWORK, INCLUDE_CELLS, INCLUDE_FOCAL_ADHESIONS, ORGANOID_ASSAY
    ecm_agent_counts = [None] * len(steps)
    counter = 0
    BFORCE = make_dataclass("BFORCE",
                            [("fxpos", float), ("fxneg", float), ("fypos", float), ("fyneg", float), ("fzpos", float),
                             ("fzneg", float)])
    BFORCE_SHEAR = make_dataclass("BFORCE_SHEAR",
                                  [("fxpos_y", float), ("fxpos_z", float), ("fxneg_y", float), ("fxneg_z", float),
                                   ("fypos_x", float), ("fypos_z", float), ("fyneg_x", float), ("fyneg_z", float),
                                   ("fzpos_x", float), ("fzpos_y", float), ("fzneg_x", float), ("fzneg_y", float)])
    BATTACH = make_dataclass(
        "BATTACH",
        [("n_bx_pos", float), ("n_bx_neg", float), ("n_by_pos", float), ("n_by_neg", float), ("n_bz_pos", float), ("n_bz_neg", float)],
    )
    BFORCE_OVER_TIME = []
    BFORCE_SHEAR_OVER_TIME = []
    BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME = []
    FOCAD_METRICS_OVER_TIME = []
    FOCAD_POLARITY_METRICS_OVER_TIME = []
    FNODE_METRICS_OVER_TIME = []
    CELL_METRICS_OVER_TIME = []

    if INCLUDE_FIBRE_NETWORK:
        for step in steps:
            stepcount = step.getStepCount()
            if stepcount % SAVE_EVERY_N_STEPS == 0 or stepcount == 1:
                fnode_agents = step.getAgent("FNODE")
                ecm_agent_counts[counter] = fnode_agents.getCount()
                f_bx_pos = fnode_agents.getSumFloat("f_bx_pos")
                f_bx_neg = fnode_agents.getSumFloat("f_bx_neg")
                f_by_pos = fnode_agents.getSumFloat("f_by_pos")
                f_by_neg = fnode_agents.getSumFloat("f_by_neg")
                f_bz_pos = fnode_agents.getSumFloat("f_bz_pos")
                f_bz_neg = fnode_agents.getSumFloat("f_bz_neg")
                f_bx_pos_y = fnode_agents.getSumFloat("f_bx_pos_y")
                f_bx_pos_z = fnode_agents.getSumFloat("f_bx_pos_z")
                f_bx_neg_y = fnode_agents.getSumFloat("f_bx_neg_y")
                f_bx_neg_z = fnode_agents.getSumFloat("f_bx_neg_z")
                f_by_pos_x = fnode_agents.getSumFloat("f_by_pos_x")
                f_by_pos_z = fnode_agents.getSumFloat("f_by_pos_z")
                f_by_neg_x = fnode_agents.getSumFloat("f_by_neg_x")
                f_by_neg_z = fnode_agents.getSumFloat("f_by_neg_z")
                f_bz_pos_x = fnode_agents.getSumFloat("f_bz_pos_x")
                f_bz_pos_y = fnode_agents.getSumFloat("f_bz_pos_y")
                f_bz_neg_x = fnode_agents.getSumFloat("f_bz_neg_x")
                f_bz_neg_y = fnode_agents.getSumFloat("f_bz_neg_y")
                n_bx_pos = fnode_agents.getSumUInt8("clamped_bx_pos")
                n_bx_neg = fnode_agents.getSumUInt8("clamped_bx_neg")
                n_by_pos = fnode_agents.getSumUInt8("clamped_by_pos")
                n_by_neg = fnode_agents.getSumUInt8("clamped_by_neg")
                n_bz_pos = fnode_agents.getSumUInt8("clamped_bz_pos")
                n_bz_neg = fnode_agents.getSumUInt8("clamped_bz_neg")

                step_bforce = pd.DataFrame([BFORCE(f_bx_pos, f_bx_neg, f_by_pos, f_by_neg, f_bz_pos, f_bz_neg)])
                step_bforce_shear = pd.DataFrame([BFORCE_SHEAR(f_bx_pos_y, f_bx_pos_z, f_bx_neg_y, f_bx_neg_z,
                                                            f_by_pos_x, f_by_pos_z, f_by_neg_x, f_by_neg_z,
                                                            f_bz_pos_x, f_bz_pos_y, f_bz_neg_x, f_bz_neg_y)])
                step_battach = pd.DataFrame([BATTACH(n_bx_pos, n_bx_neg, n_by_pos, n_by_neg, n_bz_pos, n_bz_neg)])
                if counter == 0:
                    BFORCE_OVER_TIME = pd.DataFrame([BFORCE(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
                    BFORCE_SHEAR_OVER_TIME = pd.DataFrame(
                        [BFORCE_SHEAR(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
                    BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME = step_battach
                else:
                    # BFORCE_OVER_TIME = BFORCE_OVER_TIME.append(step_bforce, ignore_index=True) # deprecated
                    BFORCE_OVER_TIME = pd.concat([BFORCE_OVER_TIME, step_bforce], ignore_index=True)
                    # BFORCE_SHEAR_OVER_TIME = BFORCE_SHEAR_OVER_TIME.append(step_bforce_shear, ignore_index=True) # deprecated
                    BFORCE_SHEAR_OVER_TIME = pd.concat([BFORCE_SHEAR_OVER_TIME, step_bforce_shear], ignore_index=True)
                    BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME = pd.concat([BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME, step_battach], ignore_index=True)

                # Accumulate FNODE matrix remodeling metrics
                n_fnodes = fnode_agents.getCount()
                sum_degradation = fnode_agents.getSumFloat("degradation")
                sum_reinforcement = fnode_agents.getSumFloat("reinforcement")
                sum_elastic_energy = fnode_agents.getSumFloat("elastic_energy")
                n_secreted = fnode_agents.getSumInt("secreted")
                mean_degradation = (sum_degradation / n_fnodes) if n_fnodes > 0 else 0.0
                mean_reinforcement = (sum_reinforcement / n_fnodes) if n_fnodes > 0 else 0.0
                mean_elastic_energy = (sum_elastic_energy / n_fnodes) if n_fnodes > 0 else 0.0

                step_fnode_met = pd.DataFrame([{
                    "n_fnodes_total": n_fnodes,
                    "n_fnodes_secreted_cumulative": n_secreted,
                    "sum_degradation": sum_degradation,
                    "sum_reinforcement": sum_reinforcement,
                    "sum_elastic_energy": sum_elastic_energy,
                    "mean_degradation": mean_degradation,
                    "mean_reinforcement": mean_reinforcement,
                    "mean_elastic_energy": mean_elastic_energy,
                    "net_remodeling_total": sum_reinforcement - sum_degradation,
                }])
                if len(FNODE_METRICS_OVER_TIME) == 0:
                    FNODE_METRICS_OVER_TIME = step_fnode_met
                else:
                    FNODE_METRICS_OVER_TIME = pd.concat([FNODE_METRICS_OVER_TIME, step_fnode_met], ignore_index=True)

                counter += 1

    if INCLUDE_FOCAL_ADHESIONS:
        FMET = make_dataclass("FMET", [("attached", float), ("total", float), ("attached_ratio", float), ("mean_f_mag", float)])
        FPOL = make_dataclass(
            "FPOL",
            [
                ("front_count", float),
                ("rear_count", float),
                ("front_attached", float),
                ("rear_attached", float),
                ("front_attached_ratio", float),
                ("rear_attached_ratio", float),
                ("frontness_front_mean", float),
                ("frontness_rear_mean", float),
                ("k_on_eff_front_mean", float),
                ("k_on_eff_rear_mean", float),
                ("k_off_0_eff_front_mean", float),
                ("k_off_0_eff_rear_mean", float),
            ],
        )
        for step in steps:
            stepcount = step.getStepCount()
            if stepcount % SAVE_EVERY_N_STEPS == 0 or stepcount == 1:
                focad_agents = step.getAgent("FOCAD")
                n_focad = focad_agents.getCount()
                attached = focad_agents.getSumInt("attached")
                total_f_mag = focad_agents.getSumFloat("f_mag")
                ratio = (attached / n_focad) if n_focad > 0 else 0.0
                mean_f_mag = (total_f_mag / n_focad) if n_focad > 0 else 0.0
                step_fmet = pd.DataFrame([FMET(attached, n_focad, ratio, mean_f_mag)])
                if len(FOCAD_METRICS_OVER_TIME) == 0:
                    FOCAD_METRICS_OVER_TIME = step_fmet
                else:
                    FOCAD_METRICS_OVER_TIME = pd.concat([FOCAD_METRICS_OVER_TIME, step_fmet], ignore_index=True)

                front_count = focad_agents.getSumInt("is_front")
                rear_count = focad_agents.getSumInt("is_rear")
                front_attached = focad_agents.getSumInt("attached_front")
                rear_attached = focad_agents.getSumInt("attached_rear")

                front_attached_ratio = (front_attached / front_count) if front_count > 0 else 0.0
                rear_attached_ratio = (rear_attached / rear_count) if rear_count > 0 else 0.0

                frontness_front_sum = focad_agents.getSumFloat("frontness_front")
                frontness_rear_sum = focad_agents.getSumFloat("frontness_rear")

                k_on_eff_front_sum = focad_agents.getSumFloat("k_on_eff_front")
                k_on_eff_rear_sum = focad_agents.getSumFloat("k_on_eff_rear")

                k_off_0_eff_front_sum = focad_agents.getSumFloat("k_off_0_eff_front")
                k_off_0_eff_rear_sum = focad_agents.getSumFloat("k_off_0_eff_rear")

                frontness_front_mean = (frontness_front_sum / front_count) if front_count > 0 else 0.0
                frontness_rear_mean = (frontness_rear_sum / rear_count) if rear_count > 0 else 0.0
                k_on_eff_front_mean = (k_on_eff_front_sum / front_count) if front_count > 0 else 0.0
                k_on_eff_rear_mean = (k_on_eff_rear_sum / rear_count) if rear_count > 0 else 0.0
                k_off_0_eff_front_mean = (k_off_0_eff_front_sum / front_count) if front_count > 0 else 0.0
                k_off_0_eff_rear_mean = (k_off_0_eff_rear_sum / rear_count) if rear_count > 0 else 0.0

                step_fpol = pd.DataFrame([
                    FPOL(
                        front_count,
                        rear_count,
                        front_attached,
                        rear_attached,
                        front_attached_ratio,
                        rear_attached_ratio,
                        frontness_front_mean,
                        frontness_rear_mean,
                        k_on_eff_front_mean,
                        k_on_eff_rear_mean,
                        k_off_0_eff_front_mean,
                        k_off_0_eff_rear_mean,
                    )
                ])
                if len(FOCAD_POLARITY_METRICS_OVER_TIME) == 0:
                    FOCAD_POLARITY_METRICS_OVER_TIME = step_fpol
                else:
                    FOCAD_POLARITY_METRICS_OVER_TIME = pd.concat([FOCAD_POLARITY_METRICS_OVER_TIME, step_fpol], ignore_index=True)

    if INCLUDE_CELLS:
        for step in steps:
            stepcount = step.getStepCount()
            if stepcount % SAVE_EVERY_N_STEPS == 0 or stepcount == 1 or stepcount == STEPS:
                cell_agents = step.getAgent("CELL")
                n_cells = cell_agents.getCount()
                n_dead = cell_agents.getSumInt("dead")
                n_alive = n_cells - n_dead
                step_cell_met = pd.DataFrame([{
                    "step": stepcount,
                    "n_cells_total": n_cells,
                    "n_cells_alive": n_alive,
                    "n_cells_dead": n_dead,
                }])
                if len(CELL_METRICS_OVER_TIME) == 0:
                    CELL_METRICS_OVER_TIME = step_cell_met
                else:
                    CELL_METRICS_OVER_TIME = pd.concat([CELL_METRICS_OVER_TIME, step_cell_met], ignore_index=True)

    if not is_ensemble:
        print()
        print("============================")
        print("BOUNDARY POSITIONS OVER TIME")
        print(BPOS_OVER_TIME)
        print()
        print("============================")
        print("BOUNDARY FORCES OVER TIME")
        print(BFORCE_OVER_TIME)
        print()
        print("============================")
        print("BOUNDARY SHEAR FORCES OVER TIME")
        print(BFORCE_SHEAR_OVER_TIME)
        print()
        print("============================")
        print("BOUNDARY ATTACHMENT COUNTS OVER TIME")
        print(BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME)
        print()
        print("============================")
        print("POISSON RATIO OVER TIME")
        print(POISSON_RATIO_OVER_TIME)
        print()
        print("============================")
        print("STRAIN OVER TIME")
        print(OSCILLATORY_STRAIN_OVER_TIME)
        print()
        if INCLUDE_FOCAL_ADHESIONS and len(FOCAD_METRICS_OVER_TIME) > 0:
            print("============================")
            print("FA METRICS OVER TIME")
            print(FOCAD_METRICS_OVER_TIME)
            print()
        if INCLUDE_FOCAL_ADHESIONS and len(FOCAD_POLARITY_METRICS_OVER_TIME) > 0:
            print("============================")
            print("FA POLARITY METRICS OVER TIME")
            print(FOCAD_POLARITY_METRICS_OVER_TIME)
            print()
        if INCLUDE_FIBRE_NETWORK and len(FNODE_METRICS_OVER_TIME) > 0:
            print("============================")
            print("FNODE MATRIX REMODELING METRICS OVER TIME")
            print(FNODE_METRICS_OVER_TIME)
            print()
        if INCLUDE_CELLS and len(CELL_METRICS_OVER_TIME) > 0:
            print("============================")
            print("CELL POPULATION METRICS OVER TIME")
            print(CELL_METRICS_OVER_TIME)
            print()
        if INCLUDE_CELLS and len(CELL_SPEED_METRICS) > 0:
            print("============================")
            print("FINAL CELL SPEED METRICS")
            print(CELL_SPEED_METRICS)
            print()
        if INCLUDE_RG_VARIABLES and len(RG_METRICS) > 0:
            print("============================")
            print("RG FINAL CELL METRICS")
            print(RG_METRICS)
            print()
        if INCLUDE_RG_VARIABLES and len(RG_ROSETTE_METRICS_OVER_TIME) > 0:
            print("============================")
            print("RG ROSETTE METRICS OVER TIME")
            print(RG_ROSETTE_METRICS_OVER_TIME)
            print()
        if ORGANOID_ASSAY and len(ORGANOID_METRICS_OVER_TIME) > 0:
            print("============================")
            print("ORGANOID METRICS OVER TIME")
            print(ORGANOID_METRICS_OVER_TIME)
            print()
    # Saving pickle
    print(f"[DIAG] manageLogs: SAVE_PICKLE={SAVE_PICKLE}, RES_PATH={RES_PATH}, idx={idx}")
    if SAVE_PICKLE:
        file_name = f'output_data_{idx}.pickle'
        file_path = RES_PATH / file_name
        print(f"[DIAG] Saving pickle to: {file_path}")
        with open(str(file_path), 'wb') as file:
            pickle.dump({'BPOS_OVER_TIME': BPOS_OVER_TIME,
                         'BFORCE_OVER_TIME': BFORCE_OVER_TIME,
                         'BFORCE_SHEAR_OVER_TIME': BFORCE_SHEAR_OVER_TIME,
                         'BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME': BOUNDARY_ATTACHMENT_COUNTS_OVER_TIME,
                         'FIBRE_SECTION_AREA_UM2': FIBRE_SECTION_AREA_UM2,
                         'FOCAD_METRICS_OVER_TIME': FOCAD_METRICS_OVER_TIME,
                         'FOCAD_POLARITY_METRICS_OVER_TIME': FOCAD_POLARITY_METRICS_OVER_TIME,
                         'FNODE_METRICS_OVER_TIME': FNODE_METRICS_OVER_TIME,
                         'CELL_METRICS_OVER_TIME': CELL_METRICS_OVER_TIME,
                         'CELL_SPEED_METRICS': CELL_SPEED_METRICS,
                         'RG_FINAL_METRICS': RG_METRICS,
                         'RG_ROSETTE_METRICS_OVER_TIME': RG_ROSETTE_METRICS_OVER_TIME,
                         'ORGANOID_METRICS_OVER_TIME': ORGANOID_METRICS_OVER_TIME,
                         'POISSON_RATIO_OVER_TIME': POISSON_RATIO_OVER_TIME,
                         'OSCILLATORY_STRAIN_OVER_TIME': OSCILLATORY_STRAIN_OVER_TIME,
                         'MODEL_CONFIG': MODEL_CONFIG,
                         'EXECUTION_TIME': EXECUTION_TIME},
                        file, protocol=pickle.HIGHEST_PROTOCOL)

            print('Results successfully saved to {0}'.format(file_path))
    # Plotting
    if SHOW_PLOTS and not is_ensemble:
        MODEL_CONFIG.plot_all(
            bpos_over_time=BPOS_OVER_TIME,
            bforce_over_time=BFORCE_OVER_TIME,
            bforce_shear_over_time=BFORCE_SHEAR_OVER_TIME,
            poisson_ratio_over_time=POISSON_RATIO_OVER_TIME,
            show=True,
        )
        if OSCILLATORY_SHEAR_ASSAY:
            MODEL_CONFIG.plot_oscillatory_shear_scatter(
                oscillatory_strain_over_time=OSCILLATORY_STRAIN_OVER_TIME,
                bforce_shear_over_time=BFORCE_SHEAR_OVER_TIME,
                max_strain=MAX_STRAIN,
                show=True,
            )

# Deal with logs
print("[DIAG] Processing logs...")
if ENSEMBLE:
    logs = simulation.getLogs()
    for i in range(len(logs)):
        steps = logs[i].getStepLog()
        manageLogs(steps, ENSEMBLE, i)
else:
    logs = simulation.getRunLog()
    steps = logs.getStepLog()
    print(f"[DIAG] Got {len(steps)} step log entries")
    manageLogs(steps, ENSEMBLE, 0)
print("[DIAG] All done.")