# +====================================================================+
# | Model: CELLFOUNDRY                                                 |
# | Last update: 13/02/2026 - 13:28:03                                 |
# +====================================================================+


# +====================================================================+
# | IMPORTS                                                            |
# +====================================================================+
from pyflamegpu import *
import pathlib, time, math, sys
from dataclasses import make_dataclass
import pandas as pd
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
from helper_module import compute_expected_boundary_pos_from_corners, getRandomVectors3D, build_model_config_from_namespace, load_fibre_network, getRandomCoordsAroundPoint, getRandomCoords3D, compute_u_ref_from_anchor_pos, build_save_data_context, save_data_to_file_step, print_fibre_calibration_summary

# TODO LIST:
# Add cell-fnode repulsion
# Add FOCAD interaction with other FOCADs from other cells?
# Include new FOCAD agent generation? (e.g. when a cell starts migrating, it generates new FOCAD agents at its leading edge, which then try to find fibres to attach to. When a FOCAD agent detaches, it can be removed from the simulation or moved back to the cell center to be reused later)
# Add cell guidance by fibre orientation (cells prefer to move along the main fibre orientation, which could be implemented by making them prefer to move towards areas where the fibre segments are more aligned in a certain direction)
# Add matrix degradation / deposition. Easy: Modifying FNODE properties, Complex: removing / adding FNODE agents (which would require updating the connectivity matrix)

start_time = time.time()

# +====================================================================+
# | GLOBAL SIMULATION PARAMETERS                                       |
# +====================================================================+
# Set whether to run single model or ensemble, agent population size, and simulation steps 
ENSEMBLE = False
ENSEMBLE_RUNS = 0
VISUALISATION = False  # Change to false if pyflamegpu has not been built with visualisation support
DEBUG_PRINTING = False
PAUSE_EVERY_STEP = False  # If True, the visualization stops every step until P is pressed
SAVE_PICKLE = True  # If True, dumps model configuration into a pickle file for post-processing
SHOW_PLOTS = False  # Show plots at the end of the simulation
SAVE_DATA_TO_FILE = True  # If true, agent data is exported to .vtk file every SAVE_EVERY_N_STEPS steps
SAVE_EVERY_N_STEPS = 1 # Affects both the .vtk files and the Dataframes storing boundary data

CURR_PATH = pathlib.Path().absolute()
RES_PATH = CURR_PATH / 'result_files'
RES_PATH.mkdir(parents=True, exist_ok=True)
EPSILON = 0.0000000001

print("Executing in ", CURR_PATH)
# Minimum number of ECM agents per direction (x,y,z). 
# If domain is not cubical, N is asigned to the shorter dimension and more agents are added to the longer ones
# NOTE: ECM agents are always present (mandatory) eventhough they are only used when INCLUDE_DIFFUSION is True. If there is no diffusion, set N to a small value to reduce computational cost.
# ----------------------------------------------------------------------
N = 21

# Time simulation parameters
# ----------------------------------------------------------------------
TIME_STEP = 0.1 # s. WARNING: diffusion and cell migration events might need different scales
STEPS = 50

# +====================================================================+
# | BOUNDARY CONDITIONS                                                |
# +====================================================================+

# Boundary interactions and mechanical parameters
# ----------------------------------------------------------------------
ECM_K_ELAST = 0.2  # [nN/um]
ECM_D_DUMPING = 0.04  # [nN·s/um]
ECM_ETA = 2.0  # [nN·s/µm] Effective drag for overdamped FNODE motion (calibration parameter)

#BOUNDARY_COORDS = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]  # +X,-X,+Y,-Y,+Z,-Z
BOUNDARY_COORDS = [100.0, -100.0, 100.0, -100.0, 100.0, -100.0]# microdevice dimensions in um
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
#CLAMP_AGENT_TOUCHING_BOUNDARY = [0, 0, 1, 1, 0, 0]# +X,-X,+Y,-Y,+Z,-Z [bool] - shear assay
CLAMP_AGENT_TOUCHING_BOUNDARY = [1, 1, 1, 1, 1, 1]# +X,-X,+Y,-Y,+Z,-Z [bool]
ALLOW_AGENT_SLIDING = [1, 1, 1, 1, 1, 1]# +X,-X,+Y,-Y,+Z,-Z [bool]

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
MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION = 2 * ECM_ECM_EQUILIBRIUM_DISTANCE # this radius is used to check if cells interact with each other

OSCILLATORY_SHEAR_ASSAY = False  # if True, BOUNDARY_DISP_RATES_PARALLEL options are overrun but used to make the boundaries oscillate in their corresponding planes following a sin() function
MAX_STRAIN = 0.25  # maximum strain applied during oscillatory shear assay (used to compute OSCILLATORY_AMPLITUDE)
OSCILLATORY_AMPLITUDE = MAX_STRAIN * (BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3])  # range [0-1] * domain size in the direction of oscillation
OSCILLATORY_FREQ = 0.05  # strain oscillation frequency [s^-1]
OSCILLATORY_W = 2 * math.pi * OSCILLATORY_FREQ * TIME_STEP
# Compute expected boundary positions after motion, WARNING: make sure the direction matches with OSCILLATORY_AMPLITUDE definition
MAX_EXPECTED_BOUNDARY_POS_OSCILLATORY = 0.25 * (BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3]) + BOUNDARY_COORDS[2]  # max pos reached at sin()=1

# Fitting parameters for the fiber strain-stiffening phenomena
# Ref: https://bio.physik.fau.de/publications/Steinwachs%20Nat%20Meth%202016.pdf
# ----------------------------------------------------------------------
BUCKLING_COEFF_D0 = 0.1
STRAIN_STIFFENING_COEFF_DS = 0.25
CRITICAL_STRAIN = 0.1

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
INCLUDE_FIBRE_NETWORK = True

MAX_CONNECTIVITY = 8 # must match hard-coded C++ values
# NOTE: These are calibrated model parameters (effective segment-level mechanics), not universal material constants.
# They depend on collagen type/concentration, crosslinking, architecture and coarse-graining choices.
FIBRE_SEGMENT_K_ELAST = 0.5  # [nN/um] Effective fibre-segment stiffness (baseline for tuning)
FIBRE_SEGMENT_D_DUMPING = 0.2  # [nN*s/um] Effective fibre-segment damping (baseline for tuning)
FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = 15 # WARNING: must match the value used in network generation
FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS = 0.05
FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE = 0.0
MAX_SEARCH_RADIUS_FNODES = FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE / 10.0 # must me smaller than FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE
FIBRE_NODE_REPULSION_K = 0.2 * FIBRE_SEGMENT_K_ELAST  # [nN/um] Short-range FNODE-FNODE exclusion stiffness (kept below segment stiffness)


# +====================================================================+
# | DIFFUSION PARAMETERS                                               |
# +====================================================================+
INCLUDE_DIFFUSION = True
N_SPECIES = 2  # number of diffusing species.WARNING: make sure that the value coincides with the one declared in TODO
DIFFUSION_COEFF_MULTI = [300.0, 300.0]  # diffusion coefficient in [um^2/s] per specie
BOUNDARY_CONC_INIT_MULTI = [[50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            # initial concentration at each surface (+X,-X,+Y,-Y,+Z,-Z) [um^2/s]. -1.0 means no condition assigned. All agents are assigned 0 by default.
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # add as many lines as different species

BOUNDARY_CONC_FIXED_MULTI = [[50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             # concentration boundary conditions at each surface. WARNING: -1.0 means initial condition prevails. Don't use 0.0 as initial condition if that value is not fixed. Use -1.0 instead
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # add as many lines as different species
HETEROGENEOUS_DIFFUSION = False  # if True, diffusion coefficient is multiplied by (1 - local ECM density) to simulate hindered diffusion through the ECM. WARNING: this is a very simple approximation of the phenomenon and highly depends on grid density (N). 
# +====================================================================+
# | CELL PARAMETERS                                                    |
# +====================================================================+
INCLUDE_CELLS = True
INCLUDE_CELL_ORIENTATION = True
INCLUDE_CELL_CELL_INTERACTION = False
INCLUDE_CELL_CYCLE = False
PERIODIC_BOUNDARIES_FOR_CELLS = False
CELL_ORIENTATION_RATE = 1.0  # [1/s] TODO: check whether cell reorient themselves faster than ECM
N_CELLS = 50
CELL_K_ELAST = 2.0  # [nN/um]
CELL_D_DUMPING = 0.4  # [nN·s/um]
CELL_RADIUS = 8.412 #ECM_ECM_EQUILIBRIUM_DISTANCE / 2 # [um]
CELL_NUCLEUS_RADIUS = CELL_RADIUS / 2 # [um]
CELL_SPEED_REF = 0.75 # [um/s] Another option is to define it according to grid distance ECM_ECM_EQUILIBRIUM_DISTANCE / TIME_STEP / X -> e.g. in how many steps a cell would move the distance between ECM agents. This is important to avoid missing interactions with ECM agents due to large jumps. WARNING: if cell speed is too high, consider increasing the number of ECM agents (N) or reducing the time step (TIME_STEP) to avoid missing interactions.
BROWNIAN_MOTION_STRENGTH = CELL_SPEED_REF / 10.0 # [um/s] Strength of random movement added to cell velocity to represent Brownian motion and other non-directed motility.
print(f'Initial cell speed reference: {CELL_SPEED_REF} um/s')   
print(f'Initial Brownian motion strength: {BROWNIAN_MOTION_STRENGTH} um/s')
CYCLE_PHASE_G1_DURATION = 10.0 #[h]
CYCLE_PHASE_S_DURATION = 8.0
CYCLE_PHASE_G2_DURATION = 4.0
CYCLE_PHASE_M_DURATION = 2.0
CYCLE_PHASE_G1_START = 0.0 #[h]
CYCLE_PHASE_S_START = CYCLE_PHASE_G1_DURATION
CYCLE_PHASE_G2_START = CYCLE_PHASE_G1_DURATION + CYCLE_PHASE_S_DURATION
CYCLE_PHASE_M_START = CYCLE_PHASE_G1_DURATION + CYCLE_PHASE_S_DURATION + CYCLE_PHASE_G2_DURATION
CELL_CYCLE_DURATION = CYCLE_PHASE_G1_DURATION + CYCLE_PHASE_S_DURATION + CYCLE_PHASE_G2_DURATION + CYCLE_PHASE_M_DURATION # typically 24h [h]
INIT_ECM_CONCENTRATION_VALS = [0.0, 0.0]  # initial concentration of each species on the ECM agents
INIT_CELL_CONCENTRATION_VALS = [15.0, 0.0]  # initial concentration of each species on the CELL agents
INIT_CELL_CONC_MASS_VALS = [x * (4/3 * 3.1415926 * CELL_RADIUS**3) for x in INIT_CELL_CONCENTRATION_VALS]  # initial mass of each species on the CELL agents
INIT_ECM_SAT_CONCENTRATION_VALS = [0.0, 10.0]  # initial saturation concentration of each species on the ECM agents
INIT_CELL_CONSUMPTION_RATES = [0.001, 0.0]  # consumption rate of each species by the CELL agents 
INIT_CELL_PRODUCTION_RATES = [0.0, 10.0]  # production rate of each species by the CELL agents 
INIT_CELL_REACTION_RATES = [0.00018, 0.00018]  # metabolic reaction rates of each species by the CELL agents 
# +====================================================================+
# | FOCAL ADHESION PARAMETERS  (units: um, s, nN)                      |
# +====================================================================+
INCLUDE_FOCAL_ADHESIONS = True
INIT_N_FOCAD_PER_CELL = 10 # initial number of focal adhesions per cell. 
N_ANCHOR_POINTS = 100 # number of anchor points to which focal adhesions can attach on the nucleus surface. Their positions change with nucleus deformation
MAX_SEARCH_RADIUS_FOCAD = 3.0 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE  # TEMP(debug attach): increased to strongly favor FA-node encounters. Reasonable baseline: 1.0 * FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE
MAX_FOCAD_ARM_LENGTH = 4 * CELL_RADIUS  # maximum length of the focal adhesion "arm" (distance between the focal adhesion and its anchor point on the nucleus). If the arm is stretched beyond this length, the focal adhesion moves back towards the anchor point. This is a simple way to represent the limited reach of cellular protrusions and avoid unrealistic stretching of focal adhesions. WARNING: make sure this value is consistent with CELL_RADIUS and MAX_SEARCH_RADIUS_FOCAD to avoid unrealistic behavior.
# WARNING: rate values below assume global timestep ~ 1.0 s
FOCAD_REST_LENGTH_0 = CELL_RADIUS - CELL_NUCLEUS_RADIUS # [um] Initial rest/target length at creation time.
FOCAD_MIN_REST_LENGTH = FOCAD_REST_LENGTH_0 / 10.0 # [um] Minimum rest length to prevent collapse. 
FOCAD_K_FA = 10.0 # [nN/um] Adhesion stiffness (effective spring constant). Typical range: ~0.1–10 nN/um; 
FOCAD_F_MAX= 0.0 # [nN] Maximum force per adhesion (cap to avoid runaway and represent myosin/structural limits).Typical range: ~5–50 nN. WARNING: 0 means "no cap" 
FOCAD_V_C = 0.2 # [um/s] Contractile shortening speed of L(t) (actomyosin-driven).
FOCAD_K_ON = 5.0 # [1/s] TEMP(debug attach): high binding rate to force attachments. Reasonable baseline: 0.01 [1/s]
FOCAD_K_OFF_0 = 0.0002 # [1/s] TEMP(debug attach): low baseline detachment to retain attachments. Reasonable baseline: 0.003 [1/s]
FOCAD_F_C = 5.0 # [nN] Force scale controlling force sensitivity in koff(F) (catch/slip style). Typical range: ~2–10 nN. Sets how quickly detachment probability changes as traction builds.
# Example (simple slip): koff(F)=K_OFF_0*exp(|F|/F_C) => faster turnover under high force.
USE_CATCH_BOND = True  # If True, use a two-pathway catch+slip off-rate instead of pure slip-bond.
CATCH_BOND_CATCH_SCALE = 4.0  # Multiplier of K_OFF_0 for catch branch (larger -> stronger stabilization window).
CATCH_BOND_SLIP_SCALE = 0.2  # Multiplier of K_OFF_0 for slip branch (larger -> faster high-force failure).
CATCH_BOND_F_CATCH = 2.0  # [nN] Force scale for catch branch exp(-|F|/F_catch).
CATCH_BOND_F_SLIP = 4.0  # [nN] Force scale for slip branch exp(+|F|/F_slip).
# Suggested starting point when USE_CATCH_BOND=True (to avoid over-stabilization):
#   FOCAD_K_ON ~ 0.02-0.1 [1/s], FOCAD_K_OFF_0 ~ 0.001-0.01 [1/s], FOCAD_K_REINF <= 0.001 [1/s].
FOCAD_K_REINF = 0.001 # [1/s] Reinforcement rate for adhesion strengthening. Timescale ~1/K_REINF = 1000 s (~17 min). E.g. something like k_fa <- k_fa + K_REINF * g(|F|) * DT, adhesions gradually stiffen over tens of minutes when they carry load.
FOCAD_F_REINF = 1.0 # [nN] Force scale for reinforcement saturation: g(F)=F/(F+F_REINF).
FOCAD_K_FA_MAX = 50.0 # [nN/um] Upper bound for reinforced adhesion stiffness.
FOCAD_K_FA_DECAY = 0.0 # [1/s] Optional decay towards baseline FOCAD_K_FA when unloaded/detached. 0 disables decay.
FOCAD_POLARITY_KON_FRONT_GAIN = 2.0  # [-] Frontness gain for attachment probability (k_on).
FOCAD_POLARITY_KOFF_FRONT_REDUCTION = 0.5  # [-] Fractional reduction of k_off_0 at the front.
FOCAD_POLARITY_KOFF_REAR_GAIN = 1.0  # [-] Fractional increase of k_off_0 at the rear.
# +====================================================================+
# | LINC coupling between cell nucleus and FOCAD                       |
# +====================================================================+
INCLUDE_LINC_COUPLING = False
LINC_K_ELAST = 10.0 # [nN/um] Effective LINC stiffness in series with FOCAD stiffness.
LINC_D_DUMPING = 0.0 # [nN·s/um] Optional damping along FOCAD-LINC axis.
LINC_REST_LENGTH = 0.0 # [um] Rest length of virtual LINC segment.

# +====================================================================+
# | NUCLEAR MECHANICS  (ONLY USED IF FOCAL ADHESIONS ARE INCLUDED)     |
# +====================================================================+
# Elasticity (small-strain linear)
NUCLEUS_E = 2.0               # [nN/µm² = kPa] Young’s modulus of the nucleus (effective stiffness). Typical: 0.5–5.0 Pa depending on cell type/lamina.
NUCLEUS_NU = 0.48             # [-] Poisson ratio. Nearly incompressible nucleus. Typical: 0.45–0.49. WARNING: must be < 0.5.
# Viscoelastic relaxation
NUCLEUS_TAU = 0.2            # [s] Relaxation time controlling how fast strain follows the instantaneous elastic strain. Typical: 10–100 s.
NUCLEUS_EPS_CLAMP = 0.30      # [-] Clamp for each strain component to preserve small-strain assumptions and avoid numerical blow-up. Typical: 0.1–0.3.
# +====================================================================+
# | CHEMOTAXIS                                                         |
# +====================================================================+
INCLUDE_CHEMOTAXIS = True
CHEMOTAXIS_SENSITIVITY = [1.0, 0.0] # [-1.0 to +1.0] Chemotactic sensitivity for each species. Positive: attraction, Negative: repulsion towards higher concentrations.
CHEMOTAXIS_ONLY_DIR = True # if True, chemotaxis only affects cell orientation, not speed. If False, chemotaxis affects both orientation and speed (e.g. by making cells move faster when they are oriented towards higher concentration gradient)
CHEMOTAXIS_CHI = 10.0 # [um^2/s] Chemotactic coefficient (χ) used to compute chemotactic velocity as v_chem = χ * ∇C. Typical range: 0.1–10 µm²/s depending on cell type and chemoattractant. Only used if CHEMOTAXIS_ONLY_DIR is False.
# +====================================================================+
# | CELL MIGRATION RELATED PARAMETERS                                  |
# +====================================================================+
INCLUDE_DUROTAXIS = True   # if True, cells prefer to move towards stiffer regions, which is implemented by making them prefer to move in the direction of maximum stress/strain. 
DUROTAXIS_ONLY_DIR = True  # if True, stress/strain direction changes movement vector (keeps speed), False: changes speed too
FOCAD_MOBILITY_MU  = 1e-4   # Mobility scaling for stress contribution (start small)
INCLUDE_ORIENTATION_ALIGN = True  # True: enable gradual alignment to principal direction
ORIENTATION_ALIGN_RATE  = 1.0  # Alignment rate [1/time] -> ~ ORIENTATION_ALIGN_RATE/TIME_STEP steps to achive full aligment
ORIENTATION_ALIGN_USE_STRESS = True  # True: align to stress eigvec1, False: align to strain eigvec1
DUROTAXIS_BLEND_BETA = 0.5   # 0: traction only, 1: principal direction only
DUROTAXIS_USE_STRESS = True   # True: use stress eigenpair, False: use strain eigenpair


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

# Checking for incompatible conditions
# ----------------------------------------------------------------------
critical_error = False
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
        file_name='network_3d.pkl',
        boundary_coords=BOUNDARY_COORDS,
        epsilon=EPSILON,
        fibre_segment_equilibrium_distance=FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE,
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
    if MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION < (2 * CELL_RADIUS):
        print('MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION: {0} must be higher than 2 * CELL_RADIUS: 2 * {1}'.format(MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION, CELL_RADIUS))
        critical_error = True
    if INCLUDE_FOCAL_ADHESIONS and not INCLUDE_FIBRE_NETWORK: 
        print('ERROR: focal adhesions cannot be included if there is no fibre network to interact with')
        critical_error = True
    if INCLUDE_FOCAL_ADHESIONS and MAX_FOCAD_ARM_LENGTH < CELL_RADIUS:
        print('ERROR: MAX_FOCAD_ARM_LENGTH: {0} must be bigger than CELL_RADIUS: {1}, as focal adhesions are initiated at the cell surface and should be able to grow away'.format(MAX_FOCAD_ARM_LENGTH, CELL_RADIUS))
elif INCLUDE_FOCAL_ADHESIONS:
    print('ERROR: focal adhesions cannot be included if there are no cells to form them')
    critical_error= True

if INCLUDE_FIBRE_NETWORK:
    print_fibre_calibration_summary(
        fibre_segment_k_elast=FIBRE_SEGMENT_K_ELAST,
        fibre_segment_d_dumping=FIBRE_SEGMENT_D_DUMPING,
        fibre_segment_equilibrium_distance=FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE,
        dt = TIME_STEP,
    )


if critical_error:
    quit()

MODEL_CONFIG = build_model_config_from_namespace(globals())
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

"""
  CELL
"""
cell_spatial_location_data_file = "cell_spatial_location_data.cpp"
cell_ecm_interaction_metabolism_file = "cell_ecm_interaction_metabolism.cpp"
cell_move_file = "cell_move.cpp"
cell_bucket_location_data_file = "cell_bucket_location_data.cpp"
cell_update_stress_file = "cell_update_stress.cpp"

"""
  FOCAD
"""
focad_bucket_location_data_file = "focad_bucket_location_data.cpp"
focad_spatial_location_data_file = "focad_spatial_location_data.cpp"
focad_anchor_update_file = "focad_anchor_update.cpp"
focad_fnode_interaction_file = "focad_fnode_interaction.cpp"
focad_move_file = "focad_move.cpp"

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
fnode_fnode_spatial_interaction_file = "fnode_fnode_spatial_interaction.cpp"
fnode_fnode_bucket_interaction_file = "fnode_fnode_bucket_interaction.cpp"
fnode_move_file = "fnode_move.cpp"
fnode_focad_interaction_file = "fnode_focad_interaction.cpp"


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
env.newPropertyUInt("AVG_NETWORK_VOXEL_DENSITY", AVG_NETWORK_VOXEL_DENSITY)
env.newPropertyArrayFloat("DIFFUSION_COEFF_MULTI", DIFFUSION_COEFF_MULTI)
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
env.newPropertyUInt("ECM_POPULATION_SIZE", ECM_POPULATION_SIZE)

# Fibre network parameters
env.newPropertyUInt("INCLUDE_FIBRE_NETWORK", INCLUDE_FIBRE_NETWORK)
env.newPropertyFloat("MAX_SEARCH_RADIUS_FNODES",MAX_SEARCH_RADIUS_FNODES)
env.newPropertyFloat("FIBRE_SEGMENT_K_ELAST",FIBRE_SEGMENT_K_ELAST)
env.newPropertyFloat("FIBRE_SEGMENT_D_DUMPING",FIBRE_SEGMENT_D_DUMPING)
env.newPropertyFloat("FIBRE_NODE_REPULSION_K", FIBRE_NODE_REPULSION_K)

# Cell properties TODO: MOVE SOME OF THESE PROPERTIES TO THE CELL AGENT 
env.newPropertyUInt("INCLUDE_CELL_ORIENTATION", INCLUDE_CELL_ORIENTATION)
env.newPropertyUInt("INCLUDE_CELL_CELL_INTERACTION", INCLUDE_CELL_CELL_INTERACTION)
env.newPropertyUInt("PERIODIC_BOUNDARIES_FOR_CELLS", PERIODIC_BOUNDARIES_FOR_CELLS)
env.newPropertyUInt("N_CELLS", N_CELLS)
env.newPropertyFloat("CELL_K_ELAST", CELL_K_ELAST)
env.newPropertyFloat("CELL_D_DUMPING", CELL_D_DUMPING)
env.newPropertyFloat("CELL_RADIUS", CELL_RADIUS)
env.newPropertyFloat("CELL_NUCLEUS_RADIUS", CELL_NUCLEUS_RADIUS)
env.newPropertyFloat("CELL_SPEED_REF", CELL_SPEED_REF)
env.newPropertyFloat("BROWNIAN_MOTION_STRENGTH", BROWNIAN_MOTION_STRENGTH)
env.newPropertyFloat("CELL_ORIENTATION_RATE", CELL_ORIENTATION_RATE)
env.newPropertyFloat("MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION", MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION)
env.newPropertyFloat("MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION", MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION)
env.newPropertyFloat("CELL_CYCLE_DURATION", CELL_CYCLE_DURATION)
env.newPropertyFloat("CYCLE_PHASE_G1_DURATION", CYCLE_PHASE_G1_DURATION)
env.newPropertyFloat("CYCLE_PHASE_S_DURATION", CYCLE_PHASE_S_DURATION)
env.newPropertyFloat("CYCLE_PHASE_G2_DURATION", CYCLE_PHASE_G2_DURATION)
env.newPropertyFloat("CYCLE_PHASE_M_DURATION", CYCLE_PHASE_M_DURATION)
env.newPropertyFloat("CYCLE_PHASE_G1_START", CYCLE_PHASE_G1_START)
env.newPropertyFloat("CYCLE_PHASE_S_START", CYCLE_PHASE_S_START)
env.newPropertyFloat("CYCLE_PHASE_G2_START", CYCLE_PHASE_G2_START)
env.newPropertyFloat("CYCLE_PHASE_M_START", CYCLE_PHASE_M_START)

# Focal adhesion properties
env.newPropertyUInt("INCLUDE_FOCAL_ADHESIONS", INCLUDE_FOCAL_ADHESIONS)
env.newPropertyUInt("INIT_N_FOCAD_PER_CELL", INIT_N_FOCAD_PER_CELL)
env.newPropertyUInt("N_ANCHOR_POINTS", N_ANCHOR_POINTS)
env.newPropertyFloat("MAX_SEARCH_RADIUS_FOCAD", MAX_SEARCH_RADIUS_FOCAD)
env.newPropertyFloat("MAX_FOCAD_ARM_LENGTH", MAX_FOCAD_ARM_LENGTH)
env.newPropertyFloat("FOCAD_REST_LENGTH_0", FOCAD_REST_LENGTH_0)
env.newPropertyFloat("FOCAD_MIN_REST_LENGTH", FOCAD_MIN_REST_LENGTH)
env.newPropertyFloat("FOCAD_K_FA", FOCAD_K_FA)
env.newPropertyFloat("FOCAD_F_MAX", FOCAD_F_MAX)
env.newPropertyFloat("FOCAD_V_C", FOCAD_V_C)
env.newPropertyFloat("FOCAD_K_ON", FOCAD_K_ON)
env.newPropertyFloat("FOCAD_K_OFF_0", FOCAD_K_OFF_0)
env.newPropertyFloat("FOCAD_F_C", FOCAD_F_C)
env.newPropertyUInt("USE_CATCH_BOND", USE_CATCH_BOND)
env.newPropertyFloat("CATCH_BOND_CATCH_SCALE", CATCH_BOND_CATCH_SCALE)
env.newPropertyFloat("CATCH_BOND_SLIP_SCALE", CATCH_BOND_SLIP_SCALE)
env.newPropertyFloat("CATCH_BOND_F_CATCH", CATCH_BOND_F_CATCH)
env.newPropertyFloat("CATCH_BOND_F_SLIP", CATCH_BOND_F_SLIP)
env.newPropertyFloat("FOCAD_K_REINF", FOCAD_K_REINF)
env.newPropertyFloat("FOCAD_F_REINF", FOCAD_F_REINF)
env.newPropertyFloat("FOCAD_K_FA_MAX", FOCAD_K_FA_MAX)
env.newPropertyFloat("FOCAD_K_FA_DECAY", FOCAD_K_FA_DECAY)
env.newPropertyFloat("FOCAD_POLARITY_KON_FRONT_GAIN", FOCAD_POLARITY_KON_FRONT_GAIN)
env.newPropertyFloat("FOCAD_POLARITY_KOFF_FRONT_REDUCTION", FOCAD_POLARITY_KOFF_FRONT_REDUCTION)
env.newPropertyFloat("FOCAD_POLARITY_KOFF_REAR_GAIN", FOCAD_POLARITY_KOFF_REAR_GAIN)
env.newPropertyUInt("INCLUDE_LINC_COUPLING", INCLUDE_LINC_COUPLING)
env.newPropertyFloat("LINC_K_ELAST", LINC_K_ELAST)
env.newPropertyFloat("LINC_D_DUMPING", LINC_D_DUMPING)
env.newPropertyFloat("LINC_REST_LENGTH", LINC_REST_LENGTH)

# Nucleus mechanical properties
env.newPropertyFloat("NUCLEUS_E", NUCLEUS_E)
env.newPropertyFloat("NUCLEUS_NU", NUCLEUS_NU)
env.newPropertyFloat("NUCLEUS_TAU", NUCLEUS_TAU)
env.newPropertyFloat("NUCLEUS_EPS_CLAMP", NUCLEUS_EPS_CLAMP)

# Chemotaxis properties
env.newPropertyUInt("INCLUDE_CHEMOTAXIS", INCLUDE_CHEMOTAXIS)
env.newPropertyFloat("CHEMOTAXIS_CHI", CHEMOTAXIS_CHI)
env.newPropertyUInt("CHEMOTAXIS_ONLY_DIR", CHEMOTAXIS_ONLY_DIR)

# Cell migration (durotaxis/orientation alignment) properties
env.newPropertyUInt("INCLUDE_DUROTAXIS", INCLUDE_DUROTAXIS)
env.newPropertyUInt("DUROTAXIS_ONLY_DIR", DUROTAXIS_ONLY_DIR)
env.newPropertyFloat("FOCAD_MOBILITY_MU", FOCAD_MOBILITY_MU)
env.newPropertyUInt("INCLUDE_ORIENTATION_ALIGN", INCLUDE_ORIENTATION_ALIGN)
env.newPropertyFloat("ORIENTATION_ALIGN_RATE", ORIENTATION_ALIGN_RATE)
env.newPropertyUInt("ORIENTATION_ALIGN_USE_STRESS", ORIENTATION_ALIGN_USE_STRESS)
env.newPropertyFloat("DUROTAXIS_BLEND_BETA", DUROTAXIS_BLEND_BETA)
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

# Other globals
env.newPropertyFloat("PI", 3.1415)
env.newPropertyUInt("DEBUG_PRINTING", DEBUG_PRINTING)
env.newPropertyUInt("DEBUG_DIFFUSION", False)
env.newPropertyFloat("EPSILON", EPSILON)
env.newPropertyUInt("MOVING_BOUNDARIES", MOVING_BOUNDARIES)

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
    if (MAX_SEARCH_RADIUS_FNODES < ECM_ECM_EQUILIBRIUM_DISTANCE) and INCLUDE_DIFFUSION and HETEROGENEOUS_DIFFUSION:
        FNODE_spatial_location_message.setRadius(ECM_ECM_EQUILIBRIUM_DISTANCE) 
    else:
        FNODE_spatial_location_message.setRadius(MAX_SEARCH_RADIUS_FNODES)  
    FNODE_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS,MIN_EXPECTED_BOUNDARY_POS)
    FNODE_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS,MAX_EXPECTED_BOUNDARY_POS)
    FNODE_spatial_location_message.newVariableInt("id") # as an edge can have multiple inner agents, this stores the position within the edge

    FNODE_bucket_location_message = model.newMessageBucket("fnode_bucket_location_message")
    # Set the range and bounds.
    # setBounds(min, max) where min and max are the min and max ids of the message buckets. This is independent of the number of agents (there can be more agents than buckets and vice versa).
    # Here, we assign one bucket per fibre node so that each fibre node can be found in its own bucket when searching for neighbours.
    FNODE_bucket_location_message.setBounds(8+1,N_NODES + 8) # +8 because domain corners have idx from 1 to 8. WARNING: make sure to initialize fibre nodes starting from index 9

    FNODE_bucket_location_message.newVariableInt("id")
    FNODE_bucket_location_message.newVariableFloat("x")
    FNODE_bucket_location_message.newVariableFloat("y")
    FNODE_bucket_location_message.newVariableFloat("z")
    FNODE_bucket_location_message.newVariableFloat("vx")
    FNODE_bucket_location_message.newVariableFloat("vy")
    FNODE_bucket_location_message.newVariableFloat("vz")
    FNODE_bucket_location_message.newVariableFloat("k_elast")
    FNODE_bucket_location_message.newVariableFloat("d_dumping")
    FNODE_bucket_location_message.newVariableArrayFloat("equilibrium_distance", MAX_CONNECTIVITY) # each segment can have a different equilibrium distance depending on the rest length assigned during network generation
    FNODE_bucket_location_message.newVariableArrayInt("linked_nodes", MAX_CONNECTIVITY) # store the index of the linked nodes, which is a proxy for the bucket id


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
        
    # Set the range and bounds.
    if INCLUDE_FOCAL_ADHESIONS:
        CELL_bucket_location_message = model.newMessageBucket("cell_bucket_location_message")
        cell_bucket_min = 8 + N_NODES + 1
        cell_bucket_max = 8 + N_NODES + N_CELLS
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
BCORNER_agent.newVariableInt("id")
BCORNER_agent.newVariableFloat("x")
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
    FNODE_agent.newVariableInt("id")
    FNODE_agent.newVariableFloat("x")
    FNODE_agent.newVariableFloat("y")
    FNODE_agent.newVariableFloat("z")
    FNODE_agent.newVariableFloat("vx", 0.0)
    FNODE_agent.newVariableFloat("vy", 0.0)
    FNODE_agent.newVariableFloat("vz", 0.0)
    FNODE_agent.newVariableFloat("fx", 0.0)
    FNODE_agent.newVariableFloat("fy", 0.0)
    FNODE_agent.newVariableFloat("fz", 0.0)
    FNODE_agent.newVariableFloat("k_elast")
    FNODE_agent.newVariableFloat("d_dumping")
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
    FNODE_agent.newVariableFloat("f_extension")
    FNODE_agent.newVariableFloat("f_compression")
    FNODE_agent.newVariableFloat("elastic_energy")
    FNODE_agent.newVariableArrayFloat("linked_nodes", MAX_CONNECTIVITY)
    FNODE_agent.newVariableUInt8("clamped_bx_pos")
    FNODE_agent.newVariableUInt8("clamped_bx_neg")
    FNODE_agent.newVariableUInt8("clamped_by_pos")
    FNODE_agent.newVariableUInt8("clamped_by_neg")
    FNODE_agent.newVariableUInt8("clamped_bz_pos")
    FNODE_agent.newVariableUInt8("clamped_bz_neg")

    FNODE_agent.newRTCFunctionFile("fnode_spatial_location_data", fnode_spatial_location_data_file).setMessageOutput("fnode_spatial_location_message")
    FNODE_agent.newRTCFunctionFile("fnode_bucket_location_data", fnode_bucket_location_data_file).setMessageOutput("fnode_bucket_location_message")
    FNODE_agent.newRTCFunctionFile("fnode_boundary_interaction", fnode_boundary_interaction_file)
    FNODE_agent.newRTCFunctionFile("fnode_fnode_spatial_interaction", fnode_fnode_spatial_interaction_file).setMessageInput("fnode_spatial_location_message")
    FNODE_agent.newRTCFunctionFile("fnode_fnode_bucket_interaction", fnode_fnode_bucket_interaction_file).setMessageInput("fnode_bucket_location_message")
    FNODE_agent.newRTCFunctionFile("fnode_move", fnode_move_file)
    if INCLUDE_FOCAL_ADHESIONS:
        FNODE_agent.newRTCFunctionFile("fnode_focad_interaction", fnode_focad_interaction_file).setMessageInput("focad_spatial_location_message")


"""
  ECM agent
"""
ECM_agent = model.newAgent("ECM")
ECM_agent.newVariableInt("id", 0)
ECM_agent.newVariableFloat("x", 0.0)
ECM_agent.newVariableFloat("y", 0.0)
ECM_agent.newVariableFloat("z", 0.0)
ECM_agent.newVariableInt("grid_lin_id", 0) # linear index in the 3D grid that maps to i,j,k positions
ECM_agent.newVariableUInt8("grid_i", 0)
ECM_agent.newVariableUInt8("grid_j", 0)
ECM_agent.newVariableUInt8("grid_k", 0)
ECM_agent.newVariableArrayFloat("D_sp", N_SPECIES)  # diffusion coefficient of each species at the agent location (used for heterogeneous diffusion)
ECM_agent.newVariableArrayFloat("C_sp", N_SPECIES) 
ECM_agent.newVariableArrayFloat("C_sp_sat", N_SPECIES) 
ECM_agent.newVariableFloat("k_elast")
ECM_agent.newVariableFloat("d_dumping")
ECM_agent.newVariableFloat("vx")
ECM_agent.newVariableFloat("vy")
ECM_agent.newVariableFloat("vz")
ECM_agent.newVariableFloat("fx")
ECM_agent.newVariableFloat("fy")
ECM_agent.newVariableFloat("fz")
ECM_agent.newVariableUInt8("clamped_bx_pos")
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
if MOVING_BOUNDARIES:
    ECM_agent.newRTCFunctionFile("ecm_move", ecm_move_file)

"""
  CELL agent
"""
if INCLUDE_CELLS:
    CELL_agent = model.newAgent("CELL")
    CELL_agent.newVariableInt("id", 0)
    CELL_agent.newVariableFloat("x", 0.0)
    CELL_agent.newVariableFloat("y", 0.0)
    CELL_agent.newVariableFloat("z", 0.0)
    CELL_agent.newVariableFloat("vx", 0.0)
    CELL_agent.newVariableFloat("vy", 0.0)
    CELL_agent.newVariableFloat("vz", 0.0)
    CELL_agent.newVariableFloat("orx")
    CELL_agent.newVariableFloat("ory")
    CELL_agent.newVariableFloat("orz")
    CELL_agent.newVariableFloat("k_elast")
    CELL_agent.newVariableFloat("d_dumping")
    CELL_agent.newVariableFloat("alignment", 0.0)
    CELL_agent.newVariableArrayFloat("k_consumption", N_SPECIES) 
    CELL_agent.newVariableArrayFloat("k_production", N_SPECIES) 
    CELL_agent.newVariableArrayFloat("k_reaction", N_SPECIES) 
    CELL_agent.newVariableArrayFloat("C_sp", N_SPECIES) 
    CELL_agent.newVariableArrayFloat("M_sp", N_SPECIES) 
    CELL_agent.newVariableFloat("speed_ref", CELL_SPEED_REF)   
    CELL_agent.newVariableFloat("radius", CELL_RADIUS)
    CELL_agent.newVariableFloat("nucleus_radius", CELL_NUCLEUS_RADIUS)
    CELL_agent.newVariableInt("cycle_phase", 1) # [1:G1] [2:S] [3:G2] [4:M]
    CELL_agent.newVariableFloat("clock", 0.0) # internal clock of the cell to switch phases
    CELL_agent.newVariableInt("completed_cycles", 0)
    CELL_agent.newRTCFunctionFile("cell_spatial_location_data", cell_spatial_location_data_file).setMessageOutput("cell_spatial_location_message")
    CELL_agent.newRTCFunctionFile("cell_ecm_interaction_metabolism", cell_ecm_interaction_metabolism_file).setMessageInput("ecm_grid_location_message")
    CELL_agent.newRTCFunctionFile("cell_move", cell_move_file)
    CELL_agent.newVariableArrayFloat("x_i", N_ANCHOR_POINTS) # store the position of the anchor points on the cell. Unused if INCLUDE_FOCAL_ADHESIONS is False
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
    CELL_agent.newVariableFloat("sig_xx", 0.0) # stress tensor
    CELL_agent.newVariableFloat("sig_yy", 0.0)
    CELL_agent.newVariableFloat("sig_zz", 0.0)
    CELL_agent.newVariableFloat("sig_xy", 0.0)
    CELL_agent.newVariableFloat("sig_xz", 0.0)
    CELL_agent.newVariableFloat("sig_yz", 0.0)
    CELL_agent.newVariableFloat("sig_eig_1", 0.0)
    CELL_agent.newVariableFloat("sig_eig_2", 0.0)
    CELL_agent.newVariableFloat("sig_eig_3", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec1_x", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec1_y", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec1_z", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec2_x", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec2_y", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec2_z", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec3_x", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec3_y", 0.0)
    CELL_agent.newVariableFloat("sig_eigvec3_z", 0.0)
    CELL_agent.newVariableFloat("eps_eig_1", 0.0)
    CELL_agent.newVariableFloat("eps_eig_2", 0.0)
    CELL_agent.newVariableFloat("eps_eig_3", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec1_x", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec1_y", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec1_z", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec2_x", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec2_y", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec2_z", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec3_x", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec3_y", 0.0)
    CELL_agent.newVariableFloat("eps_eigvec3_z", 0.0)
    CELL_agent.newVariableArrayFloat("chemotaxis_sensitivity", N_SPECIES)
    if INCLUDE_FOCAL_ADHESIONS:  
        CELL_agent.newRTCFunctionFile("cell_bucket_location_data", cell_bucket_location_data_file).setMessageOutput("cell_bucket_location_message")
        CELL_agent.newRTCFunctionFile("cell_update_stress", cell_update_stress_file).setMessageInput("focad_bucket_location_message")
        
"""
  FOCAD agent
"""
if INCLUDE_FOCAL_ADHESIONS:
    FOCAD_agent = model.newAgent("FOCAD")
    FOCAD_agent.newVariableInt("id", 0)
    FOCAD_agent.newVariableInt("cell_id")
    FOCAD_agent.newVariableInt("fnode_id")    
    FOCAD_agent.newVariableFloat("x", 0.0)
    FOCAD_agent.newVariableFloat("y", 0.0)
    FOCAD_agent.newVariableFloat("z", 0.0)
    FOCAD_agent.newVariableFloat("vx", 0.0)
    FOCAD_agent.newVariableFloat("vy", 0.0)
    FOCAD_agent.newVariableFloat("vz", 0.0)
    FOCAD_agent.newVariableFloat("fx", 0.0)
    FOCAD_agent.newVariableFloat("fy", 0.0)
    FOCAD_agent.newVariableFloat("fz", 0.0)
    FOCAD_agent.newVariableInt("anchor_id",-1)
    FOCAD_agent.newVariableFloat("x_i", 0.0)
    FOCAD_agent.newVariableFloat("y_i", 0.0)
    FOCAD_agent.newVariableFloat("z_i", 0.0)
    FOCAD_agent.newVariableFloat("x_c", 0.0)
    FOCAD_agent.newVariableFloat("y_c", 0.0)
    FOCAD_agent.newVariableFloat("z_c", 0.0)
    FOCAD_agent.newVariableFloat("orx", 1.0)
    FOCAD_agent.newVariableFloat("ory", 0.0)
    FOCAD_agent.newVariableFloat("orz", 0.0)
    FOCAD_agent.newVariableFloat("rest_length_0")
    FOCAD_agent.newVariableFloat("rest_length")
    FOCAD_agent.newVariableFloat("k_fa")
    FOCAD_agent.newVariableFloat("f_max")
    FOCAD_agent.newVariableInt("attached")
    FOCAD_agent.newVariableUInt8("active")
    FOCAD_agent.newVariableFloat("v_c")
    FOCAD_agent.newVariableUInt8("fa_state")
    FOCAD_agent.newVariableFloat("age")
    FOCAD_agent.newVariableFloat("k_on")
    FOCAD_agent.newVariableFloat("k_off_0")
    FOCAD_agent.newVariableFloat("f_c")
    FOCAD_agent.newVariableFloat("k_reinf")
    FOCAD_agent.newVariableFloat("f_mag", 0.0)  # |F_FA| traction magnitude [nN] at current step
    FOCAD_agent.newVariableInt("is_front", 0)  # 1 if adhesion is classified in the cell front hemisphere, else 0
    FOCAD_agent.newVariableInt("is_rear", 0)  # 1 if adhesion is classified in the cell rear hemisphere, else 0
    FOCAD_agent.newVariableInt("attached_front", 0)  # 1 if attached and in front; diagnostic aggregate helper
    FOCAD_agent.newVariableInt("attached_rear", 0)  # 1 if attached and in rear; diagnostic aggregate helper
    FOCAD_agent.newVariableFloat("frontness_front", 0.0)  # frontness score used for front-biased kinetics (front branch) -> Polarity score p in [-1,1] from orientation vs anchor direction (cell center -> anchor)
    FOCAD_agent.newVariableFloat("frontness_rear", 0.0)  # rearness/frontness-derived score used for rear-biased kinetics
    FOCAD_agent.newVariableFloat("k_on_eff_front", 0.0)  # effective attachment rate used for front-side update [1/s]
    FOCAD_agent.newVariableFloat("k_on_eff_rear", 0.0)  # effective attachment rate used for rear-side update [1/s]
    FOCAD_agent.newVariableFloat("k_off_0_eff_front", 0.0)  # effective baseline detachment rate at front [1/s]
    FOCAD_agent.newVariableFloat("k_off_0_eff_rear", 0.0)  # effective baseline detachment rate at rear [1/s]
    FOCAD_agent.newVariableFloat("linc_prev_total_length", 0.0)  # previous-step LINC internal length state for BE Kelvin-Voigt-in-series solve [um]


    FOCAD_agent.newRTCFunctionFile("focad_bucket_location_data", focad_bucket_location_data_file).setMessageOutput("focad_bucket_location_message")
    FOCAD_agent.newRTCFunctionFile("focad_spatial_location_data", focad_spatial_location_data_file).setMessageOutput("focad_spatial_location_message")
    FOCAD_agent.newRTCFunctionFile("focad_anchor_update", focad_anchor_update_file).setMessageInput("cell_bucket_location_message")
    FOCAD_agent.newRTCFunctionFile("focad_fnode_interaction", focad_fnode_interaction_file).setMessageInput("fnode_spatial_location_message")
    FOCAD_agent.newRTCFunctionFile("focad_move", focad_move_file).setMessageInput("fnode_bucket_location_message")        



"""
  Population initialisation functions
"""

# Agent population initialization 
# ----------------------------------------------------------------------    
# IMPORTANT NOTE: agents must be initialized in the following order to make sure that their ids are consistent with the assumptions made in the RTC functions and bucket message bounds:
# 1) Boundary corners (idx 1 to 8)
# 2) Fibre nodes (idx 9 to 8+N_NODES) if INCLUDE_FIBRE_NETWORK is True
# 3) Cell agents (idx 8+N_NODES+1 to 8+N_NODES+N_CELLS)
# 4) Focal adhesions (idx 8+N_NODES+N_CELLS+1 to 8+N_NODES+N_CELLS+(INIT_N_FOCAD_PER_CELL*N_CELLS)) if INCLUDE_FOCAL_ADHESIONS is True.
# 5) ECM agents (idx starting from 8+N_NODES+N_CELLS+(INIT_N_FOCAD_PER_CELL*N_CELLS)+1)
class initAgentPopulations(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        global INCLUDE_CELLS, N_CELLS, INIT_CELL_CONCENTRATION_VALS, INIT_CELL_REACTION_RATES
        global INIT_CELL_CONC_MASS_VALS, INIT_CELL_CONSUMPTION_RATES, INIT_CELL_PRODUCTION_RATES
        global INCLUDE_FOCAL_ADHESIONS, N_ANCHOR_POINTS, INIT_N_FOCAD_PER_CELL, CELL_RADIUS, CELL_NUCLEUS_RADIUS
        global FOCAD_REST_LENGTH_0, FOCAD_K_FA, FOCAD_F_MAX, FOCAD_K_ON, FOCAD_K_OFF_0, FOCAD_F_C, FOCAD_K_REINF
        global INCLUDE_DIFFUSION, N_SPECIES, DIFFUSION_COEFF_MULTI
        global INIT_ECM_CONCENTRATION_VALS, INIT_ECM_SAT_CONCENTRATION_VALS
        global INCLUDE_FIBRE_NETWORK, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, MAX_CONNECTIVITY
        # BOUNDARY CORNERS
        current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
        coord_boundary = FLAMEGPU.environment.getPropertyArrayFloat("COORDS_BOUNDARIES")
        coord_boundary_x_pos = coord_boundary[0]
        coord_boundary_x_neg = coord_boundary[1]
        coord_boundary_y_pos = coord_boundary[2]
        coord_boundary_y_neg = coord_boundary[3]
        coord_boundary_z_pos = coord_boundary[4]
        coord_boundary_z_neg = coord_boundary[5]
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
                instance.setVariableUInt8("clamped_bx_pos", 0)
                instance.setVariableUInt8("clamped_bx_neg", 0)
                instance.setVariableUInt8("clamped_by_pos", 0)
                instance.setVariableUInt8("clamped_by_neg", 0)
                instance.setVariableUInt8("clamped_bz_pos", 0)
                instance.setVariableUInt8("clamped_bz_neg", 0)
                instance.setVariableArrayFloat("linked_nodes", linked_nodes.tolist())            


            FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id + count)
            
        # CELLS
        if INCLUDE_CELLS:
            current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
            current_id += 1
            print(f"--- Initializing CELLS ({N_CELLS})")
            print("  |-> current_id:", current_id)
            count = -1
            cell_pos = getRandomCoords3D(N_CELLS,
                                        coord_boundary[0], coord_boundary[1],
                                        coord_boundary[2], coord_boundary[3],
                                        coord_boundary[4], coord_boundary[5])
            if N_CELLS == 1: # DEBUGGING. FIX CELL POSITION TO 0,0,0
                cell_pos = np.array([[0.0, 0.0, 0.0]], dtype=float) # for testing with 1 cell. 
            cell_orientations = getRandomVectors3D(N_CELLS)
            k_elast = FLAMEGPU.environment.getPropertyFloat("CELL_K_ELAST")
            d_dumping = FLAMEGPU.environment.getPropertyFloat("CELL_D_DUMPING")
            cell_id_list = []
            for i in range(N_CELLS):
                count += 1
                cell_id_list.append(current_id + count) # store the cell ids in a list to be used for focal adhesion initialization if INCLUDE_FOCAL_ADHESIONS is True
                instance = FLAMEGPU.agent("CELL").newAgent()
                instance.setVariableInt("id", current_id + count)
                instance.setVariableFloat("x", cell_pos[i, 0])
                instance.setVariableFloat("y", cell_pos[i, 1])
                instance.setVariableFloat("z", cell_pos[i, 2])
                instance.setVariableFloat("vx", 0.0)
                instance.setVariableFloat("vy", 0.0)
                instance.setVariableFloat("vz", 0.0)
                instance.setVariableFloat("orx", cell_orientations[i, 0])
                instance.setVariableFloat("ory", cell_orientations[i, 1])
                instance.setVariableFloat("orz", cell_orientations[i, 2])
                instance.setVariableFloat("alignment", 0.0)
                instance.setVariableFloat("k_elast", k_elast)
                instance.setVariableFloat("d_dumping", d_dumping)
                instance.setVariableArrayFloat("C_sp", INIT_CELL_CONCENTRATION_VALS)
                instance.setVariableArrayFloat("M_sp", INIT_CELL_CONC_MASS_VALS)                
                instance.setVariableArrayFloat("k_consumption", INIT_CELL_CONSUMPTION_RATES)
                instance.setVariableArrayFloat("k_production", INIT_CELL_PRODUCTION_RATES)
                instance.setVariableArrayFloat("k_reaction", INIT_CELL_REACTION_RATES)
                instance.setVariableFloat("radius", CELL_RADIUS)
                instance.setVariableFloat("nucleus_radius", CELL_NUCLEUS_RADIUS)
                instance.setVariableFloat("speed_ref", CELL_SPEED_REF)
                cycle_phase = random.randint(1, 4) # [1:G1] [2:S] [3:G2] [4:M]
                instance.setVariableInt("cycle_phase", cycle_phase)
                cycle_clock = 0.0
                if cycle_phase == 1:
                    cycle_clock = FLAMEGPU.environment.getPropertyFloat("CYCLE_PHASE_G1_START") 
                    + np.random.uniform(0.0, 1.0) * FLAMEGPU.environment.getPropertyFloat("CYCLE_PHASE_G1_DURATION")                
                elif cycle_phase == 2:
                    cycle_clock = FLAMEGPU.environment.getPropertyFloat("CYCLE_PHASE_S_START")
                    + np.random.uniform(0.0, 1.0) * FLAMEGPU.environment.getPropertyFloat("CYCLE_PHASE_S_DURATION")                    
                elif cycle_phase == 3:
                    cycle_clock = FLAMEGPU.environment.getPropertyFloat("CYCLE_PHASE_G2_START")
                    + np.random.uniform(0.0, 1.0) * FLAMEGPU.environment.getPropertyFloat("CYCLE_PHASE_G2_DURATION")                    
                elif cycle_phase == 4:
                    cycle_clock = FLAMEGPU.environment.getPropertyFloat("CYCLE_PHASE_M_START")
                    + np.random.uniform(0.0, 1.0) * FLAMEGPU.environment.getPropertyFloat("CYCLE_PHASE_M_DURATION")                    
                instance.setVariableFloat("clock", cycle_clock)
                instance.setVariableInt("completed_cycles",0)
                
                anchor_pos = getRandomCoordsAroundPoint(N_ANCHOR_POINTS, cell_pos[i, 0], cell_pos[i, 1], cell_pos[i, 2], CELL_NUCLEUS_RADIUS, on_surface=True)
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
                instance.setVariableArrayFloat("chemotaxis_sensitivity", CHEMOTAXIS_SENSITIVITY) 


            FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id + count)
            
        if INCLUDE_FOCAL_ADHESIONS:
            current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
            current_id += 1
            print(f"--- Initializing FOCAL ADHESIONS ({N_CELLS * INIT_N_FOCAD_PER_CELL})")
            print("  |-> current_id:", current_id)
            count = -1
            for i in range(N_CELLS):
                focad_pos = getRandomCoordsAroundPoint(INIT_N_FOCAD_PER_CELL, cell_pos[i, 0], cell_pos[i, 1], cell_pos[i, 2], CELL_RADIUS, on_surface=True)
                for j in range(INIT_N_FOCAD_PER_CELL):
                    count += 1
                    instance = FLAMEGPU.agent("FOCAD").newAgent()
                    instance.setVariableInt("id", current_id + count)
                    instance.setVariableInt("fnode_id", -1) # initialized as not attached to any fibre node
                    instance.setVariableInt("cell_id", cell_id_list[i])
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
                    anchor_pos = cell_pos[i, :] + (focad_dir / np.linalg.norm(focad_dir)) * CELL_NUCLEUS_RADIUS
                    instance.setVariableFloat("x_i", anchor_pos[0])
                    instance.setVariableFloat("y_i", anchor_pos[1])
                    instance.setVariableFloat("z_i", anchor_pos[2])
                    instance.setVariableFloat("x_c", cell_pos[i, 0])
                    instance.setVariableFloat("y_c", cell_pos[i, 1])
                    instance.setVariableFloat("z_c", cell_pos[i, 2])
                    instance.setVariableFloat("orx", cell_orientations[i, 0])
                    instance.setVariableFloat("ory", cell_orientations[i, 1])
                    instance.setVariableFloat("orz", cell_orientations[i, 2])
                    instance.setVariableFloat("rest_length_0", FOCAD_REST_LENGTH_0)
                    instance.setVariableFloat("rest_length", FOCAD_REST_LENGTH_0) # initialized at rest length, can be updated during the simulation if needed
                    instance.setVariableFloat("k_fa", FOCAD_K_FA)
                    instance.setVariableFloat("f_max", FOCAD_F_MAX) # WARNING: 0 means "no cap" 
                    instance.setVariableInt("attached", 0) # initialized as not attached
                    instance.setVariableUInt8("active", 1) # initialized as active (can form new attachments)
                    instance.setVariableFloat("v_c", FOCAD_V_C)
                    instance.setVariableUInt8("fa_state", 1) # [1: nascent] [2: mature] [3: disassembling]
                    instance.setVariableFloat("age", 0.0)
                    instance.setVariableFloat("k_on", FOCAD_K_ON)
                    instance.setVariableFloat("k_off_0", FOCAD_K_OFF_0)
                    instance.setVariableFloat("f_c", FOCAD_F_C)
                    instance.setVariableFloat("k_reinf", FOCAD_K_REINF)
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



class SaveDataToFile(pyflamegpu.HostFunction):
    def __init__(self):
        global ECM_AGENTS_PER_DIR, INCLUDE_FIBRE_NETWORK, N_NODES
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
        global INCLUDE_FOCAL_ADHESIONS
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
                "pyflamegpu": pyflamegpu,
            },
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

        print(
            f"FA metrics (step {stepCounter:04d}) -> attached={int(attached_count)}/{n_focad} "
            f"(ratio={attached_ratio:.3f}), mean|F|={mean_force_mag:.4f} nN"
        )


                          

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

sdf = SaveDataToFile()
# SaveDataToFile host function; behavior is controlled by SAVE_DATA_TO_FILE flag.
model.addStepFunction(sdf)

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
"""

# L1: Agent_Locations
model.newLayer("L1_Agent_Locations").addAgentFunction("BCORNER", "bcorner_output_location_data")
if INCLUDE_DIFFUSION:
    model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_grid_location_data")
if INCLUDE_CELLS:
    model.Layer("L1_Agent_Locations").addAgentFunction("CELL", "cell_spatial_location_data")
    if INCLUDE_FOCAL_ADHESIONS:
        model.newLayer("L1_CELL_Locations_2").addAgentFunction("CELL", "cell_bucket_location_data")  # these functions share data of the same agent, so must be in separate layers
        model.newLayer("L1_FOCAD_Update_Anchors").addAgentFunction("FOCAD", "focad_anchor_update")
if INCLUDE_FIBRE_NETWORK:
    model.newLayer("L1_FNODE_Locations_1").addAgentFunction("FNODE", "fnode_spatial_location_data")
    # These functions share data of the same agent, so must be in separate layers
    model.newLayer("L1_FNODE_Locations_2").addAgentFunction("FNODE", "fnode_bucket_location_data")
    

# L2: Boundary_Interactions  
if INCLUDE_DIFFUSION:
    model.newLayer("L2_ECM_Boundary_Interactions").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
if INCLUDE_FIBRE_NETWORK:
    model.newLayer("L2_FNODE_Boundary_Interactions").addAgentFunction("FNODE", "fnode_boundary_interaction")

if INCLUDE_CELLS and INCLUDE_DIFFUSION:
    # L3_Metabolism
    model.newLayer("L3_Metabolism").addAgentFunction("CELL", "cell_ecm_interaction_metabolism")
if INCLUDE_DIFFUSION:
    # L4_ECM_Csp_Update
    model.newLayer("L4_ECM_Csp_Update").addAgentFunction("ECM", "ecm_Csp_update")
    if HETEROGENEOUS_DIFFUSION and INCLUDE_FIBRE_NETWORK:
        model.newLayer("L4_ECM_Dsp_Update").addAgentFunction("ECM", "ecm_Dsp_update")
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
        model.newLayer("L7_CELL_Stress_Update").addAgentFunction("CELL", "cell_update_stress") 

# L8_Agent_Movement
if INCLUDE_CELLS:
    model.newLayer("L8_CELL_Movement").addAgentFunction("CELL", "cell_move")
if INCLUDE_FIBRE_NETWORK:
    model.newLayer("L8_FNODE_Movement").addAgentFunction("FNODE", "fnode_move")
if INCLUDE_FOCAL_ADHESIONS:
    model.newLayer("L8_FOCAD_Movement").addAgentFunction("FOCAD", "focad_move")
# If boundaries are not moving, the ECM grid does not need to be updated
if MOVING_BOUNDARIES:
    model.newLayer("L8_BCORNER_Movement").addAgentFunction("BCORNER", "bcorner_move")
    model.newLayer("L8_ECM_Movement").addAgentFunction("ECM", "ecm_move")
    
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
if ENSEMBLE:
    # Execute the ensemble using the specified RunPlans
    errs = ensemble.simulate(ensemble_runs)
else:
    simulation.simulate()


if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
    vis.join() # join the visualisation thread and stops the visualisation closing after the simulation finishes

EXECUTION_TIME = time.time() - start_time
print("--- EXECUTION TIME: %s seconds ---" % EXECUTION_TIME)

incL_dir1 = (BPOS_OVER_TIME.iloc[:, POISSON_DIRS[0] * 2] - BPOS_OVER_TIME.iloc[:, POISSON_DIRS[0] * 2 + 1]) - (
        BPOS_OVER_TIME.iloc[0, POISSON_DIRS[0] * 2] - BPOS_OVER_TIME.iloc[0, POISSON_DIRS[0] * 2 + 1])
incL_dir2 = (BPOS_OVER_TIME.iloc[:, POISSON_DIRS[1] * 2] - BPOS_OVER_TIME.iloc[:, POISSON_DIRS[1] * 2 + 1]) - (
        BPOS_OVER_TIME.iloc[0, POISSON_DIRS[1] * 2] - BPOS_OVER_TIME.iloc[0, POISSON_DIRS[1] * 2 + 1])

POISSON_RATIO_OVER_TIME = -1 * incL_dir1 / incL_dir2


def manageLogs(steps, is_ensemble, idx):
    global SAVE_EVERY_N_STEPS, SAVE_PICKLE, SHOW_PLOTS, RES_PATH, MODEL_CONFIG, EXECUTION_TIME
    global BPOS_OVER_TIME, BFORCE_OVER_TIME, BFORCE_SHEAR_OVER_TIME, POISSON_RATIO_OVER_TIME, OSCILLATORY_STRAIN_OVER_TIME
    ecm_agent_counts = [None] * len(steps)
    counter = 0
    BFORCE = make_dataclass("BFORCE",
                            [("fxpos", float), ("fxneg", float), ("fypos", float), ("fyneg", float), ("fzpos", float),
                             ("fzneg", float)])
    BFORCE_SHEAR = make_dataclass("BFORCE_SHEAR",
                                  [("fxpos_y", float), ("fxpos_z", float), ("fxneg_y", float), ("fxneg_z", float),
                                   ("fypos_x", float), ("fypos_z", float), ("fyneg_x", float), ("fyneg_z", float),
                                   ("fzpos_x", float), ("fzpos_y", float), ("fzneg_x", float), ("fzneg_y", float)])
    BFORCE_OVER_TIME = []
    BFORCE_SHEAR_OVER_TIME = []
    FOCAD_METRICS_OVER_TIME = []
    FOCAD_POLARITY_METRICS_OVER_TIME = []

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

                step_bforce = pd.DataFrame([BFORCE(f_bx_pos, f_bx_neg, f_by_pos, f_by_neg, f_bz_pos, f_bz_neg)])
                step_bforce_shear = pd.DataFrame([BFORCE_SHEAR(f_bx_pos_y, f_bx_pos_z, f_bx_neg_y, f_bx_neg_z,
                                                            f_by_pos_x, f_by_pos_z, f_by_neg_x, f_by_neg_z,
                                                            f_bz_pos_x, f_bz_pos_y, f_bz_neg_x, f_bz_neg_y)])
                if counter == 0:
                    BFORCE_OVER_TIME = pd.DataFrame([BFORCE(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
                    BFORCE_SHEAR_OVER_TIME = pd.DataFrame(
                        [BFORCE_SHEAR(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
                else:
                    # BFORCE_OVER_TIME = BFORCE_OVER_TIME.append(step_bforce, ignore_index=True) # deprecated
                    BFORCE_OVER_TIME = pd.concat([BFORCE_OVER_TIME, step_bforce], ignore_index=True)
                    # BFORCE_SHEAR_OVER_TIME = BFORCE_SHEAR_OVER_TIME.append(step_bforce_shear, ignore_index=True) # deprecated
                    BFORCE_SHEAR_OVER_TIME = pd.concat([BFORCE_SHEAR_OVER_TIME, step_bforce_shear], ignore_index=True)
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
    # Saving pickle
    if SAVE_PICKLE:
        file_name = f'output_data_{idx}.pickle'
        file_path = RES_PATH / file_name
        with open(str(file_path), 'wb') as file:
            pickle.dump({'BPOS_OVER_TIME': BPOS_OVER_TIME,
                         'BFORCE_OVER_TIME': BFORCE_OVER_TIME,
                         'BFORCE_SHEAR_OVER_TIME': BFORCE_SHEAR_OVER_TIME,
                         'FOCAD_METRICS_OVER_TIME': FOCAD_METRICS_OVER_TIME,
                         'FOCAD_POLARITY_METRICS_OVER_TIME': FOCAD_POLARITY_METRICS_OVER_TIME,
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
if ENSEMBLE:
    logs = simulation.getLogs()
    for i in range(len(logs)):
        steps = logs[i].getStepLog()
        manageLogs(steps, ENSEMBLE, i)
else:
    logs = simulation.getRunLog()
    steps = logs.getStepLog()
    manageLogs(steps, ENSEMBLE, 0)