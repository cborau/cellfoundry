# +====================================================================+
# | Model: metabolism                                                  |
# | Last update: 02/02/2026 - 13:28:03                                 |
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
from helper_module import compute_expected_boundary_pos_from_corners, getRandomVectors3D, build_model_config_from_namespace, load_fibre_network


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
SAVE_EVERY_N_STEPS = 5 # Affects both the .vtk files and the Dataframes storing boundary data

CURR_PATH = pathlib.Path().absolute()
RES_PATH = CURR_PATH / 'result_files'
RES_PATH.mkdir(parents=True, exist_ok=True)
EPSILON = 0.0000000001

print("Executing in ", CURR_PATH)
# Minimum number of agents per direction (x,y,z). 
# If domain is not cubical, N is asigned to the shorter dimension and more agents are added to the longer ones
# ----------------------------------------------------------------------
N = 21

# Time simulation parameters
# ----------------------------------------------------------------------
TIME_STEP = 0.01# time. WARNING: diffusion and cell migration events might need different scales
STEPS = 50

# +====================================================================+
# | BOUNDARY CONDITIONS                                                |
# +====================================================================+

# Boundary interactions and mechanical parameters
# ----------------------------------------------------------------------
ECM_K_ELAST = 0.2  # [N/units/kg]
ECM_D_DUMPING = 0.04  # [N*s/units/kg]
ECM_ETA = 1  # [1/time]

#BOUNDARY_COORDS = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]  # +X,-X,+Y,-Y,+Z,-Z
BOUNDARY_COORDS = [100.0, -100.0, 100.0, -100.0, 100.0, -100.0]# microdevice dimensions in um
#BOUNDARY_COORDS = [coord / 1000.0 for coord in BOUNDARY_COORDS] # in mm
BOUNDARY_DISP_RATES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# perpendicular to each surface (+X,-X,+Y,-Y,+Z,-Z) [units/time]
BOUNDARY_DISP_RATES_PARALLEL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# parallel to each surface (+X_y,+X_z,-X_y,-X_z,+Y_x,+Y_z,-Y_x,-Y_z,+Z_x,+Z_y,-Z_x,-Z_y)[units/time]

POISSON_DIRS = [0, 1]  # 0: xdir, 1:ydir, 2:zdir. poisson_ratio ~= -incL(dir1)/incL(dir2) dir2 is the direction in which the load is applied
ALLOW_BOUNDARY_ELASTIC_MOVEMENT = [0, 0, 0, 0, 0, 0]  # [bool]
RELATIVE_BOUNDARY_STIFFNESS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
BOUNDARY_STIFFNESS_VALUE = 10.0  # N/units
BOUNDARY_DUMPING_VALUE = 5.0
BOUNDARY_STIFFNESS = [BOUNDARY_STIFFNESS_VALUE * x for x in RELATIVE_BOUNDARY_STIFFNESS]
BOUNDARY_DUMPING = [BOUNDARY_DUMPING_VALUE * x for x in RELATIVE_BOUNDARY_STIFFNESS]
#CLAMP_AGENT_TOUCHING_BOUNDARY = [0, 0, 1, 1, 0, 0]# +X,-X,+Y,-Y,+Z,-Z [bool] - shear assay
CLAMP_AGENT_TOUCHING_BOUNDARY = [1, 1, 1, 1, 1, 1]# +X,-X,+Y,-Y,+Z,-Z [bool]
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
MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION = 2 * ECM_ECM_EQUILIBRIUM_DISTANCE # this radius is used to check if cells interact with each other

OSCILLATORY_SHEAR_ASSAY = False  # if True, BOUNDARY_DISP_RATES_PARALLEL options are overrun but used to make the boundaries oscillate in their corresponding planes following a sin() function
MAX_STRAIN = 0.25  # maximum strain applied during oscillatory shear assay (used to compute OSCILLATORY_AMPLITUDE)
OSCILLATORY_AMPLITUDE = MAX_STRAIN * (BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3])  # range [0-1] * domain size in the direction of oscillation
OSCILLATORY_FREQ = 0.05  # strain oscillation frequency [time^-1]
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
FIBRE_SEGMENT_K_ELAST = 0.2  # [N/units/kg]
FIBRE_SEGMENT_D_DUMPING = 0.04  # [N*s/units/kg]
FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = 15 # WARNING: must match the value used in network generation
FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS = 0.05
FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE = 0.0
MAX_SEARCH_RADIUS_FNODES = FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE / 10.0 # must me smaller than FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE


# +====================================================================+
# | DIFFUSION PARAMETERS                                               |
# +====================================================================+
INCLUDE_DIFFUSION = True
N_SPECIES = 2  # number of diffusing species.WARNING: make sure that the value coincides with the one declared in TODO
DIFFUSION_COEFF_MULTI = [300.0, 300.0]  # diffusion coefficient in [units^2/s] per specie
BOUNDARY_CONC_INIT_MULTI = [[50.0,50.0, 50.0, 50.0, 50.0, 50.0],
                            # initial concentration at each surface (+X,-X,+Y,-Y,+Z,-Z) [units^2/s]. -1.0 means no condition assigned. All agents are assigned 0 by default.
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # add as many lines as different species

BOUNDARY_CONC_FIXED_MULTI = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             # concentration boundary conditions at each surface. WARNING: -1.0 means initial condition prevails. Don't use 0.0 as initial condition if that value is not fixed. Use -1.0 instead
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # add as many lines as different species

# +====================================================================+
# | CELL PARAMETERS                                                   |
# +====================================================================+
INCLUDE_CELLS = True
INCLUDE_CELL_ORIENTATION = True
INCLUDE_CELL_CELL_INTERACTION = False
INCLUDE_CELL_CYCLE = False
PERIODIC_BOUNDARIES_FOR_CELLS = False
CELL_ORIENTATION_RATE = 1.0  # [1/time] TODO: check whether cell reorient themselves faster than ECM
N_CELLS = 1
CELL_K_ELAST = 2.0  # [N/units/kg]
CELL_D_DUMPING = 0.4  # [N*time/units/kg]
CELL_RADIUS = 8.412 #ECM_ECM_EQUILIBRIUM_DISTANCE / 2 # [units]
CELL_SPEED_REF = ECM_ECM_EQUILIBRIUM_DISTANCE / TIME_STEP / 10.0 # [units/time]
CYCLE_PHASE_G1_DURATION = 10.0 #[h]
CYCLE_PHASE_S_DURATION = 8.0
CYCLE_PHASE_G2_DURATION = 4.0
CYCLE_PHASE_M_DURATION = 2.0
CYCLE_PHASE_G1_START = 0.0 #[h]
CYCLE_PHASE_S_START = CYCLE_PHASE_G1_DURATION
CYCLE_PHASE_G2_START = CYCLE_PHASE_G1_DURATION + CYCLE_PHASE_S_DURATION
CYCLE_PHASE_M_START = CYCLE_PHASE_G1_DURATION + CYCLE_PHASE_S_DURATION + CYCLE_PHASE_G2_DURATION
CELL_CYCLE_DURATION = CYCLE_PHASE_G1_DURATION + CYCLE_PHASE_S_DURATION + CYCLE_PHASE_G2_DURATION + CYCLE_PHASE_M_DURATION # typically 24h [h]
INIT_ECM_CONCENTRATION_VALS = [50.0, 0.0]  # initial concentration of each species on the ECM agents
INIT_CELL_CONCENTRATION_VALS = [15.0, 0.0]  # initial concentration of each species on the CELL agents
INIT_CELL_CONC_MASS_VALS = [x * (4/3 * 3.1415926 * CELL_RADIUS**3) for x in INIT_CELL_CONCENTRATION_VALS]  # initial mass of each species on the CELL agents
INIT_ECM_SAT_CONCENTRATION_VALS = [0.0, 10.0]  # initial saturation concentration of each species on the ECM agents
INIT_CELL_CONSUMPTION_RATES = [0.001, 0.0]  # consumption rate of each species by the CELL agents 
INIT_CELL_PRODUCTION_RATES = [0.0, 10.0]  # production rate of each species by the CELL agents 
INIT_CELL_REACTION_RATES = [0.00018, 0.00018]  # metabolic reaction rates of each species by the CELL agents 

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
    nodes, connectivity, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, fibre_critical_error = load_fibre_network(
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

    if nodes is not None and connectivity is not None:
        N_FIBRES = len(connectivity)
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
                'WARNING: diffusion problem is ill conditioned (Fi_x should be < 0.5), check parameters and consider decreasing time step\nSemi-implicit diffusion will be used instead')
            UNSTABLE_DIFFUSION = True
    dy = L0_y / (ECM_AGENTS_PER_DIR[1] - 1)
    for i in range(N_SPECIES):
        Fi_y = 3 * (DIFFUSION_COEFF_MULTI[i] * TIME_STEP / (dy * dy))  # this value should be < 0.5
        # print('Fi_y value: {0} for species {1}'.format(Fi_y, i + 1))
        if Fi_y > 0.5:
            print(
                'WARNING: diffusion problem is ill conditioned (Fi_y should be < 0.5), check parameters and consider decreasing time step\nSemi-implicit diffusion will be used instead')
            UNSTABLE_DIFFUSION = True
    dz = L0_z / (ECM_AGENTS_PER_DIR[2] - 1)
    for i in range(N_SPECIES):
        Fi_z = 3 * (DIFFUSION_COEFF_MULTI[i] * TIME_STEP / (dz * dz))  # this value should be < 0.5
        # print('Fi_z value: {0} for species {1}'.format(Fi_z, i + 1))
        if Fi_z > 0.5:
            print(
                'WARNING: diffusion problem is ill conditioned (Fi_z should be < 0.5), check parameters and consider decreasing time step\nSemi-implicit diffusion will be used instead')
            UNSTABLE_DIFFUSION = True

if INCLUDE_CELLS:
    if MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION < (2 * CELL_RADIUS):
        print('MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION: {0} must be higher than 2 * CELL_RADIUS: 2 * {1}'.format(MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION, CELL_RADIUS))
        critical_error = True


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
ecm_output_grid_location_data_file = "ecm_output_grid_location_data.cpp"
ecm_ecm_interaction_file = "ecm_ecm_interaction.cpp"
ecm_boundary_concentration_conditions_file = "ecm_boundary_concentration_conditions.cpp"
ecm_move_file = "ecm_move.cpp"
ecm_Csp_update_file = "ecm_Csp_update.cpp"

"""
  CELL
"""
cell_output_location_data_file = "cell_output_location_data.cpp"
cell_ecm_interaction_metabolism_file = "cell_ecm_interaction_metabolism.cpp"
cell_move_file = "cell_move.cpp"

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


model = pyflamegpu.ModelDescription("metabolism")

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
env.newPropertyUInt("UNSTABLE_DIFFUSION", UNSTABLE_DIFFUSION)
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

# Cell properties
env.newPropertyUInt("INCLUDE_CELL_ORIENTATION", INCLUDE_CELL_ORIENTATION)
env.newPropertyUInt("INCLUDE_CELL_CELL_INTERACTION", INCLUDE_CELL_CELL_INTERACTION)
env.newPropertyUInt("PERIODIC_BOUNDARIES_FOR_CELLS", PERIODIC_BOUNDARIES_FOR_CELLS)
env.newPropertyUInt("N_CELLS", N_CELLS)
env.newPropertyFloat("CELL_K_ELAST", CELL_K_ELAST)
env.newPropertyFloat("CELL_D_DUMPING", CELL_D_DUMPING)
env.newPropertyFloat("CELL_RADIUS", CELL_RADIUS)
env.newPropertyFloat("CELL_SPEED_REF", CELL_SPEED_REF)
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
bcorner_location_message = model.newMessageSpatial3D("bcorner_location_message")
# Set the range and bounds.
bcorner_location_message.setRadius(MAX_EXPECTED_BOUNDARY_POS - MIN_EXPECTED_BOUNDARY_POS)  # corners are not actually interacting with anything
bcorner_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS)
bcorner_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS)
# A message to hold the location of an agent. WARNING: spatial3D messages already define x,y,z variables internally.
bcorner_location_message.newVariableInt("id")


if INCLUDE_FIBRE_NETWORK:
    fnode_spatial_location_message = model.newMessageSpatial3D("fnode_spatial_location_message")
    fnode_spatial_location_message.setRadius(MAX_SEARCH_RADIUS_FNODES)  
    fnode_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS,MIN_EXPECTED_BOUNDARY_POS)
    fnode_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS,MAX_EXPECTED_BOUNDARY_POS)
    fnode_spatial_location_message.newVariableInt("id") # as an edge can have multiple inner agents, this stores the position within the edge

    fnode_bucket_location_message = model.newMessageBucket("fnode_bucket_location_message")
    # Set the range and bounds.
    # setBounds(min, max) where min and max are the min and max ids of the message buckets. This is independent of the number of agents (there can be more agents than buckets and vice versa).
    # Here, we assign one bucket per fibre node so that each fibre node can be found in its own bucket when searching for neighbours.
    fnode_bucket_location_message.setBounds(8,N_NODES + 8) # +8 because domain corners have idx from 1 to 8. WARNING: make sure to initialize fibre nodes starting from index 9

    fnode_bucket_location_message.newVariableInt("id")
    fnode_bucket_location_message.newVariableFloat("x")
    fnode_bucket_location_message.newVariableFloat("y")
    fnode_bucket_location_message.newVariableFloat("z")
    fnode_bucket_location_message.newVariableFloat("vx")
    fnode_bucket_location_message.newVariableFloat("vy")
    fnode_bucket_location_message.newVariableFloat("vz")
    fnode_bucket_location_message.newVariableFloat("k_elast")
    fnode_bucket_location_message.newVariableFloat("d_dumping")
    fnode_bucket_location_message.newVariableArrayFloat("equilibrium_distance", MAX_CONNECTIVITY) # each segment can have a different equilibrium distance depending on the rest length assigned during network generation
    fnode_bucket_location_message.newVariableArrayInt("linked_nodes", MAX_CONNECTIVITY) # store the index of the linked nodes, which is a proxy for the bucket id


ECM_grid_location_message = model.newMessageArray3D("ECM_grid_location_message")
ECM_grid_location_message.setDimensions(ECM_AGENTS_PER_DIR[0], ECM_AGENTS_PER_DIR[1], ECM_AGENTS_PER_DIR[2])
ECM_grid_location_message.newVariableInt("id")
ECM_grid_location_message.newVariableFloat("x")
ECM_grid_location_message.newVariableFloat("y")
ECM_grid_location_message.newVariableFloat("z")
ECM_grid_location_message.newVariableInt("grid_lin_id")
ECM_grid_location_message.newVariableUInt8("grid_i")
ECM_grid_location_message.newVariableUInt8("grid_j")
ECM_grid_location_message.newVariableUInt8("grid_k")
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
    CELL_spatial_location_message = model.newMessageSpatial3D("CELL_spatial_location_message")
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

# ++==================================================================++
# ++ Agents                                                            |
# ++==================================================================++
"""
  AGENTS
"""

"""
  BCORNER agent
"""
bcorner_agent = model.newAgent("BCORNER") # boundary corner agent to track boundary positions
bcorner_agent.newVariableInt("id")
bcorner_agent.newVariableFloat("x")
bcorner_agent.newVariableFloat("y")
bcorner_agent.newVariableFloat("z")

bcorner_agent.newRTCFunctionFile("bcorner_output_location_data", bcorner_output_location_data_file).setMessageOutput("bcorner_location_message")
if MOVING_BOUNDARIES:
    bcorner_agent.newRTCFunctionFile("bcorner_move", bcorner_move_file)

"""
  FIBRE NODE agent
"""
if INCLUDE_FIBRE_NETWORK:
    fnode_agent = model.newAgent("FNODE")
    fnode_agent.newVariableInt("id")
    fnode_agent.newVariableFloat("x")
    fnode_agent.newVariableFloat("y")
    fnode_agent.newVariableFloat("z")
    fnode_agent.newVariableFloat("vx", 0.0)
    fnode_agent.newVariableFloat("vy", 0.0)
    fnode_agent.newVariableFloat("vz", 0.0)
    fnode_agent.newVariableFloat("fx", 0.0)
    fnode_agent.newVariableFloat("fy", 0.0)
    fnode_agent.newVariableFloat("fz", 0.0)
    fnode_agent.newVariableFloat("k_elast")
    fnode_agent.newVariableFloat("d_dumping")
    fnode_agent.newVariableArrayFloat("equilibrium_distance", MAX_CONNECTIVITY) # each segment can have a different equilibrium distance depending on the rest length assigned during network generation
    fnode_agent.newVariableFloat("boundary_fx")  # boundary_f[A]: normal force coming from boundary [A] when elastic boundaries option is selected.
    fnode_agent.newVariableFloat("boundary_fy")
    fnode_agent.newVariableFloat("boundary_fz")
    fnode_agent.newVariableFloat("f_bx_pos")  # f_b[A]_[B]: normal force transmitted to the boundary [A]_[B] when agent is clamped
    fnode_agent.newVariableFloat("f_bx_neg")
    fnode_agent.newVariableFloat("f_by_pos")
    fnode_agent.newVariableFloat("f_by_neg")
    fnode_agent.newVariableFloat("f_bz_pos")
    fnode_agent.newVariableFloat("f_bz_neg")
    fnode_agent.newVariableFloat("f_bx_pos_y")  # f_b[A]_[B]_[C]: shear force transmitted to the boundary [A]_[B] in the direction [C] when agent is clamped
    fnode_agent.newVariableFloat("f_bx_pos_z")
    fnode_agent.newVariableFloat("f_bx_neg_y")
    fnode_agent.newVariableFloat("f_bx_neg_z")
    fnode_agent.newVariableFloat("f_by_pos_x")
    fnode_agent.newVariableFloat("f_by_pos_z")
    fnode_agent.newVariableFloat("f_by_neg_x")
    fnode_agent.newVariableFloat("f_by_neg_z")
    fnode_agent.newVariableFloat("f_bz_pos_x")
    fnode_agent.newVariableFloat("f_bz_pos_y")
    fnode_agent.newVariableFloat("f_bz_neg_x")
    fnode_agent.newVariableFloat("f_bz_neg_y")
    fnode_agent.newVariableFloat("f_extension")
    fnode_agent.newVariableFloat("f_compression")
    fnode_agent.newVariableFloat("elastic_energy")
    fnode_agent.newVariableArrayFloat("linked_nodes", MAX_CONNECTIVITY)
    fnode_agent.newVariableUInt8("clamped_bx_pos")
    fnode_agent.newVariableUInt8("clamped_bx_neg")
    fnode_agent.newVariableUInt8("clamped_by_pos")
    fnode_agent.newVariableUInt8("clamped_by_neg")
    fnode_agent.newVariableUInt8("clamped_bz_pos")
    fnode_agent.newVariableUInt8("clamped_bz_neg")

    fnode_agent.newRTCFunctionFile("fnode_spatial_location_data", fnode_spatial_location_data_file).setMessageOutput("fnode_spatial_location_message")
    fnode_agent.newRTCFunctionFile("fnode_bucket_location_data", fnode_bucket_location_data_file).setMessageOutput("fnode_bucket_location_message")
    fnode_agent.newRTCFunctionFile("fnode_boundary_interaction", fnode_boundary_interaction_file)
    fnode_agent.newRTCFunctionFile("fnode_fnode_spatial_interaction", fnode_fnode_spatial_interaction_file).setMessageInput("fnode_spatial_location_message")
    fnode_agent.newRTCFunctionFile("fnode_fnode_bucket_interaction", fnode_fnode_bucket_interaction_file).setMessageInput("fnode_bucket_location_message")
    fnode_agent.newRTCFunctionFile("fnode_move", fnode_move_file)


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
ECM_agent.newRTCFunctionFile("ecm_output_grid_location_data", ecm_output_grid_location_data_file).setMessageOutput("ECM_grid_location_message")
ECM_agent.newRTCFunctionFile("ecm_ecm_interaction", ecm_ecm_interaction_file).setMessageInput("ECM_grid_location_message")
ECM_agent.newRTCFunctionFile("ecm_boundary_concentration_conditions", ecm_boundary_concentration_conditions_file)
ECM_agent.newRTCFunctionFile("ecm_Csp_update", ecm_Csp_update_file)
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
    CELL_agent.newVariableFloat("radius", CELL_RADIUS)
    CELL_agent.newVariableInt("cycle_phase", 1) # [1:G1] [2:S] [3:G2] [4:M]
    CELL_agent.newVariableFloat("clock", 0.0) # internal clock of the cell to switch phases
    CELL_agent.newVariableInt("completed_cycles", 0)
    CELL_agent.newRTCFunctionFile("cell_output_location_data", cell_output_location_data_file).setMessageOutput("CELL_spatial_location_message")
    CELL_agent.newRTCFunctionFile("cell_ecm_interaction_metabolism", cell_ecm_interaction_metabolism_file).setMessageInput("ECM_grid_location_message")
    CELL_agent.newRTCFunctionFile("cell_move", cell_move_file)


"""
  Population initialisation functions
"""

# Agent population initialization 
# ----------------------------------------------------------------------    
# This class is used to ensure that corner agents are assigned the first 8 ids
class initAgentPopulations(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # TODO: clean this line of globals
        global INIT_ECM_CONCENTRATION_VALS, INIT_CELL_CONCENTRATION_VALS,INIT_CELL_CONC_MASS_VALS, INIT_ECM_SAT_CONCENTRATION_VALS, INIT_CELL_CONSUMPTION_RATES, INIT_CELL_PRODUCTION_RATES,INIT_CELL_REACTION_RATES, N_SPECIES, INCLUDE_DIFFUSION,INCLUDE_FIBRE_NETWORK, INCLUDE_CELLS, N_CELLS, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, MAX_CONNECTIVITY
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

        # TODO: make this dependent on INCLUDE_DIFFUSION
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

        # CELLS
        if INCLUDE_CELLS:
            current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID")
            current_id += 1
            print(f"--- Initializing CELLS ({N_CELLS})")
            print("  |-> current_id:", current_id)
            count = -1
            cell_orientations = getRandomVectors3D(N_CELLS)
            k_elast = FLAMEGPU.environment.getPropertyFloat("CELL_K_ELAST")
            d_dumping = FLAMEGPU.environment.getPropertyFloat("CELL_D_DUMPING")
            radius = FLAMEGPU.environment.getPropertyFloat("CELL_RADIUS")
            for i in range(N_CELLS):
                count += 1
                instance = FLAMEGPU.agent("CELL").newAgent()
                instance.setVariableInt("id", current_id + count)
                instance.setVariableFloat("x", 0.0)
                instance.setVariableFloat("y", 0.0)
                instance.setVariableFloat("z", 0.0)
                instance.setVariableFloat("vx", 0.0)
                instance.setVariableFloat("vy", 0.0)
                instance.setVariableFloat("vz", 0.0)
                instance.setVariableFloat("orx", cell_orientations[count, 0])
                instance.setVariableFloat("ory", cell_orientations[count, 1])
                instance.setVariableFloat("orz", cell_orientations[count, 2])
                instance.setVariableFloat("alignment", 0.0)
                instance.setVariableFloat("k_elast", k_elast)
                instance.setVariableFloat("d_dumping", d_dumping)
                instance.setVariableArrayFloat("C_sp", INIT_CELL_CONCENTRATION_VALS)
                instance.setVariableArrayFloat("M_sp", INIT_CELL_CONC_MASS_VALS)                
                instance.setVariableArrayFloat("k_consumption", INIT_CELL_CONSUMPTION_RATES)
                instance.setVariableArrayFloat("k_production", INIT_CELL_PRODUCTION_RATES)
                instance.setVariableArrayFloat("k_reaction", INIT_CELL_REACTION_RATES)
                instance.setVariableFloat("radius", radius)
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
        self.header = list()
        self.header.append("# vtk DataFile Version 3.0")
        self.header.append("ECM data")
        self.header.append("ASCII")
        self.header.append("DATASET POLYDATA")
        self.header.append("POINTS {} float".format(8 + ECM_AGENTS_PER_DIR[0] * ECM_AGENTS_PER_DIR[1] * ECM_AGENTS_PER_DIR[2]))  # number of ECM agents + 8 corners
        # Domain data for boundary visualization in ECM files
        self.domaindata = list()
        self.domaindata.append("POLYGONS 6 30")        
        cube_conn = [[4, 0, 3, 7, 4], 
                     [4, 1, 2, 6, 5], 
                     [4, 1, 0, 4, 5], 
                     [4, 2, 3, 7, 6], 
                     [4, 0, 1, 2, 3],
                     [4, 4, 5, 6, 7]]
        for i in range(len(cube_conn)):
            for j in range(len(cube_conn[i])):
                if j > 0:
                    cube_conn[i][j] = cube_conn[i][j] + ECM_AGENTS_PER_DIR[0] * ECM_AGENTS_PER_DIR[1] * ECM_AGENTS_PER_DIR[2]
            self.domaindata.append(' '.join(str(x) for x in cube_conn[i]))

        if INCLUDE_FIBRE_NETWORK:
            # Domain data for boundary visualization in FIBRE network files (contains different data)
            self.domaindata_network = list()
            cube_conn_network = [[4, 0, 3, 7, 4],
                     [4, 1, 2, 6, 5], 
                     [4, 1, 0, 4, 5], 
                     [4, 2, 3, 7, 6], 
                     [4, 0, 1, 2, 3],
                     [4, 4, 5, 6, 7]]
            for i in range(len(cube_conn_network)):
                for j in range(len(cube_conn_network[i])):
                    if j > 0:
                        cube_conn_network[i][j] = cube_conn_network[i][j] + N_NODES
                self.domaindata_network.append(' '.join(str(x) for x in cube_conn_network[i]))

        self.domaindata.append("CELL_DATA 6")
        self.domaindata.append("SCALARS boundary_index int 1")
        self.domaindata.append("LOOKUP_TABLE default")
        self.domaindata.append("0")
        self.domaindata.append("1")
        self.domaindata.append("2")
        self.domaindata.append("3")
        self.domaindata.append("4")
        self.domaindata.append("5")
        self.domaindata.append("NORMALS boundary_normals float")
        self.domaindata.append("1 0 0")
        self.domaindata.append("-1 0 0")
        self.domaindata.append("0 1 0")
        self.domaindata.append("0 -1 0")
        self.domaindata.append("0 0 1")
        self.domaindata.append("0 0 -1")
        # VASCULARIZATION (TODO: UNUSED FOR NOW)
        self.vascularizationdata = list()  # a different file is created to show the position of the vascularization points
        self.vascularizationdata.append("# vtk DataFile Version 3.0")
        self.vascularizationdata.append("Vascularization points")
        self.vascularizationdata.append("ASCII")
        self.vascularizationdata.append("DATASET UNSTRUCTURED_GRID")
        # FIBRE NODES 
        self.fibrenodedata = list()  # a different file is created to show fibre node agent data
        self.fibrenodedata.append("# vtk DataFile Version 3.0") 
        self.fibrenodedata.append("Fibre node agents")
        self.fibrenodedata.append("ASCII")
        self.fibrenodedata.append("DATASET UNSTRUCTURED_GRID")
        # CELLS
        self.celldata = list()  # a different file is created to show cell agent data
        self.celldata.append("# vtk DataFile Version 3.0")
        self.celldata.append("Cell agents")
        self.celldata.append("ASCII")
        self.celldata.append("DATASET UNSTRUCTURED_GRID")

    def run(self, FLAMEGPU):
        global SAVE_DATA_TO_FILE, SAVE_EVERY_N_STEPS, N_SPECIES
        global RES_PATH, ENSEMBLE
        global INCLUDE_FIBRE_NETWORK, INITIAL_NETWORK_CONNECTIVITY,N_NODES, INCLUDE_CELLS, ECM_POPULATION_SIZE
        BUCKLING_COEFF_D0 = FLAMEGPU.environment.getPropertyFloat("BUCKLING_COEFF_D0")
        STRAIN_STIFFENING_COEFF_DS = FLAMEGPU.environment.getPropertyFloat("STRAIN_STIFFENING_COEFF_DS")
        CRITICAL_STRAIN = FLAMEGPU.environment.getPropertyFloat("CRITICAL_STRAIN")
        stepCounter = FLAMEGPU.getStepCounter() + 1
        
        coord_boundary = list(FLAMEGPU.environment.getPropertyArrayFloat("COORDS_BOUNDARIES"))

        if SAVE_DATA_TO_FILE:
            if stepCounter % SAVE_EVERY_N_STEPS == 0 or stepCounter == 1:

                if INCLUDE_FIBRE_NETWORK:
                    file_name = 'fibre_network_data_t{:04d}.vtk'.format(stepCounter)
                    file_path = RES_PATH / file_name

                    agent = FLAMEGPU.agent("FNODE")
                    # reaction forces, thus, opposite to agent-applied forces
                    sum_bx_pos = -agent.sumFloat("f_bx_pos")
                    sum_bx_neg = -agent.sumFloat("f_bx_neg")
                    sum_by_pos = -agent.sumFloat("f_by_pos")
                    sum_by_neg = -agent.sumFloat("f_by_neg")
                    sum_bz_pos = -agent.sumFloat("f_bz_pos")
                    sum_bz_neg = -agent.sumFloat("f_bz_neg")
                    sum_bx_pos_y = -agent.sumFloat("f_bx_pos_y")
                    sum_bx_pos_z = -agent.sumFloat("f_bx_pos_z")
                    sum_bx_neg_y = -agent.sumFloat("f_bx_neg_y")
                    sum_bx_neg_z = -agent.sumFloat("f_bx_neg_z")
                    sum_by_pos_x = -agent.sumFloat("f_by_pos_x")
                    sum_by_pos_z = -agent.sumFloat("f_by_pos_z")
                    sum_by_neg_x = -agent.sumFloat("f_by_neg_x")
                    sum_by_neg_z = -agent.sumFloat("f_by_neg_z")
                    sum_bz_pos_x = -agent.sumFloat("f_bz_pos_x")
                    sum_bz_pos_y = -agent.sumFloat("f_bz_pos_y")
                    sum_bz_neg_x = -agent.sumFloat("f_bz_neg_x")
                    sum_bz_neg_y = -agent.sumFloat("f_bz_neg_y")

                    ids = list()
                    coords = list()
                    velocity = list()
                    force = list()
                    elastic_energy = list()

                    av = agent.getPopulationData()  # this returns a DeviceAgentVector
                    for ai in av:
                        id_ai = ai.getVariableInt("id") - 9 # for vtk visualization, FNODE agents start at 0
                        coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
                        velocity_ai = (ai.getVariableFloat("vx"), ai.getVariableFloat("vy"), ai.getVariableFloat("vz"))
                        force_ai = (ai.getVariableFloat("fx"), ai.getVariableFloat("fy"), ai.getVariableFloat("fz"))
                        ids.append(id_ai)
                        coords.append(coords_ai)
                        velocity.append(velocity_ai)
                        force.append(force_ai)
                        elastic_energy.append(ai.getVariableFloat("elastic_energy"))
                        
                    # Get sorting indices based on ids
                    sorted_indices = np.argsort(ids)

                    # Rearrange lists according to the sorted indices
                    ids = [ids[i] for i in sorted_indices]
                    coords = [coords[i] for i in sorted_indices]
                    velocity = [velocity[i] for i in sorted_indices]
                    force = [force[i] for i in sorted_indices]
                    elastic_energy = [elastic_energy[i] for i in sorted_indices]
                    # Extract lines from INITIAL_NETWORK_CONNECTIVITY
                    # To avoid duplicating cells, use a set to keep track of added lines
                    added_lines = set()
                    cell_connectivity = []
                    
                    for node_index, connections in INITIAL_NETWORK_CONNECTIVITY.items():
                        for connected_node_index in connections:
                            if connected_node_index != -1:
                                # Create a tuple with sorted indices to ensure uniqueness
                                line = tuple(sorted((node_index, connected_node_index)))
                                if line not in added_lines:
                                    added_lines.add(line)
                                    cell_connectivity.append(line)
                    
                    num_cells = len(cell_connectivity)
                    
                    
                    with open(str(file_path), 'w') as file:
                        for line in self.fibrenodedata:
                            file.write(line + '\n')
                            
                        file.write("POINTS {} float \n".format(8 + N_NODES))
                        for coords_ai in coords:
                            file.write("{} {} {} \n".format(coords_ai[0], coords_ai[1], coords_ai[2]))

                        # Write boundary positions at the end so that corner points don't cover the points underneath
                        file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[2], coord_boundary[4]))
                        file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[2], coord_boundary[4]))
                        file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[3], coord_boundary[4]))
                        file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[3], coord_boundary[4]))
                        file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[2], coord_boundary[5]))
                        file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[2], coord_boundary[5]))
                        file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[3], coord_boundary[5]))
                        file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[3], coord_boundary[5]))
                            
                        # Write the cell connectivity
                        file.write(f"CELLS {num_cells + 6} {num_cells * 3 + 6 * 5}\n") # each of the 6 boundaries is a cell of type POLYGON with 4 nodes
                        for conn in cell_connectivity:
                            file.write(f"2 {conn[0]} {conn[1]}\n") 
                        for line in self.domaindata_network:
                            file.write(line + '\n')
                        
                        # Write the cell types (3 for VTK_LINE)
                        file.write(f"CELL_TYPES {num_cells + 6}\n")
                        for _ in range(num_cells):
                            file.write("3\n")  # VTK_LINE  
                        for _ in range(6):
                            file.write("7\n")  # VTK_POLYGON 
                            
                        file.write(f"CELL_DATA {num_cells + 6}\n")
                        file.write("SCALARS boundary_idx int 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')                    
                        for _ in range(num_cells):
                            file.write("0\n")    
                        for bidx in range(6):
                            file.write(f"{bidx + 1}\n") 

                        file.write("SCALARS boundary_normal_forces float 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')                    
                        for _ in range(num_cells):
                            file.write("0.0\n")    
                        file.write(str(sum_bx_pos) + '\n')
                        file.write(str(sum_bx_neg) + '\n')
                        file.write(str(sum_by_pos) + '\n')
                        file.write(str(sum_by_neg) + '\n')
                        file.write(str(sum_bz_pos) + '\n')
                        file.write(str(sum_bz_neg) + '\n')
                        
                        file.write("SCALARS boundary_normal_force_scaling float 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')
                        for _ in range(num_cells):
                            file.write("0.0\n")    
                        file.write(str(abs(sum_bx_pos)) + '\n')
                        file.write(str(abs(sum_bx_neg)) + '\n')
                        file.write(str(abs(sum_by_pos)) + '\n')
                        file.write(str(abs(sum_by_neg)) + '\n')
                        file.write(str(abs(sum_bz_pos)) + '\n')
                        file.write(str(abs(sum_bz_neg)) + '\n')
                        # must be divided in blocks of 6 (one value per face of the cube)
                        file.write("SCALARS boundary_shear_forces_pos float 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')
                        for _ in range(num_cells):
                            file.write("0.0\n")    
                        file.write(str(sum_bx_pos_y) + '\n')
                        file.write(str(sum_bx_pos_z) + '\n')
                        file.write(str(sum_by_pos_x) + '\n')
                        file.write(str(sum_by_pos_z) + '\n')
                        file.write(str(sum_bz_pos_x) + '\n')
                        file.write(str(sum_bz_pos_y) + '\n')
                        file.write("SCALARS boundary_shear_forces_neg float 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')
                        for _ in range(num_cells):
                            file.write("0.0\n")    
                        file.write(str(sum_bx_neg_y) + '\n')
                        file.write(str(sum_bx_neg_z) + '\n')
                        file.write(str(sum_by_neg_x) + '\n')
                        file.write(str(sum_by_neg_z) + '\n')
                        file.write(str(sum_bz_neg_x) + '\n')
                        file.write(str(sum_bz_neg_y) + '\n')


                        file.write("POINT_DATA {} \n".format(8 + N_NODES))  # 8 corners + number of FNODE agents

                        file.write("SCALARS is_corner int 1" + '\n')  # create this variable to remove them from representations
                        file.write("LOOKUP_TABLE default" + '\n')

                        for ee_ai in elastic_energy:
                            file.write("{0} \n".format(0))
                        for _ in range(8):
                            file.write("1 \n")  # boundary corners


                        file.write("SCALARS elastic_energy float 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')
                        for ee_ai in elastic_energy:
                            file.write("{:.4f} \n".format(ee_ai))
                        for _ in range(8):
                            file.write("0.0 \n")  # boundary corners

                        file.write("VECTORS velocity float" + '\n')
                        for v_ai in velocity:
                            file.write("{:.4f} {:.4f} {:.4f} \n".format(v_ai[0], v_ai[1], v_ai[2]))
                        for _ in range(8):
                            file.write("0.0 0.0 0.0 \n")  # boundary corners

                        file.write("VECTORS force float" + '\n')
                        for f_ai in force:
                            file.write("{:.4f} {:.4f} {:.4f} \n".format(f_ai[0], f_ai[1], f_ai[2]))
                        for _ in range(8):
                            file.write("0.0 0.0 0.0 \n")  # boundary corners

                if INCLUDE_CELLS:
                    cell_coords = list()
                    cell_velocity = list()
                    cell_orientation = list()
                    cell_alignment = list()
                    cell_radius = list()
                    cell_clock = list()
                    cell_cycle_phase = list()
                    concentration_multi = list()  # this is a list of tuples. Each tuple has N_SPECIES elements
                    file_name = 'cells_t{:04d}.vtk'.format(stepCounter)
                    file_path = RES_PATH / file_name
                    cell_agent = FLAMEGPU.agent("CELL")
                    cell_agent.sortInt("id", pyflamegpu.HostAgentAPI.Asc); # this is critical to ensure cell ids are kept in order for visualization
                    av = cell_agent.getPopulationData()  # this returns a DeviceAgentVector
                    for ai in av:
                        coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
                        velocity_ai = (ai.getVariableFloat("vx"), ai.getVariableFloat("vy"), ai.getVariableFloat("vz"))
                        orientation_ai = (ai.getVariableFloat("orx"), ai.getVariableFloat("ory"), ai.getVariableFloat("orz"))
                        alignment_ai = ai.getVariableFloat("alignment")
                        radius_ai = ai.getVariableFloat("radius")
                        clock_ai = ai.getVariableFloat("clock")
                        cycle_phase_ai = ai.getVariableInt("cycle_phase")
                        concentration_multi.append(ai.getVariableArrayFloat("C_sp"))
                        cell_coords.append(coords_ai)
                        cell_velocity.append(velocity_ai)
                        cell_orientation.append(orientation_ai)
                        cell_alignment.append(alignment_ai)
                        cell_radius.append(radius_ai)
                        cell_clock.append(clock_ai)
                        cell_cycle_phase.append(cycle_phase_ai)
                    with open(str(file_path), 'w') as file:
                        for line in self.celldata:
                            file.write(line + '\n')
                        file.write("POINTS {} float \n".format(
                            FLAMEGPU.environment.getPropertyUInt("N_CELLS")))  # number of cell agents
                        for coords_ai in cell_coords:
                            file.write("{} {} {} \n".format(coords_ai[0], coords_ai[1], coords_ai[2]))
                        file.write("POINT_DATA {} \n".format(
                            FLAMEGPU.environment.getPropertyUInt("N_CELLS")))  
                        file.write("SCALARS alignment float 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')
                        for a_ai in cell_alignment:
                            file.write("{:.4f} \n".format(a_ai))                            
                        file.write("SCALARS radius float 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')
                        for r_ai in cell_radius:
                            file.write("{:.4f} \n".format(r_ai))                            
                        file.write("SCALARS clock float 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')
                        for c_ai in cell_clock:
                            file.write("{:.4f} \n".format(c_ai))                        
                        file.write("SCALARS cycle_phase int 1" + '\n')
                        file.write("LOOKUP_TABLE default" + '\n')
                        for ccp_ai in cell_cycle_phase:
                            file.write("{} \n".format(ccp_ai))    
                        for s in range(N_SPECIES):
                            file.write("SCALARS concentration_species_{0} float 1 \n".format(s))
                            file.write("LOOKUP_TABLE default" + '\n')
                            for c_ai in concentration_multi:
                                file.write("{:.4f} \n".format(c_ai[s]))              
                        file.write("VECTORS velocity float" + '\n')
                        for v_ai in cell_velocity:
                            file.write("{:.4f} {:.4f} {:.4f} \n".format(v_ai[0], v_ai[1], v_ai[2]))
                        file.write("VECTORS orientation float" + '\n')
                        for o_ai in cell_orientation:
                            file.write("{:.4f} {:.4f} {:.4f} \n".format(o_ai[0], o_ai[1], o_ai[2]))

                file_name = 'ecm_data_t{:04d}.vtk'.format(stepCounter)
                if ENSEMBLE:
                    dir_name = f"BUCKLING_COEFF_D0_{BUCKLING_COEFF_D0:.3f}_STRAIN_STIFFENING_COEFF_DS_{STRAIN_STIFFENING_COEFF_DS:.3f}_CRITICAL_STRAIN_{CRITICAL_STRAIN:.3f}"
                    # Combine the base directory with the current directory name
                    file_path = RES_PATH / dir_name / file_name
                else:
                    file_path = RES_PATH / file_name

                agent = FLAMEGPU.agent("ECM")
                # reaction forces, currently unused in ECM agents. FNODE agents bear the load instead
                sum_bx_pos = 0.0
                sum_bx_neg = 0.0
                sum_by_pos = 0.0
                sum_by_neg = 0.0
                sum_bz_pos = 0.0
                sum_bz_neg = 0.0
                sum_bx_pos_y = 0.0
                sum_bx_pos_z = 0.0
                sum_bx_neg_y = 0.0
                sum_bx_neg_z = 0.0
                sum_by_pos_x = 0.0
                sum_by_pos_z = 0.0
                sum_by_neg_x = 0.0
                sum_by_neg_z = 0.0
                sum_bz_pos_x = 0.0
                sum_bz_pos_y = 0.0
                sum_bz_neg_x = 0.0
                sum_bz_neg_y = 0.0

                coords = list()
                velocity = list()
                force = list()
                concentration_multi = list()  # this is a list of tuples. Each tuple has N_SPECIES elements
                av = agent.getPopulationData()  # this returns a DeviceAgentVector
                for ai in av:
                    coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
                    velocity_ai = (ai.getVariableFloat("vx"), ai.getVariableFloat("vy"), ai.getVariableFloat("vz"))
                    force_ai = (ai.getVariableFloat("fx"), ai.getVariableFloat("fy"), ai.getVariableFloat("fz"))
                    coords.append(coords_ai)
                    velocity.append(velocity_ai)
                    force.append(force_ai)
                    concentration_multi.append(ai.getVariableArrayFloat("C_sp"))
                print("====== SAVING DATA FROM Step {:03d} TO FILE ======".format(stepCounter))
                with open(str(file_path), 'w') as file:
                    for line in self.header:
                        file.write(line + '\n')
                    for coords_ai in coords:
                        file.write("{} {} {} \n".format(coords_ai[0], coords_ai[1], coords_ai[2]))
                    # Write boundary positions at the end so that corner points don't cover the points underneath
                    file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[2], coord_boundary[4]))
                    file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[2], coord_boundary[4]))
                    file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[3], coord_boundary[4]))
                    file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[3], coord_boundary[4]))
                    file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[2], coord_boundary[5]))
                    file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[2], coord_boundary[5]))
                    file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[3], coord_boundary[5]))
                    file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[3], coord_boundary[5]))
                    for line in self.domaindata:
                        file.write(line + '\n')
                    file.write("SCALARS boundary_normal_forces float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    file.write(str(sum_bx_pos) + '\n')
                    file.write(str(sum_bx_neg) + '\n')
                    file.write(str(sum_by_pos) + '\n')
                    file.write(str(sum_by_neg) + '\n')
                    file.write(str(sum_bz_pos) + '\n')
                    file.write(str(sum_bz_neg) + '\n')
                    file.write("SCALARS boundary_normal_force_scaling float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    file.write(str(abs(sum_bx_pos)) + '\n')
                    file.write(str(abs(sum_bx_neg)) + '\n')
                    file.write(str(abs(sum_by_pos)) + '\n')
                    file.write(str(abs(sum_by_neg)) + '\n')
                    file.write(str(abs(sum_bz_pos)) + '\n')
                    file.write(str(abs(sum_bz_neg)) + '\n')
                    file.write("VECTORS boundary_normal_force_dir float" + '\n')
                    file.write("1 0 0 \n" if sum_bx_pos > 0 else "-1 0 0 \n")
                    file.write("1 0 0 \n" if sum_bx_neg > 0 else "-1 0 0 \n")
                    file.write("0 1 0 \n" if sum_by_pos > 0 else "0 -1 0 \n")
                    file.write("0 1 0 \n" if sum_by_neg > 0 else "0 -1 0 \n")
                    file.write("0 0 1 \n" if sum_bz_pos > 0 else "0 0 -1 \n")
                    file.write("0 0 1 \n" if sum_bz_neg > 0 else "0 0 -1 \n")
                    # must be divided in blocks of 6 (one value per face of the cube)
                    file.write("SCALARS boundary_shear_forces_pos float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    file.write(str(sum_bx_pos_y) + '\n')
                    file.write(str(sum_bx_pos_z) + '\n')
                    file.write(str(sum_by_pos_x) + '\n')
                    file.write(str(sum_by_pos_z) + '\n')
                    file.write(str(sum_bz_pos_x) + '\n')
                    file.write(str(sum_bz_pos_y) + '\n')
                    file.write("SCALARS boundary_shear_forces_neg float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    file.write(str(sum_bx_neg_y) + '\n')
                    file.write(str(sum_bx_neg_z) + '\n')
                    file.write(str(sum_by_neg_x) + '\n')
                    file.write(str(sum_by_neg_z) + '\n')
                    file.write(str(sum_bz_neg_x) + '\n')
                    file.write(str(sum_bz_neg_y) + '\n')
                    file.write("SCALARS boundary_shear_force_scaling_pos float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    file.write(str(abs(sum_bx_pos_y)) + '\n')
                    file.write(str(abs(sum_bx_pos_z)) + '\n')
                    file.write(str(abs(sum_by_pos_x)) + '\n')
                    file.write(str(abs(sum_by_pos_z)) + '\n')
                    file.write(str(abs(sum_bz_pos_x)) + '\n')
                    file.write(str(abs(sum_bz_pos_y)) + '\n')
                    file.write("SCALARS boundary_shear_force_scaling_neg float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    file.write(str(abs(sum_bx_neg_y)) + '\n')
                    file.write(str(abs(sum_bx_neg_z)) + '\n')
                    file.write(str(abs(sum_by_neg_x)) + '\n')
                    file.write(str(abs(sum_by_neg_z)) + '\n')
                    file.write(str(abs(sum_bz_neg_x)) + '\n')
                    file.write(str(abs(sum_bz_neg_y)) + '\n')

                    file.write("VECTORS boundary_shear_force_dir_pos float" + '\n')
                    file.write("0 1 0 \n" if sum_bx_pos_y > 0 else "0 -1 0 \n")
                    file.write("0 0 1 \n" if sum_bx_pos_z > 0 else "0 0 -1 \n")
                    file.write("1 0 0 \n" if sum_by_pos_x > 0 else "-1 0 0 \n")
                    file.write("0 0 1 \n" if sum_by_pos_z > 0 else "0 0 -1 \n")
                    file.write("1 0 0 \n" if sum_bz_pos_x > 0 else "-1 0 0 \n")
                    file.write("0 1 0 \n" if sum_bz_pos_y > 0 else "0 -1 0 \n")
                    file.write("VECTORS boundary_shear_force_dir_neg float" + '\n')
                    file.write("0 1 0 \n" if sum_bx_neg_y > 0 else "0 -1 0 \n")
                    file.write("0 0 1 \n" if sum_bx_neg_z > 0 else "0 0 -1 \n")
                    file.write("1 0 0 \n" if sum_by_neg_x > 0 else "-1 0 0 \n")
                    file.write("0 0 1 \n" if sum_by_neg_z > 0 else "0 0 -1 \n")
                    file.write("1 0 0 \n" if sum_bz_neg_x > 0 else "-1 0 0 \n")
                    file.write("0 1 0 \n" if sum_bz_neg_y > 0 else "0 -1 0 \n")

                    file.write("POINT_DATA {} \n".format(8 + ECM_POPULATION_SIZE))  # 8 corners + number of ECM agents

                    file.write(
                        "SCALARS is_corner int 1" + '\n')  # create this variable to remove them from representations
                    file.write("LOOKUP_TABLE default" + '\n')

                    for i in range(ECM_POPULATION_SIZE):
                        file.write("{0} \n".format(0))
                    for i in range(8):
                        file.write("1 \n")  # boundary corners

                    for s in range(N_SPECIES):
                        file.write("SCALARS concentration_species_{0} float 1 \n".format(s))
                        file.write("LOOKUP_TABLE default" + '\n')

                        for c_ai in concentration_multi:
                            file.write("{:.4f} \n".format(c_ai[s]))
                        for i in range(8):
                            file.write("0.0 \n")  # boundary corners

                    file.write("VECTORS velocity float" + '\n')
                    for v_ai in velocity:
                        file.write("{:.4f} {:.4f} {:.4f} \n".format(v_ai[0], v_ai[1], v_ai[2]))
                    for i in range(8):
                        file.write("0.0 0.0 0.0 \n")  # boundary corners

                    file.write("VECTORS force float" + '\n')
                    for f_ai in force:
                        file.write("{:.4f} {:.4f} {:.4f} \n".format(f_ai[0], f_ai[1], f_ai[2]))
                    for i in range(8):
                        file.write("0.0 0.0 0.0 \n")  # boundary corners

                print("... succesful save ")
                print("=================================")
                          

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
model.addStepFunction(sdf)


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
    model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_output_grid_location_data")
if INCLUDE_CELLS:
    model.Layer("L1_Agent_Locations").addAgentFunction("CELL", "cell_output_location_data")
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
    # L5_Diffusion
    model.newLayer("L5_Diffusion").addAgentFunction("ECM", "ecm_ecm_interaction")
    # L6_Diffusion_Boundary (called twice to ensure concentration at boundaries is properly shown visually)
    model.newLayer("L6_Diffusion_Boundary").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
if INCLUDE_FIBRE_NETWORK:
    # L7_Fibre_Network Mechanical interactions
    model.newLayer("L7_FNODE_Repulsion").addAgentFunction("FNODE", "fnode_fnode_spatial_interaction")
    model.newLayer("L7_FNODE_Network_Mechanics").addAgentFunction("FNODE", "fnode_fnode_bucket_interaction")

# L8_Agent_Movement
if INCLUDE_CELLS:
    model.newLayer("L8_CELL_Movement").addAgentFunction("CELL", "cell_move")
if INCLUDE_FIBRE_NETWORK:
    model.newLayer("L8_FNODE_Movement").addAgentFunction("FNODE", "fnode_move")
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
    # Saving pickle
    if SAVE_PICKLE:
        file_name = f'output_data_{idx}.pickle'
        file_path = RES_PATH / file_name
        with open(str(file_path), 'wb') as file:
            pickle.dump({'BPOS_OVER_TIME': BPOS_OVER_TIME,
                         'BFORCE_OVER_TIME': BFORCE_OVER_TIME,
                         'BFORCE_SHEAR_OVER_TIME': BFORCE_SHEAR_OVER_TIME,
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