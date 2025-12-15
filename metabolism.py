# +====================================================================+
# | Model: metabolism                                    |
# | Last update: 15/12/2025 - 13:28:03                                    |
# +====================================================================+


# +====================================================================+
# | IMPORTS                                                            |
# +====================================================================+
from pyflamegpu import *
import pathlib, time, math
from dataclasses import make_dataclass
import pandas as pd

start_time = time.time()

# +====================================================================+
# | GLOBAL PARAMETERS                                                  |
# +====================================================================+
# Set whether to run single model or ensemble, agent population size, and simulation steps 
ENSEMBLE = False
ENSEMBLE_RUNS = 0
VISUALISATION = True  # Change to false if pyflamegpu has not been built with visualisation support
DEBUG_PRINTING = False
PAUSE_EVERY_STEP = False  # If True, the visualization stops every step until P is pressed
SAVE_PICKLE = True  # If True, dumps agent and boudary force data into a pickle file for post-processing
SHOW_PLOTS = False  # Show plots at the end of the simulation
SAVE_DATA_TO_FILE = True  # If true, agent data is exported to .vtk file every SAVE_EVERY_N_STEPS steps
SAVE_EVERY_N_STEPS = 20  # Affects both the .vtk files and the Dataframes storing boundary data

CURR_PATH = pathlib.Path().absolute()
RES_PATH = CURR_PATH / 'result_files'
RES_PATH.mkdir(parents=True, exist_ok=True)
EPSILON = 0.0000000001

print("Executing in ", CURR_PATH)
# Minimum number of agents per direction (x,y,z). 
# If domain is not cubical, N is asigned to the shorter dimension and more agents are added to the longer ones
# +--------------------------------------------------------------------+
N = 16

# Time simulation parameters
# +--------------------------------------------------------------------+
TIME_STEP = 0.025  # time. WARNING: diffusion and cell migration events might need different scales
STEPS = 5000

# Boundary interactions and mechanical parameters
# +--------------------------------------------------------------------+
ECM_K_ELAST = 2.0  # [N/units/kg]
ECM_D_DUMPING = 0.4  # [N*s/units/kg]
ECM_MASS = 1.0  # [dimensionless to make K and D mass dependent]

#BOUNDARY_COORDS = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]  # +X,-X,+Y,-Y,+Z,-Z
BOUNDARY_COORDS = [1000.0, -1000.0, 650.0, -650.0, 150.0, -150.0] # microdevice dimensions in um
#BOUNDARY_COORDS = [coord / 1000.0 for coord in BOUNDARY_COORDS] # in mm
BOUNDARY_DISP_RATES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # perpendicular to each surface (+X,-X,+Y,-Y,+Z,-Z) [units/time]
BOUNDARY_DISP_RATES_PARALLEL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # parallel to each surface (+X_y,+X_z,-X_y,-X_z,+Y_x,+Y_z,-Y_x,-Y_z,+Z_x,+Z_y,-Z_x,-Z_y)[units/time]

POISSON_DIRS = [0, 1]  # 0: xdir, 1:ydir, 2:zdir. poisson_ratio ~= -incL(dir1)/incL(dir2) dir2 is the direction in which the load is applied
ALLOW_BOUNDARY_ELASTIC_MOVEMENT = [0, 0, 0, 0, 0, 0]  # [bool]
RELATIVE_BOUNDARY_STIFFNESS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
BOUNDARY_STIFFNESS_VALUE = 10.0  # N/units
BOUNDARY_DUMPING_VALUE = 5.0
BOUNDARY_STIFFNESS = [BOUNDARY_STIFFNESS_VALUE * x for x in RELATIVE_BOUNDARY_STIFFNESS]
BOUNDARY_DUMPING = [BOUNDARY_DUMPING_VALUE * x for x in RELATIVE_BOUNDARY_STIFFNESS]
#CLAMP_AGENT_TOUCHING_BOUNDARY = [0, 0, 1, 1, 0, 0]  # +X,-X,+Y,-Y,+Z,-Z [bool] - shear assay
CLAMP_AGENT_TOUCHING_BOUNDARY = [1, 1, 1, 1, 1, 1]  # +X,-X,+Y,-Y,+Z,-Z [bool]
ALLOW_AGENT_SLIDING = [0, 0, 0, 0, 0, 0]  # +X,-X,+Y,-Y,+Z,-Z [bool]

# Adjusting number of agents if domain is not cubical
# +--------------------------------------------------------------------+
# Calculate the differences between opposite pairs along each axis
diff_x = abs(BOUNDARY_COORDS[0] - BOUNDARY_COORDS[1])
diff_y = abs(BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3])
diff_z = abs(BOUNDARY_COORDS[4] - BOUNDARY_COORDS[5])

# Check if the differences are equal
if diff_x == diff_y == diff_z:
    ECM_AGENTS_PER_DIR = [N, N, N]
    print("The domain is cubical.")    
else:
    print("The domain is not cubical.")
    min_length = min(diff_x, diff_y, diff_z)
    dist_agents = min_length / (N - 1)
    ECM_AGENTS_PER_DIR = [int(diff_x / dist_agents) + 1, int(diff_y / dist_agents) + 1, int(diff_z / dist_agents) + 1]
    # Redefine BOUNDARY_COORDS due to rounding values
    diff_x = dist_agents * (ECM_AGENTS_PER_DIR[0] - 1)
    diff_y = dist_agents * (ECM_AGENTS_PER_DIR[1] - 1)
    diff_z = dist_agents * (ECM_AGENTS_PER_DIR[2] - 1)
    BOUNDARY_COORDS = [round(diff_x / 2, 2), -round(diff_x / 2, 2), round(diff_y / 2, 2), -round(diff_y / 2, 2), round(diff_z / 2, 2), -round(diff_z / 2, 2)] 
    
print('DOMAIN SIZE: {0},{1},{2}'.format(
    abs(BOUNDARY_COORDS[0] - BOUNDARY_COORDS[1]),
    abs(BOUNDARY_COORDS[3] - BOUNDARY_COORDS[2]),
    abs(BOUNDARY_COORDS[5] - BOUNDARY_COORDS[4])
))
print('ECM_AGENTS_PER_DIR: {0},{1},{2}'.format(ECM_AGENTS_PER_DIR[0], ECM_AGENTS_PER_DIR[1], ECM_AGENTS_PER_DIR[2]))
ECM_POPULATION_SIZE = ECM_AGENTS_PER_DIR[0] * ECM_AGENTS_PER_DIR[1] * ECM_AGENTS_PER_DIR[2]
print('Total number of agents: {0}'.format(ECM_POPULATION_SIZE))
ECM_ECM_EQUILIBRIUM_DISTANCE = (BOUNDARY_COORDS[0] - BOUNDARY_COORDS[1]) / (ECM_AGENTS_PER_DIR[0] - 1) 
print("ECM_ECM_EQUILIBRIUM_DISTANCE [units]: ", ECM_ECM_EQUILIBRIUM_DISTANCE)
ECM_BOUNDARY_INTERACTION_RADIUS = 0.05
ECM_BOUNDARY_EQUILIBRIUM_DISTANCE = 0.0

INCLUDE_FIBER_ALIGNMENT = True

MAX_SEARCH_RADIUS_VASCULARIZATION = ECM_ECM_EQUILIBRIUM_DISTANCE  # this strongly affects the number of bins and therefore the memory allocated for simulations (more bins -> more memory -> faster (in theory))
MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION = ECM_ECM_EQUILIBRIUM_DISTANCE # this radius is used to find ECM agents
print("MAX_SEARCH_RADIUS for CELLS [units]: ", MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION)
MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION = ECM_ECM_EQUILIBRIUM_DISTANCE # this radius is used to check if cells interact with each other

OSCILLATORY_SHEAR_ASSAY = False  # if true, BOUNDARY_DISP_RATES_PARALLEL options are overrun but used to make the boundaries oscillate in their corresponding planes following a sin() function
OSCILLATORY_AMPLITUDE = 0.25  # range [0-1]
OSCILLATORY_FREQ = 0.1  # strain oscillation frequency [time^-1]
OSCILLATORY_W = 2 * math.pi * OSCILLATORY_FREQ * TIME_STEP

# Fitting parameters for the fiber strain-stiffening phenomena
# Ref: https://bio.physik.fau.de/publications/Steinwachs%20Nat%20Meth%202016.pdf
# +--------------------------------------------------------------------+
BUCKLING_COEFF_D0 = 0.1
STRAIN_STIFFENING_COEFF_DS = 0.25
CRITICAL_STRAIN = 0.1

# Parallel disp rate values are overrun in oscillatory assays
# +--------------------------------------------------------------------+
if OSCILLATORY_SHEAR_ASSAY:
    for d in range(12):
        if abs(BOUNDARY_DISP_RATES_PARALLEL[d]) > 0.0:
            BOUNDARY_DISP_RATES_PARALLEL[d] = OSCILLATORY_AMPLITUDE * math.cos(
                OSCILLATORY_W * 0.0) * OSCILLATORY_W / TIME_STEP  # cos(w*t)*w is used because the slope of the sin(w*t) function is needed. Expressed in units/sec

# Diffusion related paramenters
# +--------------------------------------------------------------------+
INCLUDE_DIFFUSION = True
N_SPECIES = 2  # number of diffusing species.WARNING: make sure that the value coincides with the one declared in TODO
DIFFUSION_COEFF_MULTI = [0.02, 0.02]  # diffusion coefficient in [units^2/s] per specie
BOUNDARY_CONC_INIT_MULTI = [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                            # initial concentration at each surface (+X,-X,+Y,-Y,+Z,-Z) [units^2/s]. -1.0 means no condition assigned. All agents are assigned 0 by default.
                            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]  # add as many lines as different species

BOUNDARY_CONC_FIXED_MULTI = [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                             # concentration boundary conditions at each surface. WARNING: -1.0 means initial condition prevails. Don't use 0.0 as initial condition if that value is not fixed. Use -1.0 instead
                             [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]  # add as many lines as different species

INIT_ECM_CONCENTRATION_VALS = [0.0, 0.0]  # initial concentration of each species on the ECM agents


# Cell agent related paramenters
# +--------------------------------------------------------------------+
INCLUDE_CELLS = True
INCLUDE_CELL_ORIENTATION = True
INCLUDE_CELL_CELL_INTERACTION = False
INCLUDE_CELL_CYCLE = False
PERIODIC_BOUNDARIES_FOR_CELLS = False
CELL_ORIENTATION_RATE = 1.0  # [1/time] TODO: check whether cell reorient themselves faster than ECM
N_CELLS = 1
CELL_K_ELAST = 2.0  # [N/units/kg]
CELL_D_DUMPING = 0.4  # [N*time/units/kg]
CELL_RADIUS = ECM_ECM_EQUILIBRIUM_DISTANCE / 2 # [units]
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

# Other simulation parameters: TODO: INCLUDE PARALLEL DISP RATES
# +--------------------------------------------------------------------+
MAX_EXPECTED_BOUNDARY_POS = max(BOUNDARY_DISP_RATES) * STEPS * TIME_STEP + max(diff_x, diff_y, diff_z) / 2
MIN_EXPECTED_BOUNDARY_POS = min(BOUNDARY_DISP_RATES) * STEPS * TIME_STEP - max(diff_x, diff_y, diff_z) / 2
print("Max expected boundary position: ", MAX_EXPECTED_BOUNDARY_POS)
print("Min expected boundary position: ", MIN_EXPECTED_BOUNDARY_POS)

# Dataframe initialization data storage
# +--------------------------------------------------------------------+
BPOS = make_dataclass("BPOS", [("xpos", float), ("xneg", float), ("ypos", float), ("yneg", float), ("zpos", float),
                               ("zneg", float)])
# Use a dataframe to store boundary positions over time
BPOS_OVER_TIME = pd.DataFrame([BPOS(BOUNDARY_COORDS[0], BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[3],
                                    BOUNDARY_COORDS[4], BOUNDARY_COORDS[5])])
OSOT = make_dataclass("OSOT", [("strain", float)])
OSCILLATORY_STRAIN_OVER_TIME = pd.DataFrame([OSOT(0)])

# Checking for incompatible conditions
# +--------------------------------------------------------------------+
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

if INCLUDE_DIFFUSION:
    if (len(DIFFUSION_COEFF_MULTI) != N_SPECIES) or (len(BOUNDARY_CONC_INIT_MULTI) != N_SPECIES) or (
            len(BOUNDARY_CONC_FIXED_MULTI) != N_SPECIES):
        print('ERROR: you must define a diffusion coefficient and the boundary conditions for each species simulated')
        critical_error = True
    # Check diffusion values for numerical stability
    dx = 1.0 / (ECM_AGENTS_PER_DIR[0] - 1)
    for i in range(N_SPECIES):
        Fi_x = 3 * (DIFFUSION_COEFF_MULTI[i] * TIME_STEP / (dx * dx))  # this value should be < 0.5
        print('Fi_x value: {0} for species {1}'.format(Fi_x, i + 1))
        if Fi_x > 0.5:
            print(
                'ERROR: diffusion problem is ill conditioned (Fi_x should be < 0.5), check parameters and consider decreasing time step')
            critical_error = True
    dy = 1.0 / (ECM_AGENTS_PER_DIR[1] - 1)
    for i in range(N_SPECIES):
        Fi_y = 3 * (DIFFUSION_COEFF_MULTI[i] * TIME_STEP / (dy * dy))  # this value should be < 0.5
        print('Fi_y value: {0} for species {1}'.format(Fi_y, i + 1))
        if Fi_y > 0.5:
            print(
                'ERROR: diffusion problem is ill conditioned (Fi_y should be < 0.5), check parameters and consider decreasing time step')
            critical_error = True
    dz = 1.0 / (ECM_AGENTS_PER_DIR[2] - 1)
    for i in range(N_SPECIES):
        Fi_z = 3 * (DIFFUSION_COEFF_MULTI[i] * TIME_STEP / (dz * dz))  # this value should be < 0.5
        print('Fi_z value: {0} for species {1}'.format(Fi_z, i + 1))
        if Fi_z > 0.5:
            print(
                'ERROR: diffusion problem is ill conditioned (Fi_z should be < 0.5), check parameters and consider decreasing time step')
            critical_error = True

if INCLUDE_CELLS:
    if MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION < (2 * CELL_RADIUS):
        print('MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION: {0} must be higher than 2 * CELL_RADIUS: 2 * {1}'.format(MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION, CELL_RADIUS))
        critical_error = True

if critical_error:
    quit()

# +====================================================================+
# | FLAMEGPU2 IMPLEMENTATION                                           |
# +====================================================================+


"""
AGENT Files
"""
# Files containing agent functions for agents, which outputs publicly visible properties to a message list

# Agent function files
"""
  ECM
"""
ecm_output_grid_location_data_file = "ecm_output_grid_location_data.cpp"
ecm_boundary_interaction_file = "ecm_boundary_interaction.cpp"
ecm_ecm_interaction_file = "ecm_ecm_interaction.cpp"
ecm_boundary_concentration_conditions_file = "ecm_boundary_concentration_conditions.cpp"
ecm_move_file = "ecm_move.cpp"
ecm_output_spatial_location_data_file = "ecm_output_spatial_location_data.cpp"
ecm_Csp_update_file = "ecm_Csp_update.cpp"

"""
  CELL
"""
cell_output_location_data_file = "cell_output_location_data.cpp"
cell_ecm_interaction_metabolism_file = "cell_ecm_interaction_metabolism.cpp"
cell_move_file = "cell_move.cpp"


model = pyflamegpu.ModelDescription("metabolism")

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
# Model globals
env.newMacroPropertyFloat("C_SP_MACRO", N_SPECIES, ECM_POPULATION_SIZE)
env.newMacroPropertyFloat("BOUNDARY_CONC_INIT_MULTI", N_SPECIES,
                          6)  # a 2D matrix with the 6 boundary conditions (columns) for each species (rows)
env.newMacroPropertyFloat("BOUNDARY_CONC_FIXED_MULTI", N_SPECIES,
                          6)  # a 2D matrix with the 6 boundary conditions (columns) for each species (rows)

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


# Other globals
env.newPropertyFloat("PI", 3.1415)
env.newPropertyUInt("DEBUG_PRINTING", DEBUG_PRINTING)
env.newPropertyFloat("EPSILON", EPSILON)

"""
  LOCATION MESSAGES
"""
ECM_grid_location_message = model.newMessageArray3D("ECM_grid_location_message")
ECM_grid_location_message.setDimensions(ECM_AGENTS_PER_DIR[0], ECM_AGENTS_PER_DIR[1], ECM_AGENTS_PER_DIR[2])
ECM_grid_location_message.newVariableInt("id")
ECM_grid_location_message.newVariableFloat("x")
ECM_grid_location_message.newVariableFloat("y")
ECM_grid_location_message.newVariableFloat("z")
ECM_grid_location_message.newVariableUInt8("grid_i")
ECM_grid_location_message.newVariableUInt8("grid_j")
ECM_grid_location_message.newVariableUInt8("grid_k")
ECM_grid_location_message.newVariableArrayFloat("C_sp", N_SPECIES) 
ECM_grid_location_message.newVariableFloat("k_elast")
ECM_grid_location_message.newVariableUInt8("d_dumping")
ECM_grid_location_message.newVariableFloat("vx")
ECM_grid_location_message.newVariableFloat("vy")
ECM_grid_location_message.newVariableFloat("vz")
# TODO: add or remove variables manually to leave only those that need to be reported. If message type is MessageSpatial3D, variables x, y, z are included internally.

ECM_spatial_location_message = model.newMessageSpatial3D("ECM_spatial_location_message")
ECM_spatial_location_message.setRadius(MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION)
ECM_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS)
ECM_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS)
ECM_spatial_location_message.newVariableInt("id")
ECM_spatial_location_message.newVariableUInt8("grid_i")
ECM_spatial_location_message.newVariableUInt8("grid_j")
ECM_spatial_location_message.newVariableUInt8("grid_k")
ECM_spatial_location_message.newVariableArrayFloat("C_sp", N_SPECIES) 
ECM_spatial_location_message.newVariableFloat("k_elast")
ECM_spatial_location_message.newVariableUInt8("d_dumping")
ECM_spatial_location_message.newVariableFloat("vx")
ECM_spatial_location_message.newVariableFloat("vy")
ECM_spatial_location_message.newVariableFloat("vz")
# TODO: add or remove variables manually to leave only those that need to be reported. If message type is MessageSpatial3D, variables x, y, z are included internally.

CELL_spatial_location_message = model.newMessageSpatial3D("CELL_spatial_location_message")
CELL_spatial_location_message.setRadius(MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION)
CELL_spatial_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS)
CELL_spatial_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS)
CELL_spatial_location_message.newVariableInt("id")
CELL_spatial_location_message.newVariableFloat("vx")
CELL_spatial_location_message.newVariableFloat("vy")
CELL_spatial_location_message.newVariableFloat("vz")
CELL_spatial_location_message.newVariableArrayFloat("k_consumption", N_SPECIES) 
CELL_spatial_location_message.newVariableArrayFloat("k_production", N_SPECIES) 
CELL_spatial_location_message.newVariableArrayFloat("C_sp", N_SPECIES) 
# TODO: add or remove variables manually to leave only those that need to be reported. If message type is MessageSpatial3D, variables x, y, z are included internally.


"""
  AGENTS
"""
"""
  ECM agent
"""
ECM_agent = model.newAgent("ECM")
ECM_agent.newVariableInt("id", 0)
ECM_agent.newVariableFloat("x", 0.0)
ECM_agent.newVariableFloat("y", 0.0)
ECM_agent.newVariableFloat("z", 0.0)
ECM_agent.newVariableUInt8("grid_i", 0)
ECM_agent.newVariableUInt8("grid_j", 0)
ECM_agent.newVariableUInt8("grid_k", 0)
ECM_agent.newVariableArrayFloat("C_sp", N_SPECIES) 
# TODO: default array values must be explicitly defined when initializing agent populations
ECM_agent.newVariableFloat("k_elast")
ECM_agent.newVariableUInt8("d_dumping")
ECM_agent.newVariableFloat("vx")
ECM_agent.newVariableFloat("vy")
ECM_agent.newVariableFloat("vz")
ECM_agent.newRTCFunctionFile("ecm_output_grid_location_data", ecm_output_grid_location_data_file).setMessageOutput("ECM_grid_location_message ")
ECM_agent.newRTCFunctionFile("ecm_boundary_interaction", ecm_boundary_interaction_file)
# TODO: connect message input for ECM::ecm_ecm_interaction
ECM_agent.newRTCFunctionFile("ecm_ecm_interaction", ecm_ecm_interaction_file)
ECM_agent.newRTCFunctionFile("ecm_boundary_concentration_conditions", ecm_boundary_concentration_conditions_file)
ECM_agent.newRTCFunctionFile("ecm_move", ecm_move_file)
ECM_agent.newRTCFunctionFile("ecm_output_spatial_location_data", ecm_output_spatial_location_data_file).setMessageOutput("ECM_spatial_location_message ")
ECM_agent.newRTCFunctionFile("ecm_Csp_update", ecm_Csp_update_file)

"""
  CELL agent
"""
CELL_agent = model.newAgent("CELL")
CELL_agent.newVariableInt("id", 0)
CELL_agent.newVariableFloat("x", 0.0)
CELL_agent.newVariableFloat("y", 0.0)
CELL_agent.newVariableFloat("z", 0.0)
CELL_agent.newVariableFloat("vx", 0.0)
CELL_agent.newVariableFloat("vy", 0.0)
CELL_agent.newVariableFloat("vz", 0.0)
CELL_agent.newVariableArrayFloat("k_consumption", N_SPECIES) 
# TODO: default array values must be explicitly defined when initializing agent populations
CELL_agent.newVariableArrayFloat("k_production", N_SPECIES) 
# TODO: default array values must be explicitly defined when initializing agent populations
CELL_agent.newVariableArrayFloat("C_sp", N_SPECIES) 
# TODO: default array values must be explicitly defined when initializing agent populations
CELL_agent.newRTCFunctionFile("cell_output_location_data", cell_output_location_data_file).setMessageOutput("CELL_spatial_location_message ")
CELL_agent.newRTCFunctionFile("cell_ecm_interaction_metabolism", cell_ecm_interaction_metabolism_file).setMessageInput("ECM_spatial_location_message ")
CELL_agent.newRTCFunctionFile("cell_move", cell_move_file)


"""
  Population initialisation functions
"""


# This class is used to ensure that corner agents are assigned the first 8 ids
class initAgentPopulations(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # TODO: code the initialization of agents. For example:
        # instance = FLAMEGPU.agent("AGENT_NAME").newAgent()
        # instance.setVariableFloat("VARX", 0.0)
        
        return


# Add function callback to INIT functions for population generation
initialAgentPopulation = initAgentPopulations()
model.addInitFunction(initialAgentPopulation)
# Initialize the MacroProperties
class initMacroProperties(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # Get property handles and modify their values.  Replace getMacroPropertyFloat by getMacroPropertyInt if needed
        C_SP_MACRO = FLAMEGPU.environment.getMacroPropertyFloat("C_SP_MACRO")
        # TODO: initialize values. All 0 by default

        return

initialMacroProperties = initMacroProperties()
model.addInitFunction(initialMacroProperties)

"""
  STEP FUNCTIONS
"""
# pyflamegpu requires step functions to be a class which extends the StepFunction base class.
# This class must extend the handle function
# TODO: remove unnecessary parts
class SaveDataToFile(pyflamegpu.HostFunction):
    def __init__(self):
        global ECM_AGENTS_PER_DIR, N_VASCULARIZATION_POINTS
        super().__init__()
        self.header = list()
        self.header.append("# vtk DataFile Version 3.0")
        self.header.append("ECM data")
        self.header.append("ASCII")
        self.header.append("DATASET POLYDATA")
        self.header.append("POINTS {} float".format(8 + ECM_AGENTS_PER_DIR[0] * ECM_AGENTS_PER_DIR[1] * ECM_AGENTS_PER_DIR[2]))  # number of ECM agents + 8 corners
        # self.header.append("POINTS {} float".format(8))
        self.domaindata = list()
        self.domaindata.append("POLYGONS 6 30")
        cube_conn = [[4, 0, 3, 7, 4], [4, 1, 2, 6, 5], [4, 1, 0, 4, 5], [4, 2, 3, 7, 6], [4, 0, 1, 2, 3],
                     [4, 4, 5, 6, 7]]
        for i in range(len(cube_conn)):
            for j in range(len(cube_conn[i])):
                if j > 0:
                    cube_conn[i][j] = cube_conn[i][j] + ECM_AGENTS_PER_DIR[0] * ECM_AGENTS_PER_DIR[1] * ECM_AGENTS_PER_DIR[2]
            self.domaindata.append(' '.join(str(x) for x in cube_conn[i]))

        # self.domaindata.append("4 0 3 7 4")
        # self.domaindata.append("4 1 2 6 5")
        # self.domaindata.append("4 1 0 4 5")
        # self.domaindata.append("4 2 3 7 6")
        # self.domaindata.append("4 0 1 2 3")
        # self.domaindata.append("4 4 5 6 7")
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
        # VASCULARIZATION
        self.vascularizationdata = list()  # a different file is created to show the position of the vascularization points
        self.vascularizationdata.append("# vtk DataFile Version 3.0")
        self.vascularizationdata.append("Vascularization points")
        self.vascularizationdata.append("ASCII")
        self.vascularizationdata.append("DATASET UNSTRUCTURED_GRID")
        # CELLS
        self.celldata = list()  # a different file is created to show cell agent data
        self.celldata.append("# vtk DataFile Version 3.0")
        self.celldata.append("Cell agents")
        self.celldata.append("ASCII")
        self.celldata.append("DATASET UNSTRUCTURED_GRID")

    def run(self, FLAMEGPU):
        global SAVE_DATA_TO_FILE, SAVE_EVERY_N_STEPS, N_SPECIES
        global RES_PATH, ENSEMBLE
        global fileCounter, INCLUDE_VASCULARIZATION
        global INCLUDE_CELLS
        BUCKLING_COEFF_D0 = FLAMEGPU.environment.getPropertyFloat("BUCKLING_COEFF_D0")
        STRAIN_STIFFENING_COEFF_DS = FLAMEGPU.environment.getPropertyFloat("STRAIN_STIFFENING_COEFF_DS")
        CRITICAL_STRAIN = FLAMEGPU.environment.getPropertyFloat("CRITICAL_STRAIN")
        stepCounter = FLAMEGPU.getStepCounter() + 1
        
        coord_boundary = list(FLAMEGPU.environment.getPropertyArrayFloat("COORDS_BOUNDARIES"))

        if SAVE_DATA_TO_FILE:
            if stepCounter % SAVE_EVERY_N_STEPS == 0 or stepCounter == 1:

                if INCLUDE_VASCULARIZATION:
                    vasc_coords = list()
                    file_name = 'vascularization_points_t{:04d}.vtk'.format(stepCounter)
                    file_path = RES_PATH / file_name
                    vasc_agent = FLAMEGPU.agent("VASCULARIZATION")
                    av = vasc_agent.getPopulationData()  # this returns a DeviceAgentVector
                    for ai in av:
                        coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
                        vasc_coords.append(coords_ai)
                    with open(str(file_path), 'w') as file:
                        for line in self.vascularizationdata:
                            file.write(line + '\n')
                        file.write("POINTS {} float \n".format(FLAMEGPU.environment.getPropertyUInt(
                            "N_VASCULARIZATION_POINTS")))  # number of vascularization agents
                        for coords_ai in vasc_coords:
                            file.write("{} {} {} \n".format(coords_ai[0], coords_ai[1], coords_ai[2]))

                if INCLUDE_CELLS:
                    cell_coords = list()
                    cell_velocity = list()
                    cell_orientation = list()
                    cell_alignment = list()
                    cell_radius = list()
                    cell_clock = list()
                    cell_cycle_phase = list()
                    file_name = 'cells_t{:04d}.vtk'.format(stepCounter)
                    file_path = RES_PATH / file_name
                    cell_agent = FLAMEGPU.agent("CELL")
                    cell_agent.sortInt("id", pyflamegpu.HostAgentAPI.Asc); # this is critical to ensure cell ids are kept in order for visualization
                    av = cell_agent.getPopulationData()  # this returns a DeviceAgentVector
                    for ai in av:
                        coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
                        velocity_ai = (ai.getVariableFloat("vx"), ai.getVariableFloat("vy"), ai.getVariableFloat("vz"))
                        orientation_ai = (
                        ai.getVariableFloat("orx"), ai.getVariableFloat("ory"), ai.getVariableFloat("orz"))
                        alignment_ai = ai.getVariableFloat("alignment")
                        radius_ai = ai.getVariableFloat("radius")
                        clock_ai = ai.getVariableFloat("clock")
                        cycle_phase_ai = ai.getVariableInt("cycle_phase")
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

                coords = list()
                velocity = list()
                orientation = list()
                alignment = list()
                gel_conc = list()
                force = list()
                elastic_energy = list()
                concentration_multi = list()  # this is a list of tuples. Each tuple has N_SPECIES elements
                av = agent.getPopulationData()  # this returns a DeviceAgentVector
                for ai in av:
                    coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
                    velocity_ai = (ai.getVariableFloat("vx"), ai.getVariableFloat("vy"), ai.getVariableFloat("vz"))
                    force_ai = (ai.getVariableFloat("fx"), ai.getVariableFloat("fy"), ai.getVariableFloat("fz"))
                    orientation_ai = (
                    ai.getVariableFloat("orx"), ai.getVariableFloat("ory"), ai.getVariableFloat("orz"))
                    alignment.append(ai.getVariableFloat("alignment"))
                    gel_conc.append(ai.getVariableFloat("gel_conc"))
                    coords.append(coords_ai)
                    velocity.append(velocity_ai)
                    force.append(force_ai)
                    orientation.append(orientation_ai)
                    elastic_energy.append(ai.getVariableFloat("elastic_energy"))
                    concentration_multi.append(ai.getVariableArrayFloat("concentration_multi"))
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

                    file.write("POINT_DATA {} \n".format(8 + ECM_AGENTS_PER_DIR[0] * ECM_AGENTS_PER_DIR[1] * ECM_AGENTS_PER_DIR[2]))  # 8 corners + number of ECM agents

                    file.write(
                        "SCALARS is_corner int 1" + '\n')  # create this variable to remove them from representations
                    file.write("LOOKUP_TABLE default" + '\n')

                    for ee_ai in elastic_energy:
                        file.write("{0} \n".format(0))
                    for i in range(8):
                        file.write("1 \n")  # boundary corners

                    file.write("SCALARS elastic_energy float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    for ee_ai in elastic_energy:
                        file.write("{:.4f} \n".format(ee_ai))
                    for i in range(8):
                        file.write("0.0 \n")  # boundary corners

                    file.write("SCALARS alignment float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    for a_ai in alignment:
                        file.write("{:.4f} \n".format(a_ai))
                    for i in range(8):
                        file.write("0.0 \n")  # boundary corners

                    file.write("SCALARS gel_conc float 1" + '\n')
                    file.write("LOOKUP_TABLE default" + '\n')
                    for gc_ai in gel_conc:
                        file.write("{:.4f} \n".format(gc_ai))
                    for i in range(8):
                        file.write("0.0 \n")  # boundary corners

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

                    file.write("VECTORS orientation float" + '\n')
                    for o_ai in orientation:
                        file.write("{:.4f} {:.4f} {:.4f} \n".format(o_ai[0], o_ai[1], o_ai[2]))
                    for i in range(8):
                        file.write("0.0 0.0 0.0 \n")  # boundary corners

                print("... succesful save ")
                print("=================================")
                          

sdf = SaveDataToFile()
model.addStepFunction(sdf)


"""
  END OF STEP FUNCTIONS
"""

"""
  Control flow
"""
layer_count = 0
# L1_Agent_Locations
layer_count += 1
model.newLayer("L1_Agent_Locations").addAgentFunction("CELL", "cell_output_location_data")
model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_output_grid_location_data")
model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_output_spatial_location_data")
# L2_Boundary_Interactions
layer_count += 1
model.newLayer("L2_Boundary_Interactions").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
model.Layer("L2_Boundary_Interactions").addAgentFunction("ECM", "ecm_boundary_interaction")
# L3_Metabolism
layer_count += 1
model.newLayer("L3_Metabolism").addAgentFunction("CELL", "cell_ecm_interaction_metabolism")
# L4_ECM_Csp_Update
layer_count += 1
model.newLayer("L4_ECM_Csp_Update").addAgentFunction("ECM", "ecm_Csp_update")
# L5_Diffusion
layer_count += 1
model.newLayer("L5_Diffusion").addAgentFunction("ECM", "ecm_ecm_interaction")
# L6_Diffusion_Boundary
layer_count += 1
model.newLayer("L6_Diffusion_Boundary").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
# L7_Agent_Movement
layer_count += 1
model.newLayer("L7_Agent_Movement").addAgentFunction("CELL", "cell_move")
model.Layer("L7_Agent_Movement").addAgentFunction("ECM", "ecm_move")


"""
  Logging
"""

# Create and configure logging details 
logging_config = pyflamegpu.LoggingConfig(model)

ECM_agent_log = logging_config.agent("ECM")
ECM_agent_log.logCount()

CELL_agent_log = logging_config.agent("CELL")
CELL_agent_log.logCount()

step_log = pyflamegpu.StepLoggingConfig(logging_config)
step_log.setFrequency(1) # if 1, data will be logged every step

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

"""
  Create Visualisation
"""
if pyflamegpu.VISUALISATION and VISUALISATION:
    vis = simulation.getVisualisation()
    # Configure vis
    domain_width = ?
    INIT_CAM = ? # A value of the position of the domain by the end of the simulation, multiplied by 5, looks nice
    vis.setInitialCameraLocation(0.0, 0.0, INIT_CAM)
    vis.setCameraSpeed(? * domain_width) # values <<1 (e.g. 0.002) work fine
    if DEBUG_PRINTING:
        vis.setSimulationSpeed(1)
    vis.setBeginPaused(True)

    CELL_vis_agent = vis.addAgent("CELL")
    # Position vars are named x, y, z so they are used by default
    CELL_vis_agent.setModel(pyflamegpu.ICOSPHERE)
    CELL_vis_agent.setModelScale(? * domain_width) # values <<1 (e.g. 0.03) work fine
    CELL_vis_agent.setColor(pyflamegpu.Color("#00aaff"))

    ECM_vis_agent = vis.addAgent("ECM")
    # Position vars are named x, y, z so they are used by default
    ECM_vis_agent.setModel(pyflamegpu.CUBE)
    ECM_vis_agent.setModelScale(? * domain_width) # values <<1 (e.g. 0.03) work fine
    ECM_vis_agent.setColor(pyflamegpu.Color("#ffaa00"))

    coord_boundary = list(env.getPropertyArrayFloat("BOUNDARY_COORDS"))
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



def manageLogs(steps, is_ensemble, idx):
    counter = 0
    for step in steps:
        stepcount = step.getStepCount()
        if stepcount % SAVE_EVERY_N_STEPS == 0 or stepcount == 1:
            ECM_agents = step.getAgent("ECM")
            ECM_agent_counts[counter] = ECM_agents.getCount()

            CELL_agents = step.getAgent("CELL")
            CELL_agent_counts[counter] = CELL_agents.getCount()
            # TODO: print/plot/save data as needed

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