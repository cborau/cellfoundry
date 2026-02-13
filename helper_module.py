import math
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

def compute_expected_boundary_pos_from_corners(
    BOUNDARY_COORDS,
    BOUNDARY_DISP_RATES,
    BOUNDARY_DISP_RATES_PARALLEL,
    STEPS,
    TIME_STEP):
    """
    Compute MIN_EXPECTED_BOUNDARY_POS and MAX_EXPECTED_BOUNDARY_POS as the global min/max
    across (x,y,z) of the 8 corners after applying boundary motion.
    """
    x_max0, x_min0, y_max0, y_min0, z_max0, z_min0 = BOUNDARY_COORDS
    R = BOUNDARY_DISP_RATES
    P = BOUNDARY_DISP_RATES_PARALLEL
    T = STEPS * TIME_STEP

    # Face displacement vectors (vx, vy, vz)
    # +X: normal -> x, parallel -> y,z
    v_plusX  = (R[0],  P[0],  P[1])
    v_minusX = (R[1],  P[2],  P[3])

    # +Y: normal -> y, parallel -> x,z
    v_plusY  = (P[4],  R[2],  P[5])
    v_minusY = (P[6],  R[3],  P[7])

    # +Z: normal -> z, parallel -> x,y
    v_plusZ  = (P[8],  P[9],  R[4])
    v_minusZ = (P[10], P[11], R[5])

    # Helper: sum three face vectors
    def add3(a, b, c):
        return (a[0] + b[0] + c[0],
                a[1] + b[1] + c[1],
                a[2] + b[2] + c[2])

    # 8 corners: (x choice, y choice, z choice) and their 3 contributing faces
    corners = [
        # x_max, y_max, z_max affected by +X, +Y, +Z
        ((x_max0, y_max0, z_max0), add3(v_plusX,  v_plusY,  v_plusZ)),
        ((x_max0, y_max0, z_min0), add3(v_plusX,  v_plusY,  v_minusZ)),
        ((x_max0, y_min0, z_max0), add3(v_plusX,  v_minusY, v_plusZ)),
        ((x_max0, y_min0, z_min0), add3(v_plusX,  v_minusY, v_minusZ)),

        ((x_min0, y_max0, z_max0), add3(v_minusX, v_plusY,  v_plusZ)),
        ((x_min0, y_max0, z_min0), add3(v_minusX, v_plusY,  v_minusZ)),
        ((x_min0, y_min0, z_max0), add3(v_minusX, v_minusY, v_plusZ)),
        ((x_min0, y_min0, z_min0), add3(v_minusX, v_minusY, v_minusZ)),
    ]

    moved_corners = []
    for (x0, y0, z0), (vx, vy, vz) in corners:
        moved_corners.append((x0 + vx * T, y0 + vy * T, z0 + vz * T))

    # global min/max across all coordinates of all moved corners
    flat = [c for pt in moved_corners for c in pt]
    min_expected_pos = min(flat)
    max_expected_pos = max(flat)

    return min_expected_pos, max_expected_pos, moved_corners


def load_fibre_network(
    file_name,
    boundary_coords,
    epsilon,
    fibre_segment_equilibrium_distance,
):
    critical_error = False
    nodes = None
    connectivity = None
    n_fiber = None

    if os.path.exists(file_name):
        # print(f'Loading network from {file_name}')
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            nodes = data['node_coords']
            connectivity = data['connectivity']
            network_parameters = data.get('network_parameters')
        if network_parameters:
            # print('Loaded network parameters:')
            # for key, value in network_parameters.items():
            #     print(f'  {key}: {value}')

            domain_lx = abs(boundary_coords[1] - boundary_coords[0])
            domain_ly = abs(boundary_coords[3] - boundary_coords[2])
            domain_lz = abs(boundary_coords[5] - boundary_coords[4])

            expected_lx = network_parameters.get('LX')
            expected_ly = network_parameters.get('LY')
            expected_lz = network_parameters.get('LZ')
            expected_edge_length = network_parameters.get('EDGE_LENGTH')
            n_fiber = network_parameters.get('N_FIBER')

            if expected_lx is not None and not math.isclose(domain_lx, expected_lx, rel_tol=0.0, abs_tol=epsilon):
                print('ERROR: Network LX does not match domain size.')
                critical_error = True
            if expected_ly is not None and not math.isclose(domain_ly, expected_ly, rel_tol=0.0, abs_tol=epsilon):
                print('ERROR: Network LY does not match domain size.')
                critical_error = True
            if expected_lz is not None and not math.isclose(domain_lz, expected_lz, rel_tol=0.0, abs_tol=epsilon):
                print('ERROR: Network LZ does not match domain size.')
                critical_error = True
            if expected_edge_length is not None and not math.isclose(
                fibre_segment_equilibrium_distance,
                expected_edge_length,
                rel_tol=0.0,
                abs_tol=epsilon,
            ):
                print(
                    'WARNING: FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE does not match EDGE_LENGTH from network file. '
                    f'Updating FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE to match EDGE_LENGTH {expected_edge_length}.'
                )
                fibre_segment_equilibrium_distance = expected_edge_length
        else:
            print('WARNING: network_parameters not found in network_3d.pkl. Skipping compatibility checks.')
        # print(f'Network loaded: {nodes.shape[0]} nodes, {len(connectivity)} fibers')
    else:
        print(f"ERROR: file {file_name} containing network nodes and connectivity was not found")
        critical_error = True
        return nodes, connectivity, n_fiber, fibre_segment_equilibrium_distance, critical_error

    msg_wrong_network_dimensions = (
        "WARNING: Fibre network nodes do not coincide with boundary faces on at least two axes. "
        "Check NODE_COORDS vs BOUNDARY_COORDS or regenerate the network."
    )

    x_max, x_min, y_max, y_min, z_max, z_min = boundary_coords
    axes_with_both_faces = 0

    has_x_pos = np.any(np.isclose(nodes[:, 0], x_max, atol=epsilon))
    has_x_neg = np.any(np.isclose(nodes[:, 0], x_min, atol=epsilon))
    if has_x_pos and has_x_neg:
        axes_with_both_faces += 1

    has_y_pos = np.any(np.isclose(nodes[:, 1], y_max, atol=epsilon))
    has_y_neg = np.any(np.isclose(nodes[:, 1], y_min, atol=epsilon))
    if has_y_pos and has_y_neg:
        axes_with_both_faces += 1

    has_z_pos = np.any(np.isclose(nodes[:, 2], z_max, atol=epsilon))
    has_z_neg = np.any(np.isclose(nodes[:, 2], z_min, atol=epsilon))
    if has_z_pos and has_z_neg:
        axes_with_both_faces += 1

    if axes_with_both_faces < 2:
        print(msg_wrong_network_dimensions)
        critical_error = True

    return nodes, connectivity, n_fiber, fibre_segment_equilibrium_distance, critical_error


#Helper functions for agent initialization
# +--------------------------------------------------------------------+
def getRandomCoords3D(n, minx, maxx, miny, maxy, minz, maxz):
    """
    Generates an array (nx3 matrix) of random numbers with specific ranges for each column.

    Args:
        n (int): Number of rows in the array.
        minx, maxx (float): Range for the values in the first column [minx, maxx].
        miny, maxy (float): Range for the values in the second column [miny, maxy].
        minz, maxz (float): Range for the values in the third column [minz, maxz].

    Returns:
        numpy.ndarray: Array of random numbers with shape (n, 3).
    """
    np.random.seed()
    random_array = np.random.uniform(low=[minx, miny, minz], high=[maxx, maxy, maxz], size=(n, 3))
    return random_array
    

def randomVector3D():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution

    Returns
    -------
    (x,y,z) : tuple
        Coordinates of the vector.
    """
    np.random.seed()
    phi = np.random.uniform(0.0, np.pi * 2.0)
    costheta = np.random.uniform(-1.0, 1.0)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (x, y, z)


def getRandomVectors3D(n_vectors: int):
    """
    Generates an array of random 3D unit vectors (directions) with a uniform spherical distribution

    Parameters
    ----------
    n_vectors : int
        Number of vectors to be generated
    Returns
    -------
    v_array : Numpy array
        Coordinates of the vectors. Shape: [n_vectors, 3].
    """
    v_array = np.zeros((n_vectors, 3))
    for i in range(n_vectors):
        vi = randomVector3D()
        v_array[i, :] = np.array(vi, dtype='float')

    return v_array


def getFixedVectors3D(n_vectors: int, v_dir: np.array):
    """
    Generates an array of 3D unit vectors (directions) in the specified direction

    Parameters
    ----------
    n_vectors : int
        Number of vectors to be generated
    v_dir : Numpy array
        Direction of the vectors
    Returns
    -------
    v_array : Numpy array
        Coordinates of the vectors. Shape: [n_vectors, 3].
    """
    v_array = np.tile(v_dir, (n_vectors, 1))

    return v_array
    
    
def getRandomCoordsAroundPoint(n, px, py, pz, radius, on_surface=False):
    """
    Generates N random 3D coordinates within a sphere of a specific radius around a central point.

    Parameters
    ----------
    n : int
        The number of random coordinates to generate.
    px : float
        The x-coordinate of the central point.
    py : float
        The y-coordinate of the central point.
    pz : float
        The z-coordinate of the central point.
    radius : float
        The radius of the sphere.
    on_surface : bool
        If True, points lie on the sphere surface; otherwise, points are within the sphere.

    Returns
    -------
    coords
        A numpy array of randomly generated 3D coordinates with shape (n, 3).
    """
    central_point = np.array([px, py, pz])
    rand_dirs = getRandomVectors3D(n)
    coords = np.zeros((n, 3))
    np.random.seed()
    for i in range(n):
        if on_surface:
            radius_i = radius
        else:
            radius_i = np.random.uniform(0.0, 1.0) * radius
        coords[i, :] = central_point + np.array(rand_dirs[i, :] * radius_i, dtype='float')
    

    return coords


class ModelParameterConfig:
    def __init__(
        self,
        save_every_n_steps: int = None,
        ecm_agents_per_dir: list = None,
        time_step: float = None,
        steps: int = None,
        # Domain / boundary
        boundary_coords: list = None,
        boundary_disp_rates: list = None,
        boundary_disp_rates_parallel: list = None,
        poisson_dirs: list = None,
        allow_boundary_elastic_movement: list = None,
        boundary_stiffness: list = None,
        boundary_dumping: list = None,
        clamp_agent_touching_boundary: list = None,
        allow_agent_sliding: list = None,
        moving_boundaries: bool = None,
        epsilon: float = None,
        # ECM mechanics
        ecm_k_elast: float = None,
        ecm_d_dumping: float = None,
        ecm_mass: float = None,
        ecm_eta: float = None,
        ecm_gel_concentration: float = None,
        ecm_ecm_equilibrium_distance: float = None,
        ecm_boundary_interaction_radius: float = None,
        ecm_boundary_equilibrium_distance: float = None,
        ecm_voxel_volume: float = None,
        ecm_population_size: int = None,
        include_fiber_alignment: int = None,
        ecm_orientation_rate: float = None,
        buckling_coeff_d0: float = None,
        strain_stiffening_coeff_ds: float = None,
        critical_strain: float = None,
        # Fibre network
        include_fibre_network: bool = None,
        max_connectivity: int = None,
        fibre_segment_k_elast: float = None,
        fibre_segment_d_dumping: float = None,
        fibre_segment_mass: float = None,
        fibre_segment_equilibrium_distance: float = None,
        fibre_node_boundary_interaction_radius: float = None,
        fibre_node_boundary_equilibrium_distance: float = None,
        max_search_radius_fnodes: float = None,
        # Diffusion
        include_diffusion: bool = None,
        n_species: int = None,
        diffusion_coeff_multi: list = None,
        boundary_conc_init_multi: list = None,
        boundary_conc_fixed_multi: list = None,
        init_ecm_concentration_vals: list = None,
        init_ecm_sat_concentration_vals: list = None,
        unstable_diffusion: bool = None,
        # Cells
        include_cells: bool = None,
        include_cell_orientation: bool = None,
        include_cell_cell_interaction: bool = None,
        include_cell_cycle: bool = None,
        periodic_boundaries_for_cells: bool = None,
        n_cells: int = None,
        cell_k_elast: float = None,
        cell_d_dumping: float = None,
        cell_radius: float = None,
        cell_speed_ref: float = None,
        cell_orientation_rate: float = None,
        max_search_radius_cell_ecm_interaction: float = None,
        max_search_radius_cell_cell_interaction: float = None,
        cell_cycle_duration: float = None,
        cycle_phase_g1_duration: float = None,
        cycle_phase_s_duration: float = None,
        cycle_phase_g2_duration: float = None,
        cycle_phase_m_duration: float = None,
        cycle_phase_g1_start: float = None,
        cycle_phase_s_start: float = None,
        cycle_phase_g2_start: float = None,
        cycle_phase_m_start: float = None,
        init_cell_concentration_vals: list = None,
        init_cell_conc_mass_vals: list = None,
        init_cell_consumption_rates: list = None,
        init_cell_production_rates: list = None,
        init_cell_reaction_rates: list = None,
        # Focal adhesions
        include_focal_adhesions: bool = None,
        init_n_focad_per_cell: int = None,
        n_anchor_points: int = None,
        max_search_radius_focad: float = None,
        focad_rest_length_0: float = None,
        focad_k_fa: float = None,
        focad_f_max: float = None,
        focad_v_c: float = None,
        focad_k_on: float = None,
        focad_k_off_0: float = None,
        focad_f_c: float = None,
        focad_k_reinf: float = None,
        # Oscillatory assay
        oscillatory_shear_assay: bool = None,
        max_strain: float = None,
        oscillatory_amplitude: float = None,
        oscillatory_freq: float = None,
        oscillatory_w: float = None,
        min_expected_boundary_pos: float = None,
        max_expected_boundary_pos: float = None,
        # Vascularization
        include_vascularization: bool = None,
        init_vascularization_concentration_vals: list = None,
        # Misc / logging
        save_pickle: bool = None,
        show_plots: bool = None,
        save_data_to_file: bool = None,
        res_path: str = None,
        **kwargs,
    ):
        self.SAVE_EVERY_N_STEPS = save_every_n_steps
        self.ECM_AGENTS_PER_DIR = ecm_agents_per_dir
        self.TIME_STEP = time_step
        self.STEPS = steps
        self.BOUNDARY_COORDS = boundary_coords
        self.BOUNDARY_DISP_RATES = boundary_disp_rates
        self.BOUNDARY_DISP_RATES_PARALLEL = boundary_disp_rates_parallel
        self.POISSON_DIRS = poisson_dirs
        self.ALLOW_BOUNDARY_ELASTIC_MOVEMENT = allow_boundary_elastic_movement
        self.BOUNDARY_STIFFNESS = boundary_stiffness
        self.BOUNDARY_DUMPING = boundary_dumping
        self.CLAMP_AGENT_TOUCHING_BOUNDARY = clamp_agent_touching_boundary
        self.ALLOW_AGENT_SLIDING = allow_agent_sliding
        self.MOVING_BOUNDARIES = moving_boundaries
        self.EPSILON = epsilon
        self.ECM_K_ELAST = ecm_k_elast
        self.ECM_D_DUMPING = ecm_d_dumping
        self.ECM_MASS = ecm_mass
        self.ECM_ETA = ecm_eta
        self.ECM_GEL_CONCENTRATION = ecm_gel_concentration
        self.ECM_ECM_EQUILIBRIUM_DISTANCE = ecm_ecm_equilibrium_distance
        self.ECM_BOUNDARY_INTERACTION_RADIUS = ecm_boundary_interaction_radius
        self.ECM_BOUNDARY_EQUILIBRIUM_DISTANCE = ecm_boundary_equilibrium_distance
        self.ECM_VOXEL_VOLUME = ecm_voxel_volume
        self.ECM_POPULATION_SIZE = ecm_population_size
        self.INCLUDE_FIBER_ALIGNMENT = include_fiber_alignment
        self.ECM_ORIENTATION_RATE = ecm_orientation_rate
        self.BUCKLING_COEFF_D0 = buckling_coeff_d0
        self.STRAIN_STIFFENING_COEFF_DS = strain_stiffening_coeff_ds
        self.CRITICAL_STRAIN = critical_strain
        self.INCLUDE_FIBRE_NETWORK = include_fibre_network
        self.MAX_CONNECTIVITY = max_connectivity
        self.FIBRE_SEGMENT_K_ELAST = fibre_segment_k_elast
        self.FIBRE_SEGMENT_D_DUMPING = fibre_segment_d_dumping
        self.FIBRE_SEGMENT_MASS = fibre_segment_mass
        self.FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = fibre_segment_equilibrium_distance
        self.FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS = fibre_node_boundary_interaction_radius
        self.FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE = fibre_node_boundary_equilibrium_distance
        self.MAX_SEARCH_RADIUS_FNODES = max_search_radius_fnodes
        self.INCLUDE_DIFFUSION = include_diffusion
        self.N_SPECIES = n_species
        self.DIFFUSION_COEFF_MULTI = diffusion_coeff_multi
        self.BOUNDARY_CONC_INIT_MULTI = boundary_conc_init_multi
        self.BOUNDARY_CONC_FIXED_MULTI = boundary_conc_fixed_multi
        self.INIT_ECM_CONCENTRATION_VALS = init_ecm_concentration_vals
        self.INIT_ECM_SAT_CONCENTRATION_VALS = init_ecm_sat_concentration_vals
        self.UNSTABLE_DIFFUSION = unstable_diffusion
        self.INCLUDE_CELLS = include_cells
        self.INCLUDE_CELL_ORIENTATION = include_cell_orientation
        self.INCLUDE_CELL_CELL_INTERACTION = include_cell_cell_interaction
        self.INCLUDE_CELL_CYCLE = include_cell_cycle
        self.PERIODIC_BOUNDARIES_FOR_CELLS = periodic_boundaries_for_cells
        self.N_CELLS = n_cells
        self.CELL_K_ELAST = cell_k_elast
        self.CELL_D_DUMPING = cell_d_dumping
        self.CELL_RADIUS = cell_radius
        self.CELL_SPEED_REF = cell_speed_ref
        self.CELL_ORIENTATION_RATE = cell_orientation_rate
        self.MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION = max_search_radius_cell_ecm_interaction
        self.MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION = max_search_radius_cell_cell_interaction
        self.CELL_CYCLE_DURATION = cell_cycle_duration
        self.CYCLE_PHASE_G1_DURATION = cycle_phase_g1_duration
        self.CYCLE_PHASE_S_DURATION = cycle_phase_s_duration
        self.CYCLE_PHASE_G2_DURATION = cycle_phase_g2_duration
        self.CYCLE_PHASE_M_DURATION = cycle_phase_m_duration
        self.CYCLE_PHASE_G1_START = cycle_phase_g1_start
        self.CYCLE_PHASE_S_START = cycle_phase_s_start
        self.CYCLE_PHASE_G2_START = cycle_phase_g2_start
        self.CYCLE_PHASE_M_START = cycle_phase_m_start
        self.INIT_CELL_CONCENTRATION_VALS = init_cell_concentration_vals
        self.INIT_CELL_CONC_MASS_VALS = init_cell_conc_mass_vals
        self.INIT_CELL_CONSUMPTION_RATES = init_cell_consumption_rates
        self.INIT_CELL_PRODUCTION_RATES = init_cell_production_rates
        self.INIT_CELL_REACTION_RATES = init_cell_reaction_rates
        self.INCLUDE_FOCAL_ADHESIONS = include_focal_adhesions
        self.INIT_N_FOCAD_PER_CELL = init_n_focad_per_cell
        self.N_ANCHOR_POINTS = n_anchor_points
        self.MAX_SEARCH_RADIUS_FOCAD = max_search_radius_focad
        self.FOCAD_REST_LENGTH_0 = focad_rest_length_0
        self.FOCAD_K_FA = focad_k_fa
        self.FOCAD_F_MAX = focad_f_max
        self.FOCAD_V_C = focad_v_c
        self.FOCAD_K_ON = focad_k_on
        self.FOCAD_K_OFF_0 = focad_k_off_0
        self.FOCAD_F_C = focad_f_c
        self.FOCAD_K_REINF = focad_k_reinf
        self.OSCILLATORY_SHEAR_ASSAY = oscillatory_shear_assay
        self.MAX_STRAIN = max_strain
        self.OSCILLATORY_AMPLITUDE = oscillatory_amplitude
        self.OSCILLATORY_FREQ = oscillatory_freq
        self.OSCILLATORY_W = oscillatory_w
        self.MIN_EXPECTED_BOUNDARY_POS = min_expected_boundary_pos
        self.MAX_EXPECTED_BOUNDARY_POS = max_expected_boundary_pos
        self.INCLUDE_VASCULARIZATION = include_vascularization
        self.INIT_VASCULARIZATION_CONCENTRATION_VALS = init_vascularization_concentration_vals
        self.SAVE_PICKLE = save_pickle
        self.SHOW_PLOTS = show_plots
        self.SAVE_DATA_TO_FILE = save_data_to_file
        self.RES_PATH = res_path
        self.EXTRA_PARAMS = kwargs

    def print_all(self):
        attributes = vars(self)
        for attribute, value in attributes.items():
            print(f"{attribute}: {value}")

    def print_summary(self):
        print()
        print("=== ModelParameterConfig Summary ===")
        print(f"STEPS: {self.STEPS} | TIME_STEP: {self.TIME_STEP}")
        print(f"ECM_AGENTS_PER_DIR: {self.ECM_AGENTS_PER_DIR}")
        print(f"INCLUDE_DIFFUSION: {self.INCLUDE_DIFFUSION} | N_SPECIES: {self.N_SPECIES}")
        print(f"INCLUDE_CELLS: {self.INCLUDE_CELLS} | N_CELLS: {self.N_CELLS}")
        if self.INCLUDE_FOCAL_ADHESIONS and self.N_CELLS is not None and self.INIT_N_FOCAD_PER_CELL is not None:
            focad_count = self.N_CELLS * self.INIT_N_FOCAD_PER_CELL
            print(
                f"INCLUDE_FOCAL_ADHESIONS: True | FOCAD_PER_CELL: {self.INIT_N_FOCAD_PER_CELL} | FOCAD_COUNT: {focad_count}"
            )
        else:
            print(f"INCLUDE_FOCAL_ADHESIONS: {self.INCLUDE_FOCAL_ADHESIONS}")
        print(f"INCLUDE_FIBRE_NETWORK: {self.INCLUDE_FIBRE_NETWORK}")
        print(f"MOVING_BOUNDARIES: {self.MOVING_BOUNDARIES}")

    def print_configuration_summary(self, n_nodes=None, n_fibres=None):
        domain_lx = abs(self.BOUNDARY_COORDS[1] - self.BOUNDARY_COORDS[0])
        domain_ly = abs(self.BOUNDARY_COORDS[3] - self.BOUNDARY_COORDS[2])
        domain_lz = abs(self.BOUNDARY_COORDS[5] - self.BOUNDARY_COORDS[4])

        print("=========================================")
        print("====== Model Configuration Summary ======")
        print("=========================================")
        print("Time step: {0} | Steps: {1}".format(self.TIME_STEP, self.STEPS))
        print("Save every N steps: {0}".format(self.SAVE_EVERY_N_STEPS))
        print("Save data to file: {0}".format(self.SAVE_DATA_TO_FILE))
        print("Show plots: {0}".format(self.SHOW_PLOTS))
        print(
            "Domain size (LX, LY, LZ): {0}, {1}, {2}".format(
                domain_lx, domain_ly, domain_lz
            )
        )
        print()
        self.print_boundary_config()
        self.print_agent_config(n_nodes, n_fibres)

        print("=========================================\n")
        print("MODEL RUNNING...\n")
        
    def print_agent_config(self, n_nodes=None, n_fibres=None):
        print("=========== Agent Configuration =========")
        print("== ECM: ")
        print("ECM agents per dir: {0}".format(tuple(self.ECM_AGENTS_PER_DIR)))
        print("ECM population size: {0}".format(self.ECM_POPULATION_SIZE))
        print("ECM voxel volume: {0}".format(self.ECM_VOXEL_VOLUME)) 
        total_number_of_agents = self.ECM_POPULATION_SIZE
        if self.INCLUDE_DIFFUSION:
            coeff_text = (
                "unknown" if self.DIFFUSION_COEFF_MULTI is None else self.DIFFUSION_COEFF_MULTI
            )
            unstable_text = (
                "unknown" if self.UNSTABLE_DIFFUSION is None else self.UNSTABLE_DIFFUSION
            )
            print(
                "Diffusion: enabled | Species: {0} | Coefficients: {1} | Unstable: {2}".format(
                    self.N_SPECIES,
                    coeff_text,
                    unstable_text,
                )
            )
        else:
            print("Diffusion: disabled")
        print()
        if self.INCLUDE_FIBRE_NETWORK:
            total_number_of_agents += n_nodes if n_nodes is not None else 0
            print("== FIBRE NETWORK: ")
            nodes_text = "unknown" if n_nodes is None else str(n_nodes)
            fibres_text = "unknown" if n_fibres is None else str(n_fibres)
            print(
                "Fibre network: enabled | Nodes: {0} | Fibres: {1} | Segment length: {2}".format(
                    nodes_text,
                    fibres_text,
                    self.FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE,
                )
            )
        else:
            print("Fibre network: disabled")
        print()

        if self.INCLUDE_CELLS:
            total_number_of_agents += self.N_CELLS if self.N_CELLS is not None else 0
            print("== CELLS: ")
            print("Cells: enabled")
            print(f" - N_CELLS: {self.N_CELLS}")
            print(f" - Radius: {self.CELL_RADIUS}")
            print(f" - Reference speed: {self.CELL_SPEED_REF}")
            print(
            " - Search radius (cell-ECM, cell-cell): {0}, {1}".format(
                self.MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION,
                self.MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION,
            )
        )
        else:
            print("Cells: disabled")
        print()
        
        if self.INCLUDE_FOCAL_ADHESIONS and self.N_CELLS is not None and self.INIT_N_FOCAD_PER_CELL is not None:
            focad_count = self.N_CELLS * self.INIT_N_FOCAD_PER_CELL
            total_number_of_agents += focad_count
            print("== FOCAL ADHESIONS: ")
            print("Focal adhesions: enabled")
            print(f" - FOCAD_PER_CELL: {self.INIT_N_FOCAD_PER_CELL}")
            print(f" - FOCAD_COUNT: {focad_count}")
            print(f" - Search radius: {self.MAX_SEARCH_RADIUS_FOCAD}")

        print()
        print(f"TOTAL NUMBER OF AGENTS: {total_number_of_agents}")

    def print_boundary_config(self):
        print("========= Boundary Configuration ========")
        print(f"BOUNDARY_COORDS: {self.BOUNDARY_COORDS}")
        print(f"CLAMP_AGENT_TOUCHING_BOUNDARY: {self.CLAMP_AGENT_TOUCHING_BOUNDARY}")
        print(f"ALLOW_BOUNDARY_ELASTIC_MOVEMENT: {self.ALLOW_BOUNDARY_ELASTIC_MOVEMENT}")
        if self.MOVING_BOUNDARIES:
            print(f"BOUNDARY_DISP_RATES: {self.BOUNDARY_DISP_RATES}")
            print(f"BOUNDARY_DISP_RATES_PARALLEL: {self.BOUNDARY_DISP_RATES_PARALLEL}")
            print("Moving boundaries: {0}".format(self.MOVING_BOUNDARIES))
            print(
                "Max expected boundary position (min, max): {0}, {1}".format(
                    self.MIN_EXPECTED_BOUNDARY_POS,
                    self.MAX_EXPECTED_BOUNDARY_POS,
                )
            )
            print("Oscillatory shear assay: {0}".format(self.OSCILLATORY_SHEAR_ASSAY))
        print("\n")    
         
    def plot_boundary_positions(self, bpos_over_time, ax=None, show=True):
        if bpos_over_time is None:
            return None
        if ax is None:
            fig, ax = plt.subplots()
        bpos_over_time.plot(ax=ax)
        ax.set_xlabel("time step")
        ax.set_ylabel("pos")
        if show:
            plt.show()
        return ax

    def plot_boundary_forces(self, bforce_over_time, ax=None, show=True):
        if bforce_over_time is None:
            return None
        if ax is None:
            fig, ax = plt.subplots()
        bforce_over_time.plot(ax=ax)
        ax.set_ylabel("normal force")
        ax.set_xlabel("time step")
        if show:
            plt.show()
        return ax

    def plot_boundary_shear_forces(self, bforce_shear_over_time, ax=None, show=True):
        if bforce_shear_over_time is None:
            return None
        if ax is None:
            fig, ax = plt.subplots()
        bforce_shear_over_time.plot(ax=ax)
        ax.set_ylabel("shear force")
        ax.set_xlabel("time step")
        if show:
            plt.show()
        return ax

    def plot_poisson_ratio(self, poisson_ratio_over_time, ax=None, show=True):
        if poisson_ratio_over_time is None:
            return None
        if ax is None:
            fig, ax = plt.subplots()
        poisson_ratio_over_time.plot(ax=ax)
        ax.set_ylabel("poisson ratio")
        ax.set_xlabel("time step")
        if show:
            plt.show()
        return ax

    def plot_oscillatory_strain(self, oscillatory_strain_over_time, ax=None, show=True):
        if oscillatory_strain_over_time is None:
            return None
        if ax is None:
            fig, ax = plt.subplots()
        oscillatory_strain_over_time.plot(ax=ax)
        ax.set_ylabel("strain")
        ax.set_xlabel("time step")
        if show:
            plt.show()
        return ax

    def plot_all(
        self,
        bpos_over_time=None,
        bforce_over_time=None,
        bforce_shear_over_time=None,
        poisson_ratio_over_time=None,
        show=True,
    ):
        fig = plt.figure()
        gs = fig.add_gridspec(2, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[:, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])

        if bpos_over_time is not None:
            bpos_over_time.plot(ax=ax1)
            ax1.set_xlabel("time step")
            ax1.set_ylabel("pos")

        if bforce_over_time is not None:
            bforce_over_time.plot(ax=ax2)
            ax2.set_ylabel("normal force")
            ax2.set_xlabel("time step")

        if bforce_shear_over_time is not None:
            bforce_shear_over_time.plot(ax=ax3)
            ax3.set_ylabel("shear force")
            ax3.set_xlabel("time step")

        if poisson_ratio_over_time is not None:
            poisson_ratio_over_time.plot(ax=ax4)
            ax4.set_ylabel("poisson ratio")
            ax4.set_xlabel("time step")

        if bpos_over_time is not None and bforce_over_time is not None:
            for pos_col, force_col in [("xpos", "fxpos"), ("ypos", "fypos"), ("zpos", "fzpos")]:
                if pos_col in bpos_over_time and force_col in bforce_over_time:
                    x_vals = bpos_over_time[pos_col] - bpos_over_time[pos_col].iloc[0]
                    y_vals = bforce_over_time[force_col]
                    common_len = min(len(x_vals), len(y_vals))
                    if common_len < 2:
                        continue
                    ax5.plot(
                        x_vals.iloc[:common_len],
                        y_vals.iloc[:common_len],
                        label=pos_col,
                    )
            if len(ax5.get_lines()) > 0:
                ax5.legend()
                ax5.set_ylabel("axis normal force")
                ax5.set_xlabel("axis disp")

        fig.tight_layout()

        if show:
            plt.show()
        return fig

    def plot_oscillatory_shear_scatter(
        self,
        oscillatory_strain_over_time,
        bforce_shear_over_time,
        max_strain=None,
        show=True,
    ):
        if oscillatory_strain_over_time is None or bforce_shear_over_time is None:
            return None
        strain_series = oscillatory_strain_over_time["strain"]
        if max_strain not in (None, 0):
            strain_series = strain_series / max_strain

        force_series = bforce_shear_over_time["fypos_x"]
        strain_abs = strain_series.abs()
        force_abs = force_series.abs()

        # Use the actual number of saved samples (data length), not the total sim steps.
        n_samples = len(strain_series)
        sample_idx = np.arange(0, n_samples, 1)
        colors = sample_idx.tolist()

        fig2, ax2 = plt.subplots()
        sc2 = ax2.scatter(
            strain_abs,
            force_abs,
            marker="o",
            c=colors,
            alpha=0.3,
            cmap="viridis",
        )
        ax2.set_xlabel("strain")
        ax2.set_ylabel("shear force")
        cbar2 = fig2.colorbar(sc2, ax=ax2)
        cbar2.set_label("time (sample index)")
        fig2.tight_layout()

        fig3, ax3 = plt.subplots()
        sc3 = ax3.scatter(
            strain_abs,
            force_series,
            marker="o",
            c=colors,
            alpha=0.3,
            cmap="viridis",
        )
        ax3.set_xlabel("strain")
        ax3.set_ylabel("shear force")
        cbar3 = fig3.colorbar(sc3, ax=ax3)
        cbar3.set_label("time (sample index)")
        fig3.tight_layout()

        fig4, ax41 = plt.subplots()
        ax42 = ax41.twinx()
        ax41.plot(sample_idx, strain_series, "g-", label="strain")
        ax42.plot(sample_idx, force_series, "b-", label="shear force")

        ax41.set_xlabel("samples")
        ax41.set_ylabel("strain", color="g")
        ax42.set_ylabel("shear force", color="b")

        force_min = np.nanmin(force_series)
        force_max = np.nanmax(force_series)
        pad = (force_max - force_min) * 0.05 if force_max != force_min else 1.0
        ax42.set_ylim(force_min - pad, force_max + pad)

        lines_1, labels_1 = ax41.get_legend_handles_labels()
        lines_2, labels_2 = ax42.get_legend_handles_labels()
        ax41.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        fig4.tight_layout()

        if show:
            plt.show()
        return (fig2, fig3, fig4)


def build_model_config_from_namespace(ns: dict) -> ModelParameterConfig:
    return ModelParameterConfig(
        save_every_n_steps=ns.get("SAVE_EVERY_N_STEPS"),
        ecm_agents_per_dir=ns.get("ECM_AGENTS_PER_DIR"),
        time_step=ns.get("TIME_STEP"),
        steps=ns.get("STEPS"),
        boundary_coords=ns.get("BOUNDARY_COORDS"),
        boundary_disp_rates=ns.get("BOUNDARY_DISP_RATES"),
        boundary_disp_rates_parallel=ns.get("BOUNDARY_DISP_RATES_PARALLEL"),
        poisson_dirs=ns.get("POISSON_DIRS"),
        allow_boundary_elastic_movement=ns.get("ALLOW_BOUNDARY_ELASTIC_MOVEMENT"),
        boundary_stiffness=ns.get("BOUNDARY_STIFFNESS"),
        boundary_dumping=ns.get("BOUNDARY_DUMPING"),
        clamp_agent_touching_boundary=ns.get("CLAMP_AGENT_TOUCHING_BOUNDARY"),
        allow_agent_sliding=ns.get("ALLOW_AGENT_SLIDING"),
        moving_boundaries=ns.get("MOVING_BOUNDARIES"),
        epsilon=ns.get("EPSILON"),
        ecm_k_elast=ns.get("ECM_K_ELAST"),
        ecm_d_dumping=ns.get("ECM_D_DUMPING"),
        ecm_mass=ns.get("ECM_MASS"),
        ecm_eta=ns.get("ECM_ETA"),
        ecm_gel_concentration=ns.get("ECM_GEL_CONCENTRATION"),
        ecm_ecm_equilibrium_distance=ns.get("ECM_ECM_EQUILIBRIUM_DISTANCE"),
        ecm_boundary_interaction_radius=ns.get("ECM_BOUNDARY_INTERACTION_RADIUS"),
        ecm_boundary_equilibrium_distance=ns.get("ECM_BOUNDARY_EQUILIBRIUM_DISTANCE"),
        ecm_voxel_volume=ns.get("ECM_VOXEL_VOLUME"),
        ecm_population_size=ns.get("ECM_POPULATION_SIZE"),
        include_fiber_alignment=ns.get("INCLUDE_FIBER_ALIGNMENT"),
        ecm_orientation_rate=ns.get("ECM_ORIENTATION_RATE"),
        buckling_coeff_d0=ns.get("BUCKLING_COEFF_D0"),
        strain_stiffening_coeff_ds=ns.get("STRAIN_STIFFENING_COEFF_DS"),
        critical_strain=ns.get("CRITICAL_STRAIN"),
        include_fibre_network=ns.get("INCLUDE_FIBRE_NETWORK"),
        max_connectivity=ns.get("MAX_CONNECTIVITY"),
        fibre_segment_k_elast=ns.get("FIBRE_SEGMENT_K_ELAST"),
        fibre_segment_d_dumping=ns.get("FIBRE_SEGMENT_D_DUMPING"),
        fibre_segment_mass=ns.get("FIBRE_SEGMENT_MASS"),
        fibre_segment_equilibrium_distance=ns.get("FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE"),
        fibre_node_boundary_interaction_radius=ns.get("FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS"),
        fibre_node_boundary_equilibrium_distance=ns.get("FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE"),
        max_search_radius_fnodes=ns.get("MAX_SEARCH_RADIUS_FNODES"),
        include_diffusion=ns.get("INCLUDE_DIFFUSION"),
        n_species=ns.get("N_SPECIES"),
        diffusion_coeff_multi=ns.get("DIFFUSION_COEFF_MULTI"),
        boundary_conc_init_multi=ns.get("BOUNDARY_CONC_INIT_MULTI"),
        boundary_conc_fixed_multi=ns.get("BOUNDARY_CONC_FIXED_MULTI"),
        init_ecm_concentration_vals=ns.get("INIT_ECM_CONCENTRATION_VALS"),
        init_ecm_sat_concentration_vals=ns.get("INIT_ECM_SAT_CONCENTRATION_VALS"),
        unstable_diffusion=ns.get("UNSTABLE_DIFFUSION"),
        include_cells=ns.get("INCLUDE_CELLS"),
        include_cell_orientation=ns.get("INCLUDE_CELL_ORIENTATION"),
        include_cell_cell_interaction=ns.get("INCLUDE_CELL_CELL_INTERACTION"),
        include_cell_cycle=ns.get("INCLUDE_CELL_CYCLE"),
        periodic_boundaries_for_cells=ns.get("PERIODIC_BOUNDARIES_FOR_CELLS"),
        n_cells=ns.get("N_CELLS"),
        cell_k_elast=ns.get("CELL_K_ELAST"),
        cell_d_dumping=ns.get("CELL_D_DUMPING"),
        cell_radius=ns.get("CELL_RADIUS"),
        cell_speed_ref=ns.get("CELL_SPEED_REF"),
        cell_orientation_rate=ns.get("CELL_ORIENTATION_RATE"),
        max_search_radius_cell_ecm_interaction=ns.get("MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION"),
        max_search_radius_cell_cell_interaction=ns.get("MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION"),
        cell_cycle_duration=ns.get("CELL_CYCLE_DURATION"),
        cycle_phase_g1_duration=ns.get("CYCLE_PHASE_G1_DURATION"),
        cycle_phase_s_duration=ns.get("CYCLE_PHASE_S_DURATION"),
        cycle_phase_g2_duration=ns.get("CYCLE_PHASE_G2_DURATION"),
        cycle_phase_m_duration=ns.get("CYCLE_PHASE_M_DURATION"),
        cycle_phase_g1_start=ns.get("CYCLE_PHASE_G1_START"),
        cycle_phase_s_start=ns.get("CYCLE_PHASE_S_START"),
        cycle_phase_g2_start=ns.get("CYCLE_PHASE_G2_START"),
        cycle_phase_m_start=ns.get("CYCLE_PHASE_M_START"),
        init_cell_concentration_vals=ns.get("INIT_CELL_CONCENTRATION_VALS"),
        init_cell_conc_mass_vals=ns.get("INIT_CELL_CONC_MASS_VALS"),
        init_cell_consumption_rates=ns.get("INIT_CELL_CONSUMPTION_RATES"),
        init_cell_production_rates=ns.get("INIT_CELL_PRODUCTION_RATES"),
        init_cell_reaction_rates=ns.get("INIT_CELL_REACTION_RATES"),
        include_focal_adhesions=ns.get("INCLUDE_FOCAL_ADHESIONS"),
        init_n_focad_per_cell=ns.get("INIT_N_FOCAD_PER_CELL"),
        n_anchor_points=ns.get("N_ANCHOR_POINTS"),
        max_search_radius_focad=ns.get("MAX_SEARCH_RADIUS_FOCAD"),
        focad_rest_length_0=ns.get("FOCAD_REST_LENGTH_0"),
        focad_k_fa=ns.get("FOCAD_K_FA"),
        focad_f_max=ns.get("FOCAD_F_MAX"),
        focad_v_c=ns.get("FOCAD_V_C"),
        focad_k_on=ns.get("FOCAD_K_ON"),
        focad_k_off_0=ns.get("FOCAD_K_OFF_0"),
        focad_f_c=ns.get("FOCAD_F_C"),
        focad_k_reinf=ns.get("FOCAD_K_REINF"),
        oscillatory_shear_assay=ns.get("OSCILLATORY_SHEAR_ASSAY"),
        max_strain=ns.get("MAX_STRAIN"),
        oscillatory_amplitude=ns.get("OSCILLATORY_AMPLITUDE"),
        oscillatory_freq=ns.get("OSCILLATORY_FREQ"),
        oscillatory_w=ns.get("OSCILLATORY_W"),
        min_expected_boundary_pos=ns.get("MIN_EXPECTED_BOUNDARY_POS"),
        max_expected_boundary_pos=ns.get("MAX_EXPECTED_BOUNDARY_POS"),
        include_vascularization=ns.get("INCLUDE_VASCULARIZATION"),
        init_vascularization_concentration_vals=ns.get("INIT_VASCULARIZATION_CONCENTRATION_VALS"),
        save_pickle=ns.get("SAVE_PICKLE"),
        show_plots=ns.get("SHOW_PLOTS"),
        save_data_to_file=ns.get("SAVE_DATA_TO_FILE"),
        res_path=str(ns.get("RES_PATH")) if ns.get("RES_PATH") is not None else None,
    )


