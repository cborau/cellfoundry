import math
import os
import pickle
import time

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
    allow_warning_on_mismatch=False, # for special cases like single-fibre network tests
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

    if axes_with_both_faces < 2 and not allow_warning_on_mismatch:
        print(msg_wrong_network_dimensions)
        critical_error = True

    return nodes, connectivity, n_fiber, fibre_segment_equilibrium_distance, critical_error


import math

def print_fibre_calibration_summary(
    fibre_segment_k_elast,
    fibre_segment_d_dumping,
    fibre_segment_equilibrium_distance,
    dt,
    reference_modulus_mpa=(1.5, 100.0, 700.0),
    reference_diameter_nm=(20.0, 60.0, 120.0),
    tau_multipliers=(10.0, 50.0, 100.0),
):
    """
    FNODE-FNODE Kelvin-Voigt link:

        F = k_pair * (d - d0) + d_pair * d_dot

    Model structure:
      - Two identical springs in series:
            k_pair = k_node / 2
      - One dashpot in parallel at link level:
            d_pair = fibre_segment_d_dumping

    Relaxation time:
            tau = d_pair / k_pair

    Units:
      k in nN/um
      d in nN*s/um
      L in um
      dt in s
      1 MPa = 1000 nN/um^2
    """

    eps = 1e-20

    k_node = float(fibre_segment_k_elast)        # per spring
    d_pair = float(fibre_segment_d_dumping)      # dashpot at link level
    L = float(fibre_segment_equilibrium_distance)
    dt = float(dt)

    # Two springs in series
    k_pair = 0.5 * k_node

    # Kelvin-Voigt relaxation time
    tau = d_pair / max(k_pair, eps)
    tau_steps = tau / max(dt, eps)

    print("\n--- Fibre calibration summary ---")
    print(f"k_node = {k_node:.4g} nN/um")
    print(f"k_pair = {k_pair:.4g} nN/um  (2 springs in series)")
    print(f"d_dumping = {d_pair:.4g} nN*s/um  (dashpot in parallel)")
    print(f"L0 = {L:.4g} um, dt = {dt:.4g} s")
    print(f"Relaxation time tau = d_pair/k_pair = {tau:.4g} s")
    print(f"That is about {tau_steps:.3g} timesteps if dt = {dt:.4g} s")

    # Suggested damping values for stable explicit integration
    print("\nSuggested stabilization targets (tau ~ 10-100 timesteps):")
    for m in tau_multipliers:
        tau_target = m * dt
        d_suggest = tau_target * k_pair
        print(f"  tau = {m:.0f}*dt = {tau_target:.4g} s  ->  d_dumping ~= {d_suggest:.4g} nN*s/um")

    print("\nTuning guideline:")
    print("  - Too jittery or oscillatory: increase tau (increase d_dumping)")
    print("  - Too sluggish / takes forever to settle: decrease tau (decrease d_dumping)")

    # Forward mapping: modulus -> implied diameter
    print("\nForward mapping (given E -> implied fibre diameter):")
    for E_mpa in reference_modulus_mpa:
        E = E_mpa * 1000.0  # MPa -> nN/um^2
        area = (k_pair * L) / max(E, eps)
        r = math.sqrt(max(area, 0.0) / math.pi)
        d_nm = 2.0 * r * 1000.0
        print(f"  E = {E_mpa:.4g} MPa -> diameter ~= {d_nm:.3f} nm")

    # Inverse mapping: target diameter -> required stiffness and damping
    print("\nInverse mapping (target E, diameter -> required k_node and d_dumping):")
    for E_mpa in reference_modulus_mpa:
        E = E_mpa * 1000.0
        for diam_nm in reference_diameter_nm:
            diam_um = diam_nm / 1000.0
            r = 0.5 * diam_um
            area = math.pi * r * r

            k_pair_req = E * area / max(L, eps)
            k_node_req = 2.0 * k_pair_req

            # keep same relaxation time tau
            d_req = tau * k_pair_req
            tau_req_steps = tau / max(dt, eps)

            print(
                f"  E={E_mpa:.4g} MPa, d={diam_nm:.4g} nm -> "
                f"k_node~={k_node_req:.4g} nN/um, "
                f"d_dumping~={d_req:.4g} nN*s/um "
                f"(tau~={tau:.4g} s ~= {tau_req_steps:.3g} steps)"
            )
    print()

    return {
        "k_node": k_node,
        "k_pair": k_pair,
        "d_pair": d_pair,
        "tau_s": tau,
        "tau_steps": tau_steps,
    }


def print_focad_birth_calibration_summary(
    *,
    dt,
    init_n_focad_per_cell,
    n_min,
    n_max,
    k0,
    kmax,
    refractory_s,
    k_sigma,
    hill_sigma,
    k_c,
    hill_conc,
    species_index,
):
    """
    Print a compact calibration summary for CELL-driven FOCAD birth dynamics.

    Birth model in cell_focad_update:
      h_sigma = sigma_+^m_sigma / (k_sigma^m_sigma + sigma_+^m_sigma)
      h_c     = C^n_conc / (k_c^n_conc + C^n_conc)
      h_birth = h_sigma * h_c
      k_birth = k0 + kmax * h_birth
      p_step  = 1 - exp(-k_birth * dt)

    Notes:
      - Single-birth-attempt per CELL per step (agent_out semantics)
      - Additional refractory timer further caps effective birth frequency
    """

    eps = 1e-20
    dt = float(dt)
    k0 = float(k0)
    kmax = float(kmax)
    refractory_s = float(refractory_s)
    n_min = int(n_min)
    n_max = int(n_max)
    init_n = int(init_n_focad_per_cell)

    k_birth_min = max(0.0, k0)
    k_birth_max = max(0.0, k0 + kmax)

    p_step_min = 1.0 - math.exp(-k_birth_min * dt)
    p_step_max = 1.0 - math.exp(-k_birth_max * dt)

    births_per_min_min = 60.0 * k_birth_min
    births_per_min_max = 60.0 * k_birth_max

    delta_to_max = max(0, n_max - init_n)
    if delta_to_max == 0:
        est_fill_time_s = 0.0
    else:
        effective_births_per_s = k_birth_max
        if refractory_s > eps:
            effective_births_per_s = min(effective_births_per_s, 1.0 / refractory_s)
        if effective_births_per_s > eps:
            est_fill_time_s = delta_to_max / effective_births_per_s
        else:
            est_fill_time_s = float("inf")

    if refractory_s > eps:
        refractory_steps = refractory_s / max(dt, eps)
        max_births_per_min_refractory = 60.0 / refractory_s
    else:
        refractory_steps = 0.0
        max_births_per_min_refractory = float("inf")

    print("\n--- FOCAD birth calibration summary ---")
    print(f"dt = {dt:.4g} s")
    print(f"species index for biochemical gate = {species_index}")
    print(f"target count bounds: n_min = {n_min}, n_max = {n_max} (init = {init_n})")
    print(f"kinetic rates: k0 = {k0:.4g} 1/s, kmax = {kmax:.4g} 1/s")
    print(
        f"gate half-saturation: k_sigma = {k_sigma:.4g} kPa (hill_sigma = {hill_sigma:.4g}), "
        f"k_c = {k_c:.4g} (hill_conc = {hill_conc:.4g})"
    )
    print(f"k_birth range = [{k_birth_min:.4g}, {k_birth_max:.4g}] 1/s")
    print(f"p_step range = [{p_step_min:.4g}, {p_step_max:.4g}] per step")
    print(f"expected births per cell per minute (rate-only) = [{births_per_min_min:.4g}, {births_per_min_max:.4g}]")

    if math.isfinite(max_births_per_min_refractory):
        print(
            f"refractory = {refractory_s:.4g} s (~{refractory_steps:.3g} steps), "
            f"absolute cap ~= {max_births_per_min_refractory:.4g} births/cell/min"
        )
    else:
        print("refractory disabled (<=0): no refractory cap")

    if math.isfinite(est_fill_time_s):
        print(
            f"estimated time to go from init ({init_n}) to n_max ({n_max}) at max drive (ignoring deaths) "
            f"~= {est_fill_time_s:.4g} s ({est_fill_time_s/60.0:.4g} min)"
        )
    else:
        print(
            f"estimated time to go from init ({init_n}) to n_max ({n_max}) at max drive (ignoring deaths): infinite "
            f"(effective birth rate <= 0)"
        )

    print("tuning guideline:")
    print("  - Too many births: decrease kmax, increase refractory, or increase k_sigma/k_c")
    print("  - Too few births: increase kmax or decrease k_sigma/k_c")
    print("  - Bursty births: increase refractory (and/or lower k0)")
    print()

    return {
        "k_birth_min": k_birth_min,
        "k_birth_max": k_birth_max,
        "p_step_min": p_step_min,
        "p_step_max": p_step_max,
        "births_per_min_min": births_per_min_min,
        "births_per_min_max": births_per_min_max,
        "refractory_steps": refractory_steps,
        "max_births_per_min_refractory": max_births_per_min_refractory,
        "delta_to_max": float(delta_to_max),
        "est_fill_time_s": est_fill_time_s,
    }



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
    return np.random.uniform(low=[minx, miny, minz], high=[maxx, maxy, maxz], size=(n, 3))
    

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
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=n_vectors)
    costheta = np.random.uniform(-1.0, 1.0, size=n_vectors)
    sintheta = np.sqrt(np.maximum(0.0, 1.0 - costheta * costheta))

    v_array = np.empty((n_vectors, 3), dtype=float)
    v_array[:, 0] = sintheta * np.cos(phi)
    v_array[:, 1] = sintheta * np.sin(phi)
    v_array[:, 2] = costheta
    return v_array


def getCellInitCachePath(cache_dir, n_cells: int):
    cache_dir = os.fspath(cache_dir)
    return os.path.join(cache_dir, f"cell_pos_ori_{n_cells}.pkl")


def generateCellInitializationData(n_cells: int, boundary_coords):
    boundary_coords = np.asarray(boundary_coords, dtype=float)
    cell_pos = getRandomCoords3D(
        n_cells,
        boundary_coords[0], boundary_coords[1],
        boundary_coords[2], boundary_coords[3],
        boundary_coords[4], boundary_coords[5],
    )
    cell_orientations = getRandomVectors3D(n_cells)
    return cell_pos, cell_orientations


def saveCellInitializationCache(n_cells: int, boundary_coords, cache_dir, extra_data=None):
    os.makedirs(cache_dir, exist_ok=True)
    boundary_coords = np.asarray(boundary_coords, dtype=float)

    pos_start = time.perf_counter()
    cell_pos = getRandomCoords3D(
        n_cells,
        boundary_coords[0], boundary_coords[1],
        boundary_coords[2], boundary_coords[3],
        boundary_coords[4], boundary_coords[5],
    )
    pos_seconds = time.perf_counter() - pos_start

    ori_start = time.perf_counter()
    cell_orientations = getRandomVectors3D(n_cells)
    ori_seconds = time.perf_counter() - ori_start

    cache_data = {
        "n_cells": int(n_cells),
        "boundary_coords": boundary_coords,
        "domain_size": boundary_coords.copy(),
        "cell_pos": np.asarray(cell_pos, dtype=float),
        "cell_orientations": np.asarray(cell_orientations, dtype=float),
        "generation_time_seconds": {
            "positions": pos_seconds,
            "orientations": ori_seconds,
            "total": pos_seconds + ori_seconds,
        },
    }
    if extra_data:
        cache_data.update(extra_data)

    output_path = getCellInitCachePath(cache_dir, n_cells)
    with open(output_path, 'wb') as fh:
        pickle.dump(cache_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path, cache_data


def loadCachedCellInitialization(n_cells: int, boundary_coords, cache_dir, atol=1.0e-10):
    cache_path = getCellInitCachePath(cache_dir, n_cells)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'rb') as fh:
            cache_data = pickle.load(fh)
    except Exception as exc:
        print(f"WARNING: Could not read cached cell initialization from {cache_path}: {exc}")
        return None

    cached_boundary = cache_data.get('boundary_coords', cache_data.get('domain_size'))
    if cached_boundary is None:
        print(f"WARNING: Cached cell initialization file {cache_path} does not contain boundary coordinates.")
        return None

    cached_boundary = np.asarray(cached_boundary, dtype=float)
    current_boundary = np.asarray(boundary_coords, dtype=float)
    if cached_boundary.shape != (6,) or current_boundary.shape != (6,):
        print(f"WARNING: Cached cell initialization file {cache_path} has invalid boundary shape.")
        return None

    if not np.allclose(cached_boundary, current_boundary, rtol=0.0, atol=atol):
        print(
            f"WARNING: Cached cell initialization file {cache_path} was generated for a different domain. "
            "Falling back to on-the-fly generation."
        )
        return None

    cell_pos = np.asarray(cache_data.get('cell_pos'), dtype=float)
    cell_orientations = np.asarray(cache_data.get('cell_orientations'), dtype=float)
    expected_shape = (n_cells, 3)
    if cell_pos.shape != expected_shape or cell_orientations.shape != expected_shape:
        print(
            f"WARNING: Cached cell initialization file {cache_path} has unexpected array shapes "
            f"(positions={cell_pos.shape}, orientations={cell_orientations.shape}, expected={expected_shape})."
        )
        return None

    return cell_pos, cell_orientations, cache_path, cache_data


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
    central_point = np.asarray([px, py, pz], dtype=float)
    rand_dirs = getRandomVectors3D(n)
    if on_surface:
        radii = np.full((n, 1), radius, dtype=float)
    else:
        radii = np.random.uniform(0.0, 1.0, size=(n, 1)) * radius

    return central_point + rand_dirs * radii


def compute_u_ref_from_anchor_pos(anchor_pos: np.ndarray,
                                 cell_center: np.ndarray,
                                 eps: float = 1e-12) -> np.ndarray:
    """
    Compute reference unit vectors u_ref for nucleus anchors.

    Parameters
    ----------
    anchor_pos : (N, 3) np.ndarray
        Anchor positions in world coordinates.
    cell_center : (3,) np.ndarray
        Nucleus center in world coordinates (cell_pos[i, :]).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    u_ref : (N, 3) np.ndarray
        Unit vectors pointing from nucleus center to each anchor.
    """
    anchor_pos = np.asarray(anchor_pos, dtype=np.float64)
    cell_center = np.asarray(cell_center, dtype=np.float64).reshape(3,)

    if anchor_pos.ndim != 2 or anchor_pos.shape[1] != 3:
        raise ValueError(f"anchor_pos must have shape (N, 3), got {anchor_pos.shape}")
    if cell_center.shape != (3,):
        raise ValueError(f"cell_center must have shape (3,), got {cell_center.shape}")

    # Vectors from center to anchors
    u = anchor_pos - cell_center[None, :]  # (N, 3)

    # Normalize safely
    norm = np.linalg.norm(u, axis=1)  # (N,)
    norm = np.maximum(norm, eps)      # avoid divide by zero
    u_ref = u / norm[:, None]

    return u_ref


def build_save_data_context(ecm_agents_per_dir, include_fibre_network, n_nodes):
    context = {}
    context["header"] = [
        "# vtk DataFile Version 3.0",
        "ECM data",
        "ASCII",
        "DATASET POLYDATA",
        "POINTS {} float".format(8 + ecm_agents_per_dir[0] * ecm_agents_per_dir[1] * ecm_agents_per_dir[2]),
    ]

    domaindata = ["POLYGONS 6 30"]
    cube_conn = [
        [4, 0, 3, 7, 4],
        [4, 1, 2, 6, 5],
        [4, 1, 0, 4, 5],
        [4, 2, 3, 7, 6],
        [4, 0, 1, 2, 3],
        [4, 4, 5, 6, 7],
    ]
    for i in range(len(cube_conn)):
        for j in range(len(cube_conn[i])):
            if j > 0:
                cube_conn[i][j] = cube_conn[i][j] + ecm_agents_per_dir[0] * ecm_agents_per_dir[1] * ecm_agents_per_dir[2]
        domaindata.append(' '.join(str(x) for x in cube_conn[i]))

    context["domaindata"] = domaindata
      
    if include_fibre_network:
        domaindata_network = []
        cube_conn_network = [
            [4, 0, 3, 7, 4],
            [4, 1, 2, 6, 5],
            [4, 1, 0, 4, 5],
            [4, 2, 3, 7, 6],
            [4, 0, 1, 2, 3],
            [4, 4, 5, 6, 7],
        ]
        for i in range(len(cube_conn_network)):
            for j in range(len(cube_conn_network[i])):
                if j > 0:
                    cube_conn_network[i][j] = cube_conn_network[i][j] + n_nodes
            domaindata_network.append(' '.join(str(x) for x in cube_conn_network[i]))
        context["domaindata_network"] = domaindata_network

    context["domaindata"] += [
        "CELL_DATA 6",
        "SCALARS boundary_index int 1",
        "LOOKUP_TABLE default",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "NORMALS boundary_normals float",
        "1 0 0",
        "-1 0 0",
        "0 1 0",
        "0 -1 0",
        "0 0 1",
        "0 0 -1",
    ]

    context["vascularizationdata"] = [
        "# vtk DataFile Version 3.0",
        "Vascularization points",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
    ]
    context["fibrenodedata"] = [
        "# vtk DataFile Version 3.0",
        "Fibre node agents",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
    ]
    context["celldata"] = [
        "# vtk DataFile Version 3.0",
        "Cell agents",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
    ]
    context["nucleusdata"] = [
        "# vtk DataFile Version 3.0",
        "Cell agents - nucleus data",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
    ]
    context["focaladhesionsdata"] = [
        "# vtk DataFile Version 3.0",
        "Focal adhesions",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
    ]

    return context


def save_data_to_file_step(FLAMEGPU, save_context, config):
    save_data_to_file = config["SAVE_DATA_TO_FILE"]
    save_every_n_steps = config["SAVE_EVERY_N_STEPS"]
    n_species = config["N_SPECIES"]
    res_path = config["RES_PATH"]
    include_fibre_network = config["INCLUDE_FIBRE_NETWORK"]
    heterogeneous_diffusion = config["HETEROGENEOUS_DIFFUSION"]
    initial_network_connectivity = config["INITIAL_NETWORK_CONNECTIVITY"]
    n_nodes = config["N_NODES"]
    include_cells = config["INCLUDE_CELLS"]
    ecm_population_size = config["ECM_POPULATION_SIZE"]
    include_focal_adhesions = config["INCLUDE_FOCAL_ADHESIONS"]
    include_network_remodeling = config["INCLUDE_NETWORK_REMODELING"]
    pyflamegpu = config["pyflamegpu"]

    stepCounter = FLAMEGPU.getStepCounter() + 1
    coord_boundary = list(FLAMEGPU.environment.getPropertyArrayFloat("COORDS_BOUNDARIES"))

    if not save_data_to_file:
        return
    if stepCounter % save_every_n_steps != 0 and stepCounter != 1:
        return

    if include_fibre_network:
        file_name = 'fibre_network_data_t{:04d}.vtk'.format(stepCounter)
        file_path = res_path / file_name

        agent = FLAMEGPU.agent("FNODE")
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
        degradation = list()
        reinforcement = list()
        secreted = list()
        linked_nodes_all = list()
        k_elast = list()

        av = agent.getPopulationData()
        for ai in av:
            id_ai = ai.getVariableInt("id")
            coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
            velocity_ai = (ai.getVariableFloat("vx"), ai.getVariableFloat("vy"), ai.getVariableFloat("vz"))
            force_ai = (ai.getVariableFloat("fx"), ai.getVariableFloat("fy"), ai.getVariableFloat("fz"))
            ids.append(id_ai)
            coords.append(coords_ai)
            velocity.append(velocity_ai)
            force.append(force_ai)
            elastic_energy.append(ai.getVariableFloat("elastic_energy"))
            degradation.append(ai.getVariableFloat("degradation"))
            reinforcement.append(ai.getVariableFloat("reinforcement"))
            secreted.append(ai.getVariableInt("secreted"))
            linked_nodes_all.append(ai.getVariableArrayFloat("linked_nodes"))
            k_elast.append(ai.getVariableFloat("k_elast"))
        if len(ids) > 0:
            min_id = min(ids)
            ids = [fid - min_id for fid in ids if fid > 0]
            for i in range(len(linked_nodes_all)):
                linked_nodes_all[i] = [linked_id - min_id for linked_id in linked_nodes_all[i] if linked_id > 0] + [linked_id for linked_id in linked_nodes_all[i] if linked_id <= 0]

        sorted_indices = np.argsort(ids)
        ids = [ids[i] for i in sorted_indices]
        coords = [coords[i] for i in sorted_indices]
        velocity = [velocity[i] for i in sorted_indices]
        force = [force[i] for i in sorted_indices]
        elastic_energy = [elastic_energy[i] for i in sorted_indices]
        degradation = [degradation[i] for i in sorted_indices]
        reinforcement = [reinforcement[i] for i in sorted_indices]
        secreted = [secreted[i] for i in sorted_indices]
        linked_nodes_all = [linked_nodes_all[i] for i in sorted_indices]

        n_fnodes = len(ids)
        id_to_point_idx = {fid: i for i, fid in enumerate(ids)}

        added_lines = set()
        cell_connectivity = []
        for i, fid in enumerate(ids):
            links = linked_nodes_all[i]
            for linked_id_raw in links:
                linked_id = int(round(linked_id_raw))
                if linked_id < 0 or linked_id == fid:
                    continue
                if linked_id in id_to_point_idx:
                    line = tuple(sorted((id_to_point_idx[fid], id_to_point_idx[linked_id])))
                    if line not in added_lines:
                        added_lines.add(line)
                        cell_connectivity.append(line)

        num_cells = len(cell_connectivity) # WARNING: Vtk cells, nothing to do with cell agents

        with open(str(file_path), 'w') as file:
            for line in save_context["fibrenodedata"]:
                file.write(line + '\n')

            file.write("POINTS {} float \n".format(8 + n_fnodes))
            for coords_ai in coords:
                file.write("{} {} {} \n".format(coords_ai[0], coords_ai[1], coords_ai[2]))

            file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[2], coord_boundary[4]))
            file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[2], coord_boundary[4]))
            file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[3], coord_boundary[4]))
            file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[3], coord_boundary[4]))
            file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[2], coord_boundary[5]))
            file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[2], coord_boundary[5]))
            file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[3], coord_boundary[5]))
            file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[3], coord_boundary[5]))

            file.write(f"CELLS {num_cells + 6} {num_cells * 3 + 6 * 5}\n")
            for conn in cell_connectivity:
                file.write(f"2 {conn[0]} {conn[1]}\n")
            if include_network_remodeling:
                domaindata_network = []
                cube_conn_network = [
                    [4, 0, 3, 7, 4],
                    [4, 1, 2, 6, 5],
                    [4, 1, 0, 4, 5],
                    [4, 2, 3, 7, 6],
                    [4, 0, 1, 2, 3],
                    [4, 4, 5, 6, 7],
                ]
                for i in range(len(cube_conn_network)):
                    for j in range(len(cube_conn_network[i])):
                        if j > 0:
                            cube_conn_network[i][j] = cube_conn_network[i][j] + n_fnodes
                    domaindata_network.append(' '.join(str(x) for x in cube_conn_network[i]))
                for line in domaindata_network:
                    file.write(line + '\n')                
            else:
                for line in save_context["domaindata_network"]:
                    file.write(line + '\n')

            file.write(f"CELL_TYPES {num_cells + 6}\n")
            for _ in range(num_cells):
                file.write("3\n")
            for _ in range(6):
                file.write("7\n")

            file.write(f"CELL_DATA {num_cells + 6}\n")
            file.write("SCALARS boundary_idx int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for _ in range(num_cells):
                file.write("0\n")
            for bidx in range(6):
                file.write(f"{bidx + 1}\n")

            file.write("SCALARS boundary_normal_forces float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for _ in range(num_cells):
                file.write("0.0\n")
            file.write(str(sum_bx_pos) + '\n')
            file.write(str(sum_bx_neg) + '\n')
            file.write(str(sum_by_pos) + '\n')
            file.write(str(sum_by_neg) + '\n')
            file.write(str(sum_bz_pos) + '\n')
            file.write(str(sum_bz_neg) + '\n')

            file.write("SCALARS boundary_normal_force_scaling float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for _ in range(num_cells):
                file.write("0.0\n")
            file.write(str(abs(sum_bx_pos)) + '\n')
            file.write(str(abs(sum_bx_neg)) + '\n')
            file.write(str(abs(sum_by_pos)) + '\n')
            file.write(str(abs(sum_by_neg)) + '\n')
            file.write(str(abs(sum_bz_pos)) + '\n')
            file.write(str(abs(sum_bz_neg)) + '\n')

            file.write("SCALARS boundary_shear_forces_pos float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for _ in range(num_cells):
                file.write("0.0\n")
            file.write(str(sum_bx_pos_y) + '\n')
            file.write(str(sum_bx_pos_z) + '\n')
            file.write(str(sum_by_pos_x) + '\n')
            file.write(str(sum_by_pos_z) + '\n')
            file.write(str(sum_bz_pos_x) + '\n')
            file.write(str(sum_bz_pos_y) + '\n')

            file.write("SCALARS boundary_shear_forces_neg float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for _ in range(num_cells):
                file.write("0.0\n")
            file.write(str(sum_bx_neg_y) + '\n')
            file.write(str(sum_bx_neg_z) + '\n')
            file.write(str(sum_by_neg_x) + '\n')
            file.write(str(sum_by_neg_z) + '\n')
            file.write(str(sum_bz_neg_x) + '\n')
            file.write(str(sum_bz_neg_y) + '\n')

            file.write("POINT_DATA {} \n".format(8 + n_fnodes))
            file.write("SCALARS is_corner int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for _ in elastic_energy:
                file.write("0 \n")
            for _ in range(8):
                file.write("1 \n")

            file.write("SCALARS k_elast float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for k_ai in k_elast:
                file.write("{:.4f} \n".format(k_ai))
            for _ in range(8):
                file.write("0.0 \n")

            file.write("SCALARS elastic_energy float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for ee_ai in elastic_energy:
                file.write("{:.4f} \n".format(ee_ai))
            for _ in range(8):
                file.write("0.0 \n")

            file.write("SCALARS degradation float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for d_ai in degradation:
                file.write("{:.4f} \n".format(d_ai))
            for _ in range(8):
                file.write("0.0 \n")

            file.write("SCALARS reinforcement float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for r_ai in reinforcement:
                file.write("{:.4f} \n".format(r_ai))
            for _ in range(8):
                file.write("0.0 \n")

            file.write("SCALARS secreted int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for s_ai in secreted:
                file.write("{} \n".format(s_ai))
            for _ in range(8):
                file.write("0 \n")

            file.write("VECTORS velocity float\n")
            for v_ai in velocity:
                file.write("{:.4f} {:.4f} {:.4f} \n".format(v_ai[0], v_ai[1], v_ai[2]))
            for _ in range(8):
                file.write("0.0 0.0 0.0 \n")

            file.write("VECTORS force float\n")
            for f_ai in force:
                file.write("{:.4f} {:.4f} {:.4f} \n".format(f_ai[0], f_ai[1], f_ai[2]))
            for _ in range(8):
                file.write("0.0 0.0 0.0 \n")

    if include_cells:
        cell_ids = list()
        cell_coords = list()
        cell_velocity = list()
        cell_orientation = list()
        cell_alignment = list()
        cell_radius = list()
        cell_clock = list()
        cell_cycle_phase = list()
        cell_completed_cycles = list()
        cell_type = list()
        cell_damage = list()
        cell_dead = list()
        cell_dead_by = list()
        cell_anchor_points_x = list()
        cell_anchor_points_y = list()
        cell_anchor_points_z = list()
        c_sp_multi = list()
        file_name = 'cells_t{:04d}.vtk'.format(stepCounter)
        file_path = res_path / file_name
        cell_agent = FLAMEGPU.agent("CELL")
        cell_agent.sortInt("id", pyflamegpu.HostAgentAPI.Asc)
        av = cell_agent.getPopulationData()
        for ai in av:
            cell_id_ai = ai.getVariableInt("id")
            coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
            velocity_ai = (ai.getVariableFloat("vx"), ai.getVariableFloat("vy"), ai.getVariableFloat("vz"))
            orientation_ai = (ai.getVariableFloat("orx"), ai.getVariableFloat("ory"), ai.getVariableFloat("orz"))
            alignment_ai = ai.getVariableFloat("alignment")
            radius_ai = ai.getVariableFloat("radius")
            clock_ai = ai.getVariableFloat("clock")
            cycle_phase_ai = ai.getVariableInt("cycle_phase")
            completed_cycles_ai = ai.getVariableInt("completed_cycles")
            cell_type_ai = ai.getVariableInt("cell_type")
            damage_ai = ai.getVariableFloat("damage")
            dead_ai = ai.getVariableInt("dead")
            dead_by_ai = ai.getVariableInt("dead_by")
            cell_anchor_points_x.append(ai.getVariableArrayFloat("x_i"))
            cell_anchor_points_y.append(ai.getVariableArrayFloat("y_i"))
            cell_anchor_points_z.append(ai.getVariableArrayFloat("z_i"))
            c_sp_multi.append(ai.getVariableArrayFloat("C_sp"))
            cell_coords.append(coords_ai)
            cell_velocity.append(velocity_ai)
            cell_orientation.append(orientation_ai)
            cell_alignment.append(alignment_ai)
            cell_radius.append(radius_ai)
            cell_clock.append(clock_ai)
            cell_completed_cycles.append(completed_cycles_ai)
            cell_cycle_phase.append(cycle_phase_ai)
            cell_type.append(cell_type_ai)
            cell_damage.append(damage_ai)
            cell_dead.append(dead_ai)
            cell_dead_by.append(dead_by_ai)
            cell_ids.append(cell_id_ai)
        with open(str(file_path), 'w') as file:
            for line in save_context["celldata"]:
                file.write(line + '\n')
            num_cells = len(cell_ids)
            num_anchor_points = FLAMEGPU.environment.getPropertyUInt("N_ANCHOR_POINTS")
            num_total_anchor_points = num_cells * num_anchor_points
            num_points = num_cells + num_total_anchor_points

            file.write("POINTS {} float \n".format(num_cells + num_total_anchor_points))
            for coords_ai in cell_coords:
                file.write("{} {} {} \n".format(coords_ai[0], coords_ai[1], coords_ai[2]))
            for i in range(num_cells):
                for j in range(num_anchor_points):
                    file.write("{} {} {} \n".format(cell_anchor_points_x[i][j], cell_anchor_points_y[i][j], cell_anchor_points_z[i][j]))
            
            # Vertex cells (one cell per point), needed for thresholding filters
            # VTK legacy format: CELLS <ncells> <size>
            # Each vertex cell line is: "1 <pointId>"
            # So size = ncells * (1 + 1) = 2 * num_points
            file.write(f"CELLS {num_points} {2 * num_points}\n")
            for pid in range(num_points):
                file.write(f"1 {pid}\n")

            # CELL_TYPES: VTK_VERTEX = 1
            file.write(f"CELL_TYPES {num_points}\n")
            for _ in range(num_points):
                file.write("1\n")
            
            file.write("POINT_DATA {} \n".format(num_cells + num_total_anchor_points))

            file.write("SCALARS id int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for id_ai in cell_ids:
                file.write("{} \n".format(id_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{} \n".format(cell_ids[i]))
            
            file.write("SCALARS alignment float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for a_ai in cell_alignment:
                file.write("{:.4f} \n".format(a_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{:.4f} \n".format(cell_alignment[i]))

            file.write("SCALARS radius float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for r_ai in cell_radius:
                file.write("{:.4f} \n".format(r_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{:.4f} \n".format(cell_radius[i] / 10.0))

            file.write("SCALARS clock float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for c_ai in cell_clock:
                file.write("{:.4f} \n".format(c_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{:.4f} \n".format(cell_clock[i]))

            file.write("SCALARS cycle_phase int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for ccp_ai in cell_cycle_phase:
                file.write("{} \n".format(ccp_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{} \n".format(cell_cycle_phase[i]))
                    
            file.write("SCALARS completed_cycles int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for cc_ai in cell_completed_cycles:
                file.write("{} \n".format(cc_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{} \n".format(cell_completed_cycles[i]))

            file.write("SCALARS cell_type int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for ct_ai in cell_type:
                file.write("{} \n".format(ct_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{} \n".format(cell_type[i]))

            file.write("SCALARS damage float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for d_ai in cell_damage:
                file.write("{:.4f} \n".format(d_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{:.4f} \n".format(cell_damage[i]))

            file.write("SCALARS dead int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for d_ai in cell_dead:
                file.write("{} \n".format(d_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{} \n".format(cell_dead[i]))

            file.write("SCALARS dead_by int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for d_ai in cell_dead_by:
                file.write("{} \n".format(d_ai))
            for i in range(num_cells):
                for _ in range(num_anchor_points):
                    file.write("{} \n".format(cell_dead_by[i]))

            for s in range(n_species):
                file.write("SCALARS concentration_species_{0} float 1 \n".format(s))
                file.write("LOOKUP_TABLE default\n")
                for c_ai in c_sp_multi:
                    file.write("{:.4f} \n".format(c_ai[s]))
                for i in range(num_cells):
                    for _ in range(num_anchor_points):
                        file.write("{:.4f} \n".format(c_sp_multi[i][s]))

            file.write("VECTORS velocity float\n")
            for v_ai in cell_velocity:
                file.write("{:.4f} {:.4f} {:.4f} \n".format(v_ai[0], v_ai[1], v_ai[2]))
            for _ in range(num_total_anchor_points):
                file.write("0.0 0.0 0.0 \n")

            file.write("VECTORS orientation float\n")
            for o_ai in cell_orientation:
                file.write("{:.4f} {:.4f} {:.4f} \n".format(o_ai[0], o_ai[1], o_ai[2]))
            for _ in range(num_total_anchor_points):
                file.write("0.0 0.0 0.0 \n")

    if include_focal_adhesions:
        focad_coords = list()
        focad_velocity = list()
        focad_force = list()
        focad_ori = list()
        focad_cell_id = list()
        focad_rest_length = list()
        focad_k_fa = list()
        focad_f_max = list()
        focad_attached = list()
        focad_active = list()
        focad_v_c = list()
        focad_fa_state = list()
        focad_age = list()
        focad_k_on = list()
        focad_k_off_0 = list()
        focad_f_c = list()
        focad_k_reinf = list()
        focad_f_mag = list()
        focad_is_front = list()
        focad_is_rear = list()
        focad_attached_front = list()
        focad_attached_rear = list()
        focad_frontness_front = list()
        focad_frontness_rear = list()
        focad_k_on_eff_front = list()
        focad_k_on_eff_rear = list()
        focad_k_off_0_eff_front = list()
        focad_k_off_0_eff_rear = list()
        focad_linc_prev_total_length = list()
        focad_x_i = list()
        focad_y_i = list()
        focad_z_i = list()

        file_name = 'focad_t{:04d}.vtk'.format(stepCounter)
        file_path = res_path / file_name

        focad_agent = FLAMEGPU.agent("FOCAD")
        focad_agent.sortInt("id", pyflamegpu.HostAgentAPI.Asc)
        av = focad_agent.getPopulationData()
        num_focad = len(av)

        for ai in av:
            x = ai.getVariableFloat("x")
            y = ai.getVariableFloat("y")
            z = ai.getVariableFloat("z")
            vx = ai.getVariableFloat("vx")
            vy = ai.getVariableFloat("vy")
            vz = ai.getVariableFloat("vz")
            fx = ai.getVariableFloat("fx")
            fy = ai.getVariableFloat("fy")
            fz = ai.getVariableFloat("fz")
            x_i = ai.getVariableFloat("x_i")
            y_i = ai.getVariableFloat("y_i")
            z_i = ai.getVariableFloat("z_i")
            ox = x_i - x
            oy = y_i - y
            oz = z_i - z

            focad_coords.append((x, y, z))
            focad_velocity.append((vx, vy, vz))
            focad_force.append((fx, fy, fz))
            focad_ori.append((ox, oy, oz))
            focad_x_i.append(x_i)
            focad_y_i.append(y_i)
            focad_z_i.append(z_i)
            focad_cell_id.append(ai.getVariableInt("cell_id"))
            focad_rest_length.append(ai.getVariableFloat("rest_length"))
            focad_k_fa.append(ai.getVariableFloat("k_fa"))
            focad_f_max.append(ai.getVariableFloat("f_max"))
            focad_attached.append(ai.getVariableInt("attached"))
            focad_active.append(ai.getVariableUInt8("active"))
            focad_v_c.append(ai.getVariableFloat("v_c"))
            focad_fa_state.append(ai.getVariableUInt8("fa_state"))
            focad_age.append(ai.getVariableFloat("age"))
            focad_k_on.append(ai.getVariableFloat("k_on"))
            focad_k_off_0.append(ai.getVariableFloat("k_off_0"))
            focad_f_c.append(ai.getVariableFloat("f_c"))
            focad_k_reinf.append(ai.getVariableFloat("k_reinf"))
            focad_f_mag.append(ai.getVariableFloat("f_mag"))
            focad_is_front.append(ai.getVariableInt("is_front"))
            focad_is_rear.append(ai.getVariableInt("is_rear"))
            focad_attached_front.append(ai.getVariableInt("attached_front"))
            focad_attached_rear.append(ai.getVariableInt("attached_rear"))
            focad_frontness_front.append(ai.getVariableFloat("frontness_front"))
            focad_frontness_rear.append(ai.getVariableFloat("frontness_rear"))
            focad_k_on_eff_front.append(ai.getVariableFloat("k_on_eff_front"))
            focad_k_on_eff_rear.append(ai.getVariableFloat("k_on_eff_rear"))
            focad_k_off_0_eff_front.append(ai.getVariableFloat("k_off_0_eff_front"))
            focad_k_off_0_eff_rear.append(ai.getVariableFloat("k_off_0_eff_rear"))
            focad_linc_prev_total_length.append(ai.getVariableFloat("linc_prev_total_length"))

        with open(str(file_path), 'w') as file:
            for line in save_context["focaladhesionsdata"]:
                file.write(line + '\n')

            file.write("POINTS {} float \n".format(num_focad))
            for coords_ai in focad_coords:
                file.write("{} {} {} \n".format(coords_ai[0], coords_ai[1], coords_ai[2]))

            file.write("POINT_DATA {} \n".format(num_focad))

            file.write("SCALARS cell_id int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_cell_id:
                file.write("{} \n".format(v))

            file.write("SCALARS attached int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_attached:
                file.write("{} \n".format(int(v)))

            file.write("SCALARS active int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_active:
                file.write("{} \n".format(int(v)))

            file.write("SCALARS fa_state int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_fa_state:
                file.write("{} \n".format(int(v)))

            file.write("SCALARS rest_length float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_rest_length:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS k_fa float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_k_fa:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS f_max float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_f_max:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS v_c float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_v_c:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS age float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_age:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS f_mag float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_f_mag:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS is_front int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_is_front:
                file.write("{} \n".format(int(v)))

            file.write("SCALARS is_rear int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_is_rear:
                file.write("{} \n".format(int(v)))

            file.write("SCALARS attached_front int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_attached_front:
                file.write("{} \n".format(int(v)))

            file.write("SCALARS attached_rear int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_attached_rear:
                file.write("{} \n".format(int(v)))

            file.write("SCALARS frontness_front float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_frontness_front:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS frontness_rear float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_frontness_rear:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS k_on float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_k_on:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS k_on_eff_front float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_k_on_eff_front:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS k_on_eff_rear float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_k_on_eff_rear:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS k_off_0 float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_k_off_0:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS k_off_0_eff_front float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_k_off_0_eff_front:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS k_off_0_eff_rear float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_k_off_0_eff_rear:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS f_c float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_f_c:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS k_reinf float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_k_reinf:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS linc_prev_total_length float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_linc_prev_total_length:
                file.write("{:.4f} \n".format(v))

            file.write("SCALARS x_i float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_x_i:
                file.write("{:.6f} \n".format(v))

            file.write("SCALARS y_i float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_y_i:
                file.write("{:.6f} \n".format(v))

            file.write("SCALARS z_i float 1\n")
            file.write("LOOKUP_TABLE default\n")
            for v in focad_z_i:
                file.write("{:.6f} \n".format(v))

            file.write("VECTORS velocity float\n")
            for v_ai in focad_velocity:
                file.write("{:.6f} {:.6f} {:.6f} \n".format(v_ai[0], v_ai[1], v_ai[2]))

            file.write("VECTORS force float\n")
            for f_ai in focad_force:
                file.write("{:.6f} {:.6f} {:.6f} \n".format(f_ai[0], f_ai[1], f_ai[2]))

            file.write("VECTORS ori float\n")
            for o_ai in focad_ori:
                file.write("{:.6f} {:.6f} {:.6f} \n".format(o_ai[0], o_ai[1], o_ai[2]))
                
                
            # Write nucleus data in a separate file. For these files:
            #   - Apply 'Tensor Glyph' and choose tensor 'U' (or 'eps') to render ellipsoids.
            #   - Set Glyph scale factor to nucleus radius (or use 'nucleus_radius' scalar as Scale Array).
            file_name = "nucleus_t{:04d}.vtk".format(stepCounter)
            file_path = res_path / file_name
            # -----------------------------
            # Collect CELL data
            # -----------------------------
            cell_coords = list()
            cell_id = list()
            cell_radius = list()
            nucleus_radius = list()

            # raw components
            eps = list()   # (xx, yy, zz, xy, xz, yz)
            sig = list()   # (xx, yy, zz, xy, xz, yz)

            cell_agent = FLAMEGPU.agent("CELL")
            cell_agent.sortInt("id", pyflamegpu.HostAgentAPI.Asc)  # keep ids ordered for viz
            av = cell_agent.getPopulationData()
            num_cells = len(av)

            for ai in av:
                x = ai.getVariableFloat("x")
                y = ai.getVariableFloat("y")
                z = ai.getVariableFloat("z")
                r = ai.getVariableFloat("radius")
                nr = ai.getVariableFloat("nucleus_radius")

                exx = ai.getVariableFloat("eps_xx")
                eyy = ai.getVariableFloat("eps_yy")
                ezz = ai.getVariableFloat("eps_zz")
                exy = ai.getVariableFloat("eps_xy")
                exz = ai.getVariableFloat("eps_xz")
                eyz = ai.getVariableFloat("eps_yz")

                sxx = ai.getVariableFloat("sig_xx")
                syy = ai.getVariableFloat("sig_yy")
                szz = ai.getVariableFloat("sig_zz")
                sxy = ai.getVariableFloat("sig_xy")
                sxz = ai.getVariableFloat("sig_xz")
                syz = ai.getVariableFloat("sig_yz")

                cell_coords.append((x, y, z))
                cell_id.append(ai.getVariableInt("id"))
                cell_radius.append(r)
                nucleus_radius.append(nr)

                eps.append((exx, eyy, ezz, exy, exz, eyz))
                sig.append((sxx, syy, szz, sxy, sxz, syz))

            eps = np.asarray(eps, dtype=np.float64)  # (N,6)
            sig = np.asarray(sig, dtype=np.float64)  # (N,6)
            cell_radius = np.asarray(cell_radius, dtype=np.float64)
            nucleus_radius = np.asarray(nucleus_radius, dtype=np.float64)

            # -----------------------------
            # Build full 3x3 symmetric tensors
            # -----------------------------
            def voigt6_to_sym33(v6):
                """v6 = (xx,yy,zz,xy,xz,yz) -> 3x3 symmetric"""
                xx, yy, zz, xy, xz, yz = v6
                return np.array([[xx, xy, xz],
                                [xy, yy, yz],
                                [xz, yz, zz]], dtype=np.float64)

            eps33 = np.zeros((num_cells, 3, 3), dtype=np.float64)
            sig33 = np.zeros((num_cells, 3, 3), dtype=np.float64)
            U33   = np.zeros((num_cells, 3, 3), dtype=np.float64)

            for i in range(num_cells):
                eps33[i] = voigt6_to_sym33(eps[i])
                sig33[i] = voigt6_to_sym33(sig[i])
                U33[i]   = np.eye(3, dtype=np.float64) + eps33[i]  # U = I + eps

            # -----------------------------
            # Derived scalars and vectors for mechanical interpretation
            # -----------------------------
            #
            # eps33  : symmetric small-strain tensor (dimensionless)
            # sig33  : Cauchy stress tensor (units: nN/um^2 = kPa)
            #
            # The following derived quantities provide compact,
            # orientation-independent mechanical metrics suitable
            # for visualization and quantitative analysis.
            #

            # Frobenius norm of strain tensor:
            #   ||eps|| = sqrt(sum_ij eps_ij^2)
            # Measures total magnitude of nucleus deformation,
            # independent of coordinate system.
            # 0 corresponds to a perfect sphere (no deformation).
            eps_norm = np.sqrt(np.sum(eps33 * eps33, axis=(1, 2)))

            # Frobenius norm of stress tensor:
            #   ||sigma|| = sqrt(sum_ij sigma_ij^2)
            # Represents overall stress intensity inside the nucleus.
            # Units are kPa under the nN/um^2 unit convention.
            sig_norm = np.sqrt(np.sum(sig33 * sig33, axis=(1, 2)))

            # Hydrostatic stress:
            #   sigma_hydro = (1/3) tr(sigma)
            # Mean normal stress component.
            # Positive values indicate net tension,
            # negative values indicate net compression.
            sig_hydro = (sig33[:, 0, 0] + sig33[:, 1, 1] + sig33[:, 2, 2]) / 3.0

            # Deviatoric stress tensor:
            #   sigma' = sigma - sigma_hydro * I
            # Captures the anisotropic (shape-changing) component of stress.
            # This component is responsible for distortional deformation.
            I = np.eye(3, dtype=np.float64)
            sig_dev = sig33 - sig_hydro[:, None, None] * I[None, :, :]

            # Frobenius norm of deviatoric stress:
            # Measures magnitude of anisotropic loading.
            # High values indicate strong directional traction imbalance.
            sig_dev_norm = np.sqrt(np.sum(sig_dev * sig_dev, axis=(1, 2)))

            # Principal strain decomposition:
            # Since eps33 is symmetric, eigen-decomposition is stable.
            # Eigenvalues correspond to principal strains.
            # Eigenvectors correspond to principal deformation directions.
            eps_evals = np.zeros((num_cells, 3), dtype=np.float64)
            eps_evecs = np.zeros((num_cells, 3, 3), dtype=np.float64)

            for i in range(num_cells):
                w, v = np.linalg.eigh(eps33[i])  # ascending order
                eps_evals[i] = w
                eps_evecs[i] = v

            # Largest principal strain:
            # Represents maximum extension (if positive)
            # or strongest compression (if negative).
            idx_max = 2  # eigenvalues sorted ascending
            eps_max = eps_evals[:, idx_max]

            # Corresponding principal direction:
            # Orientation of maximum extension.
            # Useful for analyzing nucleus elongation alignment.
            dir_max = eps_evecs[:, :, idx_max]  # (N,3)

            # Aspect ratio proxy derived from principal stretches:
            #
            # Under small-strain mapping:
            #   U = I + eps
            # Principal stretches approx:
            #   lambda_i = 1 + e_i
            #
            # Aspect ratio proxy:
            #   AR = max(lambda_i) / min(lambda_i)
            #
            # AR = 1  -> spherical nucleus
            # AR > 1  -> ellipsoidal deformation
            lam = 1.0 + eps_evals
            lam_min = np.minimum(np.minimum(lam[:, 0], lam[:, 1]), lam[:, 2])
            lam_max = np.maximum(np.maximum(lam[:, 0], lam[:, 1]), lam[:, 2])

            # Numerical safeguard against division by zero or negative stretch
            lam_min = np.maximum(lam_min, 1e-6)
            ar_proxy = lam_max / lam_min

            # Principal direction scaled by nucleus radius:
            # Useful for glyph-based visualization of elongation axes.
            # Scaling improves visual clarity in ParaView.
            dir_max_scaled = dir_max * nucleus_radius[:, None]



            with open(str(file_path), "w") as file:
                for line in save_context["nucleusdata"]:
                    file.write(line + "\n")

                # Points: nucleus centers
                file.write("POINTS {} float \n".format(num_cells))
                for x, y, z in cell_coords:
                    file.write("{:.6f} {:.6f} {:.6f}\n".format(x, y, z))

                file.write("POINT_DATA {} \n".format(num_cells))

                # ---- Scalars ----
                file.write("SCALARS id int 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in cell_id:
                    file.write("{}\n".format(v))

                file.write("SCALARS radius float 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in cell_radius:
                    file.write("{:.6f}\n".format(v))
                    
                file.write("SCALARS nucleus_radius float 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in nucleus_radius:
                    file.write("{:.6f}\n".format(v))

                file.write("SCALARS eps_norm float 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in eps_norm:
                    file.write("{:.6f}\n".format(v))

                file.write("SCALARS sig_norm float 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in sig_norm:
                    file.write("{:.6f}\n".format(v))

                file.write("SCALARS sig_hydro float 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in sig_hydro:
                    file.write("{:.6f}\n".format(v))

                file.write("SCALARS sig_dev_norm float 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in sig_dev_norm:
                    file.write("{:.6f}\n".format(v))

                file.write("SCALARS eps_max float 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in eps_max:
                    file.write("{:.6f}\n".format(v))

                file.write("SCALARS ar_proxy float 1\n")
                file.write("LOOKUP_TABLE default\n")
                for v in ar_proxy:
                    file.write("{:.6f}\n".format(v))

                # ---- Vectors ----
                file.write("VECTORS eps_dir_max float\n")
                for vx, vy, vz in dir_max_scaled:
                    file.write("{:.6f} {:.6f} {:.6f}\n".format(vx, vy, vz))

                # ---- Tensors ----
                # 3 rows per tensor per point.

                file.write("TENSORS eps float\n")
                for i in range(num_cells):
                    M = eps33[i]
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[0, 0], M[0, 1], M[0, 2]))
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[1, 0], M[1, 1], M[1, 2]))
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[2, 0], M[2, 1], M[2, 2]))

                file.write("TENSORS sigma float\n")
                for i in range(num_cells):
                    M = sig33[i]
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[0, 0], M[0, 1], M[0, 2]))
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[1, 0], M[1, 1], M[1, 2]))
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[2, 0], M[2, 1], M[2, 2]))

                # U = I + eps (dimensionless). Use nucelus radius as scale factor in ParaView.
                file.write("TENSORS U float\n")
                for i in range(num_cells):
                    M = U33[i]
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[0, 0], M[0, 1], M[0, 2]))
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[1, 0], M[1, 1], M[1, 2]))
                    file.write("{:.6e} {:.6e} {:.6e}\n".format(M[2, 0], M[2, 1], M[2, 2]))

    file_name = 'ecm_data_t{:04d}.vtk'.format(stepCounter)
    file_path = res_path / file_name
    agent = FLAMEGPU.agent("ECM")

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
    c_sp_multi = list()
    if heterogeneous_diffusion:
        d_sp_multi = list()
    av = agent.getPopulationData()
    for ai in av:
        coords_ai = (ai.getVariableFloat("x"), ai.getVariableFloat("y"), ai.getVariableFloat("z"))
        velocity_ai = (ai.getVariableFloat("vx"), ai.getVariableFloat("vy"), ai.getVariableFloat("vz"))
        force_ai = (ai.getVariableFloat("fx"), ai.getVariableFloat("fy"), ai.getVariableFloat("fz"))
        coords.append(coords_ai)
        velocity.append(velocity_ai)
        force.append(force_ai)
        c_sp_multi.append(ai.getVariableArrayFloat("C_sp"))
        if heterogeneous_diffusion:
            d_sp_multi.append(ai.getVariableArrayFloat("D_sp"))

    print("====== SAVING DATA FROM Step {:03d} TO FILE ======".format(stepCounter))
    with open(str(file_path), 'w') as file:
        for line in save_context["header"]:
            file.write(line + '\n')
        for coords_ai in coords:
            file.write("{} {} {} \n".format(coords_ai[0], coords_ai[1], coords_ai[2]))

        file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[2], coord_boundary[4]))
        file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[2], coord_boundary[4]))
        file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[3], coord_boundary[4]))
        file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[3], coord_boundary[4]))
        file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[2], coord_boundary[5]))
        file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[2], coord_boundary[5]))
        file.write("{} {} {} \n".format(coord_boundary[1], coord_boundary[3], coord_boundary[5]))
        file.write("{} {} {} \n".format(coord_boundary[0], coord_boundary[3], coord_boundary[5]))
        for line in save_context["domaindata"]:
            file.write(line + '\n')

        file.write("SCALARS boundary_normal_forces float 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write(str(sum_bx_pos) + '\n')
        file.write(str(sum_bx_neg) + '\n')
        file.write(str(sum_by_pos) + '\n')
        file.write(str(sum_by_neg) + '\n')
        file.write(str(sum_bz_pos) + '\n')
        file.write(str(sum_bz_neg) + '\n')

        file.write("SCALARS boundary_normal_force_scaling float 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write(str(abs(sum_bx_pos)) + '\n')
        file.write(str(abs(sum_bx_neg)) + '\n')
        file.write(str(abs(sum_by_pos)) + '\n')
        file.write(str(abs(sum_by_neg)) + '\n')
        file.write(str(abs(sum_bz_pos)) + '\n')
        file.write(str(abs(sum_bz_neg)) + '\n')

        file.write("VECTORS boundary_normal_force_dir float\n")
        file.write("1 0 0 \n" if sum_bx_pos > 0 else "-1 0 0 \n")
        file.write("1 0 0 \n" if sum_bx_neg > 0 else "-1 0 0 \n")
        file.write("0 1 0 \n" if sum_by_pos > 0 else "0 -1 0 \n")
        file.write("0 1 0 \n" if sum_by_neg > 0 else "0 -1 0 \n")
        file.write("0 0 1 \n" if sum_bz_pos > 0 else "0 0 -1 \n")
        file.write("0 0 1 \n" if sum_bz_neg > 0 else "0 0 -1 \n")

        file.write("SCALARS boundary_shear_forces_pos float 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write(str(sum_bx_pos_y) + '\n')
        file.write(str(sum_bx_pos_z) + '\n')
        file.write(str(sum_by_pos_x) + '\n')
        file.write(str(sum_by_pos_z) + '\n')
        file.write(str(sum_bz_pos_x) + '\n')
        file.write(str(sum_bz_pos_y) + '\n')

        file.write("SCALARS boundary_shear_forces_neg float 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write(str(sum_bx_neg_y) + '\n')
        file.write(str(sum_bx_neg_z) + '\n')
        file.write(str(sum_by_neg_x) + '\n')
        file.write(str(sum_by_neg_z) + '\n')
        file.write(str(sum_bz_neg_x) + '\n')
        file.write(str(sum_bz_neg_y) + '\n')

        file.write("SCALARS boundary_shear_force_scaling_pos float 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write(str(abs(sum_bx_pos_y)) + '\n')
        file.write(str(abs(sum_bx_pos_z)) + '\n')
        file.write(str(abs(sum_by_pos_x)) + '\n')
        file.write(str(abs(sum_by_pos_z)) + '\n')
        file.write(str(abs(sum_bz_pos_x)) + '\n')
        file.write(str(abs(sum_bz_pos_y)) + '\n')

        file.write("SCALARS boundary_shear_force_scaling_neg float 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write(str(abs(sum_bx_neg_y)) + '\n')
        file.write(str(abs(sum_bx_neg_z)) + '\n')
        file.write(str(abs(sum_by_neg_x)) + '\n')
        file.write(str(abs(sum_by_neg_z)) + '\n')
        file.write(str(abs(sum_bz_neg_x)) + '\n')
        file.write(str(abs(sum_bz_neg_y)) + '\n')

        file.write("VECTORS boundary_shear_force_dir_pos float\n")
        file.write("0 1 0 \n" if sum_bx_pos_y > 0 else "0 -1 0 \n")
        file.write("0 0 1 \n" if sum_bx_pos_z > 0 else "0 0 -1 \n")
        file.write("1 0 0 \n" if sum_by_pos_x > 0 else "-1 0 0 \n")
        file.write("0 0 1 \n" if sum_by_pos_z > 0 else "0 0 -1 \n")
        file.write("1 0 0 \n" if sum_bz_pos_x > 0 else "-1 0 0 \n")
        file.write("0 1 0 \n" if sum_bz_pos_y > 0 else "0 -1 0 \n")

        file.write("VECTORS boundary_shear_force_dir_neg float\n")
        file.write("0 1 0 \n" if sum_bx_neg_y > 0 else "0 -1 0 \n")
        file.write("0 0 1 \n" if sum_bx_neg_z > 0 else "0 0 -1 \n")
        file.write("1 0 0 \n" if sum_by_neg_x > 0 else "-1 0 0 \n")
        file.write("0 0 1 \n" if sum_by_neg_z > 0 else "0 0 -1 \n")
        file.write("1 0 0 \n" if sum_bz_neg_x > 0 else "-1 0 0 \n")
        file.write("0 1 0 \n" if sum_bz_neg_y > 0 else "0 -1 0 \n")

        file.write("POINT_DATA {} \n".format(8 + ecm_population_size))
        file.write("SCALARS is_corner int 1\n")
        file.write("LOOKUP_TABLE default\n")
        for _ in range(ecm_population_size):
            file.write("0 \n")
        for _ in range(8):
            file.write("1 \n")

        for s in range(n_species):
            file.write("SCALARS concentration_species_{0} float 1 \n".format(s))
            file.write("LOOKUP_TABLE default\n")
            for c_ai in c_sp_multi:
                file.write("{:.4f} \n".format(c_ai[s]))
            for _ in range(8):
                file.write("0.0 \n")

        if heterogeneous_diffusion:
            for s in range(n_species):
                file.write("SCALARS diffusion_coeff_{0} float 1 \n".format(s))
                file.write("LOOKUP_TABLE default\n")
                for d_ai in d_sp_multi:
                    file.write("{:.4f} \n".format(d_ai[s]))
                for _ in range(8):
                    file.write("0.0 \n")

        file.write("VECTORS velocity float\n")
        for v_ai in velocity:
            file.write("{:.4f} {:.4f} {:.4f} \n".format(v_ai[0], v_ai[1], v_ai[2]))
        for _ in range(8):
            file.write("0.0 0.0 0.0 \n")

        file.write("VECTORS force float\n")
        for f_ai in force:
            file.write("{:.4f} {:.4f} {:.4f} \n".format(f_ai[0], f_ai[1], f_ai[2]))
        for _ in range(8):
            file.write("0.0 0.0 0.0 \n")

    print("... succesful save ")
    print("=================================")


class ModelParameterConfig:
    def __init__(
        self,
        save_every_n_steps: int = None,
        ecm_agents_per_dir: list = None,
        time_step: float = None,
        steps: int = None,
        # Simulation control
        ensemble: bool = None,
        ensemble_runs: int = None,
        visualisation: bool = None,
        debug_printing: bool = None,
        pause_every_step: bool = None,
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
        relative_boundary_stiffness: list = None,
        boundary_stiffness_value: float = None,
        boundary_dumping_value: float = None,
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
        max_strain_k_factor: float = None,
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
        fibre_node_repulsion_k: float = None,
        # Fibre network remodeling
        include_network_remodeling: bool = None,
        fnode_degradation_rate: float = None,
        fnode_deposition_rate: float = None,
        fnode_cell_degradation_radius: float = None,
        fnode_birth_k_0: float = None,
        fnode_birth_k_max: float = None,
        fnode_birth_species_index: int = None,
        fnode_birth_k_c: float = None,
        fnode_birth_hill_conc: float = None,
        fnode_birth_k_sigma: float = None,
        fnode_birth_hill_sigma: float = None,
        fnode_birth_radius: float = None,
        fnode_birth_link_max_distance: float = None,
        fnode_birth_refractory: float = None,
        # Diffusion
        include_diffusion: bool = None,
        heterogeneous_diffusion: bool = None,
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
        # Cell interactions & derived
        dead_cells_disappear: bool = None,
        include_cell_fnode_repulsion: bool = None,
        cell_nucleus_radius: float = None,
        brownian_motion_strength: float = None,
        brownian_motion_strength_factor: bool = None,
        cell_cell_repulsion_k: float = None,
        cell_cell_adhesion_k: float = None,
        cell_cell_adhesion_range: float = None,
        cell_cell_dv_max: float = None,
        cell_fnode_repulsion_k: float = None,
        cell_fnode_exclusion_distance: float = None,
        cell_fnode_dv_max: float = None,
        max_expected_n_cells: int = None,
        # Cell damage & death
        cell_hypoxia_threshold: float = None,
        cell_nutrient_threshold: float = None,
        cell_stress_threshold: float = None,
        cell_hypoxia_damage_rate: float = None,
        cell_nutrient_damage_rate: float = None,
        cell_stress_damage_rate: float = None,
        cell_basal_damage_repair_rate: float = None,
        cell_acute_hypoxia_threshold: float = None,
        cell_acute_nutrient_threshold: float = None,
        cell_acute_stress_threshold: float = None,
        # Focal adhesions
        include_focal_adhesions: bool = None,
        init_n_focad_per_cell: int = None,
        n_anchor_points: int = None,
        max_search_radius_focad: float = None,
        max_focad_arm_length: float = None,
        focad_rest_length_0: float = None,
        focad_min_rest_length: float = None,
        focad_k_fa: float = None,
        focad_f_max: float = None,
        focad_v_c: float = None,
        focad_k_on: float = None,
        focad_k_off_0: float = None,
        focad_f_c: float = None,
        use_catch_bond: bool = None,
        catch_bond_catch_scale: float = None,
        catch_bond_slip_scale: float = None,
        catch_bond_f_catch: float = None,
        catch_bond_f_slip: float = None,
        focad_k_reinf: float = None,
        focad_f_reinf: float = None,
        focad_k_fa_max: float = None,
        focad_k_fa_decay: float = None,
        focad_polarity_kon_front_gain: float = None,
        focad_polarity_koff_front_reduction: float = None,
        focad_polarity_koff_rear_gain: float = None,
        # Focal adhesion lifecycle & birth
        focad_f_mature: float = None,
        focad_t_nascent_max: float = None,
        focad_t_detached_grace: float = None,
        focad_t_disassembly: float = None,
        enable_focad_birth: bool = None,
        focad_birth_species_index: int = None,
        focad_birth_n_min: int = None,
        focad_birth_n_max: int = None,
        focad_birth_k_0: float = None,
        focad_birth_k_max: float = None,
        focad_birth_k_sigma: float = None,
        focad_birth_hill_sigma: float = None,
        focad_birth_k_c: float = None,
        focad_birth_hill_conc: float = None,
        focad_birth_refractory: float = None,
        focad_mobility_mu: float = None,
        include_linc_coupling: bool = None,
        linc_k_elast: float = None,
        linc_d_dumping: float = None,
        linc_rest_length: float = None,
        # Nucleus mechanics
        nucleus_e: float = None,
        nucleus_nu: float = None,
        nucleus_tau: float = None,
        nucleus_eps_clamp: float = None,
        # Chemotaxis
        include_chemotaxis: bool = None,
        chemotaxis_sensitivity: list = None,
        chemotaxis_only_dir: bool = None,
        chemotaxis_chi: float = None,
        # Durotaxis / cell migration
        include_durotaxis: bool = None,
        durotaxis_only_dir: bool = None,
        include_orientation_align: bool = None,
        orientation_align_rate: float = None,
        orientation_align_use_stress: bool = None,
        durotaxis_blend_beta: float = None,
        durotaxis_use_stress: bool = None,
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
        self.ENSEMBLE = ensemble
        self.ENSEMBLE_RUNS = ensemble_runs
        self.VISUALISATION = visualisation
        self.DEBUG_PRINTING = debug_printing
        self.PAUSE_EVERY_STEP = pause_every_step
        self.BOUNDARY_COORDS = boundary_coords
        self.BOUNDARY_DISP_RATES = boundary_disp_rates
        self.BOUNDARY_DISP_RATES_PARALLEL = boundary_disp_rates_parallel
        self.POISSON_DIRS = poisson_dirs
        self.ALLOW_BOUNDARY_ELASTIC_MOVEMENT = allow_boundary_elastic_movement
        self.BOUNDARY_STIFFNESS = boundary_stiffness
        self.BOUNDARY_DUMPING = boundary_dumping
        self.CLAMP_AGENT_TOUCHING_BOUNDARY = clamp_agent_touching_boundary
        self.ALLOW_AGENT_SLIDING = allow_agent_sliding
        self.RELATIVE_BOUNDARY_STIFFNESS = relative_boundary_stiffness
        self.BOUNDARY_STIFFNESS_VALUE = boundary_stiffness_value
        self.BOUNDARY_DUMPING_VALUE = boundary_dumping_value
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
        self.MAX_STRAIN_K_FACTOR = max_strain_k_factor
        self.INCLUDE_FIBRE_NETWORK = include_fibre_network
        self.MAX_CONNECTIVITY = max_connectivity
        self.FIBRE_SEGMENT_K_ELAST = fibre_segment_k_elast
        self.FIBRE_SEGMENT_D_DUMPING = fibre_segment_d_dumping
        self.FIBRE_SEGMENT_MASS = fibre_segment_mass
        self.FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = fibre_segment_equilibrium_distance
        self.FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS = fibre_node_boundary_interaction_radius
        self.FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE = fibre_node_boundary_equilibrium_distance
        self.MAX_SEARCH_RADIUS_FNODES = max_search_radius_fnodes
        self.FIBRE_NODE_REPULSION_K = fibre_node_repulsion_k
        self.INCLUDE_NETWORK_REMODELING = include_network_remodeling
        self.FNODE_DEGRADATION_RATE = fnode_degradation_rate
        self.FNODE_DEPOSITION_RATE = fnode_deposition_rate
        self.FNODE_CELL_DEGRADATION_RADIUS = fnode_cell_degradation_radius
        self.FNODE_BIRTH_K_0 = fnode_birth_k_0
        self.FNODE_BIRTH_K_MAX = fnode_birth_k_max
        self.FNODE_BIRTH_SPECIES_INDEX = fnode_birth_species_index
        self.FNODE_BIRTH_K_C = fnode_birth_k_c
        self.FNODE_BIRTH_HILL_CONC = fnode_birth_hill_conc
        self.FNODE_BIRTH_K_SIGMA = fnode_birth_k_sigma
        self.FNODE_BIRTH_HILL_SIGMA = fnode_birth_hill_sigma
        self.FNODE_BIRTH_RADIUS = fnode_birth_radius
        self.FNODE_BIRTH_LINK_MAX_DISTANCE = fnode_birth_link_max_distance
        self.FNODE_BIRTH_REFRACTORY = fnode_birth_refractory
        self.INCLUDE_DIFFUSION = include_diffusion
        self.HETEROGENEOUS_DIFFUSION = heterogeneous_diffusion
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
        self.DEAD_CELLS_DISAPPEAR = dead_cells_disappear
        self.INCLUDE_CELL_FNODE_REPULSION = include_cell_fnode_repulsion
        self.CELL_NUCLEUS_RADIUS = cell_nucleus_radius
        self.BROWNIAN_MOTION_STRENGTH = brownian_motion_strength
        self.BROWNIAN_MOTION_STRENGTH_FACTOR = brownian_motion_strength_factor
        self.CELL_CELL_REPULSION_K = cell_cell_repulsion_k
        self.CELL_CELL_ADHESION_K = cell_cell_adhesion_k
        self.CELL_CELL_ADHESION_RANGE = cell_cell_adhesion_range
        self.CELL_CELL_DV_MAX = cell_cell_dv_max
        self.CELL_FNODE_REPULSION_K = cell_fnode_repulsion_k
        self.CELL_FNODE_EXCLUSION_DISTANCE = cell_fnode_exclusion_distance
        self.CELL_FNODE_DV_MAX = cell_fnode_dv_max
        self.MAX_EXPECTED_N_CELLS = max_expected_n_cells
        self.CELL_HYPOXIA_THRESHOLD = cell_hypoxia_threshold
        self.CELL_NUTRIENT_THRESHOLD = cell_nutrient_threshold
        self.CELL_STRESS_THRESHOLD = cell_stress_threshold
        self.CELL_HYPOXIA_DAMAGE_RATE = cell_hypoxia_damage_rate
        self.CELL_NUTRIENT_DAMAGE_RATE = cell_nutrient_damage_rate
        self.CELL_STRESS_DAMAGE_RATE = cell_stress_damage_rate
        self.CELL_BASAL_DAMAGE_REPAIR_RATE = cell_basal_damage_repair_rate
        self.CELL_ACUTE_HYPOXIA_THRESHOLD = cell_acute_hypoxia_threshold
        self.CELL_ACUTE_NUTRIENT_THRESHOLD = cell_acute_nutrient_threshold
        self.CELL_ACUTE_STRESS_THRESHOLD = cell_acute_stress_threshold
        self.INCLUDE_FOCAL_ADHESIONS = include_focal_adhesions
        self.INIT_N_FOCAD_PER_CELL = init_n_focad_per_cell
        self.N_ANCHOR_POINTS = n_anchor_points
        self.MAX_SEARCH_RADIUS_FOCAD = max_search_radius_focad
        self.MAX_FOCAD_ARM_LENGTH = max_focad_arm_length
        self.FOCAD_REST_LENGTH_0 = focad_rest_length_0
        self.FOCAD_MIN_REST_LENGTH = focad_min_rest_length
        self.FOCAD_K_FA = focad_k_fa
        self.FOCAD_F_MAX = focad_f_max
        self.FOCAD_V_C = focad_v_c
        self.FOCAD_K_ON = focad_k_on
        self.FOCAD_K_OFF_0 = focad_k_off_0
        self.FOCAD_F_C = focad_f_c
        self.USE_CATCH_BOND = use_catch_bond
        self.CATCH_BOND_CATCH_SCALE = catch_bond_catch_scale
        self.CATCH_BOND_SLIP_SCALE = catch_bond_slip_scale
        self.CATCH_BOND_F_CATCH = catch_bond_f_catch
        self.CATCH_BOND_F_SLIP = catch_bond_f_slip
        self.FOCAD_K_REINF = focad_k_reinf
        self.FOCAD_F_REINF = focad_f_reinf
        self.FOCAD_K_FA_MAX = focad_k_fa_max
        self.FOCAD_K_FA_DECAY = focad_k_fa_decay
        self.FOCAD_POLARITY_KON_FRONT_GAIN = focad_polarity_kon_front_gain
        self.FOCAD_POLARITY_KOFF_FRONT_REDUCTION = focad_polarity_koff_front_reduction
        self.FOCAD_POLARITY_KOFF_REAR_GAIN = focad_polarity_koff_rear_gain
        self.FOCAD_F_MATURE = focad_f_mature
        self.FOCAD_T_NASCENT_MAX = focad_t_nascent_max
        self.FOCAD_T_DETACHED_GRACE = focad_t_detached_grace
        self.FOCAD_T_DISASSEMBLY = focad_t_disassembly
        self.ENABLE_FOCAD_BIRTH = enable_focad_birth
        self.FOCAD_BIRTH_SPECIES_INDEX = focad_birth_species_index
        self.FOCAD_BIRTH_N_MIN = focad_birth_n_min
        self.FOCAD_BIRTH_N_MAX = focad_birth_n_max
        self.FOCAD_BIRTH_K_0 = focad_birth_k_0
        self.FOCAD_BIRTH_K_MAX = focad_birth_k_max
        self.FOCAD_BIRTH_K_SIGMA = focad_birth_k_sigma
        self.FOCAD_BIRTH_HILL_SIGMA = focad_birth_hill_sigma
        self.FOCAD_BIRTH_K_C = focad_birth_k_c
        self.FOCAD_BIRTH_HILL_CONC = focad_birth_hill_conc
        self.FOCAD_BIRTH_REFRACTORY = focad_birth_refractory
        self.FOCAD_MOBILITY_MU = focad_mobility_mu
        self.INCLUDE_LINC_COUPLING = include_linc_coupling
        self.LINC_K_ELAST = linc_k_elast
        self.LINC_D_DUMPING = linc_d_dumping
        self.LINC_REST_LENGTH = linc_rest_length
        self.NUCLEUS_E = nucleus_e
        self.NUCLEUS_NU = nucleus_nu
        self.NUCLEUS_TAU = nucleus_tau
        self.NUCLEUS_EPS_CLAMP = nucleus_eps_clamp
        self.INCLUDE_CHEMOTAXIS = include_chemotaxis
        self.CHEMOTAXIS_SENSITIVITY = chemotaxis_sensitivity
        self.CHEMOTAXIS_ONLY_DIR = chemotaxis_only_dir
        self.CHEMOTAXIS_CHI = chemotaxis_chi
        self.INCLUDE_DUROTAXIS = include_durotaxis
        self.DUROTAXIS_ONLY_DIR = durotaxis_only_dir
        self.INCLUDE_ORIENTATION_ALIGN = include_orientation_align
        self.ORIENTATION_ALIGN_RATE = orientation_align_rate
        self.ORIENTATION_ALIGN_USE_STRESS = orientation_align_use_stress
        self.DUROTAXIS_BLEND_BETA = durotaxis_blend_beta
        self.DUROTAXIS_USE_STRESS = durotaxis_use_stress
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
                f"INCLUDE_FOCAL_ADHESIONS: True | FOCAD_PER_CELL: {self.INIT_N_FOCAD_PER_CELL} | FOCAD_COUNT: {focad_count} | LINC: {self.INCLUDE_LINC_COUPLING}"
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
                "Diffusion: enabled | Species: {0} | Coefficients: {1} | Mode: {2} | Unstable: {3}".format(
                    self.N_SPECIES,
                    coeff_text,
                    "heterogeneous" if self.HETEROGENEOUS_DIFFUSION else "homogeneous",
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
            print(f" - Rest length (L0, min): {self.FOCAD_REST_LENGTH_0}, {self.FOCAD_MIN_REST_LENGTH}")
            print(f" - Stiffness (k_fa, k_fa_max): {self.FOCAD_K_FA}, {self.FOCAD_K_FA_MAX}")
            print(f" - Bond model: {'catch' if self.USE_CATCH_BOND else 'slip'}")
            if self.USE_CATCH_BOND:
                print(
                    f"   · Catch params (catch_scale, slip_scale, F_catch, F_slip): "
                    f"{self.CATCH_BOND_CATCH_SCALE}, {self.CATCH_BOND_SLIP_SCALE}, "
                    f"{self.CATCH_BOND_F_CATCH}, {self.CATCH_BOND_F_SLIP}"
                )
            else:
                print(f"   · Slip param F_c: {self.FOCAD_F_C}")
            print(f" - Reinforcement (k_reinf, f_reinf, decay): {self.FOCAD_K_REINF}, {self.FOCAD_F_REINF}, {self.FOCAD_K_FA_DECAY}")
            print(
                f" - Polarity gains (kon_front, koff_front_red, koff_rear_gain): "
                f"{self.FOCAD_POLARITY_KON_FRONT_GAIN}, {self.FOCAD_POLARITY_KOFF_FRONT_REDUCTION}, {self.FOCAD_POLARITY_KOFF_REAR_GAIN}"
            )
            print(f" - Mobility (durotaxis traction coupling): {self.FOCAD_MOBILITY_MU}")
            print(f" - LINC coupling: {self.INCLUDE_LINC_COUPLING}")
            if self.INCLUDE_LINC_COUPLING:
                print(
                    f"   · LINC (k, d, L0): {self.LINC_K_ELAST}, {self.LINC_D_DUMPING}, {self.LINC_REST_LENGTH}"
                )

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
        ensemble=ns.get("ENSEMBLE"),
        ensemble_runs=ns.get("ENSEMBLE_RUNS"),
        visualisation=ns.get("VISUALISATION"),
        debug_printing=ns.get("DEBUG_PRINTING"),
        pause_every_step=ns.get("PAUSE_EVERY_STEP"),
        boundary_coords=ns.get("BOUNDARY_COORDS"),
        boundary_disp_rates=ns.get("BOUNDARY_DISP_RATES"),
        boundary_disp_rates_parallel=ns.get("BOUNDARY_DISP_RATES_PARALLEL"),
        poisson_dirs=ns.get("POISSON_DIRS"),
        allow_boundary_elastic_movement=ns.get("ALLOW_BOUNDARY_ELASTIC_MOVEMENT"),
        boundary_stiffness=ns.get("BOUNDARY_STIFFNESS"),
        boundary_dumping=ns.get("BOUNDARY_DUMPING"),
        clamp_agent_touching_boundary=ns.get("CLAMP_AGENT_TOUCHING_BOUNDARY"),
        allow_agent_sliding=ns.get("ALLOW_AGENT_SLIDING"),
        relative_boundary_stiffness=ns.get("RELATIVE_BOUNDARY_STIFFNESS"),
        boundary_stiffness_value=ns.get("BOUNDARY_STIFFNESS_VALUE"),
        boundary_dumping_value=ns.get("BOUNDARY_DUMPING_VALUE"),
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
        max_strain_k_factor=ns.get("MAX_STRAIN_K_FACTOR"),
        include_fibre_network=ns.get("INCLUDE_FIBRE_NETWORK"),
        max_connectivity=ns.get("MAX_CONNECTIVITY"),
        fibre_segment_k_elast=ns.get("FIBRE_SEGMENT_K_ELAST"),
        fibre_segment_d_dumping=ns.get("FIBRE_SEGMENT_D_DUMPING"),
        fibre_segment_mass=ns.get("FIBRE_SEGMENT_MASS"),
        fibre_segment_equilibrium_distance=ns.get("FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE"),
        fibre_node_boundary_interaction_radius=ns.get("FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS"),
        fibre_node_boundary_equilibrium_distance=ns.get("FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE"),
        max_search_radius_fnodes=ns.get("MAX_SEARCH_RADIUS_FNODES"),
        fibre_node_repulsion_k=ns.get("FIBRE_NODE_REPULSION_K"),
        include_network_remodeling=ns.get("INCLUDE_NETWORK_REMODELING"),
        fnode_degradation_rate=ns.get("FNODE_DEGRADATION_RATE"),
        fnode_deposition_rate=ns.get("FNODE_DEPOSITION_RATE"),
        fnode_cell_degradation_radius=ns.get("FNODE_CELL_DEGRADATION_RADIUS"),
        fnode_birth_k_0=ns.get("FNODE_BIRTH_K_0"),
        fnode_birth_k_max=ns.get("FNODE_BIRTH_K_MAX"),
        fnode_birth_species_index=ns.get("FNODE_BIRTH_SPECIES_INDEX"),
        fnode_birth_k_c=ns.get("FNODE_BIRTH_K_C"),
        fnode_birth_hill_conc=ns.get("FNODE_BIRTH_HILL_CONC"),
        fnode_birth_k_sigma=ns.get("FNODE_BIRTH_K_SIGMA"),
        fnode_birth_hill_sigma=ns.get("FNODE_BIRTH_HILL_SIGMA"),
        fnode_birth_radius=ns.get("FNODE_BIRTH_RADIUS"),
        fnode_birth_link_max_distance=ns.get("FNODE_BIRTH_LINK_MAX_DISTANCE"),
        fnode_birth_refractory=ns.get("FNODE_BIRTH_REFRACTORY"),
        include_diffusion=ns.get("INCLUDE_DIFFUSION"),
        heterogeneous_diffusion=ns.get("HETEROGENEOUS_DIFFUSION"),
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
        dead_cells_disappear=ns.get("DEAD_CELLS_DISAPPEAR"),
        include_cell_fnode_repulsion=ns.get("INCLUDE_CELL_FNODE_REPULSION"),
        cell_nucleus_radius=ns.get("CELL_NUCLEUS_RADIUS"),
        brownian_motion_strength=ns.get("BROWNIAN_MOTION_STRENGTH"),
        brownian_motion_strength_factor=ns.get("BROWNIAN_MOTION_STRENGTH_FACTOR"),
        cell_cell_repulsion_k=ns.get("CELL_CELL_REPULSION_K"),
        cell_cell_adhesion_k=ns.get("CELL_CELL_ADHESION_K"),
        cell_cell_adhesion_range=ns.get("CELL_CELL_ADHESION_RANGE"),
        cell_cell_dv_max=ns.get("CELL_CELL_DV_MAX"),
        cell_fnode_repulsion_k=ns.get("CELL_FNODE_REPULSION_K"),
        cell_fnode_exclusion_distance=ns.get("CELL_FNODE_EXCLUSION_DISTANCE"),
        cell_fnode_dv_max=ns.get("CELL_FNODE_DV_MAX"),
        max_expected_n_cells=ns.get("MAX_EXPECTED_N_CELLS"),
        cell_hypoxia_threshold=ns.get("CELL_HYPOXIA_THRESHOLD"),
        cell_nutrient_threshold=ns.get("CELL_NUTRIENT_THRESHOLD"),
        cell_stress_threshold=ns.get("CELL_STRESS_THRESHOLD"),
        cell_hypoxia_damage_rate=ns.get("CELL_HYPOXIA_DAMAGE_RATE"),
        cell_nutrient_damage_rate=ns.get("CELL_NUTRIENT_DAMAGE_RATE"),
        cell_stress_damage_rate=ns.get("CELL_STRESS_DAMAGE_RATE"),
        cell_basal_damage_repair_rate=ns.get("CELL_BASAL_DAMAGE_REPAIR_RATE"),
        cell_acute_hypoxia_threshold=ns.get("CELL_ACUTE_HYPOXIA_THRESHOLD"),
        cell_acute_nutrient_threshold=ns.get("CELL_ACUTE_NUTRIENT_THRESHOLD"),
        cell_acute_stress_threshold=ns.get("CELL_ACUTE_STRESS_THRESHOLD"),
        include_focal_adhesions=ns.get("INCLUDE_FOCAL_ADHESIONS"),
        init_n_focad_per_cell=ns.get("INIT_N_FOCAD_PER_CELL"),
        n_anchor_points=ns.get("N_ANCHOR_POINTS"),
        max_search_radius_focad=ns.get("MAX_SEARCH_RADIUS_FOCAD"),
        max_focad_arm_length=ns.get("MAX_FOCAD_ARM_LENGTH"),
        focad_rest_length_0=ns.get("FOCAD_REST_LENGTH_0"),
        focad_min_rest_length=ns.get("FOCAD_MIN_REST_LENGTH"),
        focad_k_fa=ns.get("FOCAD_K_FA"),
        focad_f_max=ns.get("FOCAD_F_MAX"),
        focad_v_c=ns.get("FOCAD_V_C"),
        focad_k_on=ns.get("FOCAD_K_ON"),
        focad_k_off_0=ns.get("FOCAD_K_OFF_0"),
        focad_f_c=ns.get("FOCAD_F_C"),
        use_catch_bond=ns.get("USE_CATCH_BOND"),
        catch_bond_catch_scale=ns.get("CATCH_BOND_CATCH_SCALE"),
        catch_bond_slip_scale=ns.get("CATCH_BOND_SLIP_SCALE"),
        catch_bond_f_catch=ns.get("CATCH_BOND_F_CATCH"),
        catch_bond_f_slip=ns.get("CATCH_BOND_F_SLIP"),
        focad_k_reinf=ns.get("FOCAD_K_REINF"),
        focad_f_reinf=ns.get("FOCAD_F_REINF"),
        focad_k_fa_max=ns.get("FOCAD_K_FA_MAX"),
        focad_k_fa_decay=ns.get("FOCAD_K_FA_DECAY"),
        focad_polarity_kon_front_gain=ns.get("FOCAD_POLARITY_KON_FRONT_GAIN"),
        focad_polarity_koff_front_reduction=ns.get("FOCAD_POLARITY_KOFF_FRONT_REDUCTION"),
        focad_polarity_koff_rear_gain=ns.get("FOCAD_POLARITY_KOFF_REAR_GAIN"),
        focad_f_mature=ns.get("FOCAD_F_MATURE"),
        focad_t_nascent_max=ns.get("FOCAD_T_NASCENT_MAX"),
        focad_t_detached_grace=ns.get("FOCAD_T_DETACHED_GRACE"),
        focad_t_disassembly=ns.get("FOCAD_T_DISASSEMBLY"),
        enable_focad_birth=ns.get("ENABLE_FOCAD_BIRTH"),
        focad_birth_species_index=ns.get("FOCAD_BIRTH_SPECIES_INDEX"),
        focad_birth_n_min=ns.get("FOCAD_BIRTH_N_MIN"),
        focad_birth_n_max=ns.get("FOCAD_BIRTH_N_MAX"),
        focad_birth_k_0=ns.get("FOCAD_BIRTH_K_0"),
        focad_birth_k_max=ns.get("FOCAD_BIRTH_K_MAX"),
        focad_birth_k_sigma=ns.get("FOCAD_BIRTH_K_SIGMA"),
        focad_birth_hill_sigma=ns.get("FOCAD_BIRTH_HILL_SIGMA"),
        focad_birth_k_c=ns.get("FOCAD_BIRTH_K_C"),
        focad_birth_hill_conc=ns.get("FOCAD_BIRTH_HILL_CONC"),
        focad_birth_refractory=ns.get("FOCAD_BIRTH_REFRACTORY"),
        focad_mobility_mu=ns.get("FOCAD_MOBILITY_MU"),
        include_linc_coupling=ns.get("INCLUDE_LINC_COUPLING"),
        linc_k_elast=ns.get("LINC_K_ELAST"),
        linc_d_dumping=ns.get("LINC_D_DUMPING"),
        linc_rest_length=ns.get("LINC_REST_LENGTH"),
        nucleus_e=ns.get("NUCLEUS_E"),
        nucleus_nu=ns.get("NUCLEUS_NU"),
        nucleus_tau=ns.get("NUCLEUS_TAU"),
        nucleus_eps_clamp=ns.get("NUCLEUS_EPS_CLAMP"),
        include_chemotaxis=ns.get("INCLUDE_CHEMOTAXIS"),
        chemotaxis_sensitivity=ns.get("CHEMOTAXIS_SENSITIVITY"),
        chemotaxis_only_dir=ns.get("CHEMOTAXIS_ONLY_DIR"),
        chemotaxis_chi=ns.get("CHEMOTAXIS_CHI"),
        include_durotaxis=ns.get("INCLUDE_DUROTAXIS"),
        durotaxis_only_dir=ns.get("DUROTAXIS_ONLY_DIR"),
        include_orientation_align=ns.get("INCLUDE_ORIENTATION_ALIGN"),
        orientation_align_rate=ns.get("ORIENTATION_ALIGN_RATE"),
        orientation_align_use_stress=ns.get("ORIENTATION_ALIGN_USE_STRESS"),
        durotaxis_blend_beta=ns.get("DUROTAXIS_BLEND_BETA"),
        durotaxis_use_stress=ns.get("DUROTAXIS_USE_STRESS"),
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


def recompute_derived_params(ns: dict) -> None:
    """Re-compute derived parameters from base values after applying overrides.

    Call this after injecting parameter overrides into the model namespace so
    that values which depend on other parameters are kept consistent.  Only
    the *simple* algebraic relations are handled here; domain-level
    re-gridding (ECM_AGENTS_PER_DIR, ECM_POPULATION_SIZE, etc.) is
    intentionally left out because changing the grid seed (N) or
    BOUNDARY_COORDS has far-reaching consequences that should be set
    explicitly.

    Handles both scalar and list (per-cell-type) parameters transparently.
    """
    import math

    # Helper: element-wise list operations
    def _is_list(v):
        return isinstance(v, (list, tuple))

    def _map1(fn, a):
        """Apply fn element-wise; works for both scalar and list."""
        return [fn(x) for x in a] if _is_list(a) else fn(a)

    def _map2(fn, a, b):
        """Apply fn(a_i, b_i) element-wise; works for both scalar and list."""
        if _is_list(a) and _is_list(b):
            return [fn(ai, bi) for ai, bi in zip(a, b)]
        elif _is_list(a):
            return [fn(ai, b) for ai in a]
        elif _is_list(b):
            return [fn(a, bi) for bi in b]
        else:
            return fn(a, b)

    # --- Boundary stiffness / damping arrays ---
    if "RELATIVE_BOUNDARY_STIFFNESS" in ns and "BOUNDARY_STIFFNESS_VALUE" in ns:
        ns["BOUNDARY_STIFFNESS"] = [ns["BOUNDARY_STIFFNESS_VALUE"] * r for r in ns["RELATIVE_BOUNDARY_STIFFNESS"]]
    if "RELATIVE_BOUNDARY_STIFFNESS" in ns and "BOUNDARY_DUMPING_VALUE" in ns:
        ns["BOUNDARY_DUMPING"] = [ns["BOUNDARY_DUMPING_VALUE"] * r for r in ns["RELATIVE_BOUNDARY_STIFFNESS"]]
    ns["MOVING_BOUNDARIES"] = (
        any(r != 0.0 for r in ns.get("BOUNDARY_DISP_RATES_PARALLEL", [0]))
        or any(r != 0.0 for r in ns.get("BOUNDARY_DISP_RATES", [0]))
    )

    # --- Oscillatory ---
    if ns.get("OSCILLATORY_SHEAR_ASSAY") and "MAX_STRAIN" in ns and "BOUNDARY_COORDS" in ns:
        bc = ns["BOUNDARY_COORDS"]
        ns["OSCILLATORY_AMPLITUDE"] = ns["MAX_STRAIN"] * (bc[2] - bc[3])
        ns["OSCILLATORY_W"] = 2 * math.pi * ns.get("OSCILLATORY_FREQ", 0.05) * ns.get("TIME_STEP", 0.1)

    # --- Fibre network derived ---
    if "FIBRE_SEGMENT_K_ELAST" in ns:
        ns["FIBRE_NODE_REPULSION_K"] = 0.2 * ns["FIBRE_SEGMENT_K_ELAST"]
    if "FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE" in ns:
        eq = ns["FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE"]
        ns["FNODE_CELL_DEGRADATION_RADIUS"] = 0.75 * eq
        _nct = ns["N_CELL_TYPES"]
        ns["FNODE_BIRTH_RADIUS"] = [0.5 * eq] * _nct
        ns["FNODE_BIRTH_LINK_MAX_DISTANCE"] = [2.0 * eq] * _nct
        ns["MAX_SEARCH_RADIUS_FNODES"] = eq / 10.0

    # --- Cell derived (per-type lists or scalars) ---
    if "CELL_RADIUS" in ns:
        cr = ns["CELL_RADIUS"]
        ns["CELL_NUCLEUS_RADIUS"] = _map1(lambda r: r / 2, cr)
        ns["CELL_FNODE_EXCLUSION_DISTANCE"] = list(cr) if _is_list(cr) else cr
        ns["CELL_CELL_ADHESION_RANGE"] = _map1(lambda r: 0.5 * r, cr)
        _max_cr = max(cr) if _is_list(cr) else cr
        ns["MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION"] = 3.0 * _max_cr
    if "CELL_SPEED_REF" in ns:
        cs = ns["CELL_SPEED_REF"]
        # Use BROWNIAN_MOTION_STRENGTH_FACTOR if present, else default to 2.0
        if "BROWNIAN_MOTION_STRENGTH_FACTOR" in ns:
            f = ns["BROWNIAN_MOTION_STRENGTH_FACTOR"]
            def _mult(a, b):
                return a * b 
            ns["BROWNIAN_MOTION_STRENGTH"] = _map2(_mult, cs, f)
        else:
            print("Warning: BROWNIAN_MOTION_STRENGTH_FACTOR not found; defaulting to 2.0 for Brownian motion strength calculation.")
            ns["BROWNIAN_MOTION_STRENGTH"] = _map1(lambda s: s * 2.0, cs)
        ns["CELL_CELL_DV_MAX"] = _map1(lambda s: 0.5 * s, cs)
        ns["CELL_FNODE_DV_MAX"] = _map1(lambda s: 0.5 * s, cs)
    if "CELL_K_ELAST" in ns:
        ck = ns["CELL_K_ELAST"]
        ns["CELL_CELL_REPULSION_K"] = _map1(lambda k: 2.0 * k, ck)
        ns["CELL_CELL_ADHESION_K"] = _map1(lambda k: 0.2 * k, ck)
        ns["CELL_FNODE_REPULSION_K"] = _map1(lambda k: 0.5 * k, ck)

    # --- Cycle phase starts (per-type lists or scalars) ---
    g1 = ns.get("CYCLE_PHASE_G1_DURATION")
    s = ns.get("CYCLE_PHASE_S_DURATION")
    g2 = ns.get("CYCLE_PHASE_G2_DURATION")
    m = ns.get("CYCLE_PHASE_M_DURATION")
    if all(v is not None for v in (g1, s, g2, m)):
        if _is_list(g1):
            ns["CYCLE_PHASE_G1_START"] = [0.0] * len(g1)
            ns["CYCLE_PHASE_S_START"] = list(g1)
            ns["CYCLE_PHASE_G2_START"] = [a + b for a, b in zip(g1, s)]
            ns["CYCLE_PHASE_M_START"] = [a + b + c for a, b, c in zip(g1, s, g2)]
            ns["CELL_CYCLE_DURATION"] = [a + b + c + d for a, b, c, d in zip(g1, s, g2, m)]
        else:
            ns["CYCLE_PHASE_G1_START"] = 0.0
            ns["CYCLE_PHASE_S_START"] = g1
            ns["CYCLE_PHASE_G2_START"] = g1 + s
            ns["CYCLE_PHASE_M_START"] = g1 + s + g2
            ns["CELL_CYCLE_DURATION"] = g1 + s + g2 + m

    # --- Cell concentration mass (reference copy; actual per-agent mass uses per-type volume at init) ---
    if "INIT_CELL_CONCENTRATION_VALS" in ns:
        ns["INIT_CELL_CONC_MASS_VALS"] = list(ns["INIT_CELL_CONCENTRATION_VALS"])

    # --- FOCAD derived (scalar env property; per-agent values use actual cell-type radii at init) ---
    if "CELL_RADIUS" in ns:
        cr = ns["CELL_RADIUS"]
        if _is_list(cr):
            # Reference rest length = shortest across types (conservative for min-rest-length floor)
            rl_vals = [r - r / 2 for r in cr]
            ns["FOCAD_REST_LENGTH_0"] = min(rl_vals)
        else:
            ns["FOCAD_REST_LENGTH_0"] = cr - cr / 2
        ns["FOCAD_MIN_REST_LENGTH"] = ns["FOCAD_REST_LENGTH_0"] / 10.0
    if "INIT_N_FOCAD_PER_CELL" in ns:
        _nct = ns["N_CELL_TYPES"]
        ns["FOCAD_BIRTH_N_MAX"] = [float(3 * ns["INIT_N_FOCAD_PER_CELL"])] * _nct

    # --- MAX_FOCAD_ARM_LENGTH uses max CELL_RADIUS ---
    if "CELL_RADIUS" in ns:
        cr = ns["CELL_RADIUS"]
        _max_cr = max(cr) if _is_list(cr) else cr
        ns["MAX_FOCAD_ARM_LENGTH"] = 3 * _max_cr

    # --- Max expected N cells ---
    steps = ns.get("STEPS", 0)
    dt = ns.get("TIME_STEP", 0.1)
    n_cells = ns.get("N_CELLS", 0)
    cycle_dur = ns.get("CELL_CYCLE_DURATION", 0)
    if _is_list(cycle_dur):
        _min_cycle = min(cycle_dur) if cycle_dur else 0
    else:
        _min_cycle = cycle_dur
    if ns.get("INCLUDE_CELLS") and ns.get("INCLUDE_CELL_CYCLE") and _min_cycle > 0:
        doublings = (steps * dt) / _min_cycle
        ns["MAX_EXPECTED_N_CELLS"] = max(n_cells, int(math.ceil(n_cells * (2.0 ** doublings) * 2.0)))
    else:
        ns["MAX_EXPECTED_N_CELLS"] = n_cells + 1


def apply_param_overrides(ns: dict, overrides: dict) -> None:
    """Apply parameter overrides to a namespace dict and recompute derived values.

    Parameters
    ----------
    ns : dict
        Typically ``globals()`` from model.py.
    overrides : dict
        ``{PARAM_NAME: value}`` pairs to override.

        *   Scalar override:  ``{"CELL_D_DUMPING": 0.5}``
        *   Full list override: ``{"CELL_RADIUS": [8.0, 9.0, 7.5]}``
        *   Element override:   ``{"CELL_RADIUS[1]": 9.0}``
    """
    import re
    _idx_re = re.compile(r'^(\w+)\[(\d+)\]$')

    for key, val in overrides.items():
        if key.startswith("_"):
            continue
        m = _idx_re.match(key)
        if m:
            # Element-wise override: "PARAM[i]" -> ns["PARAM"][i] = val
            base_name = m.group(1)
            idx = int(m.group(2))
            if base_name in ns and isinstance(ns[base_name], (list, tuple)):
                lst = list(ns[base_name])
                if 0 <= idx < len(lst):
                    lst[idx] = type(lst[idx])(val)
                    ns[base_name] = lst
                else:
                    print(f"WARNING: index {idx} out of range for '{base_name}' (len={len(lst)}), ignoring")
            else:
                print(f"WARNING: override key '{key}' — base param '{base_name}' not found or not a list, ignoring")
        elif key in ns:
            existing = ns[key]
            # Scalar-to-list broadcasting: if the model param is a list but the
            # override is a scalar, broadcast to match the list length.
            if isinstance(existing, (list, tuple)) and not isinstance(val, (list, tuple)):
                ns[key] = [type(existing[0])(val)] * len(existing)
            else:
                ns[key] = val
        else:
            print(f"WARNING: override key '{key}' not found in model parameters, ignoring")
    recompute_derived_params(ns)


def load_param_overrides_from_cli(argv=None):
    """Parse --overrides and --result-dir from command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list to parse (pass a saved copy of sys.argv captured
        *before* pyflamegpu imports, since FLAMEGPU may strip or modify
        sys.argv).  Falls back to ``sys.argv`` if *None*.

    Returns
    -------
    overrides : dict
        Parameter overrides loaded from the JSON file (empty if none).
    result_dir : str or None
        Override for the result output directory.
    """
    import json
    if argv is None:
        import sys
        argv = sys.argv

    def _get_flag(flag):
        """Return the value after *flag* in *argv*, or None."""
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                return argv[i + 1]
        return None

    overrides_path = _get_flag("--overrides")
    result_dir = _get_flag("--result-dir")

    overrides = {}
    if overrides_path and os.path.isfile(overrides_path):
        with open(overrides_path, "r") as f:
            overrides = json.load(f)
    return overrides, result_dir
