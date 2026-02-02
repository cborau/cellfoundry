import numpy as np

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
    
    
def getRandomCoordsAroundPoint(n, px, py, pz, radius):
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
        radius_i = np.random.uniform(0.0, 1.0) * radius        
        coords[i, :] = central_point + np.array(rand_dirs[i, :] * radius_i, dtype='float')
    

    return coords