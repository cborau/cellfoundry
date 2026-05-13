from pathlib import Path
import pickle
import numpy as np
from scipy.io import loadmat


# Resolution of the FIRE image stack, in microns per pixel/voxel.
# Coordinates in the MATLAB file are stored in pixel units.
RESOLUTION_X = 0.1069258  # microns / pixel
RESOLUTION_Y = 0.1069258  # microns / pixel
RESOLUTION_Z = 0.1997950  # microns / voxel


def convert_fire_mat_to_pickle(
    mat_path,
    output_path,
    center=True,
    edge_length=None,
):
    """
    Convert a FIRE/MATLAB fibre-network .mat file into the pickle format used by
    Cellfoundry.

    The MATLAB file is expected to contain a structure called `data`.

    Read variables from `data`
    --------------------------
    data.Xa or data.Xnodes
        Reconstructed fibre-network node coordinates.

        Shape:
            (N, 3)

        Meaning:
            Each row stores the position of one network node as:

                [x, y, z]

            Coordinates are stored in image pixel/voxel units in the MATLAB file.
            They are converted to microns using:

                x_um = x_px * RESOLUTION_X
                y_um = y_px * RESOLUTION_Y
                z_um = z_px * RESOLUTION_Z

        These coordinates are used as:

            pickle["node_coords"]

    data.C
        Connectivity matrix of the reconstructed network.

        Shape:
            (N, c)

        Meaning:
            Each row corresponds to one node. The entries in that row are the
            indices of neighbouring nodes connected to it.

            MATLAB uses 1-based indexing, so valid node indices in `data.C`
            are converted to 0-based Python indices.

            Empty neighbour slots are expected to be marked with -1 and are kept
            as -1.

        This matrix is converted into the dictionary:

            pickle["connectivity"] = {
                node_index: [neighbor_1, neighbor_2, ..., neighbor_c]
            }

        where all neighbour lists have the same fixed length c.

    data.M
        Network statistics computed by FIRE/MATLAB.

        Relevant fields used here:
            data.M.fiber_num
                Number of fibres in the reconstructed network.

            data.M.avgL
                Average fibre length in the MATLAB/FIRE statistics.

        These values are stored in:

            pickle["network_parameters"]["N_FIBER"]
            pickle["network_parameters"]["L_FIBER"]

        Note:
            If `data.M.avgL` is stored in pixel units, it is not directly
            converted here because fibre length may depend on anisotropic voxel
            spacing. For strict physical lengths, recomputing fibre lengths from
            the scaled coordinates is safer.

    Output pickle structure
    -----------------------
    The saved pickle contains:

        {
            "node_coords": node_coords,
            "connectivity": connectivity,
            "network_parameters": {
                "LX": LX,
                "LY": LY,
                "LZ": LZ,
                "N_FIBER": N_FIBER,
                "L_FIBER": L_FIBER,
                "RHO": RHO,
                "EDGE_LENGTH": EDGE_LENGTH,
            },
        }

    where:

    node_coords
        N x 3 NumPy array with node coordinates in microns.

    connectivity
        Dictionary mapping each node index to a fixed-length list of neighbours.

    LX, LY, LZ
        Physical size of the network bounding box in microns.

    N_FIBER
        Number of fibres reported by FIRE/MATLAB.

    L_FIBER
        Average fibre length reported by FIRE/MATLAB.

    RHO
        Node density, computed as:

            RHO = number of nodes / network volume

        with volume measured in cubic microns.

    EDGE_LENGTH
        Target edge length used by the Python generator when splitting long
        fibres. Since the network is being imported rather than generated, this
        can be set manually or left as None.
    """

    mat_path = Path(mat_path)
    output_path = Path(output_path)

    avg_resolution = (RESOLUTION_X + RESOLUTION_Y + RESOLUTION_Z) / 3.0
    res_values = np.array([RESOLUTION_X, RESOLUTION_Y, RESOLUTION_Z])
    if res_values.max() / res_values.min() > 1.2:
        import warnings
        warnings.warn(
            f"Anisotropic voxel spacing detected: X={RESOLUTION_X}, Y={RESOLUTION_Y}, Z={RESOLUTION_Z} microns/px. "
            f"Average fibre length will be approximated using the mean resolution ({avg_resolution:.4f} microns/px)."
        )

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = mat["data"]

    if hasattr(data, "Xa"):
        nodes_px = np.asarray(data.Xa, dtype=float)
    elif hasattr(data, "Xnodes"):
        nodes_px = np.asarray(data.Xnodes, dtype=float)
    else:
        raise AttributeError("Could not find `data.Xa` or `data.Xnodes` in the .mat file.")

    scale = np.array([RESOLUTION_X, RESOLUTION_Y, RESOLUTION_Z], dtype=float)
    node_coords = nodes_px * scale

    if center:
        node_coords = node_coords - (node_coords.min(axis=0) + node_coords.max(axis=0)) / 2.0

    C = np.asarray(data.C, dtype=int)

    # Convert MATLAB 1-based indices to Python 0-based indices.
    # Empty values marked as -1 remain -1.
    C_python = np.where(C > 0, C - 1, -1)

    connectivity = {
        i: C_python[i].tolist()
        for i in range(C_python.shape[0])
    }

    LX, LY, LZ = node_coords.max(axis=0) - node_coords.min(axis=0)
    volume = LX * LY * LZ

    if volume <= 0:
        raise ValueError("Computed network volume is zero or negative.")

    RHO = len(node_coords) / volume

    if edge_length is None:
        mask = C_python >= 0
        src = np.where(mask)[0]
        dst = C_python[mask]
        edge_length = float(np.mean(np.linalg.norm(node_coords[src] - node_coords[dst], axis=1)))

    network_parameters = {
        "LX": float(LX),
        "LY": float(LY),
        "LZ": float(LZ),
        "N_FIBER": int(data.M.fiber_num) if hasattr(data.M, "fiber_num") else None,
        "L_FIBER": float(data.M.avgL) * edge_length if hasattr(data.M, "avgL") else None, # Check actual units of avgL in the MATLAB data (L_FIBER is not used anyway)
        "RHO": float(RHO),
        "EDGE_LENGTH": edge_length,
    }

    output = {
        "node_coords": node_coords,
        "connectivity": connectivity,
        "network_parameters": network_parameters,
    }

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved converted network to: {output_path.resolve()}")
    print(f"Number of nodes: {len(node_coords)}")
    print(f"Connectivity shape: {C_python.shape}")
    print(f"Domain size: LX={LX:.3f}, LY={LY:.3f}, LZ={LZ:.3f} microns")
    print(f"Node density RHO={RHO:.6e} nodes / micron^3")
    print(f"Average fibre length from FIRE/MATLAB: {data.M.avgL:.3f} segments = {data.M.avgL * edge_length:.3f} microns" if hasattr(data.M, "avgL") else "Average fibre length not found in FIRE/MATLAB data.")
    print(f"Average edge length: {edge_length:.3f} microns")

    return output

if __name__ == "__main__":
    convert_fire_mat_to_pickle(
    mat_path=r"C:\Users\PC\network_6mgml.mat",
    output_path="converted_network.pkl",
    edge_length=None,
)