import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def compute_node_connectivity(fibers, num_nodes, MAX_CONNECTIVITY = 8):
    node_connectivity = {i: [-1] * MAX_CONNECTIVITY for i in range(num_nodes)}
    
    for i in range(fibers.shape[0]):
        node1_idx = fibers[i, 0]
        node2_idx = fibers[i, 1]
        
        # Update connectivity for node1
        for j in range(MAX_CONNECTIVITY):
            if node_connectivity[node1_idx][j] == -1:
                node_connectivity[node1_idx][j] = node2_idx
                break
        
        # Update connectivity for node2
        for j in range(MAX_CONNECTIVITY):
            if node_connectivity[node2_idx][j] == -1:
                node_connectivity[node2_idx][j] = node1_idx
                break
    
    return node_connectivity

def add_intermediate_nodes(nodes, connectivity, edge_length, MAX_CONNECTIVITY):
    new_nodes = nodes.tolist()
    new_connectivity = {i: connectivity[i][:] for i in range(len(nodes))}

    current_node_index = len(nodes)

    for node1 in range(len(nodes)):
        for j in range(MAX_CONNECTIVITY):
            node2 = connectivity[node1][j]
            if node2 == -1:
                continue

            node2 = int(node2)
            dist = np.linalg.norm(nodes[node1] - nodes[node2])

            if dist > edge_length:
                num_new_nodes = int(np.ceil(dist / edge_length)) - 1
                direction = (nodes[node2] - nodes[node1]) / (num_new_nodes + 1)

                previous_node = node1

                for k in range(1, num_new_nodes + 1):
                    new_node = nodes[node1] + k * direction
                    new_nodes.append(new_node)

                    # Find the first available slot in connectivity
                    for idx in range(MAX_CONNECTIVITY):
                        if new_connectivity[previous_node][idx] == -1:
                            new_connectivity[previous_node][idx] = current_node_index
                            break

                    new_connectivity[current_node_index] = [-1] * MAX_CONNECTIVITY
                    new_connectivity[current_node_index][0] = previous_node

                    previous_node = current_node_index
                    current_node_index += 1

                # Connect the last new node to the original second node
                for idx in range(MAX_CONNECTIVITY):
                    if new_connectivity[previous_node][idx] == -1:
                        new_connectivity[previous_node][idx] = node2
                        break

                for idx in range(MAX_CONNECTIVITY):
                    if new_connectivity[node2][idx] == -1:
                        new_connectivity[node2][idx] = previous_node
                        break

                # Remove the old connection
                new_connectivity[node1][j] = -1
                for idx in range(MAX_CONNECTIVITY):
                    if new_connectivity[node2][idx] == node1:
                        new_connectivity[node2][idx] = -1
                        break

    return np.array(new_nodes), new_connectivity


def snap_to_boundaries(nodes, percentage, boundaries, bounds=None, mode="percentage", distance=None):
    """
    Snap nodes to selected domain boundaries using either percentage or distance mode.

        Modes:
                - "distance": equivalent to percentage mode with ``percentage=100``.
                - "percentage": snap ``percentage`` of nodes that are within ``distance``
                    of each chosen boundary (closest first). If ``distance`` is None, it
                    defaults to 10% of the corresponding axis length.

    Parameters:
        nodes (numpy.ndarray): Array of node coordinates (N x 3).
        percentage (float): Percentage of candidate nodes to snap (percentage mode).
        boundaries (list[str]): Boundaries to snap to (e.g. ['+x', '-y']).
        bounds (tuple, optional): Axis-aligned bounds as
            ((xmin, xmax), (ymin, ymax), (zmin, zmax)). If None, uses the
            min/max of the current nodes for each axis.
        mode (str): "percentage" or "distance".
        distance (float, optional): Threshold distance from boundary for candidates.
    """
    if bounds is None:
        min_vals = np.min(nodes, axis=0)
        max_vals = np.max(nodes, axis=0)
        bounds = ((min_vals[0], max_vals[0]), (min_vals[1], max_vals[1]), (min_vals[2], max_vals[2]))

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    if mode not in {"percentage", "distance"}:
        raise ValueError("Invalid mode. Choose from 'percentage' or 'distance'.")

    # Define the boundary mappings
    boundary_mapping = {
        '+x': (0, xmax),
        '-x': (0, xmin),
        '+y': (1, ymax),
        '-y': (1, ymin),
        '+z': (2, zmax),
        '-z': (2, zmin)
    }
    
    # Validate boundaries
    for boundary in boundaries:
        if boundary not in boundary_mapping:
            raise ValueError("Invalid boundary. Choose from '+x', '-x', '+y', '-y', '+z', '-z'.")
    
    # Initialize the snapped nodes array
    snapped_nodes = nodes.copy()
    
    for boundary in boundaries:
        axis, bound_value = boundary_mapping[boundary]
        
        # Calculate distances to the boundary for the specified axis
        distances = np.abs(snapped_nodes[:, axis] - bound_value)

        # Determine candidate indices within distance threshold
        if distance is None:
            axis_length = bounds[axis][1] - bounds[axis][0]
            threshold = 0.1 * axis_length
            print(
                f"Warning: distance is None for {boundary}; using default threshold of 10% "
                f"of axis length ({threshold})."
            )
        else:
            threshold = distance

        if threshold <= 0:
            print(f"Warning: distance <= 0 for {boundary}; no nodes will be snapped.")

        candidate_indices = np.where(distances <= threshold)[0]
        if candidate_indices.size == 0:
            print(f"Warning: no candidates found within distance {threshold} for {boundary}.")
        else:
            effective_percentage = 100 if mode == "distance" else percentage
            num_to_snap = int(np.ceil(candidate_indices.size * effective_percentage / 100))

            if num_to_snap <= 0:
                print(f"Warning: num_to_snap <= 0 for {boundary}; no nodes will be snapped.")
            else:
                sorted_candidate_indices = candidate_indices[np.argsort(distances[candidate_indices])]
                indices_to_snap = sorted_candidate_indices[:num_to_snap]
                snapped_nodes[indices_to_snap, axis] = bound_value
    
    return snapped_nodes

def remove_boundary_connectivity(nodes, connectivity, bounds=None):
    if bounds is None:
        min_vals = np.min(nodes, axis=0)
        max_vals = np.max(nodes, axis=0)
        bounds = ((min_vals[0], max_vals[0]), (min_vals[1], max_vals[1]), (min_vals[2], max_vals[2]))

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    boundaries_per_axis = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    
    # Initialize the updated connectivity dictionary
    updated_connectivity = {}
    num_nodes = len(nodes)
    
    for node_index, connections in connectivity.items():
        # Get the coordinates of the current node
        node_coords = nodes[node_index]
        
        # Initialize the updated connections list for the current node
        updated_connections = []
        no_connection = []
        
        for connected_node_index in connections:
            if connected_node_index == -1:
                no_connection.append(-1)
                continue
            
            # Get the coordinates of the connected node
            connected_node_coords = nodes[connected_node_index]
            
            # Check if both nodes are on the same boundary for any dimension
            same_boundary = False
            for dim in range(nodes.shape[1]):
                on_boundary = np.isclose(node_coords[dim], boundaries_per_axis[dim][0]) or np.isclose(node_coords[dim], boundaries_per_axis[dim][1])
                same_plane = np.isclose(connected_node_coords[dim], node_coords[dim])
                if on_boundary and same_plane:
                    same_boundary = True
                    break
            
            if not same_boundary:
                updated_connections.append(connected_node_index)
            else:
                no_connection.append(-1)
        
        # Combine the valid connections with the no connections (-1)
        updated_connections.extend(no_connection)
        
        # Update the connectivity for the current node
        updated_connectivity[node_index] = updated_connections
    
    # Remove nodes with no connectivity
    nodes_to_remove = [index for index, connections in updated_connectivity.items() if all(conn == -1 for conn in connections)]
    nodes_to_keep = [index for index in range(num_nodes) if index not in nodes_to_remove]
    
    # Create new nodes array and updated connectivity dictionary
    new_nodes = nodes[nodes_to_keep]
    new_connectivity = {}
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(nodes_to_keep)}
    
    for old_index, connections in updated_connectivity.items():
        if old_index not in nodes_to_remove:
            new_connections = [index_mapping[conn] if conn in index_mapping else -1 for conn in connections]
            new_connectivity[index_mapping[old_index]] = new_connections

    num_removed_nodes = len(nodes_to_remove)
    print(f"Number of removed nodes: {num_removed_nodes}")
    
    return new_nodes, new_connectivity

def scale_to_unit_cube(nodes):
    # Find the min and max values for each dimension
    min_vals = np.min(nodes, axis=0)
    max_vals = np.max(nodes, axis=0)
    
    # Calculate the center of the bounding box
    center = (min_vals + max_vals) / 2.0
    
    # Calculate the scaling factor for each dimension
    scales = max_vals - min_vals
    
    # Center the nodes around the origin
    centered_nodes = nodes - center
    
    # Scale the nodes to fit within the range -0.5 to 0.5
    scaled_nodes = centered_nodes / scales
    
    return scaled_nodes

def plot_network_3d(nodes, connectivity, title = '3D Plot of Fibers and Nodes'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the nodes as red markers
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='r', marker='o')

    # Plot the connections as blue lines
    for node_index, connections in connectivity.items():
        for connected_node_index in connections:
            if connected_node_index != -1:
                start_node = nodes[node_index]
                end_node = nodes[connected_node_index]
                ax.plot([start_node[0], end_node[0]],
                        [start_node[1], end_node[1]],
                        [start_node[2], end_node[2]], 'b-')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title(title)

    return fig, ax

def add_intermediate_nodes_to_plot(ax, nodes):
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='g', marker='o')

def save_network_to_vtk(filename, nodes, connectivity, scalar_vars=None, vector_vars=None):
    """
    Save the network to a VTK file as an unstructured grid.
    
    Parameters:
        filename (str): The name of the VTK file to save.
        nodes (numpy.ndarray): Array of node coordinates.
        connectivity (dict): Connectivity dictionary.
        scalar_vars (dict, optional): Dictionary of scalar variables.
        vector_vars (dict, optional): Dictionary of vector variables.
    """
    num_nodes = len(nodes)
    
    # To avoid duplicating cells, use a set to keep track of added lines
    added_lines = set()
    cell_connectivity = []
    
    for node_index, connections in connectivity.items():
        for connected_node_index in connections:
            if connected_node_index != -1:
                # Create a tuple with sorted indices to ensure uniqueness
                line = tuple(sorted((node_index, connected_node_index)))
                if line not in added_lines:
                    added_lines.add(line)
                    cell_connectivity.append(line)
    
    num_cells = len(cell_connectivity)
    
    with open(filename, 'w') as f:
        # Write the VTK file header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Network data\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Write the node coordinates
        f.write(f"POINTS {num_nodes} float\n")
        for node in nodes:
            f.write(f"{node[0]} {node[1]} {node[2]}\n")
        
        # Write the cell connectivity
        f.write(f"CELLS {num_cells} {num_cells * 3}\n")
        for conn in cell_connectivity:
            f.write(f"2 {conn[0]} {conn[1]}\n")
        
        # Write the cell types (3 for VTK_LINE)
        f.write(f"CELL_TYPES {num_cells}\n")
        for _ in range(num_cells):
            f.write("3\n")  # VTK_LINE
        
        # Write the scalar and vector variables if any
        if scalar_vars or vector_vars:
            f.write(f"POINT_DATA {num_nodes}\n")
        
        # Write the scalar variables
        if scalar_vars:
            for var_name, var_values in scalar_vars.items():
                f.write(f"SCALARS {var_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for value in var_values:
                    f.write(f"{value}\n")
        
        # Write the vector variables
        if vector_vars:
            for var_name, var_values in vector_vars.items():
                f.write(f"VECTORS {var_name} float\n")
                for value in var_values:
                    f.write(f"{value[0]} {value[1]} {value[2]}\n")

def get_valency_and_pore_size(nodes, connectivity, MAX_CONNECTIVITY = 8):
        # Calculate node valency
    node_valency = [sum(1 for conn in value if conn != -1) for value in connectivity.values()]

    # Create figure for plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the valency histogram
    axs[0].hist(node_valency, bins=range(1, MAX_CONNECTIVITY + 2), align='left', edgecolor='black')
    axs[0].set_xlabel('Node Valency')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of Node Valency')
    axs[0].set_xticks(range(1, MAX_CONNECTIVITY + 1))
    axs[0].grid(True)

    # Pore size calculation
    min_coords = np.min(nodes, axis=0)
    max_coords = np.max(nodes, axis=0)
    num_random_points = 10000
    random_points = min_coords + np.random.rand(num_random_points, 3) * (max_coords - min_coords)

    pore_sizes = np.zeros(num_random_points)
    nearest_nodes = np.zeros((num_random_points, 3))

    for i in range(num_random_points):
        random_point = random_points[i, :]
        distances = distance.cdist([random_point], nodes, 'euclidean')[0]
        min_distance = np.min(distances)
        nearest_node_index = np.argmin(distances)
        if np.all(random_point - min_distance >= min_coords) and np.all(random_point + min_distance <= max_coords):
            pore_sizes[i] = min_distance
            nearest_nodes[i, :] = nodes[nearest_node_index, :]
        else:
            pore_sizes[i] = 0

    valid_indices = pore_sizes > 0
    pore_sizes = pore_sizes[valid_indices]
    valid_random_points = random_points[valid_indices, :]
    nearest_nodes = nearest_nodes[valid_indices, :]

    print('Pore Sizes:')
    print(pore_sizes)
    average_pore_size = np.mean(pore_sizes)
    print(f'Average Pore Size: {average_pore_size:.3f}')

    # Plot the pore size histogram
    axs[1].hist(pore_sizes, bins=100, edgecolor='black')
    axs[1].set_xlabel('Pore Size')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of Pore Sizes')
    axs[1].grid(True)

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()
    # # Uncomment the following block to plot the nodes and valid random points with spheres
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='r', marker='o', s=5)
    # for i in range(len(pore_sizes)):
    #     u = np.linspace(0, 2 * np.pi, 100)
    #     v = np.linspace(0, np.pi, 100)
    #     x = pore_sizes[i] * np.outer(np.cos(u), np.sin(v)) + valid_random_points[i, 0]
    #     y = pore_sizes[i] * np.outer(np.sin(u), np.sin(v)) + valid_random_points[i, 1]
    #     z = pore_sizes[i] * np.outer(np.ones(np.size(u)), np.cos(v)) + valid_random_points[i, 2]
    #     ax.plot_surface(x, y, z, color='b', alpha=0.3)
    #     ax.scatter(nearest_nodes[i, 0], nearest_nodes[i, 1], nearest_nodes[i, 2], c='k', marker='x', s=50)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Plot of Pore Sizes with Debug Markers')
    # plt.grid(True)
    # plt.show()

def get_node_median_distance(nodes, connectivity, plot_histogram=False):
    distances = []

    # Calculate distances between connected nodes
    for node_index, connections in connectivity.items():
        for connected_node_index in connections:
            if connected_node_index != -1:
                dist = np.linalg.norm(nodes[node_index] - nodes[connected_node_index])
                distances.append(dist)
    
    distances = np.array(distances)

    # Plot histogram if requested
    if plot_histogram:
        plt.hist(distances, bins=30, edgecolor='black')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Histogram of Distances Between Connected Nodes')
        plt.grid(True)
        plt.show()
    
    # Calculate and return the median distance
    median_dist = np.median(distances)
    return median_dist

def generate_random_vars(nodes):
    num_nodes = len(nodes)
    
    # Generate random scalar variables
    scalar_vars = {
        "temperature": np.random.uniform(low=250, high=350, size=num_nodes),
        "pressure": np.random.uniform(low=1, high=10, size=num_nodes)
    }
    
    # Generate random vector variables
    vector_vars = {
        "velocity": np.random.uniform(low=-1, high=1, size=(num_nodes, 3)),
        "force": np.random.uniform(low=-10, high=10, size=(num_nodes, 3))
    }
    
    return scalar_vars, vector_vars

