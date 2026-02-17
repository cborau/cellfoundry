import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def median_distance(nodes, connectivity, plot_histogram=False):
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

# Example usage
nodes = np.array([
    [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.3, 0.3, 0.3]
])
connectivity = {
    0: [1, 2, 3, 4, -1, -1, -1, -1],
    1: [0, 2, 3, 4, -1, -1, -1, -1],
    2: [0, 1, 3, 4, -1, -1, -1, -1],
    3: [0, 1, 2, 4, -1, -1, -1, -1],
    4: [0, 1, 2, 3, -1, -1, -1, -1]
}

median_dist = median_distance(nodes, connectivity, plot_histogram=True)
print(f"Median Distance: {median_dist}")
