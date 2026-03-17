import numpy as np
import pickle
file_name = 'single_fibre'
# Generate a single fibre network within a cubical domain 
Lx = 1000.0
Ly = 1000.0
Lz = 1000.0


nodes = np.array([
    [0.0, Ly/2, 0.0],
    [0.0, -Ly/2, 0.0]
])

# Create random connectivity, ensuring each node connects to at least one other node
connectivity = {}
max_connectivity = 8  # Maximum number of connections for each node

for i in range(len(nodes)):
    conn = [-1] * max_connectivity
    connectivity[i] = conn

connectivity[0][0] = 1  # Connect node 0 to node 1
connectivity[1][0] = 0  # Connect node 1 to node 0
# Save to pickle file
with open(file_name + '.pkl', 'wb') as f:
    pickle.dump(
                {
                    'node_coords': nodes,
                    'connectivity': connectivity,
                    'network_parameters': {
                        'LX': Lx,
                        'LY': Ly,
                        'LZ': Lz,
                        'N_FIBER': 1,
                        'L_FIBER': 1000.0,
                        'RHO': 2/10**9,
                        'EDGE_LENGTH': 1000.0,
                    },
                },
                f,)
from helper_functions_network_gen import generate_random_vars, save_network_to_vtk  
scalar_vars, vector_vars = generate_random_vars(nodes)
save_network_to_vtk(file_name + '.vtk', nodes, connectivity, scalar_vars=scalar_vars, vector_vars=vector_vars)

print("Network saved to '" + file_name + ".pkl'")
