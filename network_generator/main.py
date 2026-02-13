import os
import numpy as np
from initial_network_generation import initial_network_generation
from network_optimization import network_optimization
from branch_optimization import branch_optimization
from helper_functions_network_gen import plot_network_3d, add_intermediate_nodes, compute_node_connectivity, scale_to_unit_cube, snap_to_boundaries, remove_boundary_connectivity, add_intermediate_nodes_to_plot, get_valency_and_pore_size, save_network_to_vtk, generate_random_vars, get_node_median_distance
import matplotlib.pyplot as plt
import pickle

def generate_network(lx, ly, lz, l_fiber,rho, enforce_bounds=False, bound_mode="reject"):

    # Initial Network Generation
    nodes, fibers, fiberenergy, total_energy, N1, N2, N_boundary_nodes, fiberlengths, valencycheck = initial_network_generation(rho, lx, ly, lz, l_fiber, 100)

    N = len(nodes)
    N_fibers = len(fibers)
    print(f'There are {N} nodes and {N_fibers} fibers')
    print(f'The total Initial Fiber Length Energy is {total_energy}')

    # Fiber Length Optimization
    # N_anneal: number of simulated-annealing sweeps per round. Simulated annealing
    #           is a stochastic search that accepts some worse moves early to escape
    #           local minima, then becomes more selective. More sweeps = more trials.
    # N_optimize: number of outer rounds with progressively smaller stepsizes and
    #            swap fractions (cooling schedule), refining the network.
    N_anneal = 15
    N_optimize = 5
    stepsize = np.linspace(1, 0.1, N_optimize)
    fraction_to_try_swap = np.linspace(0.3, 0.02, N_optimize)

    bounds = [(-lx / 2, lx / 2), (-ly / 2, ly / 2), (-lz / 2, lz / 2)]

    for j in range(N_optimize):
        swap_skip_energy = 3 * (np.median(fiberlengths) - l_fiber) ** 2
        print(f'Swap skip energy threshold = {swap_skip_energy}')
        nodes, fibers, fiberenergy, fiberlengths, total_energy, N1, N2, N_interior = network_optimization(
            fraction_to_try_swap[j], N, nodes, fibers, N_anneal, lx, ly, lz, l_fiber,
            fiberlengths, fiberenergy, N1, N2, N_boundary_nodes, stepsize[j], swap_skip_energy,
            enforce_bounds=enforce_bounds, bounds=bounds, bound_mode=bound_mode
        )
        percent_done = 100 * (j + 1) / N_optimize
        print(f'Percent optimized = {percent_done}')
        print('_________________________________________')


    plt.figure()
    plt.hist(fiberlengths, bins=50)
    plt.title('Fiber Length Distribution after Optimization')
    plt.show()

    # Branching Optimization
    N_branching_optimize = 4
    nodes, fibers, nodal_branching_energy, total_branching_energy_init, total_branching_energy_final = branch_optimization(
        N_branching_optimize, nodes, fibers, N, enforce_bounds=enforce_bounds, bounds=bounds, bound_mode=bound_mode
    )

    return nodes, fibers, bounds
  

if __name__ == "__main__":

    MAX_CONNECTIVITY = 8
    # Units: choose any consistent spatial unit (e.g., microns).
    # All lengths below (LX, LY, LZ, L_FIBER, EDGE_LENGTH, SNAP_DISTANCE) use that unit.
    LX = 200
    LY = 200
    LZ = 200
    L_FIBER = 45.0
    RHO = 0.0001 # number of nodes per unit volume

    # EDGE_LENGTH controls the target segment length when splitting long fibers.
    # Units must match l_fiber and lx/ly/lz.
    EDGE_LENGTH = 15.0
    file_name = 'network_3d.pkl'
    file_path = os.path.abspath(file_name)

    # ENFORCE_BOUNDS keeps optimization moves inside the initial box.
    # BOUND_MODE="reject" skips out-of-bounds moves; "clip" clamps them to the box.
    ENFORCE_BOUNDS = True
    BOUND_MODE = "reject"  # "reject" or "clip"

    # SNAP_* controls optional snapping of nodes to selected boundaries.
    # SNAP_MODE: "percentage" snaps a fraction of candidates near each boundary.
    #            "distance" snaps all candidates within SNAP_DISTANCE.
    # SNAP_DISTANCE is in the same units as lx/ly/lz.
    SNAP_BOUNDARIES = ['+x', '-x', '+y', '-y', '+z', '-z']
    SNAP_MODE = "distance"  # "percentage" or "distance"
    SNAP_PERCENTAGE = 10
    SNAP_DISTANCE = 5.0

    if os.path.exists(file_name):
        print(f'Loading network from {file_path}')
        # Load from the pickle file
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            nodes = data['node_coords']
            connectivity = data['connectivity']
    else: 
        nodes, fibers, bounds = generate_network(LX, LY, LZ, L_FIBER, RHO, enforce_bounds=ENFORCE_BOUNDS, bound_mode=BOUND_MODE)
        #nodes = scale_to_unit_cube(nodes)
        nodes = snap_to_boundaries(
            nodes,
            percentage=SNAP_PERCENTAGE,
            boundaries=SNAP_BOUNDARIES,
            bounds=bounds,
            mode=SNAP_MODE,
            distance=SNAP_DISTANCE
        )
        
        #
        # Plot the network
        
        num_nodes = nodes.shape[0]
        node_connectivity = compute_node_connectivity(fibers, num_nodes, MAX_CONNECTIVITY)
        #plot_network_3d(nodes, node_connectivity, title ='before fix')

        nodes, node_connectivity = remove_boundary_connectivity(nodes, node_connectivity, bounds=bounds)
        # fig, ax = plot_network_3d(nodes, node_connectivity, title ='after fix')

        new_nodes, new_connectivity = add_intermediate_nodes(nodes, node_connectivity, EDGE_LENGTH, MAX_CONNECTIVITY)
        # add_intermediate_nodes_to_plot(ax, new_nodes)

        #plt.show()
        # Save to a pickle file
        print(f'Saving network to {file_path}')
        with open('network_3d.pkl', 'wb') as f:
            pickle.dump(
                {
                    'node_coords': new_nodes,
                    'connectivity': new_connectivity,
                    'network_parameters': {
                        'LX': LX,
                        'LY': LY,
                        'LZ': LZ,
                        'N_FIBER': len(fibers),
                        'L_FIBER': L_FIBER,
                        'RHO': RHO,
                        'EDGE_LENGTH': EDGE_LENGTH,
                    },
                },
                f,
            )
        connectivity = new_connectivity.copy()
        nodes = new_nodes.copy()

    
    # get_valency_and_pore_size(nodes, connectivity, MAX_CONNECTIVITY)
    scalar_vars, vector_vars = generate_random_vars(nodes)
    save_network_to_vtk('network_3d.vtk', nodes, connectivity, scalar_vars=scalar_vars, vector_vars=vector_vars)
    # median_edge_length = get_node_median_distance(nodes, connectivity, plot_histogram=True)
    # print(f'Median edge length: {median_edge_length}')
    # plot_network_3d(nodes, connectivity, title ='before fix')
    # plt.show()
