import os
import numpy as np
from initial_network_generation import initial_network_generation
from network_optimization import network_optimization
from branch_optimization import branch_optimization
from helper_functions_network_gen import plot_network_3d, add_intermediate_nodes, compute_node_connectivity, scale_to_unit_cube, snap_to_boundaries, remove_boundary_connectivity, add_intermediate_nodes_to_plot, get_valency_and_pore_size, save_network_to_vtk, generate_random_vars, get_node_median_distance, merge_duplicate_nodes, check_duplicates
import matplotlib.pyplot as plt
import pickle

def generate_network(
    lx,
    ly,
    lz,
    l_fiber,
    rho,
    enforce_bounds=False,
    bound_mode="reject",
    branching_stepsize_mode="local_mean_fiber",
    branching_stepsize_mag=None,
    branching_stepsize_frac_min=0.007,
    branching_stepsize_frac_max=0.02,
    min_swap_sweeps_per_opt_round=1,
    n_anneal=15,
    n_optimize=5,
    n_branching_optimize=6,
):

    # Initial Network Generation
    # nodes: (N, 3) float array, fibers: (M, 2) int array of node indices.
    # fiberenergy/fiberlengths: (M,) float arrays, N1/N2: (M, 3) endpoint coords.
    nodes, fibers, fiberenergy, total_energy, N1, N2, N_boundary_nodes, fiberlengths, valencycheck = initial_network_generation(rho, lx, ly, lz, l_fiber, 100)

    check_duplicates(nodes, fibers, "after_initial_generation", edge_kind="fibers")

    N = len(nodes)
    N_fibers = len(fibers)
    print(f'There are {N} nodes and {N_fibers} fibers')
    print(f'The total Initial Fiber Length Energy is {total_energy}')

    stepsize = np.linspace(1, 0.1, n_optimize)
    fraction_to_try_swap = np.linspace(0.3, 0.08, n_optimize)
    length_energy_per_round = np.zeros(n_optimize)
    acceptance_pct_per_round = np.zeros(n_optimize)
    swap_sweeps_per_round = np.zeros(n_optimize, dtype=int)

    bounds = [(-lx / 2, lx / 2), (-ly / 2, ly / 2), (-lz / 2, lz / 2)]

    for j in range(n_optimize):
        swap_skip_energy = 3 * (np.median(fiberlengths) - l_fiber) ** 2
        print(f'Swap skip energy threshold = {swap_skip_energy}')
        nodes, fibers, fiberenergy, fiberlengths, total_energy, N1, N2, N_interior, opt_stats = network_optimization(
            fraction_to_try_swap[j], N, nodes, fibers, n_anneal, lx, ly, lz, l_fiber,
            fiberlengths, fiberenergy, N1, N2, N_boundary_nodes, stepsize[j], swap_skip_energy,
            enforce_bounds=enforce_bounds, bounds=bounds, bound_mode=bound_mode,
            min_swap_sweeps=min_swap_sweeps_per_opt_round,
            return_stats=True,
        )
        length_energy_per_round[j] = total_energy
        acceptance_pct_per_round[j] = opt_stats["percent_accepted_iterations"]
        swap_sweeps_per_round[j] = opt_stats["swap_sweeps_executed"]
        percent_done = 100 * (j + 1) / n_optimize
        print(f'Percent optimized = {percent_done}')
        print('_________________________________________')

    initial_energy = length_energy_per_round[0]
    final_energy = length_energy_per_round[-1]
    rel_drop = (initial_energy - final_energy) / max(abs(initial_energy), 1e-12)
    print('\n=== Length Optimization Convergence Summary ===')
    print(f'Energy: {initial_energy:.6g} -> {final_energy:.6g} (relative drop {100 * rel_drop:.2f}%)')
    for j in range(n_optimize):
        stage_drop = 0.0 if j == 0 else (length_energy_per_round[j - 1] - length_energy_per_round[j]) / max(abs(length_energy_per_round[j - 1]), 1e-12)
        print(
            f'  Round {j + 1}/{n_optimize}: energy={length_energy_per_round[j]:.6g}, '
            f'accept={acceptance_pct_per_round[j]:.2f}%, '
            f'swap_sweeps={swap_sweeps_per_round[j]}/{n_anneal}, '
            f'stage_drop={100 * stage_drop:.2f}%'
        )

    if n_optimize >= 2:
        last_stage_drop = (length_energy_per_round[-2] - length_energy_per_round[-1]) / max(abs(length_energy_per_round[-2]), 1e-12)
        last_accept = acceptance_pct_per_round[-1]
        print('Recommendations (length optimization):')
        if last_stage_drop < 0.005 and last_accept < 5:
            print('  - Likely converged: keep current N_anneal/N_optimize, or reduce runtime settings.')
        elif last_stage_drop < 0.005 and last_accept >= 5:
            print('  - Geometry moves are still accepted but not improving much: reduce final stepsize floor or raise swap activity slightly.')
        elif last_stage_drop >= 0.005 and last_accept < 3:
            print('  - Still improving but acceptance is low: increase N_anneal first, then consider +1 N_optimize stage.')
        else:
            print('  - Still improving with healthy acceptance: increasing N_anneal or N_optimize can still help.')


    plt.figure()
    plt.hist(fiberlengths, bins=50)
    plt.title('Fiber Length Distribution after Optimization')
    plt.show()

    # Branching Optimization
    # If no explicit per-sweep step parameter is provided, build a linear schedule
    # from max -> min over n_branching_optimize sweeps.
    if branching_stepsize_mag is None:
        if branching_stepsize_frac_min <= 0 or branching_stepsize_frac_max <= 0:
            raise ValueError("branching_stepsize_frac_min/max must be > 0")
        if branching_stepsize_frac_min > branching_stepsize_frac_max:
            raise ValueError("branching_stepsize_frac_min must be <= branching_stepsize_frac_max")

        if branching_stepsize_mode == "local_mean_fiber":
            # Dimensionless fraction schedule.
            branching_stepsize_mag = np.linspace(
                branching_stepsize_frac_max,
                branching_stepsize_frac_min,
                n_branching_optimize,
            )
        else:
            # Convert fraction schedule into absolute displacements using l_fiber.
            branching_stepsize_mag = l_fiber * np.linspace(
                branching_stepsize_frac_max,
                branching_stepsize_frac_min,
                n_branching_optimize,
            )

    schedule = np.asarray(branching_stepsize_mag, dtype=float)
    if schedule.ndim == 0:
        schedule = np.full(n_branching_optimize, float(schedule))
    print(f'Branching optimization mode: {branching_stepsize_mode}')
    print(f'Branching step schedule ({n_branching_optimize} sweeps): {schedule}')

    nodes, fibers, nodal_branching_energy, total_branching_energy_init, total_branching_energy_final, branch_diag = branch_optimization(
        n_branching_optimize,
        nodes,
        fibers,
        N,
        enforce_bounds=enforce_bounds,
        bounds=bounds,
        bound_mode=bound_mode,
        stepsize_mag=branching_stepsize_mag,
        stepsize_mode=branching_stepsize_mode,
        return_diagnostics=True,
    )

    branch_energy_hist = np.asarray(branch_diag["total_branching_energy_per_sweep"], dtype=float)
    branch_rel_drop = (total_branching_energy_init - total_branching_energy_final) / max(abs(total_branching_energy_init), 1e-12)
    print('\n=== Branching Optimization Convergence Summary ===')
    print(
        f'Branching energy: {total_branching_energy_init:.6g} -> {total_branching_energy_final:.6g} '
        f'(relative drop {100 * branch_rel_drop:.2f}%)'
    )
    for m in range(n_branching_optimize):
        sweep_drop = 0.0 if m == 0 else (branch_energy_hist[m - 1] - branch_energy_hist[m]) / max(abs(branch_energy_hist[m - 1]), 1e-12)
        print(
            f'  Sweep {m + 1}/{n_branching_optimize}: energy={branch_energy_hist[m]:.6g}, '
            f'stage_drop={100 * sweep_drop:.2f}%'
        )

    if n_branching_optimize >= 2:
        last_branch_drop = (branch_energy_hist[-2] - branch_energy_hist[-1]) / max(abs(branch_energy_hist[-2]), 1e-12)
        print('Recommendations (branching optimization):')
        if last_branch_drop < 0.003:
            print('  - Branching stage is near plateau: keep current sweeps, or increase max fraction if you want stronger late movement.')
        elif last_branch_drop < 0.01:
            print('  - Moderate improvement remains: consider +1 to +2 branching sweeps before increasing step fractions.')
        else:
            print('  - Strong improvement still present: increasing N_branching_optimize is likely beneficial.')

    # bounds: list of (min, max) per axis, used for snapping and connectivity pruning.
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

    # Branching optimization step controls.
    BRANCHING_STEPSIZE_MODE = "local_mean_fiber"  # "absolute" or "local_mean_fiber"
    BRANCHING_STEPSIZE_MAG = None  # None => build schedule from min/max fractions
    BRANCHING_STEPSIZE_FRAC_MIN = 0.007
    BRANCHING_STEPSIZE_FRAC_MAX = 0.02
    MIN_SWAP_SWEEPS_PER_OPT_ROUND = 1
    
    # Fiber Length Optimization
    # N_anneal: number of simulated-annealing sweeps per round. Simulated annealing
    #           is a stochastic search that accepts some worse moves early to escape
    #           local minima, then becomes more selective. More sweeps = more trials.
    # N_optimize: number of outer rounds with progressively smaller stepsizes and
    #            swap fractions (cooling schedule), refining the network.
    N_ANNEAL = 15
    N_OPTIMIZE = 5
    N_BRANCHING_OPTIMIZE = 6

    if os.path.exists(file_name):
        print(f'Loading network from {file_path}')
        # Load from the pickle file
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            # nodes: (N, 3) float array, connectivity: {node_index: [neighbors...]}
            nodes = data['node_coords']
            connectivity = data['connectivity']
            nodes, connectivity = merge_duplicate_nodes(nodes, connectivity)
    else: 
        # nodes: (N, 3) float array, fibers: (M, 2) int array, bounds: [(xmin,xmax), ...]
        nodes, fibers, bounds = generate_network(
            LX,
            LY,
            LZ,
            L_FIBER,
            RHO,
            enforce_bounds=ENFORCE_BOUNDS,
            bound_mode=BOUND_MODE,
            branching_stepsize_mode=BRANCHING_STEPSIZE_MODE,
            branching_stepsize_mag=BRANCHING_STEPSIZE_MAG,
            branching_stepsize_frac_min=BRANCHING_STEPSIZE_FRAC_MIN,
            branching_stepsize_frac_max=BRANCHING_STEPSIZE_FRAC_MAX,
            min_swap_sweeps_per_opt_round=MIN_SWAP_SWEEPS_PER_OPT_ROUND,
            n_anneal=N_ANNEAL,
            n_optimize=N_OPTIMIZE,
            n_branching_optimize=N_BRANCHING_OPTIMIZE,
        )
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
        # connectivity: {node_index: [neighbors...]} with fixed list length MAX_CONNECTIVITY.
        node_connectivity = compute_node_connectivity(fibers, num_nodes, MAX_CONNECTIVITY)
        #plot_network_3d(nodes, node_connectivity, title ='before fix')

        nodes, node_connectivity = remove_boundary_connectivity(nodes, node_connectivity, bounds=bounds)
        nodes, node_connectivity = merge_duplicate_nodes(nodes, node_connectivity)
        # fig, ax = plot_network_3d(nodes, node_connectivity, title ='after fix')

        # new_nodes: (N', 3) float array, new_connectivity: updated connectivity dict.
        new_nodes, new_connectivity = add_intermediate_nodes(nodes, node_connectivity, EDGE_LENGTH, MAX_CONNECTIVITY)
        check_duplicates(new_nodes, new_connectivity, "after_add_intermediate_nodes", edge_kind="connectivity")
        # new_nodes = nodes.copy()
        # new_connectivity = node_connectivity.copy()
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
