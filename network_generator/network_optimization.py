import numpy as np

def network_optimization(fraction_to_try_swap, N, nodes, fibers, N_anneal, lx, ly, lz, l_fiber, fiberlengths, fiberenergy, N1, N2, N_boundary_nodes, stepsize, swap_skip_energy, enforce_bounds=False, bounds=None, bound_mode="reject", min_swap_sweeps=0, return_stats=False):
    """
    network_optimization: use simulated annealing to iterate initial network
    """
    if not (0 <= fraction_to_try_swap <= 1):
        raise ValueError("fraction_to_try_swap must be in [0, 1]")
    if not (0 <= min_swap_sweeps <= N_anneal):
        raise ValueError("min_swap_sweeps must be in [0, N_anneal]")

    N_interior = np.arange(N_boundary_nodes, N)  # The boundary nodes are at the top of nodes, we don't want to move those
    N_int = len(N_interior)

    # Build a per-sweep swap schedule.
    # Equivalent to Bernoulli(p) per sweep when min_swap_sweeps=0, with an optional
    # floor to avoid optimization rounds with too few/no swap sweeps.
    n_swap_sweeps = np.random.binomial(N_anneal, fraction_to_try_swap)
    n_swap_sweeps = max(n_swap_sweeps, int(min_swap_sweeps))
    swap_sweep_mask = np.zeros(N_anneal, dtype=bool)
    if n_swap_sweeps > 0:
        swap_sweep_indices = np.random.choice(N_anneal, size=n_swap_sweeps, replace=False)
        swap_sweep_mask[swap_sweep_indices] = True
    
    # Initialize counters
    anneal_id = 0
    swap_id = 0
    n_swaps_accepted = 0
    n_swaps_rejected = 0
    n_accepted = 0
    n_prob = 0
    
    # Initialize node_fiber_matrix
    node_fiber_mat = np.full((len(nodes), 10), np.nan)
    
    for i in range(len(nodes)):
        nlf = np.where((fibers == i).any(axis=1))[0]
        node_fiber_mat[i, :len(nlf)] = nlf
    
    for m in range(N_anneal):
        if not swap_sweep_mask[m]:
            # Do node displacement, just on interior nodes
            for j in N_interior:
                anneal_id += 1
                # Try displacing the jth node by a random 3D spatial step
                newx = nodes[j, 0] + stepsize * (2 * (-.5 + np.random.rand())) * (((N_anneal * N_int) - n_accepted) / (N_anneal * N_int))
                newy = nodes[j, 1] + (ly / lx) * stepsize * (2 * (-.5 + np.random.rand())) * (((N_anneal * N_int) - n_accepted) / (N_anneal * N_int))
                newz = nodes[j, 2] + (lz / lx) * stepsize * (2 * (-.5 + np.random.rand())) * (((N_anneal * N_int) - n_accepted) / (N_anneal * N_int))

                if enforce_bounds and bounds is not None:
                    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
                    if bound_mode == "clip":
                        newx = np.clip(newx, xmin, xmax)
                        newy = np.clip(newy, ymin, ymax)
                        newz = np.clip(newz, zmin, zmax)
                    elif bound_mode == "reject":
                        if newx < xmin or newx > xmax or newy < ymin or newy > ymax or newz < zmin or newz > zmax:
                            continue
                
                # Find the fibers that this would affect
                n1f = node_fiber_mat[j, ~np.isnan(node_fiber_mat[j, :])].astype(int)
                othernodes = np.zeros(len(n1f), dtype=int)
                for k in range(len(n1f)):
                    node1, node2 = fibers[n1f[k]]
                    if node1 == j:
                        othernodes[k] = node2
                    else:
                        othernodes[k] = node1
                
                nfl = np.zeros(len(n1f))  # new fiber lengths
                for k in range(len(nfl)):
                    nfl[k] = np.sqrt((nodes[othernodes[k], 0] - newx) ** 2 + (nodes[othernodes[k], 1] - newy) ** 2 + (nodes[othernodes[k], 2] - newz) ** 2)
                
                nfe = (nfl - l_fiber) ** 2
                nfe_sum = np.sum(nfe)
                
                # Compare this to the sum of the corresponding fiberenergy values
                current_fes = fiberenergy[n1f]
                current_fes_sum = np.sum(current_fes)
                
                if nfe_sum < current_fes_sum:
                    nodes[j, :] = [newx, newy, newz]  # update relevant arrays
                    N1[n1f, :] = nodes[fibers[n1f, 0], :]
                    N2[n1f, :] = nodes[fibers[n1f, 1], :]
                    
                    fiberlengths[n1f] = np.sqrt((N1[n1f, 0] - N2[n1f, 0]) ** 2 + (N1[n1f, 1] - N2[n1f, 1]) ** 2 + (N1[n1f, 2] - N2[n1f, 2]) ** 2)
                    fiberenergy[n1f] = (fiberlengths[n1f] - l_fiber) ** 2
                    n_accepted += 1
                elif nfe_sum < 1.2 * current_fes_sum and np.random.rand() > .95:
                    nodes[j, :] = [newx, newy, newz]  # update relevant arrays
                    N1[n1f, :] = nodes[fibers[n1f, 0], :]
                    N2[n1f, :] = nodes[fibers[n1f, 1], :]
                    
                    fiberlengths[n1f] = np.sqrt((N1[n1f, 0] - N2[n1f, 0]) ** 2 + (N1[n1f, 1] - N2[n1f, 1]) ** 2 + (N1[n1f, 2] - N2[n1f, 2]) ** 2)
                    fiberenergy[n1f] = (fiberlengths[n1f] - l_fiber) ** 2
                    n_accepted += 1
        else:
            # Do node swapping
            for j in range(N):
                node1 = j
                n1f = node_fiber_mat[node1, ~np.isnan(node_fiber_mat[node1, :])].astype(int)
                current_nodal_energy = fiberenergy[n1f]
                cne_sum = np.sum(current_nodal_energy)
                
                if cne_sum < swap_skip_energy:
                    continue  # skip this node if its energy is already pretty low
                
                anneal_id += 1
                swap_id += 1
                
                node1fibers_nodes = np.zeros(len(n1f), dtype=int)
                for h in range(len(node1fibers_nodes)):
                    n1, n2 = fibers[n1f[h]]
                    if n1 == node1:
                        node1fibers_nodes[h] = n2
                    else:
                        node1fibers_nodes[h] = n1
                
                id_max = np.argmax(current_nodal_energy)
                nftd_1 = node1fibers_nodes[id_max]  # pick the worst fiber of node j
                ftd_1 = n1f[id_max]
                
                distances_1 = np.sqrt((nodes[node1, 0] - nodes[:, 0]) ** 2 + (nodes[node1, 1] - nodes[:, 1]) ** 2 + (nodes[node1, 2] - nodes[:, 2]) ** 2)
                neighbors_1 = np.where((distances_1 < 2 * l_fiber) & (distances_1 > 0.5 * l_fiber))[0]
                
                for k in node1fibers_nodes:
                    neighbors_1 = neighbors_1[neighbors_1 != k]
                
                nftd_1_fibers = node_fiber_mat[nftd_1, ~np.isnan(node_fiber_mat[nftd_1, :])].astype(int)
                nftd_1_nodes = np.zeros(len(nftd_1_fibers), dtype=int)
                for h in range(len(nftd_1_fibers)):
                    n1, n2 = fibers[nftd_1_fibers[h]]
                    if n1 == nftd_1:
                        nftd_1_nodes[h] = n2
                    else:
                        nftd_1_nodes[h] = n1
                
                for k in nftd_1_nodes:
                    neighbors_1 = neighbors_1[neighbors_1 != k]
                
                if len(neighbors_1) == 0:
                    continue
                
                node3 = nftd_1
                
                wc = 0
                while wc < len(neighbors_1):
                    wc += 1
                    n2try = neighbors_1[wc - 1]
                    node2 = n2try
                    n2try_fibers = node_fiber_mat[n2try, ~np.isnan(node_fiber_mat[n2try, :])].astype(int)
                    if len(n2try_fibers) == 0:
                        continue
                    
                    current_nodal_energy = fiberenergy[n2try_fibers]
                    id_max = np.argmax(current_nodal_energy)
                    fiber2 = n2try_fibers[id_max]
                    
                    n1, n2 = fibers[fiber2]
                    if n1 == node2:
                        node4 = n2
                    else:
                        node4 = n1
                    
                    if node4 in node1fibers_nodes or node4 in nftd_1_nodes:
                        n_prob += 1
                        continue
                    
                    fiber1 = ftd_1
                    
                    if node1 == node4 or node3 == node2:
                        continue
                    
                    new_fiber_1 = [node1, node4]
                    new_fiber_2 = [node2, node3]
                    
                    newfiber1length = np.sqrt((nodes[node1, 0] - nodes[node4, 0]) ** 2 + (nodes[node1, 1] - nodes[node4, 1]) ** 2 + (nodes[node1, 2] - nodes[node4, 2]) ** 2)
                    newfiber2length = np.sqrt((nodes[node2, 0] - nodes[node3, 0]) ** 2 + (nodes[node2, 1] - nodes[node3, 1]) ** 2 + (nodes[node2, 2] - nodes[node3, 2]) ** 2)
                    
                    newfibersenergy = (newfiber1length - l_fiber) ** 2 + (newfiber2length - l_fiber) ** 2
                    oldtwofibers = [fiber1, fiber2]
                    oldfibersenergy = np.sum(fiberenergy[oldtwofibers])
                    
                    if newfibersenergy < oldfibersenergy:
                        for n in range(2):
                            node_rem = fibers[fiber1, n]
                            fib_rem_ind = np.where(node_fiber_mat[node_rem, :] == fiber1)[0]
                            node_fiber_mat[node_rem, fib_rem_ind[0]] = np.nan
                        
                        for n in range(2):
                            node_rem = fibers[fiber2, n]
                            fib_rem_ind = np.where(node_fiber_mat[node_rem, :] == fiber2)[0]
                            node_fiber_mat[node_rem, fib_rem_ind[0]] = np.nan
                        
                        fibers[fiber1, :] = new_fiber_1
                        fibers[fiber2, :] = new_fiber_2
                        
                        for n in range(2):
                            node_add = new_fiber_1[n]
                            fib_add_ind = np.where(np.isnan(node_fiber_mat[node_add, :]))[0]
                            node_fiber_mat[node_add, fib_add_ind[0]] = fiber1
                        
                        for n in range(2):
                            node_add = new_fiber_2[n]
                            fib_add_ind = np.where(np.isnan(node_fiber_mat[node_add, :]))[0]
                            node_fiber_mat[node_add, fib_add_ind[0]] = fiber2
                        
                        N1[fiber1, :] = nodes[node1, :]
                        N2[fiber1, :] = nodes[node4, :]
                        N1[fiber2, :] = nodes[node2, :]
                        N2[fiber2, :] = nodes[node3, :]
                        
                        fiberlengths[n1f] = np.sqrt((N1[n1f, 0] - N2[n1f, 0]) ** 2 + (N1[n1f, 1] - N2[n1f, 1]) ** 2 + (N1[n1f, 2] - N2[n1f, 2]) ** 2)
                        fiberenergy[n1f] = (fiberlengths[n1f] - l_fiber) ** 2
                        
                        n_accepted += 1
                        n_swaps_accepted += 1
                        break
                    else:
                        n_swaps_rejected += 1
    
    percent_accepted_iterations = 100 * n_accepted / anneal_id if anneal_id > 0 else 0.0
    print(f'Percent accepted iterations for that optimization run = {percent_accepted_iterations}')
    print(f'Swap sweeps scheduled/executed = {n_swap_sweeps}/{N_anneal} (target p={fraction_to_try_swap}, min={min_swap_sweeps})')
    
    total_energy = np.sum(fiberenergy)
    print(f'Total Network Length Energy = {total_energy}')
    
    avg_fiber_length = np.mean(fiberlengths)
    median_fiber_length = np.median(fiberlengths)
    print(f'mean fiber length = {avg_fiber_length} median length = {median_fiber_length}')
    
    N_fibers = len(fibers)
    fdc = np.column_stack([fibers[:, 1], fibers[:, 0]])
    fiber_duplicates_final = np.vstack([fibers, fdc])
    _, fiber_duplicate_check_final = np.unique(fiber_duplicates_final, axis=0, return_index=True)
    final_fiber_check = 2 * N_fibers - len(fiber_duplicate_check_final)
    
    print(f'The number of duplicate fibers is {final_fiber_check}')

    if return_stats:
        stats = {
            "percent_accepted_iterations": percent_accepted_iterations,
            "swap_sweeps_executed": int(n_swap_sweeps),
            "swap_sweeps_total": int(N_anneal),
            "n_swaps_accepted": int(n_swaps_accepted),
            "n_swaps_rejected": int(n_swaps_rejected),
            "n_accepted": int(n_accepted),
            "anneal_id": int(anneal_id),
            "total_energy": float(total_energy),
            "mean_fiber_length": float(avg_fiber_length),
            "median_fiber_length": float(median_fiber_length),
        }
        return nodes, fibers, fiberenergy, fiberlengths, total_energy, N1, N2, N_interior, stats

    return nodes, fibers, fiberenergy, fiberlengths, total_energy, N1, N2, N_interior
