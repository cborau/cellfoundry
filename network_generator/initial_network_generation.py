import numpy as np
import time as _time

def initial_network_generation(rho, lx, ly, lz, l_fiber, n_neighbors_to_try):
    """
    Initial_Network_Generation: Create an initial fiber network configuration
    that satisfies the desired valency distribution, but not necessarily
    other network properties.
    """
    # Number of nodes
    N = round(rho * lx * ly * lz)
    volume = lx * ly * lz
    print(f'[init] rho={rho}, volume={volume:.0f}, N={N} nodes')
    print(f'[init] Fiber assignment loop is O(N^2) — {N}^2 = {N**2:.2e} operations')
    if N > 50000:
        print(f'[init] WARNING: N={N} is very large. This will be slow (minutes to hours).')
    
    # Number of boundary nodes
    NBx = round(.05 * (N))
    NBy = round(.05 * (N))
    NBz = round(.05 * (N))
    
    # Boundary nodes
    N_boundary_nodes = 2 * (NBx + NBy + NBz)
    nodes_boundary = np.zeros((N_boundary_nodes, 3))
    
    # X boundary
    xbnd1_x = (-lx / 2) * np.ones(NBx)
    xbnd1_y = (-ly / 2) + ly * np.random.rand(NBx)
    xbnd1_z = (-lz / 2) + lz * np.random.rand(NBx)
    
    xbnd2_x = (lx / 2) * np.ones(NBx)
    xbnd2_y = (-ly / 2) + ly * np.random.rand(NBx)
    xbnd2_z = (-lz / 2) + lz * np.random.rand(NBx)
    
    nodes_boundary[0:2*NBx, :] = np.vstack([np.column_stack((xbnd1_x, xbnd1_y, xbnd1_z)), 
                                            np.column_stack((xbnd2_x, xbnd2_y, xbnd2_z))])
    
    # Y boundary
    ybnd1_x = (-lx / 2) + lx * np.random.rand(NBy)
    ybnd1_y = (-ly / 2) * np.ones(NBy)
    ybnd1_z = (-lz / 2) + lz * np.random.rand(NBy)
    
    ybnd2_x = (-lx / 2) + lx * np.random.rand(NBy)
    ybnd2_y = (ly / 2) * np.ones(NBy)
    ybnd2_z = (-lz / 2) + lz * np.random.rand(NBy)
    
    nodes_boundary[2*NBx:2*NBx+2*NBy, :] = np.vstack([np.column_stack((ybnd1_x, ybnd1_y, ybnd1_z)), 
                                                      np.column_stack((ybnd2_x, ybnd2_y, ybnd2_z))])
    
    # Z boundary
    zbnd1_x = (-lx / 2) + lx * np.random.rand(NBz)
    zbnd1_y = (-ly / 2) + ly * np.random.rand(NBz)
    zbnd1_z = (-lz / 2) * np.ones(NBz)
    
    zbnd2_x = (-lx / 2) + lx * np.random.rand(NBz)
    zbnd2_y = (-ly / 2) + ly * np.random.rand(NBz)
    zbnd2_z = (lz / 2) * np.ones(NBz)
    
    nodes_boundary[2*NBx+2*NBy:N_boundary_nodes, :] = np.vstack([np.column_stack((zbnd1_x, zbnd1_y, zbnd1_z)), 
                                                                 np.column_stack((zbnd2_x, zbnd2_y, zbnd2_z))])
    
    # Preallocate node array
    nodes = np.zeros((N, 3))
    nodes[0:N_boundary_nodes, :] = nodes_boundary
    
    # The rest of the nodes should be uniformly randomly distributed inside the domain.
    nodes[N_boundary_nodes:N, :] = np.random.rand(N - N_boundary_nodes, 3) * np.array([lx, ly, lz]) - 0.5 * np.array([lx, ly, lz])

    # Each node needs to be assigned a valency of either 3, 4, 5, or 6.
    valency = np.random.rand(N)
    valency[valency < .7] = 2
    valency[valency < .8] = 3
    valency[valency < .9] = 4
    valency[valency < .97] = 5
    valency[valency < 1] = 6
    valency = valency.astype(int)
    
    # Preallocate the fibers array
    fibers = np.zeros((4 * N, 2), dtype=int)
    current_valency_matrix = np.zeros((N, 2), dtype=int)
    current_valency_matrix[:, 0] = np.arange(N)
    fiberid = 0
    n_wll = 0
    _t0 = _time.time()
    _last_print = _t0
    _print_interval = 5.0  # seconds between progress prints
    print(f'[init] Starting fiber assignment loop over {N} nodes...')
    
    for k in range(N):
        _now = _time.time()
        if _now - _last_print >= _print_interval or k == 0:
            elapsed = _now - _t0
            if k > 0:
                eta = elapsed / k * (N - k)
                print(f'[init]   node {k}/{N} ({100*k/N:.1f}%) — '
                      f'elapsed {elapsed:.1f}s, ETA {eta:.1f}s, '
                      f'fibers so far: {fiberid}, whileloop breaks: {n_wll}')
            else:
                print(f'[init]   node {k}/{N} — starting...')
            _last_print = _now
        currentvalency = current_valency_matrix[k, 1]
        node_loc = nodes[k, :]
        neighbors = []
        nd_val = 0.5 * (lx + ly + lz) / 3
        node_dist = nd_val
        
        while len(neighbors) < n_neighbors_to_try + 1:
            node_dist += nd_val
            x_lim = np.where((nodes[:, 0] > node_loc[0] - node_dist) & (nodes[:, 0] < node_loc[0] + node_dist))[0]
            y_lim = np.where((nodes[:, 1] > node_loc[1] - node_dist) & (nodes[:, 1] < node_loc[1] + node_dist))[0]
            z_lim = np.where((nodes[:, 2] > node_loc[2] - node_dist) & (nodes[:, 2] < node_loc[2] + node_dist))[0]
            neighbors = np.intersect1d(np.intersect1d(x_lim, y_lim), z_lim)
        
        distances = np.sqrt(np.sum((nodes[neighbors, :] - node_loc) ** 2, axis=1))
        neighbors = neighbors[np.argsort(distances)]
        
        neighbors = neighbors[1:n_neighbors_to_try + 1]  # first neighbor is the node itself
        
        for m in range(valency[k] - currentvalency):
            fibers[fiberid, 0] = k
            current_valency_matrix[k, 1] += 1
            eid = np.random.randint(n_neighbors_to_try)
            endpoint = neighbors[eid]
            currentneighborvalency = current_valency_matrix[endpoint, 1]
            whilelooplimit = 10 * n_neighbors_to_try
            wll = 0
            
            while currentneighborvalency >= valency[endpoint]:
                if wll > whilelooplimit:
                    n_wll += 1
                    break
                eid = np.random.randint(n_neighbors_to_try)
                endpoint = neighbors[eid]
                currentneighborvalency = current_valency_matrix[endpoint, 1]
                wll += 1
            
            fibers[fiberid, 1] = endpoint
            current_valency_matrix[endpoint, 1] += 1
            fiberid += 1
    
    _elapsed_total = _time.time() - _t0
    print(f'[init] Fiber assignment loop done: {fiberid} fibers in {_elapsed_total:.1f}s '
          f'({n_wll} whileloop breaks)')

    # Crop fibers
    fibers = fibers[~np.all(fibers == 0, axis=1)]
    
    # Remove zero-length fibers
    zlc = fibers[:, 0] - fibers[:, 1]
    zlcheck = np.where(zlc == 0)[0]
    fibers = np.delete(fibers, zlcheck, axis=0)
    
    # Check for literal duplicates in fibers
    _, fiber_duplicate_check = np.unique(fibers, axis=0, return_index=True)
    fibers = fibers[np.sort(fiber_duplicate_check)]
    
    # Check for "mirrored" duplicates
    fmc = np.column_stack([fibers[:, 1], fibers[:, 0]])
    fibers_plus_mirrored = np.vstack([fibers, fmc])
    _, fiber_mirror_check = np.unique(fibers_plus_mirrored, axis=0, return_index=True)
    mirrors_deleted = fibers_plus_mirrored[np.sort(fiber_mirror_check)]
    fibers_with_mirrors_deleted = np.column_stack([mirrors_deleted[len(fibers):, 1], mirrors_deleted[len(fibers):, 0]])
    
    fibers = fibers_with_mirrors_deleted
    
    # Check what the average valency is
    valencycheck = np.zeros(N, dtype=int)
    
    for k in range(N):
        valencycheck[k] = np.sum(fibers == k)
    
    val_0 = np.where(valencycheck == 0)[0]
    print(f'N valency=0 is {len(val_0)}')
    val_1 = np.where(valencycheck == 1)[0]
    print(f'N valency=1 is {len(val_1)}')
    val_2 = np.where(valencycheck == 2)[0]
    print(f'N valency=2 is {len(val_2)}')
    
    print(f'[init] Fixing under-connected nodes: '
          f'{len(val_0)} with valency 0, {len(val_1)} with valency 1, {len(val_2)} with valency 2')
    add_fibers_0_1 = np.zeros((len(val_0), 2), dtype=int)
    add_fibers_0_2 = add_fibers_0_1.copy()
    add_fibers_0_3 = add_fibers_0_1.copy()
    add_fibers_1_1 = np.zeros((len(val_1), 2), dtype=int)
    add_fibers_1_2 = add_fibers_1_1.copy()
    add_fibers_3 = np.zeros((len(val_2), 2), dtype=int)
    _t_fix = _time.time()
    
    for k in range(len(val_0)):
        if k % max(1, len(val_0)//5) == 0 and len(val_0) > 100:
            print(f'[init]   valency-0 fix: {k}/{len(val_0)}')
        distances = np.sqrt(np.sum((nodes[val_0[k], :] - nodes) ** 2, axis=1))
        neighbors = np.argsort(distances)[1:4]
        add_fibers_0_1[k, :] = [val_0[k], neighbors[0]]
        add_fibers_0_2[k, :] = [val_0[k], neighbors[1]]
        add_fibers_0_3[k, :] = [val_0[k], neighbors[2]]
    
    for k in range(len(val_1)):
        if k % max(1, len(val_1)//5) == 0 and len(val_1) > 100:
            print(f'[init]   valency-1 fix: {k}/{len(val_1)}')
        distances = np.sqrt(np.sum((nodes[val_1[k], :] - nodes) ** 2, axis=1))
        neighbors = np.argsort(distances)[1:3]
        add_fibers_1_1[k, :] = [val_1[k], neighbors[0]]
        add_fibers_1_2[k, :] = [val_1[k], neighbors[1]]
    
    for k in range(len(val_2)):
        if k % max(1, len(val_2)//5) == 0 and len(val_2) > 100:
            print(f'[init]   valency-2 fix: {k}/{len(val_2)}')
        distances = np.sqrt(np.sum((nodes[val_2[k], :] - nodes) ** 2, axis=1))
        neighbor = np.argsort(distances)[1]
        add_fibers_3[k, :] = [val_2[k], neighbor]
    
    _t_fix_done = _time.time()
    print(f'[init] Valency fix done in {_t_fix_done - _t_fix:.1f}s')
    fibers = np.vstack([fibers, add_fibers_0_1, add_fibers_0_2, add_fibers_0_3, add_fibers_1_1, add_fibers_1_2, add_fibers_3])
    
    for k in range(N):
        valencycheck[k] = np.sum(fibers == k)
    
    check = valency - valencycheck
    wrong = np.where(check)[0]
    
    n_wrong_valency = len(wrong)
    avgvalency = np.mean(valencycheck)
    
    print(f'The average valency is {avgvalency}')
    wrongvalencyfraction = n_wrong_valency / N
    print(f'The percent of nodes with the wrong valency is {100 * wrongvalencyfraction}')
    
    # Set up arrays with the fiber endpoint coordinates
    N_fibers = len(fibers)
    N1 = np.zeros((N_fibers, 3))
    N2 = N1.copy()
    
    for m in range(N_fibers):
        N1[m, :] = nodes[fibers[m, 0], :]
        N2[m, :] = nodes[fibers[m, 1], :]
    
    fiberlengths = np.sqrt(np.sum((N1 - N2) ** 2, axis=1))
    
    # Find the total_energy of the initial network
    fiberenergy = (fiberlengths - l_fiber) ** 2
    total_energy = np.sum(fiberenergy)
    
    return nodes, fibers, fiberenergy, total_energy, N1, N2, N_boundary_nodes, fiberlengths, valencycheck
