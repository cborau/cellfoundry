import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def branch_optimization(
    N_branching_optimize,
    nodes,
    fibers,
    N,
    enforce_bounds=False,
    bounds=None,
    bound_mode="reject",
    stepsize_mag=0.15,
    stepsize_mode="absolute",
    plot_diagnostics=True,
    return_diagnostics=False,
):
    """
    Improve local branch alignment at network nodes via stochastic, greedy moves.

    For each node, this routine identifies the pair of connected fibers that are
    most opposite (minimum dot product), then evaluates a branching energy that:
    - rewards the selected pair being closer to collinear/opposite,
    - penalizes remaining fibers that are not aligned with one of those two fibers.

    During optimization, each node and its directly connected neighbor nodes are
    proposed to move by a small displacement, and the move is accepted if local
    branching energy decreases (with a small probability of accepting slightly
    worse moves to avoid poor local minima).

    Parameters
    ----------
    N_branching_optimize : int
        Number of global branching-optimization sweeps.
    nodes : np.ndarray, shape (N, 3)
        Node coordinates.
    fibers : np.ndarray, shape (E, 2)
        Fiber connectivity as endpoint node indices.
    N : int
        Number of nodes (expected to match nodes.shape[0]).
    enforce_bounds : bool, optional
        If True, keep proposed moves within `bounds` using `bound_mode`.
    bounds : tuple, optional
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    bound_mode : {"reject", "clip"}, optional
        - "reject": skip proposals that move any affected node out of bounds.
        - "clip": clamp proposed coordinates to bounds.
    stepsize_mag : float or array-like, optional
        Step-size parameter whose interpretation depends on `stepsize_mode`.
        If scalar, the same value is used for all sweeps. If array-like,
        length must equal `N_branching_optimize` and each entry is used for
        the corresponding sweep.
    stepsize_mode : {"absolute", "local_mean_fiber"}, optional
        - "absolute": use `stepsize_mag` directly as displacement magnitude.
        - "local_mean_fiber": scale by local mean connected-fiber length,
          i.e., effective step = `stepsize_mag * mean(local_fiber_lengths)`.
    plot_diagnostics : bool, optional
        If True, emit diagnostic plots before/after optimization.
    return_diagnostics : bool, optional
        If True, return a diagnostics dictionary with per-sweep histories.

    Returns
    -------
    nodes : np.ndarray
        Updated node coordinates.
    fibers : np.ndarray
        Unchanged connectivity array.
    nodal_branching_energy : np.ndarray, shape (N,)
        Final nodal branching energy values.
    total_branching_energy_init : float
        Total branching energy before optimization.
    total_branching_energy_final : float
        Total branching energy after optimization.
    """

    if bound_mode not in ("reject", "clip"):
        raise ValueError("bound_mode must be 'reject' or 'clip'")
    if stepsize_mode not in ("absolute", "local_mean_fiber"):
        raise ValueError("stepsize_mode must be 'absolute' or 'local_mean_fiber'")

    stepsize_schedule = np.asarray(stepsize_mag, dtype=float)
    if stepsize_schedule.ndim == 0:
        stepsize_schedule = np.full(N_branching_optimize, float(stepsize_schedule))
    elif stepsize_schedule.ndim == 1 and len(stepsize_schedule) == N_branching_optimize:
        pass
    else:
        raise ValueError("stepsize_mag must be scalar or length N_branching_optimize")

    if np.any(stepsize_schedule <= 0):
        raise ValueError("stepsize_mag values must be > 0")

    othernodes = [None] * N
    valency = np.zeros(N, dtype=int)
    n2s = np.zeros(N, dtype=int)

    for j in range(N):
        node1 = j
        cf = np.where((fibers == node1).any(axis=1))[0]
        n_cf = len(cf)
        
        valency[j] = n_cf
        
        othernodes_jth = np.zeros(n_cf, dtype=int)
        
        for k in range(n_cf):
            n1, n2 = fibers[cf[k]]
            if n1 == node1:
                othernodes_jth[k] = n2
            else:
                othernodes_jth[k] = n1
        
        othernodes[node1] = othernodes_jth

    nodal_branching_energy = np.zeros(N)
    sf1 = np.zeros(N, dtype=int)
    sf2 = np.zeros(N, dtype=int)
    
    of_stored = [None] * N
    unit_vecs = [None] * N
    dot_prods = [None] * N
    SFs = [None] * N
    fiber_to_align_toward = np.zeros(N, dtype=int)

    for j in range(N):
        node1 = j
        othernodes_jth = othernodes[node1]
        n_cf = len(othernodes_jth)
        unit_vecs[j] = np.zeros((n_cf, 3))
        cf_lengths = np.zeros(n_cf)
        
        for k in range(n_cf):
            cf_lengths[k] = np.sqrt(np.sum((nodes[othernodes_jth[k]] - nodes[node1]) ** 2))
            unit_vecs[j][k] = (nodes[othernodes_jth[k]] - nodes[node1]) / cf_lengths[k]
        
        N_combinations = comb(n_cf, 2, exact=True)
        dot_prods[j] = np.zeros(N_combinations)
        combos = np.array([(a, b) for a in range(n_cf) for b in range(a + 1, n_cf)])
        
        for k in range(N_combinations):
            dot_prods[j][k] = np.dot(unit_vecs[j][combos[k, 0]], unit_vecs[j][combos[k, 1]])
        
        mindot = np.min(dot_prods[j])
        id_min = np.argmin(dot_prods[j])
        SFs[j] = combos[id_min]
        sf1 = SFs[j][0]
        sf2 = SFs[j][1]
        
        ofs = np.arange(n_cf)
        id2 = [sf1, sf2]
        ofs = np.delete(ofs, id2)
        of_stored[j] = ofs
        
        term_1 = 1 + mindot
        
        other_unit_vecs = np.delete(unit_vecs[j], id2, axis=0)
        sf1_uv = unit_vecs[j][sf1]
        sf2_uv = unit_vecs[j][sf2]
        
        dot_prods_sf1 = np.dot(other_unit_vecs, sf1_uv)
        dot_prods_sf2 = np.dot(other_unit_vecs, sf2_uv)
        
        align_sf1 = np.sum(dot_prods_sf1)
        align_sf2 = np.sum(dot_prods_sf2)
        
        if align_sf1 > align_sf2:
            term_2 = len(other_unit_vecs) - align_sf1
            fiber_to_align_toward[j] = 1
        else:
            term_2 = len(other_unit_vecs) - align_sf2
            fiber_to_align_toward[j] = 2
        
        nodal_branching_energy[j] = 8 * term_1 + term_2

    if plot_diagnostics:
        plt.figure()
        plt.hist(nodal_branching_energy)
        plt.title('Nodal Branching Energy Before Branching Optimization')
    
    total_branching_energy_init = np.sum(nodal_branching_energy)

    N_accepted_BO = 0
    N_accepted_wrt_m = np.zeros(N_branching_optimize)
    total_BE_intermed = np.zeros(N_branching_optimize)
    opt_id = 0
    n_baa = 0

    for m in range(N_branching_optimize):
        stepsize_param = stepsize_schedule[m]
        print(
            f'Branching sweep {m + 1}/{N_branching_optimize}: '
            f'step parameter = {stepsize_param:.6g} ({stepsize_mode})'
        )
        for j in range(N):
            opt_id += 1
            node1 = j
            oldx, oldy, oldz = nodes[node1]
            othernodes_jth = othernodes[node1]
            n_cf = len(othernodes_jth)

            if stepsize_mode == "absolute":
                stepsize = stepsize_param
            else:
                local_lengths = np.linalg.norm(nodes[othernodes_jth] - nodes[node1], axis=1)
                local_length_scale = np.mean(local_lengths) if local_lengths.size > 0 else 1.0
                stepsize = stepsize_param * local_length_scale

            endpt = unit_vecs[j][SFs[j]].sum(axis=0)
            l_endpt = np.sqrt(np.sum(endpt ** 2))
            
            dir_to_displace = endpt / l_endpt
            
            newx = oldx + stepsize * dir_to_displace[0] + (stepsize / 50) * (-0.5 + np.random.rand())
            newy = oldy + stepsize * dir_to_displace[1] + (stepsize / 50) * (-0.5 + np.random.rand())
            newz = oldz + stepsize * dir_to_displace[2] + (stepsize / 50) * (-0.5 + np.random.rand())
            
            cnis = np.concatenate(([node1], othernodes_jth))
            other_old_coords = nodes[othernodes_jth]
            other_spatial_steps = np.zeros((n_cf, 3))
            
            for k in range(n_cf - 2):
                endpt = unit_vecs[j][SFs[j][fiber_to_align_toward[j] - 1]] - unit_vecs[j][of_stored[j][k]]
                other_spatial_steps[k] = stepsize * endpt + (stepsize / 50) * (-0.5 + np.random.rand(3))
            
            other_new_coords = other_old_coords + other_spatial_steps

            if enforce_bounds and bounds is not None:
                (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
                if bound_mode == "clip":
                    newx = np.clip(newx, xmin, xmax)
                    newy = np.clip(newy, ymin, ymax)
                    newz = np.clip(newz, zmin, zmax)
                    other_new_coords[:, 0] = np.clip(other_new_coords[:, 0], xmin, xmax)
                    other_new_coords[:, 1] = np.clip(other_new_coords[:, 1], ymin, ymax)
                    other_new_coords[:, 2] = np.clip(other_new_coords[:, 2], zmin, zmax)
                elif bound_mode == "reject":
                    out_main = (newx < xmin or newx > xmax or newy < ymin or newy > ymax or newz < zmin or newz > zmax)
                    out_others = np.any(
                        (other_new_coords[:, 0] < xmin) | (other_new_coords[:, 0] > xmax) |
                        (other_new_coords[:, 1] < ymin) | (other_new_coords[:, 1] > ymax) |
                        (other_new_coords[:, 2] < zmin) | (other_new_coords[:, 2] > zmax)
                    )
                    if out_main or out_others:
                        continue
            
            new_nodal_branching_energies = np.zeros(1 + len(othernodes_jth))
            
            cf_lengths = np.zeros(n_cf)
            new_unit_vecs = np.zeros((n_cf, 3))
            
            for k in range(n_cf):
                newx_1, newy_1, newz_1 = other_new_coords[k]  # Ensure these variables are assigned correctly
                cf_lengths[k] = np.sqrt(np.sum((np.array([newx, newy, newz]) - np.array([newx_1, newy_1, newz_1])) ** 2))
                new_unit_vecs[k] = (np.array([newx, newy, newz]) - np.array([newx_1, newy_1, newz_1])) / cf_lengths[k]

            N_combinations = comb(n_cf, 2, exact=True)
            new_dot_prods = np.zeros(N_combinations)
            combos = np.array([(a, b) for a in range(n_cf) for b in range(a + 1, n_cf)])
            for k in range(N_combinations):
                new_dot_prods[k] = np.dot(new_unit_vecs[combos[k, 0]], new_unit_vecs[combos[k, 1]])
            
            mindot = np.min(new_dot_prods)
            id_min = np.argmin(new_dot_prods)
            new_SFs = combos[id_min]
            sf1 = new_SFs[0]
            sf2 = new_SFs[1]
            
            new_other_fibers = np.arange(len(othernodes_jth))
            id2 = [sf1, sf2]
            new_other_fibers = np.delete(new_other_fibers, id2)
            new_of_stored = new_other_fibers
            
            term_1 = 1 + mindot
            
            other_unit_vecs = np.delete(new_unit_vecs, id2, axis=0)
            sf1_uv = new_unit_vecs[sf1]
            sf2_uv = new_unit_vecs[sf2]
            
            dot_prods_sf1 = np.dot(other_unit_vecs, sf1_uv)
            dot_prods_sf2 = np.dot(other_unit_vecs, sf2_uv)
            
            align_sf1 = np.sum(dot_prods_sf1)
            align_sf2 = np.sum(dot_prods_sf2)
            
            if align_sf1 > align_sf2:
                term_2 = len(other_unit_vecs) - align_sf1
            else:
                term_2 = len(other_unit_vecs) - align_sf2
            
            new_nodal_branching_energies[0] = 8 * term_1 + term_2

            new_other_dot_prods = [None] * len(othernodes_jth)
            new_all_unit_vecs = [None] * len(othernodes_jth)
            new_other_SFs = [None] * len(othernodes_jth)
            new_other_of_stored = [None] * len(othernodes_jth)
            
            for l in range(len(othernodes_jth)):
                current_node = othernodes_jth[l]
                othernodes_lth = othernodes[current_node]
                n_cf = len(othernodes_lth)
                new_all_unit_vecs[l] = np.zeros((n_cf, 3))
                cf_lengths = np.zeros(n_cf)
                
                newx_1, newy_1, newz_1 = other_new_coords[l]
                jth_node_index = np.where(othernodes_lth == node1)[0][0]

                for k in range(n_cf):
                    if k == jth_node_index:
                        cf_lengths[k] = np.sqrt(np.sum((np.array([newx, newy, newz]) - np.array([newx_1, newy_1, newz_1])) ** 2))
                        new_all_unit_vecs[l][k] = (np.array([newx, newy, newz]) - np.array([newx_1, newy_1, newz_1])) / cf_lengths[k]
                    else:
                        cf_lengths[k] = np.sqrt(np.sum((nodes[othernodes_lth[k]] - np.array([newx_1, newy_1, newz_1])) ** 2))
                        new_all_unit_vecs[l][k] = (nodes[othernodes_lth[k]] - np.array([newx_1, newy_1, newz_1])) / cf_lengths[k]
                
                N_combinations = comb(n_cf, 2, exact=True)
                new_other_dot_prods[l] = np.zeros(N_combinations)
                combos = np.array([(a, b) for a in range(n_cf) for b in range(a + 1, n_cf)])
                for k in range(N_combinations):
                    new_other_dot_prods[l][k] = np.dot(new_all_unit_vecs[l][combos[k, 0]], new_all_unit_vecs[l][combos[k, 1]])
                
                mindot = np.min(new_other_dot_prods[l])
                id_min = np.argmin(new_other_dot_prods[l])
                new_other_SFs[l] = combos[id_min]
                sf1 = new_other_SFs[l][0]
                sf2 = new_other_SFs[l][1]
                new_other_fibers = np.arange(len(othernodes_lth))
                id2 = [sf1, sf2]
                new_other_fibers = np.delete(new_other_fibers, id2)
                new_other_of_stored[l] = new_other_fibers
                
                term_1 = 1 + mindot
                
                other_unit_vecs = np.delete(new_all_unit_vecs[l], id2, axis=0)
                sf1_uv = new_all_unit_vecs[l][sf1]
                sf2_uv = new_all_unit_vecs[l][sf2]
                
                dot_prods_sf1 = np.dot(other_unit_vecs, sf1_uv)
                dot_prods_sf2 = np.dot(other_unit_vecs, sf2_uv)
                
                align_sf1 = np.sum(dot_prods_sf1)
                align_sf2 = np.sum(dot_prods_sf2)
                
                if align_sf1 > align_sf2:
                    term_2 = len(other_unit_vecs) - align_sf1
                else:
                    term_2 = len(other_unit_vecs) - align_sf2
                
                new_nodal_branching_energies[1 + l] = 8 * term_1 + term_2

            old_nodal_branching_energies = nodal_branching_energy[cnis]
            rp = np.random.rand()
            
            if np.sum(new_nodal_branching_energies) < np.sum(old_nodal_branching_energies):
                nodes[cnis] = np.vstack(([newx, newy, newz], other_new_coords))
                nodal_branching_energy[cnis] = new_nodal_branching_energies
                unit_vecs[j] = new_unit_vecs
                dot_prods[j] = new_dot_prods
                SFs[j] = new_SFs
                of_stored[j] = new_of_stored
                for k in range(len(othernodes_jth)):
                    unit_vecs[othernodes_jth[k]] = new_all_unit_vecs[k]
                    dot_prods[othernodes_jth[k]] = new_other_dot_prods[k]
                    SFs[othernodes_jth[k]] = new_other_SFs[k]
                    of_stored[othernodes_jth[k]] = new_other_of_stored[k]
                N_accepted_BO += 1
            elif rp < 0.01 and np.sum(new_nodal_branching_energies) < 1.5 * np.sum(old_nodal_branching_energies):
                n_baa += 1
                nodes[cnis] = np.vstack(([newx, newy, newz], other_new_coords))
                nodal_branching_energy[cnis] = new_nodal_branching_energies
                unit_vecs[j] = new_unit_vecs
                dot_prods[j] = new_dot_prods
                SFs[j] = new_SFs
                of_stored[j] = new_of_stored
                for k in range(len(othernodes_jth)):
                    unit_vecs[othernodes_jth[k]] = new_all_unit_vecs[k]
                    dot_prods[othernodes_jth[k]] = new_other_dot_prods[k]
                    SFs[othernodes_jth[k]] = new_other_SFs[k]
                    of_stored[othernodes_jth[k]] = new_other_of_stored[k]
        
        total_BE_intermed[m] = np.sum(nodal_branching_energy)
        N_accepted_wrt_m[m] = N_accepted_BO

    if plot_diagnostics:
        plt.figure()
        plt.hist(nodal_branching_energy)
        plt.title('Nodal Branching Energy After Branching Optimization')

    total_branching_energy_final = np.sum(nodal_branching_energy)

    if plot_diagnostics:
        plt.figure()
        plt.plot(range(N_branching_optimize), N_accepted_wrt_m)
        plt.title('Accepted Changes vs Iteration Number')

        plt.figure(40)
        plt.plot(range(N_branching_optimize), total_BE_intermed)
        plt.title('Branching Energy vs Iteration Number')

    if return_diagnostics:
        diagnostics = {
            "stepsize_schedule": stepsize_schedule.copy(),
            "accepted_cumulative": N_accepted_wrt_m.copy(),
            "total_branching_energy_per_sweep": total_BE_intermed.copy(),
            "n_accepted_total": int(N_accepted_BO),
            "n_bad_accepts": int(n_baa),
        }
        return nodes, fibers, nodal_branching_energy, total_branching_energy_init, total_branching_energy_final, diagnostics

    return nodes, fibers, nodal_branching_energy, total_branching_energy_init, total_branching_energy_final
