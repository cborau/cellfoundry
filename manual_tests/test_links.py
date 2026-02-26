
import numpy as np

ids = [1, 2, 6, 4, 5, 7]
linked_nodes_all = [[2, -1, -1, -1],
                    [1, 6, -1, -1],
                    [2, 4, -1, -1],
                    [6, 5, 7, -1],
                    [4, -1, -1, -1],
                    [4, -1, -1, -1]]

min_id = min(ids)
# subtract min_id from  ids to convert to 0-based indexing (skip negative values)
ids = [fid - min_id for fid in ids if fid > 0]

# subtract min_id from linked ids to convert to 0-based indexing (don't subtract from negative values)
for i in range(len(linked_nodes_all)):
    linked_nodes_all[i] = [linked_id - min_id for linked_id in linked_nodes_all[i] if linked_id > 0] + [linked_id for linked_id in linked_nodes_all[i] if linked_id <= 0]

print("ids:", ids)
print("linked_nodes_all:", linked_nodes_all)

sorted_indices = np.argsort(ids)
ids = [ids[i] for i in sorted_indices]
linked_nodes_all = [linked_nodes_all[i] for i in sorted_indices]

print("Sorted ids:", ids)
print("Sorted linked_nodes_all:", linked_nodes_all)

n_fnodes = len(ids)
id_to_point_idx = {fid: i for i, fid in enumerate(ids)}
print("id_to_point_idx:", id_to_point_idx)



added_lines = set()
cell_connectivity = []
for i, fid in enumerate(ids):
    links = linked_nodes_all[i]
    for linked_id_raw in links:
        linked_id = int(round(linked_id_raw))
        if linked_id < 0 or linked_id == fid:
            continue
        if linked_id in id_to_point_idx:
            line = tuple(sorted((id_to_point_idx[fid], id_to_point_idx[linked_id])))
            if line not in added_lines:
                added_lines.add(line)
                cell_connectivity.append(line)
                
print("Cell connectivity (lines):", cell_connectivity)