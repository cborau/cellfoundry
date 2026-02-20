# C++ Function Reference

Generated automatically from Doxygen-style docblocks in `.cpp` files.

## bcorner_move.cpp

### bcorner_move (agent)

- **Purpose:**   Synchronize each BCORNER agent position with the current domain boundary coordinates.
- **Notes:**
  -   BCORNER ids 1..8 map to the eight corners of the simulation box.

## bcorner_output_location_data.cpp

### bcorner_output_location_data (agent)

- **Purpose:**   Publish BCORNER identifiers and coordinates to spatial messages.

## cell_bucket_location_data.cpp

### cell_bucket_location_data (agent)

- **Purpose:**   Export CELL state required by bucket-based readers (e.g., focal adhesion updates).

## cell_ecm_interaction_metabolism.cpp

### cell_ecm_interaction_metabolism (agent)

- **Purpose:** Couple each CELL to its nearest ECM voxel for species exchange and run   intracellular metabolic reactions with mass-consistent updates.

## cell_move.cpp

### cell_move (agent)

- **Purpose:** Update CELL velocity/orientation-driven migration by combining Brownian,   chemotactic, and durotactic components, then advance position.

## cell_spatial_location_data.cpp

### cell_spatial_location_data (agent)

- **Purpose:**   Broadcast CELL kinematics and metabolic parameters over a spatial message list.

## cell_update_stress.cpp

### clampf (helper)

- **Purpose:** Clamps a scalar to the closed interval [lo, hi].

### safeInv (helper)

- **Purpose:** Returns 1/x when |x| > eps, otherwise returns 0.

### normalize3 (helper)

- **Purpose:** Normalizes a 3D vector in-place; if near-zero, sets a default unit vector.

### swapf (helper)

- **Purpose:** Swaps two floats by reference.

### swap_col3 (helper)

- **Purpose:** Swaps two columns of a 3x3 matrix (used for eigenvector column reordering).

### eig_sym_3x3 (helper)

- **Purpose:** Jacobi eigendecomposition for a real symmetric 3x3 matrix:

### cell_update_stress (agent)

- **Purpose:** Reads all focal adhesion (FOCAD) messages in a bucket keyed by this cell id.

## ecm_Csp_update.cpp

### ecm_Csp_update (agent)

- **Purpose:**   Refresh each ECM voxel concentration array from the global macro property buffer.
- **Notes:**
  -   This is the synchronization bridge from macro-level concentration updates
  -   back into per-agent concentration variables.

## ecm_Dsp_update.cpp

### ecm_Dsp_update (agent)

- **Purpose:** Compute local FNODE crowding around each ECM voxel and downscale diffusion   coefficients to represent heterogeneous transport in dense regions.

## ecm_boundary_concentration_conditions.cpp

### ecm_boundary_concentration_conditions (agent)

- **Purpose:**   Apply boundary concentration conditions to ECM agents located near domain faces.

## ecm_ecm_interaction.cpp

### ecm_ecm_interaction (agent)

- **Purpose:** Execute ECM voxel-to-voxel mechanical coupling and multi-species diffusion   on the same neighborhood pass.
- **Notes:**
  -   Includes the semi-implicit diffusion branch used to prevent unstable Euler
  -   blow-up when diffusion CFL-like conditions are violated.

## ecm_grid_location_data.cpp

### ecm_grid_location_data (agent)

- **Purpose:**   Publish ECM voxel-centered state into the Array3D message for neighborhood reads.

## ecm_move.cpp

### boundPosition (helper)

- **Purpose:** Clamp ECM agent coordinates against moving boundaries and update per-axis   clamp flags when contact conditions are met.

### ecm_move (agent)

- **Purpose:** Advance ECM agent motion from accumulated forces, then enforce boundary   clamping/sliding rules and boundary-driven kinematics.

## fnode_boundary_interaction.cpp

### fnode_boundary_interaction (agent)

- **Purpose:** Compute boundary reaction forces on FNODE agents near domain boundaries,   including optional elastic and damping contributions per face.

## fnode_bucket_location_data.cpp

### fnode_bucket_location_data (agent)

- **Purpose:**   Export FNODE state and connectivity arrays into a bucket message keyed by node id.

## fnode_fnode_bucket_interaction.cpp

### fnode_fnode_bucket_interaction (agent)

- **Purpose:** Compute spring-damper forces along explicit FNODE connectivity links and   accumulate network mechanical metrics (extension/compression/elastic energy).

## fnode_fnode_spatial_interaction.cpp

### fnode_fnode_spatial_interaction (agent)

- **Purpose:**   Apply short-range repulsion between nearby FNODE agents to prevent overlap.

## fnode_focad_interaction.cpp

### fnode_focad_interaction (agent)

- **Purpose:**   Transfer precomputed FOCAD traction forces onto the corresponding FNODE.
- **Notes:**
  -   This function is scheduled after focad_fnode_interaction, which computes
  -   and stores the adhesion force on each FOCAD agent.

## fnode_move.cpp

### boundPosition (helper)

- **Purpose:** Clamp FNODE coordinates near boundaries and update clamp state flags based   on contact and configuration flags.

### fnode_move (agent)

- **Purpose:** Update FNODE positions/velocities under internal, boundary, and transmitted   forces while enforcing clamp and sliding boundary behavior.

## fnode_spatial_location_data.cpp

### fnode_spatial_location_data (agent)

- **Purpose:**   Broadcast FNODE position for spatial proximity queries.

## focad_anchor_update.cpp

### focad_anchor_update (agent)

- **Purpose:** Re-anchor each FOCAD agent to a CELL nucleus anchor point read from   bucket messages keyed by cell_id.
- **Notes:**
  -   If no fixed anchor_id exists, the closest anchor point is selected each step.

## focad_bucket_location_data.cpp

### focad_bucket_location_data (agent)

- **Purpose:**   Publish full FOCAD state for bucket-keyed readers (mainly CELL/FOCAD coupling steps).

## focad_fnode_interaction.cpp

### focad_fnode_interaction (agent)

- **Purpose:** Manage FOCAD-FNODE attachment dynamics and compute traction forces stored   on FOCAD for subsequent FNODE-side force transfer.
- **Notes:**
  -   Scheduled before fnode_focad_interaction so computed traction can be read
  -   and applied to the linked FNODE.

## focad_move.cpp

### focad_move (agent)

- **Purpose:** Update focal adhesion positions by either following attached FNODEs or   executing bounded exploratory motion when detached/inactive.

## focad_spatial_location_data.cpp

### focad_spatial_location_data (agent)

- **Purpose:**   Broadcast active adhesion position/force state for local spatial interaction queries.

## handy_device_functions_template.cpp

### vec3CrossProd (helper)

- **Purpose:**   Provide reusable device-side vector algebra helpers for interaction kernels.
- **Notes:**
  -   This file is a template/reference module and is intended for copy-paste use
  -   inside runtime-compiled FLAMEGPU agent function files.
  - vec3CrossProd: compute cross product (x1,y1,z1) x (x2,y2,z2).
