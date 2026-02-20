# C++ Function Reference

Generated automatically from Doxygen-style docblocks in `.cpp` files.

**Legend:** 🔸 Purpose  |  ⬇️ Inputs  |  ⬆️ Outputs  |  📝 Notes  |  🔗 Click function names to open source

## 📄 bcorner_move.cpp

### 🔹 [bcorner_move](../../bcorner_move.cpp#L17)
**Type:** `agent`  
**Source:** [Open bcorner_move.cpp:17](../../bcorner_move.cpp#L17)

- 🔸 **Purpose:** Synchronize each BCORNER agent position with the current domain boundary coordinates.
- ⬇️ **Inputs:**
  - Agent variable: id
  - Environment properties: COORDS_BOUNDARIES[6]
- ⬆️ **Outputs:**
  - Updated BCORNER position (x, y, z)
- 📝 **Notes:**
  - BCORNER ids 1..8 map to the eight corners of the simulation box.
- - -

## 📄 bcorner_output_location_data.cpp

### 🔹 [bcorner_output_location_data](../../bcorner_output_location_data.cpp#L13)
**Type:** `agent`  
**Source:** [Open bcorner_output_location_data.cpp:13](../../bcorner_output_location_data.cpp#L13)

- 🔸 **Purpose:** Publish BCORNER identifiers and coordinates to spatial messages.
- ⬇️ **Inputs:**
  - Agent variables: id, x, y, z
- ⬆️ **Outputs:**
  - MessageSpatial3D payload for downstream consumers
- - -

## 📄 cell_bucket_location_data.cpp

### 🔹 [cell_bucket_location_data](../../cell_bucket_location_data.cpp#L13)
**Type:** `agent`  
**Source:** [Open cell_bucket_location_data.cpp:13](../../cell_bucket_location_data.cpp#L13)

- 🔸 **Purpose:** Export CELL state required by bucket-based readers (e.g., focal adhesion updates).
- ⬇️ **Inputs:**
  - CELL variables: id, position, orientation, anchor arrays
- ⬆️ **Outputs:**
  - MessageBucket keyed by CELL id containing anchor geometry and pose
- - -

## 📄 cell_ecm_interaction_metabolism.cpp

### 🔹 [cell_ecm_interaction_metabolism](../../cell_ecm_interaction_metabolism.cpp#L16)
**Type:** `agent`  
**Source:** [Open cell_ecm_interaction_metabolism.cpp:16](../../cell_ecm_interaction_metabolism.cpp#L16)

- 🔸 **Purpose:** Couple each CELL to its nearest ECM voxel for species exchange and run intracellular metabolic reactions with mass-consistent updates.
- ⬇️ **Inputs:**
  - CELL position/volume and metabolic rate arrays
  - ECM voxel concentration fields read from Array3D
- ⬆️ **Outputs:**
  - Updated CELL species amounts/concentrations
  - Atomic updates to ECM concentration macro-property (C_SP_MACRO)
- - -

## 📄 cell_move.cpp

### 🔹 [cell_move](../../cell_move.cpp#L46)
**Type:** `agent`  
**Source:** [Open cell_move.cpp:46](../../cell_move.cpp#L46)

- 🔸 **Purpose:** Update CELL velocity/orientation-driven migration by combining Brownian, chemotactic, and durotactic components, then advance position.
- ⬇️ **Inputs:**
  - CELL kinematic state, stress/strain eigensystem, chemotaxis sensitivities
  - Environment controls for chemotaxis/durotaxis and timestep
- ⬆️ **Outputs:**
  - Updated CELL position, velocity, orientation-aligned motion state
- - -

## 📄 cell_spatial_location_data.cpp

### 🔹 [cell_spatial_location_data](../../cell_spatial_location_data.cpp#L14)
**Type:** `agent`  
**Source:** [Open cell_spatial_location_data.cpp:14](../../cell_spatial_location_data.cpp#L14)

- 🔸 **Purpose:** Broadcast CELL kinematics and metabolic parameters over a spatial message list.
- ⬇️ **Inputs:**
  - CELL variables: id, x,y,z, vx,vy,vz
  - Species arrays: k_consumption, k_production, k_reaction, C_sp, M_sp
- ⬆️ **Outputs:**
  - MessageSpatial3D record for nearby agent queries
- - -

## 📄 cell_update_stress.cpp

### 🔹 [clampf](../../cell_update_stress.cpp#L6)
**Type:** `helper`  
**Source:** [Open cell_update_stress.cpp:6](../../cell_update_stress.cpp#L6)

- 🔸 **Purpose:** Clamps a scalar to the closed interval [lo, hi].
- - -

### 🔹 [safeInv](../../cell_update_stress.cpp#L16)
**Type:** `helper`  
**Source:** [Open cell_update_stress.cpp:16](../../cell_update_stress.cpp#L16)

- 🔸 **Purpose:** Returns 1/x when |x| > eps, otherwise returns 0.
- - -

### 🔹 [normalize3](../../cell_update_stress.cpp#L25)
**Type:** `helper`  
**Source:** [Open cell_update_stress.cpp:25](../../cell_update_stress.cpp#L25)

- 🔸 **Purpose:** Normalizes a 3D vector in-place; if near-zero, sets a default unit vector.
- - -

### 🔹 [swapf](../../cell_update_stress.cpp#L44)
**Type:** `helper`  
**Source:** [Open cell_update_stress.cpp:44](../../cell_update_stress.cpp#L44)

- 🔸 **Purpose:** Swaps two floats by reference.
- - -

### 🔹 [swap_col3](../../cell_update_stress.cpp#L55)
**Type:** `helper`  
**Source:** [Open cell_update_stress.cpp:55](../../cell_update_stress.cpp#L55)

- 🔸 **Purpose:** Swaps two columns of a 3x3 matrix (used for eigenvector column reordering).
- - -

### 🔹 [eig_sym_3x3](../../cell_update_stress.cpp#L78)
**Type:** `helper`  
**Source:** [Open cell_update_stress.cpp:78](../../cell_update_stress.cpp#L78)

- 🔸 **Purpose:** Jacobi eigendecomposition for a real symmetric 3x3 matrix:
- ⬆️ **Outputs:**
  - Eigenvalues l1, l2, l3
  - Corresponding eigenvectors (v1, v2, v3)
- - -

### 🔹 [cell_update_stress](../../cell_update_stress.cpp#L198)
**Type:** `agent`  
**Source:** [Open cell_update_stress.cpp:198](../../cell_update_stress.cpp#L198)

- 🔸 **Purpose:** Reads all focal adhesion (FOCAD) messages in a bucket keyed by this cell id.
- - -

## 📄 ecm_Csp_update.cpp

### 🔹 [ecm_Csp_update](../../ecm_Csp_update.cpp#L18)
**Type:** `agent`  
**Source:** [Open ecm_Csp_update.cpp:18](../../ecm_Csp_update.cpp#L18)

- 🔸 **Purpose:** Refresh each ECM voxel concentration array from the global macro property buffer.
- ⬇️ **Inputs:**
  - Agent variable: grid_lin_id
  - Environment macro property: C_SP_MACRO
- ⬆️ **Outputs:**
  - Updated per-agent C_sp array
- 📝 **Notes:**
  - This is the synchronization bridge from macro-level concentration updates
  - back into per-agent concentration variables.
- - -

## 📄 ecm_Dsp_update.cpp

### 🔹 [ecm_Dsp_update](../../ecm_Dsp_update.cpp#L33)
**Type:** `agent`  
**Source:** [Open ecm_Dsp_update.cpp:33](../../ecm_Dsp_update.cpp#L33)

- 🔸 **Purpose:** Compute local FNODE crowding around each ECM voxel and downscale diffusion coefficients to represent heterogeneous transport in dense regions.
- ⬇️ **Inputs:**
  - Spatial FNODE messages around each ECM position
  - Environment controls: equilibrium distance, average voxel density
- ⬆️ **Outputs:**
  - Updated D_sp array per ECM agent
- - -

## 📄 ecm_boundary_concentration_conditions.cpp

### 🔹 [ecm_boundary_concentration_conditions](../../ecm_boundary_concentration_conditions.cpp#L14)
**Type:** `agent`  
**Source:** [Open ecm_boundary_concentration_conditions.cpp:14](../../ecm_boundary_concentration_conditions.cpp#L14)

- 🔸 **Purpose:** Apply boundary concentration conditions to ECM agents located near domain faces.
- ⬇️ **Inputs:**
  - ECM position and current species concentrations
  - Boundary positions and boundary concentration macro properties
- ⬆️ **Outputs:**
  - Updated agent C_sp and synchronized C_SP_MACRO values for touched boundaries
- - -

## 📄 ecm_ecm_interaction.cpp

### 🔹 [ecm_ecm_interaction](../../ecm_ecm_interaction.cpp#L56)
**Type:** `agent`  
**Source:** [Open ecm_ecm_interaction.cpp:56](../../ecm_ecm_interaction.cpp#L56)

- 🔸 **Purpose:** Execute ECM voxel-to-voxel mechanical coupling and multi-species diffusion on the same neighborhood pass.
- ⬇️ **Inputs:**
  - Array3D ECM neighborhood messages (positions, velocities, concentrations)
  - Environment controls for diffusion mode, timestep, and mechanics
- ⬆️ **Outputs:**
  - Updated ECM mechanical forces (fx, fy, fz)
  - Updated concentration state C_sp and C_SP_MACRO entries
- 📝 **Notes:**
  - Includes the semi-implicit diffusion branch used to prevent unstable Euler
  - blow-up when diffusion CFL-like conditions are violated.
- - -

## 📄 ecm_grid_location_data.cpp

### 🔹 [ecm_grid_location_data](../../ecm_grid_location_data.cpp#L14)
**Type:** `agent`  
**Source:** [Open ecm_grid_location_data.cpp:14](../../ecm_grid_location_data.cpp#L14)

- 🔸 **Purpose:** Publish ECM voxel-centered state into the Array3D message for neighborhood reads.
- ⬇️ **Inputs:**
  - ECM grid coordinates and linear index
  - Mechanical and diffusion-related voxel variables
- ⬆️ **Outputs:**
  - MessageArray3D entry indexed by (grid_i, grid_j, grid_k)
- - -

## 📄 ecm_move.cpp

### 🔹 [boundPosition](../../ecm_move.cpp#L8)
**Type:** `helper`  
**Source:** [Open ecm_move.cpp:8](../../ecm_move.cpp#L8)

- 🔸 **Purpose:** Clamp ECM agent coordinates against moving boundaries and update per-axis clamp flags when contact conditions are met.
- - -

### 🔹 [ecm_move](../../ecm_move.cpp#L104)
**Type:** `agent`  
**Source:** [Open ecm_move.cpp:104](../../ecm_move.cpp#L104)

- 🔸 **Purpose:** Advance ECM agent motion from accumulated forces, then enforce boundary clamping/sliding rules and boundary-driven kinematics.
- ⬇️ **Inputs:**
  - ECM force, velocity, clamp state, and boundary/environment parameters
- ⬆️ **Outputs:**
  - Updated position, velocity, clamp flags, and boundary force channels
- - -

## 📄 fnode_boundary_interaction.cpp

### 🔹 [fnode_boundary_interaction](../../fnode_boundary_interaction.cpp#L15)
**Type:** `agent`  
**Source:** [Open fnode_boundary_interaction.cpp:15](../../fnode_boundary_interaction.cpp#L15)

- 🔸 **Purpose:** Compute boundary reaction forces on FNODE agents near domain boundaries, including optional elastic and damping contributions per face.
- ⬇️ **Inputs:**
  - FNODE position/velocity
  - Boundary coordinates, stiffness/damping, and movement settings
- ⬆️ **Outputs:**
  - boundary_fx, boundary_fy, boundary_fz stored on each FNODE
- - -

## 📄 fnode_bucket_location_data.cpp

### 🔹 [fnode_bucket_location_data](../../fnode_bucket_location_data.cpp#L14)
**Type:** `agent`  
**Source:** [Open fnode_bucket_location_data.cpp:14](../../fnode_bucket_location_data.cpp#L14)

- 🔸 **Purpose:** Export FNODE state and connectivity arrays into a bucket message keyed by node id.
- ⬇️ **Inputs:**
  - FNODE kinematics and material parameters
  - Connectivity arrays: linked_nodes, equilibrium_distance
- ⬆️ **Outputs:**
  - MessageBucket record for direct id-based neighbor access
- - -

## 📄 fnode_fnode_bucket_interaction.cpp

### 🔹 [fnode_fnode_bucket_interaction](../../fnode_fnode_bucket_interaction.cpp#L68)
**Type:** `agent`  
**Source:** [Open fnode_fnode_bucket_interaction.cpp:68](../../fnode_fnode_bucket_interaction.cpp#L68)

- 🔸 **Purpose:** Compute spring-damper forces along explicit FNODE connectivity links and accumulate network mechanical metrics (extension/compression/elastic energy).
- ⬇️ **Inputs:**
  - Bucket messages keyed by linked node ids
  - Connectivity arrays and per-link equilibrium distances
- ⬆️ **Outputs:**
  - Updated FNODE force components and mechanical summary variables
- - -

## 📄 fnode_fnode_spatial_interaction.cpp

### 🔹 [fnode_fnode_spatial_interaction](../../fnode_fnode_spatial_interaction.cpp#L49)
**Type:** `agent`  
**Source:** [Open fnode_fnode_spatial_interaction.cpp:49](../../fnode_fnode_spatial_interaction.cpp#L49)

- 🔸 **Purpose:** Apply short-range repulsion between nearby FNODE agents to prevent overlap.
- ⬇️ **Inputs:**
  - Spatial FNODE neighbor messages
  - Environment parameters: MAX_SEARCH_RADIUS_FNODES, FIBRE_NODE_REPULSION_K
- ⬆️ **Outputs:**
  - Updated repulsive force components (fx, fy, fz) on each FNODE
- - -

## 📄 fnode_focad_interaction.cpp

### 🔹 [fnode_focad_interaction](../../fnode_focad_interaction.cpp#L18)
**Type:** `agent`  
**Source:** [Open fnode_focad_interaction.cpp:18](../../fnode_focad_interaction.cpp#L18)

- 🔸 **Purpose:** Transfer precomputed FOCAD traction forces onto the corresponding FNODE.
- ⬇️ **Inputs:**
  - Spatial FOCAD messages containing force and attachment status
  - FNODE id/position/force state
- ⬆️ **Outputs:**
  - Updated FNODE force components (fx, fy, fz)
- 📝 **Notes:**
  - This function is scheduled after focad_fnode_interaction, which computes
  - and stores the adhesion force on each FOCAD agent.
- - -

## 📄 fnode_move.cpp

### 🔹 [boundPosition](../../fnode_move.cpp#L8)
**Type:** `helper`  
**Source:** [Open fnode_move.cpp:8](../../fnode_move.cpp#L8)

- 🔸 **Purpose:** Clamp FNODE coordinates near boundaries and update clamp state flags based on contact and configuration flags.
- - -

### 🔹 [fnode_move](../../fnode_move.cpp#L104)
**Type:** `agent`  
**Source:** [Open fnode_move.cpp:104](../../fnode_move.cpp#L104)

- 🔸 **Purpose:** Update FNODE positions/velocities under internal, boundary, and transmitted forces while enforcing clamp and sliding boundary behavior.
- ⬇️ **Inputs:**
  - FNODE force channels (network + boundary), current kinematics, clamp flags
  - Boundary movement/clamping parameters from the environment
- ⬆️ **Outputs:**
  - Updated node kinematics, clamp state, and boundary force contributions
- - -

## 📄 fnode_spatial_location_data.cpp

### 🔹 [fnode_spatial_location_data](../../fnode_spatial_location_data.cpp#L13)
**Type:** `agent`  
**Source:** [Open fnode_spatial_location_data.cpp:13](../../fnode_spatial_location_data.cpp#L13)

- 🔸 **Purpose:** Broadcast FNODE position for spatial proximity queries.
- ⬇️ **Inputs:**
  - FNODE variables: id, x, y, z
- ⬆️ **Outputs:**
  - MessageSpatial3D payload used by FNODE/FOCAD interaction kernels
- - -

## 📄 focad_anchor_update.cpp

### 🔹 [focad_anchor_update](../../focad_anchor_update.cpp#L18)
**Type:** `agent`  
**Source:** [Open focad_anchor_update.cpp:18](../../focad_anchor_update.cpp#L18)

- 🔸 **Purpose:** Re-anchor each FOCAD agent to a CELL nucleus anchor point read from bucket messages keyed by cell_id.
- ⬇️ **Inputs:**
  - MessageBucket from CELL containing nucleus pose and anchor arrays
  - Current FOCAD position and cell association
- ⬆️ **Outputs:**
  - Updated FOCAD nucleus center/orientation and selected anchor (x_i,y_i,z_i)
- 📝 **Notes:**
  - If no fixed anchor_id exists, the closest anchor point is selected each step.
- - -

## 📄 focad_bucket_location_data.cpp

### 🔹 [focad_bucket_location_data](../../focad_bucket_location_data.cpp#L13)
**Type:** `agent`  
**Source:** [Open focad_bucket_location_data.cpp:13](../../focad_bucket_location_data.cpp#L13)

- 🔸 **Purpose:** Publish full FOCAD state for bucket-keyed readers (mainly CELL/FOCAD coupling steps).
- ⬇️ **Inputs:**
  - FOCAD identifiers, kinematics, mechanics, lifecycle flags and timers
- ⬆️ **Outputs:**
  - MessageBucket keyed by cell_id with adhesion state and force data
- - -

## 📄 focad_fnode_interaction.cpp

### 🔹 [focad_fnode_interaction](../../focad_fnode_interaction.cpp#L20)
**Type:** `agent`  
**Source:** [Open focad_fnode_interaction.cpp:20](../../focad_fnode_interaction.cpp#L20)

- 🔸 **Purpose:** Manage FOCAD-FNODE attachment dynamics and compute traction forces stored on FOCAD for subsequent FNODE-side force transfer.
- ⬇️ **Inputs:**
  - Spatial FNODE messages near each FOCAD
  - FOCAD mechanics/lifecycle state and environment kinetics parameters
- ⬆️ **Outputs:**
  - Updated adhesion attachment state, lifecycle timers/state, and force
- 📝 **Notes:**
  - Scheduled before fnode_focad_interaction so computed traction can be read
  - and applied to the linked FNODE.
- - -

## 📄 focad_move.cpp

### 🔹 [focad_move](../../focad_move.cpp#L16)
**Type:** `agent`  
**Source:** [Open focad_move.cpp:16](../../focad_move.cpp#L16)

- 🔸 **Purpose:** Update focal adhesion positions by either following attached FNODEs or executing bounded exploratory motion when detached/inactive.
- ⬇️ **Inputs:**
  - FOCAD state: attachment flags, anchor position, velocity, fnode_id
  - FNODE bucket messages (for attached movement)
  - Domain and adhesion motion constraints from environment
- ⬆️ **Outputs:**
  - Updated FOCAD position/velocity within boundary limits
- - -

## 📄 focad_spatial_location_data.cpp

### 🔹 [focad_spatial_location_data](../../focad_spatial_location_data.cpp#L13)
**Type:** `agent`  
**Source:** [Open focad_spatial_location_data.cpp:13](../../focad_spatial_location_data.cpp#L13)

- 🔸 **Purpose:** Broadcast active adhesion position/force state for local spatial interaction queries.
- ⬇️ **Inputs:**
  - FOCAD variables: id, x,y,z, fx,fy,fz, fnode_id, attached, active
- ⬆️ **Outputs:**
  - MessageSpatial3D payload consumed by FNODE-side force transfer
- - -

## 📄 handy_device_functions_template.cpp

### 🔹 [vec3CrossProd](../../handy_device_functions_template.cpp#L15)
**Type:** `helper`  
**Source:** [Open handy_device_functions_template.cpp:15](../../handy_device_functions_template.cpp#L15)

- 🔸 **Purpose:** Provide reusable device-side vector algebra helpers for interaction kernels.
- 📝 **Notes:**
  - This file is a template/reference module and is intended for copy-paste use
  - inside runtime-compiled FLAMEGPU agent function files.
  - vec3CrossProd: compute cross product (x1,y1,z1) x (x2,y2,z2).
- - -
