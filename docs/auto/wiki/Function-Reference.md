# C++ Function Reference

Generated automatically from Doxygen-style docblocks in `.cpp` files.

**Legend:** 🔸 Purpose  |  ⬇️ Inputs  |  ⬆️ Outputs  |  📝 Notes  |  🔗 Click function names to open source

## 📄 bcorner_move.cpp

### 🔹 [bcorner_move](https://github.com/cborau/cellfoundry/blob/master/bcorner_move.cpp)
**Type:** `agent`  
**Source:** [Open bcorner_move.cpp](https://github.com/cborau/cellfoundry/blob/master/bcorner_move.cpp)

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

### 🔹 [bcorner_output_location_data](https://github.com/cborau/cellfoundry/blob/master/bcorner_output_location_data.cpp)
**Type:** `agent`  
**Source:** [Open bcorner_output_location_data.cpp](https://github.com/cborau/cellfoundry/blob/master/bcorner_output_location_data.cpp)

- 🔸 **Purpose:** Publish BCORNER identifiers and coordinates to spatial messages.
- ⬇️ **Inputs:**
  - Agent variables: id, x, y, z
- ⬆️ **Outputs:**
  - MessageSpatial3D payload for downstream consumers
- - -

## 📄 cell_MaxID_update.cpp

### 🔹 [cell_MaxID_update](https://github.com/cborau/cellfoundry/blob/master/cell_MaxID_update.cpp)
**Type:** `agent`  
**Source:** [Open cell_MaxID_update.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_MaxID_update.cpp)

- 🔸 **Purpose:** Synchronize the agent's max_global_cell_id variable with the environment macro property.
- ⬇️ **Inputs:**
  - Environment macro property: MACRO_MAX_GLOBAL_CELL_ID
- ⬆️ **Outputs:**
  - Updated agent variable: max_global_cell_id
- - -

## 📄 cell_bucket_location_data.cpp

### 🔹 [cell_bucket_location_data](https://github.com/cborau/cellfoundry/blob/master/cell_bucket_location_data.cpp)
**Type:** `agent`  
**Source:** [Open cell_bucket_location_data.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_bucket_location_data.cpp)

- 🔸 **Purpose:** Export CELL state required by bucket-based readers (e.g., focal adhesion updates).
- ⬇️ **Inputs:**
  - CELL variables: id, position, orientation, anchor arrays
- ⬆️ **Outputs:**
  - MessageBucket keyed by CELL id containing anchor geometry and pose
- - -

## 📄 cell_cell_interaction.cpp

### 🔹 [cc_clampf](https://github.com/cborau/cellfoundry/blob/master/cell_cell_interaction.cpp)
**Type:** `helper`  
**Source:** [Open cell_cell_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_cell_interaction.cpp)

- 🔸 **Purpose:** Clamp a scalar to the closed interval [lo, hi].
- ⬇️ **Inputs:**
  - x: value to clamp
  - lo: lower bound
  - hi: upper bound
- ⬆️ **Outputs:**
  - Returns clamped value
- - -

### 🔹 [cc_normalize3](https://github.com/cborau/cellfoundry/blob/master/cell_cell_interaction.cpp)
**Type:** `helper`  
**Source:** [Open cell_cell_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_cell_interaction.cpp)

- 🔸 **Purpose:** Normalize a 3D vector in-place; if near-zero, sets a default unit vector.
- ⬇️ **Inputs:**
  - x, y, z: vector components (modified)
- ⬆️ **Outputs:**
  - x, y, z: normalized vector components
- - -

### 🔹 [cell_cell_interaction](https://github.com/cborau/cellfoundry/blob/master/cell_cell_interaction.cpp)
**Type:** `agent`  
**Source:** [Open cell_cell_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_cell_interaction.cpp)

- 🔸 **Purpose:** Compute short-range CELL-CELL mechanics with strong contact repulsion and weak finite-range adhesion shell (soft cohesion) to promote aggregate compactness while allowing escape under other motility cues.
- ⬇️ **Inputs:**
  - cell_spatial_location_message (spatial neighbors)
  - Environment interaction parameters
- ⬆️ **Outputs:**
  - Per-cell interaction velocity contribution (cc_dv*) [um/s]
- - -

## 📄 cell_cycle.cpp

### 🔹 [vec3Div](https://github.com/cborau/cellfoundry/blob/master/cell_cycle.cpp)
**Type:** `helper`  
**Source:** [Open cell_cycle.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_cycle.cpp)

- 🔸 **Purpose:** Divide a 3D vector (x, y, z) by a scalar divisor in-place.
- ⬇️ **Inputs:**
  - x, y, z: vector components (modified)
  - divisor: scalar value
- ⬆️ **Outputs:**
  - x, y, z: scaled vector components
- - -

### 🔹 [vec3Length](https://github.com/cborau/cellfoundry/blob/master/cell_cycle.cpp)
**Type:** `helper`  
**Source:** [Open cell_cycle.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_cycle.cpp)

- 🔸 **Purpose:** Compute the Euclidean length of a 3D vector (x, y, z).
- ⬇️ **Inputs:**
  - x, y, z: vector components
- ⬆️ **Outputs:**
  - Returns the magnitude of the vector
- - -

### 🔹 [cell_cycle](https://github.com/cborau/cellfoundry/blob/master/cell_cycle.cpp)
**Type:** `agent`  
**Source:** [Open cell_cycle.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_cycle.cpp)

- 🔸 **Purpose:** Agent function for cell cycle progression, division, and death. Handles cell phase transitions, damage accumulation, and division logic.
- ⬇️ **Inputs:**
  - CELL agent variables: id, cell_type, position, orientation, metabolic rates, anchor arrays, etc.
  - Environment properties: cell cycle timings, thresholds, rates
- ⬆️ **Outputs:**
  - Updated cell state variables (phase, clock, damage, division, daughter creation)
- 📝 **Notes:**
  - Dead cells may remain ALIVE for agent purposes if DEAD_CELLS_DISAPPEAR is set.
  - Division logic includes randomization and mass/anchor inheritance.
- - -

## 📄 cell_ecm_interaction_metabolism.cpp

### 🔹 [cell_ecm_interaction_metabolism](https://github.com/cborau/cellfoundry/blob/master/cell_ecm_interaction_metabolism.cpp)
**Type:** `agent`  
**Source:** [Open cell_ecm_interaction_metabolism.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_ecm_interaction_metabolism.cpp)

- 🔸 **Purpose:** Couple each CELL to its nearest ECM voxel for species exchange and run intracellular metabolic reactions with mass-consistent updates.
- ⬇️ **Inputs:**
  - CELL position/volume and metabolic rate arrays
  - ECM voxel concentration fields read from Array3D
- ⬆️ **Outputs:**
  - Updated CELL species amounts/concentrations
  - Atomic updates to ECM concentration macro-property (C_SP_MACRO)
- - -

## 📄 cell_fnode_remodel.cpp

### 🔹 [cell_fnode_remodel](https://github.com/cborau/cellfoundry/blob/master/cell_fnode_remodel.cpp)
**Type:** `agent`  
**Source:** [Open cell_fnode_remodel.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_fnode_remodel.cpp)

- 🔸 **Purpose:** Probabilistically create a single FNODE around a CELL and request reciprocal parent-link update through environment macros.
- ⬇️ **Inputs:**
  - Nearby FNODE spatial messages (id, x, y, z, connectivity_count)
  - CELL state (position, stress proxy, concentration, dead)
  - Remodeling environment properties and macro properties
- ⬆️ **Outputs:**
  - Optional newborn FNODE via agent output (max 1 per CELL per step)
  - Updated CELL cooldown variable `fnode_birth_cooldown`
- - -

## 📄 cell_fnode_repulsion.cpp

### 🔹 [cell_fnode_repulsion](https://github.com/cborau/cellfoundry/blob/master/cell_fnode_repulsion.cpp)
**Type:** `agent`  
**Source:** [Open cell_fnode_repulsion.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_fnode_repulsion.cpp)

- 🔸 **Purpose:** Prevent CELL centers from approaching FNODE points closer than an exclusion distance by adding a short-range repulsive velocity component.
- ⬇️ **Inputs:**
  - fnode_spatial_location_message (spatial neighbors)
  - Environment interaction parameters
- ⬆️ **Outputs:**
  - Per-cell FNODE interaction velocity contribution (cf_dv*) [um/s]
- - -

## 📄 cell_focad_update.cpp

### 🔹 [clampf](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)
**Type:** `helper`  
**Source:** [Open cell_focad_update.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)

- 🔸 **Purpose:** Clamps a scalar to the closed interval [lo, hi].
- - -

### 🔹 [safeInv](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)
**Type:** `helper`  
**Source:** [Open cell_focad_update.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)

- 🔸 **Purpose:** Returns 1/x when |x| > eps, otherwise returns 0.
- - -

### 🔹 [normalize3](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)
**Type:** `helper`  
**Source:** [Open cell_focad_update.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)

- 🔸 **Purpose:** Normalizes a 3D vector in-place; if near-zero, sets a default unit vector.
- - -

### 🔹 [swapf](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)
**Type:** `helper`  
**Source:** [Open cell_focad_update.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)

- 🔸 **Purpose:** Swaps two floats by reference.
- - -

### 🔹 [swap_col3](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)
**Type:** `helper`  
**Source:** [Open cell_focad_update.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)

- 🔸 **Purpose:** Swaps two columns of a 3x3 matrix (used for eigenvector column reordering).
- - -

### 🔹 [eig_sym_3x3](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)
**Type:** `helper`  
**Source:** [Open cell_focad_update.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)

- 🔸 **Purpose:** Jacobi eigendecomposition for a real symmetric 3x3 matrix:
- ⬆️ **Outputs:**
  - Eigenvalues l1, l2, l3
  - Corresponding eigenvectors (v1, v2, v3)
- - -

### 🔹 [cell_focad_update](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)
**Type:** `agent`  
**Source:** [Open cell_focad_update.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_focad_update.cpp)

- 🔸 **Purpose:** Reads all focal adhesion (FOCAD) messages in a bucket keyed by this cell id.
- - -

## 📄 cell_move.cpp

### 🔹 [cell_move](https://github.com/cborau/cellfoundry/blob/master/cell_move.cpp)
**Type:** `agent`  
**Source:** [Open cell_move.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_move.cpp)

- 🔸 **Purpose:** Update CELL velocity/orientation-driven migration by combining Brownian, chemotactic, and durotactic components, then advance position.
- ⬇️ **Inputs:**
  - CELL kinematic state, stress/strain eigensystem, chemotaxis sensitivities
  - Environment controls for chemotaxis/durotaxis and timestep
- ⬆️ **Outputs:**
  - Updated CELL position, velocity, orientation-aligned motion state
- - -

## 📄 cell_spatial_location_data.cpp

### 🔹 [cell_spatial_location_data](https://github.com/cborau/cellfoundry/blob/master/cell_spatial_location_data.cpp)
**Type:** `agent`  
**Source:** [Open cell_spatial_location_data.cpp](https://github.com/cborau/cellfoundry/blob/master/cell_spatial_location_data.cpp)

- 🔸 **Purpose:** Broadcast CELL kinematics and metabolic parameters over a spatial message list.
- ⬇️ **Inputs:**
  - CELL variables: id, x,y,z, vx,vy,vz
  - Species arrays: k_consumption, k_production, k_reaction, C_sp, M_sp
- ⬆️ **Outputs:**
  - MessageSpatial3D record for nearby agent queries
- - -

## 📄 ecm_Csp_update.cpp

### 🔹 [ecm_Csp_update](https://github.com/cborau/cellfoundry/blob/master/ecm_Csp_update.cpp)
**Type:** `agent`  
**Source:** [Open ecm_Csp_update.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_Csp_update.cpp)

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

### 🔹 [vec3CrossProd](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)
**Type:** `helper`  
**Source:** [Open ecm_Dsp_update.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)

- 🔸 **Purpose:** Compute the cross product of two 3D vectors and store result in (x, y, z).
- ⬇️ **Inputs:**
  - x1, y1, z1: first vector
  - x2, y2, z2: second vector
- ⬆️ **Outputs:**
  - x, y, z: cross product result (modified)
- - -

### 🔹 [vec3Div](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)
**Type:** `helper`  
**Source:** [Open ecm_Dsp_update.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)

- 🔸 **Purpose:** Divide a 3D vector (x, y, z) by a scalar divisor in-place.
- ⬇️ **Inputs:**
  - x, y, z: vector components (modified)
  - divisor: scalar value
- ⬆️ **Outputs:**
  - x, y, z: scaled vector components
- - -

### 🔹 [vec3Length](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)
**Type:** `helper`  
**Source:** [Open ecm_Dsp_update.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)

- 🔸 **Purpose:** Compute the Euclidean length of a 3D vector (x, y, z).
- ⬇️ **Inputs:**
  - x, y, z: vector components
- ⬆️ **Outputs:**
  - Returns the magnitude of the vector
- - -

### 🔹 [vec3Normalize](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)
**Type:** `helper`  
**Source:** [Open ecm_Dsp_update.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)

- 🔸 **Purpose:** Normalize a 3D vector in-place using its length.
- ⬇️ **Inputs:**
  - x, y, z: vector components (modified)
- ⬆️ **Outputs:**
  - x, y, z: normalized vector components
- - -

### 🔹 [ecm_Dsp_update](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)
**Type:** `agent`  
**Source:** [Open ecm_Dsp_update.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_Dsp_update.cpp)

- 🔸 **Purpose:** Compute local FNODE crowding around each ECM voxel and downscale diffusion coefficients to represent heterogeneous transport in dense regions.
- ⬇️ **Inputs:**
  - Spatial FNODE messages around each ECM position
  - Environment controls: equilibrium distance, average voxel density
- ⬆️ **Outputs:**
  - Updated D_sp array per ECM agent
- - -

## 📄 ecm_boundary_concentration_conditions.cpp

### 🔹 [ecm_boundary_concentration_conditions](https://github.com/cborau/cellfoundry/blob/master/ecm_boundary_concentration_conditions.cpp)
**Type:** `agent`  
**Source:** [Open ecm_boundary_concentration_conditions.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_boundary_concentration_conditions.cpp)

- 🔸 **Purpose:** Apply boundary concentration conditions to ECM agents located near domain faces.
- ⬇️ **Inputs:**
  - ECM position and current species concentrations
  - Boundary positions and boundary concentration macro properties
- ⬆️ **Outputs:**
  - Updated agent C_sp and synchronized C_SP_MACRO values for touched boundaries
- - -

## 📄 ecm_ecm_interaction.cpp

### 🔹 [vec3CrossProd](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)
**Type:** `helper`  
**Source:** [Open ecm_ecm_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)

- 🔸 **Purpose:** Compute the cross product of two 3D vectors and store result in (x, y, z).
- ⬇️ **Inputs:**
  - x1, y1, z1: first vector
  - x2, y2, z2: second vector
- ⬆️ **Outputs:**
  - x, y, z: cross product result (modified)
- - -

### 🔹 [vec3Div](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)
**Type:** `helper`  
**Source:** [Open ecm_ecm_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)

- 🔸 **Purpose:** Divide a 3D vector (x, y, z) by a scalar divisor in-place.
- ⬇️ **Inputs:**
  - x, y, z: vector components (modified)
  - divisor: scalar value
- ⬆️ **Outputs:**
  - x, y, z: scaled vector components
- - -

### 🔹 [vec3Length](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)
**Type:** `helper`  
**Source:** [Open ecm_ecm_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)

- 🔸 **Purpose:** Compute the Euclidean length of a 3D vector (x, y, z).
- ⬇️ **Inputs:**
  - x, y, z: vector components
- ⬆️ **Outputs:**
  - Returns the magnitude of the vector
- - -

### 🔹 [vec3Normalize](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)
**Type:** `helper`  
**Source:** [Open ecm_ecm_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)

- 🔸 **Purpose:** Normalize a 3D vector in-place using its length.
- ⬇️ **Inputs:**
  - x, y, z: vector components (modified)
- ⬆️ **Outputs:**
  - x, y, z: normalized vector components
- - -

### 🔹 [getAngleBetweenVec](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)
**Type:** `helper`  
**Source:** [Open ecm_ecm_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)

- 🔸 **Purpose:** Compute the angle (in radians) between two 3D vectors.
- ⬇️ **Inputs:**
  - x1, y1, z1: first vector
  - x2, y2, z2: second vector
- ⬆️ **Outputs:**
  - Returns angle in radians
- - -

### 🔹 [ecm_ecm_interaction](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)
**Type:** `agent`  
**Source:** [Open ecm_ecm_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_ecm_interaction.cpp)

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

### 🔹 [ecm_grid_location_data](https://github.com/cborau/cellfoundry/blob/master/ecm_grid_location_data.cpp)
**Type:** `agent`  
**Source:** [Open ecm_grid_location_data.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_grid_location_data.cpp)

- 🔸 **Purpose:** Publish ECM voxel-centered state into the Array3D message for neighborhood reads.
- ⬇️ **Inputs:**
  - ECM grid coordinates and linear index
  - Mechanical and diffusion-related voxel variables
- ⬆️ **Outputs:**
  - MessageArray3D entry indexed by (grid_i, grid_j, grid_k)
- - -

## 📄 ecm_move.cpp

### 🔹 [boundPosition](https://github.com/cborau/cellfoundry/blob/master/ecm_move.cpp)
**Type:** `helper`  
**Source:** [Open ecm_move.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_move.cpp)

- 🔸 **Purpose:** Clamp ECM agent coordinates against moving boundaries and update per-axis clamp flags when contact conditions are met.
- - -

### 🔹 [ecm_move](https://github.com/cborau/cellfoundry/blob/master/ecm_move.cpp)
**Type:** `agent`  
**Source:** [Open ecm_move.cpp](https://github.com/cborau/cellfoundry/blob/master/ecm_move.cpp)

- 🔸 **Purpose:** Advance ECM agent motion from accumulated forces, then enforce boundary clamping/sliding rules and boundary-driven kinematics.
- ⬇️ **Inputs:**
  - ECM force, velocity, clamp state, and boundary/environment parameters
- ⬆️ **Outputs:**
  - Updated position, velocity, clamp flags, and boundary force channels
- - -

## 📄 fnode_apply_remodel_updates.cpp

### 🔹 [fnode_apply_remodel_updates](https://github.com/cborau/cellfoundry/blob/master/fnode_apply_remodel_updates.cpp)
**Type:** `agent`  
**Source:** [Open fnode_apply_remodel_updates.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_apply_remodel_updates.cpp)

- 🔸 **Purpose:** Apply remodeling topology updates and optionally remove terminally degraded nodes.
- ⬇️ **Inputs:**
  - FNODE connectivity arrays
  - Spatial FNODE messages (id, x,y,z, closest_fnode_id, second_closest_fnode_id)
- ⬆️ **Outputs:**
  - Updated `linked_nodes` / `equilibrium_distance`
  - Updated `connectivity_count`
  - Returns DEAD when `marked_for_removal` is set
- - -

## 📄 fnode_boundary_interaction.cpp

### 🔹 [fnode_boundary_interaction](https://github.com/cborau/cellfoundry/blob/master/fnode_boundary_interaction.cpp)
**Type:** `agent`  
**Source:** [Open fnode_boundary_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_boundary_interaction.cpp)

- 🔸 **Purpose:** Compute boundary reaction forces on FNODE agents near domain boundaries, including optional elastic and damping contributions per face.
- ⬇️ **Inputs:**
  - FNODE position/velocity
  - Boundary coordinates, stiffness/damping, and movement settings
- ⬆️ **Outputs:**
  - boundary_fx, boundary_fy, boundary_fz stored on each FNODE
- - -

## 📄 fnode_bucket_location_data.cpp

### 🔹 [fnode_bucket_location_data](https://github.com/cborau/cellfoundry/blob/master/fnode_bucket_location_data.cpp)
**Type:** `agent`  
**Source:** [Open fnode_bucket_location_data.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_bucket_location_data.cpp)

- 🔸 **Purpose:** Export FNODE state and connectivity arrays into a bucket message keyed by node id.
- ⬇️ **Inputs:**
  - FNODE kinematics and material parameters
  - Connectivity arrays: linked_nodes, equilibrium_distance
- ⬆️ **Outputs:**
  - MessageBucket record for direct id-based neighbor access
- - -

## 📄 fnode_cell_repulsion.cpp

### 🔹 [fnode_cell_repulsion](https://github.com/cborau/cellfoundry/blob/master/fnode_cell_repulsion.cpp)
**Type:** `agent`  
**Source:** [Open fnode_cell_repulsion.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_cell_repulsion.cpp)

- 🔸 **Purpose:** Prevent FNODE points from being pushed into CELL centers by adding a short-range repulsive force (Newton-3 counterpart of cell_fnode_repulsion).
- ⬇️ **Inputs:**
  - cell_spatial_location_message (spatial neighbors)
  - Environment interaction parameters (same as cell_fnode_repulsion)
- ⬆️ **Outputs:**
  - Updated FNODE force components (fx, fy, fz) with repulsive contributions added
- - -

## 📄 fnode_fnode_bucket_interaction.cpp

### 🔹 [fnode_fnode_bucket_interaction](https://github.com/cborau/cellfoundry/blob/master/fnode_fnode_bucket_interaction.cpp)
**Type:** `agent`  
**Source:** [Open fnode_fnode_bucket_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_fnode_bucket_interaction.cpp)

- 🔸 **Purpose:** Compute spring-damper forces along explicit FNODE connectivity links and accumulate network mechanical metrics (extension/compression/elastic energy).
- ⬇️ **Inputs:**
  - Bucket messages keyed by linked node ids
  - Connectivity arrays and per-link equilibrium distances
- ⬆️ **Outputs:**
  - Updated FNODE force components and mechanical summary variables
- - -

## 📄 fnode_fnode_spatial_interaction.cpp

### 🔹 [fnode_fnode_spatial_interaction](https://github.com/cborau/cellfoundry/blob/master/fnode_fnode_spatial_interaction.cpp)
**Type:** `agent`  
**Source:** [Open fnode_fnode_spatial_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_fnode_spatial_interaction.cpp)

- 🔸 **Purpose:** Apply short-range repulsion between nearby FNODE agents to prevent overlap.
- ⬇️ **Inputs:**
  - Spatial FNODE neighbor messages
  - Environment parameters: MAX_SEARCH_RADIUS_FNODES, FIBRE_NODE_REPULSION_K
- ⬆️ **Outputs:**
  - Updated repulsive force components (fx, fy, fz) on each FNODE
- - -

## 📄 fnode_focad_interaction.cpp

### 🔹 [fnode_focad_interaction](https://github.com/cborau/cellfoundry/blob/master/fnode_focad_interaction.cpp)
**Type:** `agent`  
**Source:** [Open fnode_focad_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_focad_interaction.cpp)

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

### 🔹 [boundPosition](https://github.com/cborau/cellfoundry/blob/master/fnode_move.cpp)
**Type:** `helper`  
**Source:** [Open fnode_move.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_move.cpp)

- 🔸 **Purpose:** Clamp FNODE coordinates near boundaries and update clamp state flags based on contact and configuration flags.
- - -

### 🔹 [fnode_move](https://github.com/cborau/cellfoundry/blob/master/fnode_move.cpp)
**Type:** `agent`  
**Source:** [Open fnode_move.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_move.cpp)

- 🔸 **Purpose:** Update FNODE positions/velocities under internal, boundary, and transmitted forces while enforcing clamp and sliding boundary behavior.
- ⬇️ **Inputs:**
  - FNODE force channels (network + boundary), current kinematics, clamp flags
  - Boundary movement/clamping parameters from the environment
- ⬆️ **Outputs:**
  - Updated node kinematics, clamp state, and boundary force contributions
- - -

## 📄 fnode_remodel.cpp

### 🔹 [fnode_remodel](https://github.com/cborau/cellfoundry/blob/master/fnode_remodel.cpp)
**Type:** `agent`  
**Source:** [Open fnode_remodel.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_remodel.cpp)

- 🔸 **Purpose:** Update FNODE degradation/reinforcement state from nearby CELLs and register removal requests when net degradation reaches 1.
- ⬇️ **Inputs:**
  - CELL spatial messages (x, y, z, dead)
  - FNODE state (degradation, reinforcement, id)
  - Remodeling environment properties and removal macro buffers
- ⬆️ **Outputs:**
  - Updated FNODE `degradation`, `reinforcement`, `marked_for_removal`
- - -

## 📄 fnode_spatial_location_data.cpp

### 🔹 [fnode_spatial_location_data](https://github.com/cborau/cellfoundry/blob/master/fnode_spatial_location_data.cpp)
**Type:** `agent`  
**Source:** [Open fnode_spatial_location_data.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_spatial_location_data.cpp)

- 🔸 **Purpose:** Broadcast FNODE position for spatial proximity queries.
- ⬇️ **Inputs:**
  - FNODE variables: id, x, y, z
- ⬆️ **Outputs:**
  - MessageSpatial3D payload used by FNODE/FOCAD interaction kernels
- - -

## 📄 fnode_update_links.cpp

### 🔹 [fnode_update_links](https://github.com/cborau/cellfoundry/blob/master/fnode_update_links.cpp)
**Type:** `agent`  
**Source:** [Open fnode_update_links.cpp](https://github.com/cborau/cellfoundry/blob/master/fnode_update_links.cpp)

- 🔸 **Purpose:** Update FNODE link list using bucket messages keyed by linked node id. If a linked node has no bucket message (e.g., removed), clear that link.
- ⬇️ **Inputs:**
  - FNODE bucket messages keyed by id
  - FNODE linked_nodes / equilibrium_distance
- ⬆️ **Outputs:**
  - Updated linked_nodes
  - Updated connectivity_count
- - -

## 📄 focad_anchor_update.cpp

### 🔹 [focad_anchor_update](https://github.com/cborau/cellfoundry/blob/master/focad_anchor_update.cpp)
**Type:** `agent`  
**Source:** [Open focad_anchor_update.cpp](https://github.com/cborau/cellfoundry/blob/master/focad_anchor_update.cpp)

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

### 🔹 [focad_bucket_location_data](https://github.com/cborau/cellfoundry/blob/master/focad_bucket_location_data.cpp)
**Type:** `agent`  
**Source:** [Open focad_bucket_location_data.cpp](https://github.com/cborau/cellfoundry/blob/master/focad_bucket_location_data.cpp)

- 🔸 **Purpose:** Publish full FOCAD state for bucket-keyed readers (mainly CELL/FOCAD coupling steps).
- ⬇️ **Inputs:**
  - FOCAD identifiers, kinematics, mechanics, lifecycle flags and timers
- ⬆️ **Outputs:**
  - MessageBucket keyed by cell_id with adhesion state and force data
- - -

## 📄 focad_fnode_interaction.cpp

### 🔹 [focad_fnode_interaction](https://github.com/cborau/cellfoundry/blob/master/focad_fnode_interaction.cpp)
**Type:** `agent`  
**Source:** [Open focad_fnode_interaction.cpp](https://github.com/cborau/cellfoundry/blob/master/focad_fnode_interaction.cpp)

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

### 🔹 [focad_move](https://github.com/cborau/cellfoundry/blob/master/focad_move.cpp)
**Type:** `agent`  
**Source:** [Open focad_move.cpp](https://github.com/cborau/cellfoundry/blob/master/focad_move.cpp)

- 🔸 **Purpose:** Update focal adhesion positions by either following attached FNODEs or executing bounded exploratory motion when detached/inactive.
- ⬇️ **Inputs:**
  - FOCAD state: attachment flags, anchor position, velocity, fnode_id
  - FNODE bucket messages (for attached movement)
  - Domain and adhesion motion constraints from environment
- ⬆️ **Outputs:**
  - Updated FOCAD position/velocity within boundary limits
- - -

## 📄 focad_post_cycle_update.cpp

### 🔹 [focad_post_cycle_update](https://github.com/cborau/cellfoundry/blob/master/focad_post_cycle_update.cpp)
**Type:** `agent`  
**Source:** [Open focad_post_cycle_update.cpp](https://github.com/cborau/cellfoundry/blob/master/focad_post_cycle_update.cpp)

- 🔸 **Purpose:** Update FOCAD anchor association after cell division, switching anchor points between parent and daughter cells. Ensures spatial and orientation variables are updated for correct cell association.
- ⬇️ **Inputs:**
  - FOCAD agent variables: cell_id, anchor_id, spatial coordinates
  - MessageBucket: parent cell state
  - Environment: N_ANCHOR_POINTS
- ⬆️ **Outputs:**
  - Updated anchor_id, spatial variables, cell association
- 📝 **Notes:**
  - Uses a two-pass loop to select the correct target message, avoiding duplicated anchor update code.
  - Pass 0: For parent cell, require just_divided message (ensures correct anchor after division).
  - Pass 1: Fallback, accept any available message if none found in pass 0.
- - -

## 📄 focad_spatial_location_data.cpp

### 🔹 [focad_spatial_location_data](https://github.com/cborau/cellfoundry/blob/master/focad_spatial_location_data.cpp)
**Type:** `agent`  
**Source:** [Open focad_spatial_location_data.cpp](https://github.com/cborau/cellfoundry/blob/master/focad_spatial_location_data.cpp)

- 🔸 **Purpose:** Broadcast active adhesion position/force state for local spatial interaction queries.
- ⬇️ **Inputs:**
  - FOCAD variables: id, x,y,z, fx,fy,fz, fnode_id, attached, active
- ⬆️ **Outputs:**
  - MessageSpatial3D payload consumed by FNODE-side force transfer
- - -

## 📄 handy_device_functions_template.cpp

### 🔹 [vec3CrossProd](https://github.com/cborau/cellfoundry/blob/master/handy_device_functions_template.cpp)
**Type:** `helper`  
**Source:** [Open handy_device_functions_template.cpp](https://github.com/cborau/cellfoundry/blob/master/handy_device_functions_template.cpp)

- 🔸 **Purpose:** Provide reusable device-side vector algebra helpers for interaction kernels.
- 📝 **Notes:**
  - This file is a template/reference module and is intended for copy-paste use
  - inside runtime-compiled FLAMEGPU agent function files.
  - vec3CrossProd: compute cross product (x1,y1,z1) x (x2,y2,z2).
- - -
