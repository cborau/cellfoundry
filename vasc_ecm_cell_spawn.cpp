/**
 * vasc_ecm_cell_spawn
 *
 * Purpose:
 *   Template function for VASC-driven cell spawning.
 *   Each alive VASC node reads the local ECM species concentration from the
 *   shared macro buffer (C_SP_MACRO) and, when a user-defined condition is met,
 *   spawns a single new CELL agent at the VASC node position.
 *
 *   Spawning rules are currently unknown and are left as clearly marked
 *   TODO sections. The template demonstrates how to:
 *     - Sample ECM concentration at the nearest grid voxel via C_SP_MACRO
 *     - Test species-specific threshold conditions
 *     - Spawn different cell types depending on which condition fires
 *     - Assign all required CELL agent variables for a freshly born cell
 *
 * Inputs:
 *   - Agent variables: dead, x, y, z, id
 *   - Environment macro property: C_SP_MACRO (shared ECM concentration buffer)
 *   - Environment properties: ECM_AGENTS_PER_DIR, COORDS_BOUNDARIES, CELL_RADIUS,
 *                             CELL_NUCLEUS_RADIUS
 *
 * Outputs:
 *   - Optional new CELL agent via agent_out (at most one per VASC node per step)
 *
 * Notes:
 *   - FLAMEGPU2 permits at most one agent_out output per function execution.
 *     Use early-exit after the first successful spawn to guarantee this.
 *   - All TODO markers must be replaced with real rules before production use.
 *   - To add more cell types, duplicate the corresponding TODO block and adjust
 *     the condition and cell_type value.
 */
FLAMEGPU_AGENT_FUNCTION(vasc_ecm_cell_spawn, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Only alive VASC nodes can trigger cell birth
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::ALIVE;
    }

    const uint8_t  N_SPECIES         = 3;    // WARNING: hard-coded, must match model.py
    const uint32_t ECM_POPULATION_SIZE = 9261; // WARNING: hard-coded, must match model.py
    const uint8_t  N_ANCHOR_POINTS   = 50;   // WARNING: hard-coded, must match model.py

    float vasc_x = FLAMEGPU->getVariable<float>("x");
    float vasc_y = FLAMEGPU->getVariable<float>("y");
    float vasc_z = FLAMEGPU->getVariable<float>("z");

    // ------------------------------------------------------------------ //
    //  Sample ECM concentration at the nearest grid voxel via C_SP_MACRO  //
    // ------------------------------------------------------------------ //
    const int Nx = (int)FLAMEGPU->environment.getProperty<unsigned int>("ECM_AGENTS_PER_DIR", 0);
    const int Ny = (int)FLAMEGPU->environment.getProperty<unsigned int>("ECM_AGENTS_PER_DIR", 1);
    const int Nz = (int)FLAMEGPU->environment.getProperty<unsigned int>("ECM_AGENTS_PER_DIR", 2);

    const float X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 0);
    const float X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 1);
    const float Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 2);
    const float Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 3);
    const float Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 4);
    const float Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 5);

    // Map VASC position to nearest ECM grid index (same formula as cell_ecm_interaction_metabolism.cpp)
    int gi = (int)roundf(((vasc_x - X_NEG) / (X_POS - X_NEG)) * (float)(Nx - 1));
    int gj = (int)roundf(((vasc_y - Y_NEG) / (Y_POS - Y_NEG)) * (float)(Ny - 1));
    int gk = (int)roundf(((vasc_z - Z_NEG) / (Z_POS - Z_NEG)) * (float)(Nz - 1));

    gi = max(0, min(Nx - 1, gi));
    gj = max(0, min(Ny - 1, gj));
    gk = max(0, min(Nz - 1, gk));

    // Linear index into the ECM macro buffer
    const int grid_lin_id = gi * Ny * Nz + gj * Nz + gk;

    auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");

    // Read local ECM concentration for each species
    float local_C_sp[N_SPECIES] = {};
    for (int i = 0; i < N_SPECIES; i++) {
        local_C_sp[i] = (float)C_SP_MACRO[i][grid_lin_id];
    }

    // ------------------------------------------------------------------ //
    //  Cell-type-specific spawn conditions  (TODO: define real rules)     //
    // ------------------------------------------------------------------ //
    //
    //  General pattern:
    //    1. Evaluate a condition on local_C_sp[] (and/or other state).
    //    2. Call MACRO_MAX_GLOBAL_CELL_ID.addAtomic(1) to claim a unique id.
    //    3. Populate every CELL variable via FLAMEGPU->agent_out.setVariable<>().
    //    4. Return immediately — only one agent_out is allowed per execution.
    //
    //  Threshold constants and per-type parameters should be exposed as
    //  environment properties once the rules are finalised.
    //
    // ------------------------------------------------------------------ //

    auto MACRO_MAX_GLOBAL_CELL_ID = FLAMEGPU->environment.getMacroProperty<int, 1>("MACRO_MAX_GLOBAL_CELL_ID");

    const float CELL_RADIUS         = FLAMEGPU->environment.getProperty<float>("CELL_RADIUS");
    const float CELL_NUCLEUS_RADIUS = FLAMEGPU->environment.getProperty<float>("CELL_NUCLEUS_RADIUS");
    const float K_ELAST             = FLAMEGPU->environment.getProperty<float>("CELL_K_ELAST");
    const float D_DUMPING           = FLAMEGPU->environment.getProperty<float>("CELL_D_DUMPING");
    const float SPEED_REF           = FLAMEGPU->environment.getProperty<float>("CELL_SPEED_REF");

    // ---- Cell type 0: TODO define spawn condition ---- //
    //
    // Example skeleton (replace condition and parameters with real rules):
    //
    //   const float THRESHOLD_TYPE0_SP0 = 1.0f;  // TODO: expose as env property
    //   if (local_C_sp[0] >= THRESHOLD_TYPE0_SP0) {
    //       const int new_cell_id = MACRO_MAX_GLOBAL_CELL_ID.addAtomic(1);
    //       const int cell_type   = 0;
    //
    //       FLAMEGPU->agent_out.setVariable<int>("id", new_cell_id);
    //       FLAMEGPU->agent_out.setVariable<int>("max_global_cell_id", new_cell_id);
    //       FLAMEGPU->agent_out.setVariable<int>("cell_type", cell_type);
    //
    //       // Spawn at the VASC node position
    //       FLAMEGPU->agent_out.setVariable<float>("x", vasc_x);
    //       FLAMEGPU->agent_out.setVariable<float>("y", vasc_y);
    //       FLAMEGPU->agent_out.setVariable<float>("z", vasc_z);
    //       FLAMEGPU->agent_out.setVariable<float>("vx", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("vy", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("vz", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("trajectory_length", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("trajectory_time",   0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("birth_x", vasc_x);
    //       FLAMEGPU->agent_out.setVariable<float>("birth_y", vasc_y);
    //       FLAMEGPU->agent_out.setVariable<float>("birth_z", vasc_z);
    //
    //       // Random initial orientation
    //       float orx = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
    //       float ory = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
    //       float orz = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
    //       float ornorm = sqrtf(orx*orx + ory*ory + orz*orz);
    //       if (ornorm < 1e-12f) { orx = 1.0f; ory = 0.0f; orz = 0.0f; }
    //       else { orx /= ornorm; ory /= ornorm; orz /= ornorm; }
    //       FLAMEGPU->agent_out.setVariable<float>("orx", orx);
    //       FLAMEGPU->agent_out.setVariable<float>("ory", ory);
    //       FLAMEGPU->agent_out.setVariable<float>("orz", orz);
    //
    //       // Mechanical parameters (inherit defaults; override per type as needed)
    //       FLAMEGPU->agent_out.setVariable<float>("k_elast",         K_ELAST);
    //       FLAMEGPU->agent_out.setVariable<float>("d_dumping",        D_DUMPING);
    //       FLAMEGPU->agent_out.setVariable<float>("speed_ref",        SPEED_REF);
    //       FLAMEGPU->agent_out.setVariable<float>("radius",           CELL_RADIUS);
    //       FLAMEGPU->agent_out.setVariable<float>("nucleus_radius",   CELL_NUCLEUS_RADIUS);
    //
    //       // Velocity increments (zeroed at birth)
    //       FLAMEGPU->agent_out.setVariable<float>("cc_dvx", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("cc_dvy", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("cc_dvz", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("cf_dvx", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("cf_dvy", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("cf_dvz", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("cl_dvx", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("cl_dvy", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("cl_dvz", 0.0f);
    //
    //       // Per-species biochemical state (inherit local ECM concentration)
    //       for (int i = 0; i < N_SPECIES; i++) {
    //           FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("C_sp", i, local_C_sp[i]);
    //           FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("M_sp", i, 0.0f);
    //           FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_consumption", i, 0.0f); // TODO
    //           FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_production",  i, 0.0f); // TODO
    //           FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_reaction",    i, 0.0f); // TODO
    //           FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("chemokinesis_promotive_adapt_state",  i, 0.0f);
    //           FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("chemokinesis_inhibitory_adapt_state", i, 0.0f);
    //       }
    //
    //       // Cell cycle: start in G1
    //       FLAMEGPU->agent_out.setVariable<int>("cycle_phase",       1);
    //       FLAMEGPU->agent_out.setVariable<float>("clock",           0.0f);
    //       FLAMEGPU->agent_out.setVariable<int>("completed_cycles",  0);
    //
    //       // Damage and death state
    //       FLAMEGPU->agent_out.setVariable<float>("damage",         0.0f);
    //       FLAMEGPU->agent_out.setVariable<int>("dead",             0);
    //       FLAMEGPU->agent_out.setVariable<int>("dead_by",         -1);
    //       FLAMEGPU->agent_out.setVariable<int>("mother_id",       -1);
    //       FLAMEGPU->agent_out.setVariable<int>("daughter_id",     -1);
    //       FLAMEGPU->agent_out.setVariable<int>("just_divided",     0);
    //       FLAMEGPU->agent_out.setVariable<int>("marked_for_removal", 0);
    //       FLAMEGPU->agent_out.setVariable<float>("alignment",      0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("fnode_birth_cooldown",  0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("focad_birth_cooldown",  0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("lumen_secretion_cooldown", 0.0f);
    //
    //       // Strain/stress tensors (zeroed at birth)
    //       FLAMEGPU->agent_out.setVariable<float>("eps_xx", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_yy", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_zz", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("eps_xy", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_xz", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_yz", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("sig_xx", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_yy", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_zz", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("sig_xy", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_xz", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_yz", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("sig_eig_1", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_eig_2", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_eig_3", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("sig_eigvec1_x", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_eigvec1_y", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_eigvec1_z", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("sig_eigvec2_x", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_eigvec2_y", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_eigvec2_z", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("sig_eigvec3_x", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_eigvec3_y", 0.0f); FLAMEGPU->agent_out.setVariable<float>("sig_eigvec3_z", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("eps_eig_1", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_eig_2", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_eig_3", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("eps_eigvec1_x", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_eigvec1_y", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_eigvec1_z", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("eps_eigvec2_x", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_eigvec2_y", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_eigvec2_z", 0.0f);
    //       FLAMEGPU->agent_out.setVariable<float>("eps_eigvec3_x", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_eigvec3_y", 0.0f); FLAMEGPU->agent_out.setVariable<float>("eps_eigvec3_z", 0.0f);
    //
    //       // Focal-adhesion anchor arrays (zeroed; will be populated by cell_focad_update)
    //       for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    //           FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("x_i",       i, vasc_x);
    //           FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("y_i",       i, vasc_y);
    //           FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("z_i",       i, vasc_z);
    //           FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("u_ref_x_i", i, 0.0f);
    //           FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("u_ref_y_i", i, 0.0f);
    //           FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("u_ref_z_i", i, 0.0f);
    //       }
    //
    //       return flamegpu::ALIVE; // early exit — only one spawn per step
    //   }

    // ---- Cell type 1: TODO define spawn condition ---- //
    //   (Duplicate the block above, change cell_type = 1, and define a different condition)

    return flamegpu::ALIVE;
}
