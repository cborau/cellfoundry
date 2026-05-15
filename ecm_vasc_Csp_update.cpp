/**
 * ecm_vasc_Csp_update
 *
 * Purpose:
 *   For each ECM voxel, find nearby alive VASC nodes (via MessageSpatial3D) and
 *   impose a concentration floor: if any VASC neighbour carries a higher
 *   per-species concentration than the current ECM value, that value is adopted.
 *   This ensures the vascular network acts as a sustained source that diffusion
 *   cannot erode below the vessel-supplied level.
 *
 * Inputs:
 *   - Agent variables: x, y, z, grid_lin_id, C_sp[N_SPECIES]
 *   - MessageSpatial3D: vasc_spatial_location_message (dead flag + C_sp per VASC node)
 *   - Environment macro property: C_SP_MACRO (shared concentration buffer)
 *
 * Outputs:
 *   - Updated agent C_sp[N_SPECIES] (so the L1 ECM broadcast reflects VASC influence)
 *   - Updated C_SP_MACRO[species][grid_lin_id] (so L4 ecm_Csp_update preserves the floor)
 *
 * Notes:
 *   Must run after vasc_spatial_location_data and before the L1 ECM broadcast.
 *   Each ECM agent writes only to its own grid_lin_id slot, so no write conflicts.
 */
FLAMEGPU_AGENT_FUNCTION(ecm_vasc_Csp_update, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    float agent_z = FLAMEGPU->getVariable<float>("z");

    const uint8_t N_SPECIES = 2;           // WARNING: hard-coded, must match model.py
    const uint32_t ECM_POPULATION_SIZE = 9261; // WARNING: hard-coded, must match model.py

    int grid_lin_id = FLAMEGPU->getVariable<int>("grid_lin_id");
    auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");

    // Read current ECM concentration
    float C_sp[N_SPECIES] = {};
    for (int i = 0; i < N_SPECIES; i++) {
        C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
    }

    // Accumulate the per-species maximum concentration over alive VASC nodes within range.
    // NOTE: MessageSpatial3D's message_in(x,y,z) returns all messages in the 3x3x3 bin
    // neighbourhood (bin size = radius), which can include nodes up to ~3x the configured
    // radius away. An explicit distance check is therefore required.
    const float MAX_R = FLAMEGPU->environment.getProperty<float>("MAX_SEARCH_RADIUS_VASCULARIZATION");
    const float MAX_R2 = MAX_R * MAX_R;
    int found_alive = 0;
    float max_vasc_C_sp[N_SPECIES] = {};
    // DEBUG tracking
    int   n_vasc_in_radius  = 0;          // alive VASC nodes within MAX_R
    float min_dist2_found   = 1e30f;      // nearest alive VASC node (squared distance)

    for (const auto& msg : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        const float dx   = agent_x - msg.getVariable<float>("x");
        const float dy   = agent_y - msg.getVariable<float>("y");
        const float dz   = agent_z - msg.getVariable<float>("z");
        const float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 > MAX_R2) { continue; }
        if (msg.getVariable<int>("dead") == 0) {
            found_alive = 1;
            n_vasc_in_radius++;
            if (dist2 < min_dist2_found) { min_dist2_found = dist2; }
            for (int i = 0; i < N_SPECIES; i++) {
                float c = msg.getVariable<float, N_SPECIES>("C_sp", i);
                if (c > max_vasc_C_sp[i]) {
                    max_vasc_C_sp[i] = c;
                }
            }
        }
    }


    if (found_alive) {
        for (int i = 0; i < N_SPECIES; i++) {
            if (max_vasc_C_sp[i] > C_sp[i]) {
                // Update the per-agent variable so the L1 broadcast reflects VASC influence
                FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, max_vasc_C_sp[i]);
                // Also update the macro buffer so L4 ecm_Csp_update preserves the floor
                C_SP_MACRO[i][grid_lin_id].exchange(max_vasc_C_sp[i]);
            }
        }
    }

    return flamegpu::ALIVE;
}
