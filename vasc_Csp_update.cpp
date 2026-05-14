/**
 * vasc_Csp_update
 *
 * Purpose:
 *   Propagate species concentrations along the vascular tree.
 *   Each non-source VASC node copies its parent's concentration; if the parent is
 *   dead or absent the node's own concentration falls to zero.
 *   Source nodes (parent_id < 0) act as Dirichlet boundaries and keep their
 *   initial INIT_VASCULARIZATION_CONCENTRATION_VALS unchanged.
 *
 * Inputs:
 *   - Agent variables: dead, parent_id, C_sp[N_SPECIES]
 *   - MessageBucket: vasc_bucket_location_message (keyed by global VASC id)
 *
 * Outputs:
 *   - Updated agent C_sp[N_SPECIES]
 *
 * Notes:
 *   Must run after vasc_bucket_location_data so all parent messages are committed.
 */
FLAMEGPU_AGENT_FUNCTION(vasc_Csp_update, flamegpu::MessageBucket, flamegpu::MessageNone) {
    // Dead VASC nodes do not propagate concentration
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::ALIVE;
    }

    int parent_id = FLAMEGPU->getVariable<int>("parent_id");
    const uint8_t N_SPECIES = 2; // WARNING: hard-coded, must match model.py

    // Source / boundary nodes (no parent): keep concentration unchanged
    if (parent_id < 0) {
        return flamegpu::ALIVE;
    }

    // Look up parent state from the bucket message
    int parent_found = 0;
    int parent_alive = 0;
    float parent_C_sp[N_SPECIES] = {};
    for (const auto& msg : FLAMEGPU->message_in(parent_id)) {
        if (msg.getVariable<int>("id") == parent_id) {
            parent_found = 1;
            parent_alive = (msg.getVariable<int>("dead") == 0) ? 1 : 0;
            if (parent_alive) {
                for (int i = 0; i < N_SPECIES; i++) {
                    parent_C_sp[i] = msg.getVariable<float, N_SPECIES>("C_sp", i);
                }
            }
            break; // only one parent; stop iterating once found
        }
    }

    if (parent_found && parent_alive) {
        // Inherit parent concentration (tree propagation toward leaves)
        for (int i = 0; i < N_SPECIES; i++) {
            FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, parent_C_sp[i]);
        }
    } else {
        // Parent dead or not found: concentration falls to zero
        for (int i = 0; i < N_SPECIES; i++) {
            FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, 0.0f);
        }
    }

    return flamegpu::ALIVE;
}
