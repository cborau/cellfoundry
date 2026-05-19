/**
 * vasc_Csp_update
 *
 * Purpose:
 *   Propagate species concentrations along the vascular tree.
 *   Each non-source VASC node looks up all its parents (parent_ids array). As
 *   long as at least one parent is alive it inherits the maximum concentration
 *   across the live parents per species.  If ALL parents are dead or absent the
 *   node is marked dead (dead=1) and its concentration falls to zero, simulating
 *   vessel regression when fully disconnected from the source.
 *   Source nodes (all parent_ids entries < 0) act as Dirichlet boundaries and
 *   keep their initial INIT_VASCULARIZATION_CONCENTRATION_VALS unchanged.
 *
 * Inputs:
 *   - Agent variables: dead, parent_ids[MAX_VASC_CONNECTIVITY], C_sp[N_SPECIES]
 *   - MessageBucket: vasc_bucket_location_message (keyed by global VASC id)
 *
 * Outputs:
 *   - Updated agent C_sp[N_SPECIES]
 *   - Updated agent dead (set to 1 if all parents are dead/absent)
 *
 * Notes:
 *   Must run after vasc_bucket_location_data so all parent messages are committed.
 *   Always returns flamegpu::ALIVE; death is signalled via the dead variable.
 */
FLAMEGPU_AGENT_FUNCTION(vasc_Csp_update, flamegpu::MessageBucket, flamegpu::MessageNone) {
    // Dead VASC nodes do not propagate concentration
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::ALIVE;
    }

    const uint8_t N_SPECIES = 3;             // WARNING: hard-coded, must match model.py
    const uint8_t MAX_VASC_CONNECTIVITY = 2; // WARNING: hard-coded, must match model.py

    // Collect all parent ids; determine if this is a source node (first entry == -2)
    int parent_ids[MAX_VASC_CONNECTIVITY];
    for (int i = 0; i < MAX_VASC_CONNECTIVITY; i++) {
        parent_ids[i] = FLAMEGPU->getVariable<int, MAX_VASC_CONNECTIVITY>("parent_ids", i);
    }

    // Source / boundary nodes are flagged with -2 in the first slot
    if (parent_ids[0] == -2) {
        return flamegpu::ALIVE;
    }

    // Query all valid parents; track maximum concentration per species
    bool any_live_parent = false;
    float max_C_sp[N_SPECIES] = {};
    for (int pi = 0; pi < MAX_VASC_CONNECTIVITY; pi++) {
        int pid = parent_ids[pi];
        if (pid < 0) continue;  // empty slot
        for (const auto& msg : FLAMEGPU->message_in(pid)) {
            if (msg.getVariable<int>("id") == pid) {
                if (msg.getVariable<int>("dead") == 0) {
                    any_live_parent = true;
                    for (int i = 0; i < N_SPECIES; i++) {
                        float c = msg.getVariable<float, N_SPECIES>("C_sp", i);
                        if (c > max_C_sp[i]) max_C_sp[i] = c;
                    }
                }
                break;  // found this parent; move to next
            }
        }
    }

    if (any_live_parent) {
        for (int i = 0; i < N_SPECIES; i++) {
            FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, max_C_sp[i]);
        }
    } else {
        // All parents dead or absent: mark this node dead and zero concentration
        FLAMEGPU->setVariable<int>("dead", 1);
        for (int i = 0; i < N_SPECIES; i++) {
            FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, 0.0f);
        }
    }

    return flamegpu::ALIVE;
}
