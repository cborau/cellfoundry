/**
 * vasc_bucket_location_data
 *
 * Purpose:
 *   Broadcast full VASC node state into a MessageBucket keyed by the node's own id.
 *
 * Inputs:
 *   - Agent variables: id, x, y, z, parent_ids[MAX_VASC_CONNECTIVITY], dead, C_sp[N_SPECIES], children_ids[MAX_VASC_CONNECTIVITY]
 *
 * Outputs:
 *   - MessageBucket entry keyed by id carrying all agent state for peer lookup
 *
 * Notes:
 *   Must run before vasc_Csp_update so parent messages are available for reading.
 */
FLAMEGPU_AGENT_FUNCTION(vasc_bucket_location_data, flamegpu::MessageNone, flamegpu::MessageBucket) {
    const uint8_t N_SPECIES = 2;             // WARNING: hard-coded, must match model.py
    const uint8_t MAX_VASC_CONNECTIVITY = 2; // WARNING: hard-coded, must match model.py

    // Broadcast identity and topology
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    for (int i = 0; i < MAX_VASC_CONNECTIVITY; i++) {
        FLAMEGPU->message_out.setVariable<int, MAX_VASC_CONNECTIVITY>("parent_ids", i, FLAMEGPU->getVariable<int, MAX_VASC_CONNECTIVITY>("parent_ids", i));
    }
    FLAMEGPU->message_out.setVariable<int>("dead", FLAMEGPU->getVariable<int>("dead"));

    // Broadcast per-species concentration state
    for (int i = 0; i < N_SPECIES; i++) {
        FLAMEGPU->message_out.setVariable<float, N_SPECIES>("C_sp", i, FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i));
    }

    // Broadcast child connectivity
    for (int i = 0; i < MAX_VASC_CONNECTIVITY; i++) {
        FLAMEGPU->message_out.setVariable<int, MAX_VASC_CONNECTIVITY>("children_ids", i, FLAMEGPU->getVariable<int, MAX_VASC_CONNECTIVITY>("children_ids", i));
    }

    // Key the bucket by this node's own id so peers can look it up directly
    FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<int>("id"));
    return flamegpu::ALIVE;
}
