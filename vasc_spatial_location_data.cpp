/**
 * vasc_spatial_location_data
 *
 * Purpose:
 *   Broadcast VASC node position, liveness, and current species concentrations
 *   into a MessageSpatial3D, allowing nearby ECM agents to find and read VASC
 *   state within a configurable search radius.
 *
 * Inputs:
 *   - Agent variables: x, y, z, id, dead, C_sp[N_SPECIES]
 *
 * Outputs:
 *   - MessageSpatial3D entry (vasc_spatial_location_message) carrying x, y, z, id, dead, C_sp
 *
 * Notes:
 *   Must run before ecm_vasc_Csp_update so ECM agents see updated VASC concentrations.
 */
FLAMEGPU_AGENT_FUNCTION(vasc_spatial_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const uint8_t N_SPECIES = 2; // WARNING: hard-coded, must match model.py

    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));

    // Broadcast identity and liveness for ECM filtering
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<int>("dead", FLAMEGPU->getVariable<int>("dead"));

    // Broadcast per-species concentration so ECM can impose a concentration floor
    for (int i = 0; i < N_SPECIES; i++) {
        FLAMEGPU->message_out.setVariable<float, N_SPECIES>("C_sp", i, FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i));
    }
    return flamegpu::ALIVE;
}
