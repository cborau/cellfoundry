// variant_ids.cuh is included by register_functions() using its absolute path.
FLAMEGPU_AGENT_FUNCTION(cell_emit_marker, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") || FLAMEGPU->getVariable<int>("marker_emitted")) {
        return flamegpu::ALIVE;
    }
    auto last = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_MARKER_LAST_ID");
    auto exhausted = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_MARKER_ID_EXHAUSTED");
    const int end = FLAMEGPU->environment.getProperty<int>("VARIANT_MARKER_ID_END");
    const int id = cellfoundry_claim_variant_id(last, end, exhausted);
    if (id < 0) return flamegpu::ALIVE;  // Framework reports the allocation error.

    FLAMEGPU->agent_out.setVariable<int>("id", id);
    FLAMEGPU->agent_out.setVariable<int>("owner_cell_id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->agent_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->agent_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->agent_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->agent_out.setVariable<int>("age_steps", 0);
    FLAMEGPU->setVariable<int>("marker_emitted", 1);
    return flamegpu::ALIVE;
}
