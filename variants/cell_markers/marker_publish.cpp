FLAMEGPU_AGENT_FUNCTION(marker_publish, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<int>("owner_cell_id", FLAMEGPU->getVariable<int>("owner_cell_id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    return flamegpu::ALIVE;
}
