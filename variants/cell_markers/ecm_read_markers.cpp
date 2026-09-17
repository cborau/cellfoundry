FLAMEGPU_AGENT_FUNCTION(ecm_read_markers, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const float x = FLAMEGPU->getVariable<float>("x");
    const float y = FLAMEGPU->getVariable<float>("y");
    const float z = FLAMEGPU->getVariable<float>("z");
    const float radius = FLAMEGPU->environment.getProperty<float>("MARKER_DETECTION_RADIUS");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in) {
        const float dx = x - message.getVariable<float>("x");
        const float dy = y - message.getVariable<float>("y");
        const float dz = z - message.getVariable<float>("z");
        if (dx * dx + dy * dy + dz * dz <= radius * radius) ++count;
    }
    const float dt = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
    const float exposure = FLAMEGPU->getVariable<float>("marker_exposure");
    FLAMEGPU->setVariable<int>("marker_count", count);
    FLAMEGPU->setVariable<float>("marker_exposure", exposure + count * dt);
    return flamegpu::ALIVE;
}
