FLAMEGPU_AGENT_FUNCTION(signal_switch, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float threshold = FLAMEGPU->environment.getProperty<float>("SIGNAL_THRESHOLD");
    const float signal = FLAMEGPU->getVariable<float>("signal");
    FLAMEGPU->setVariable<int>("signal_on", signal >= threshold ? 1 : 0);
    return flamegpu::ALIVE;
}
