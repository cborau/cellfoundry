FLAMEGPU_AGENT_FUNCTION(signal_accumulate, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float rate = FLAMEGPU->environment.getProperty<float>("SIGNAL_RATE");
    const float dt = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
    const float signal = FLAMEGPU->getVariable<float>("signal");
    FLAMEGPU->setVariable<float>("signal", signal + rate * dt);
    return flamegpu::ALIVE;
}
