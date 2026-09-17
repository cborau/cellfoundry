FLAMEGPU_AGENT_FUNCTION(marker_age, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int age = FLAMEGPU->getVariable<int>("age_steps") + 1;
    const int lifetime = FLAMEGPU->environment.getProperty<int>("MARKER_LIFETIME_STEPS");
    FLAMEGPU->setVariable<int>("age_steps", age);
    return age >= lifetime ? flamegpu::DEAD : flamegpu::ALIVE;
}
