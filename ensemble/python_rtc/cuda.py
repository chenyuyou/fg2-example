AddOffset = r"""
FLAMEGPU_AGENT_FUNCTION(AddOffset, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Output each agents publicly visible properties.
    FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") + FLAMEGPU->environment.getProperty<int>("offset"));
    return flamegpu::ALIVE;
}
"""