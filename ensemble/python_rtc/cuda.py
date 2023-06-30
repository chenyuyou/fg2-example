AddOffset = r"""
FLAMEGPU_AGENT_FUNCTION(AddOffset, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Output each agents publicly visible properties.
    FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") + FLAMEGPU->environment.getProperty<int>("offset"));
    return flamegpu::ALIVE;
}
"""

init = r"""
FLAMEGPU_INIT_FUNCTION(Init) {
    const unsigned int POPULATION_TO_GENERATE = FLAMEGPU->environment.getProperty<unsigned int>("POPULATION_TO_GENERATE");
    const int init = FLAMEGPU->environment.getProperty<int>("init");
    const int init_offset = FLAMEGPU->environment.getProperty<int>("init_offset");
    auto agent = FLAMEGPU->agent("Agent");
    for (unsigned int i = 0; i < POPULATION_TO_GENERATE; ++i) {
        agent.newAgent().setVariable<int>("x", init + i * init_offset);
    }
}
"""

exit1 = r"""
FLAMEGPU_EXIT_FUNCTION(Exit) {
    atomic_init += FLAMEGPU->environment.getProperty<int>("init");
    atomic_result += FLAMEGPU->agent("Agent").sum<int>("x");
}
"""