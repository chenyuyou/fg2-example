
output1=r'''
FLAMEGPU_AGENT_FUNCTION(output, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<float>("value", FLAMEGPU->getVariable<float>("value"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return flamegpu::ALIVE;
}
'''

update1=r'''
FLAMEGPU_AGENT_FUNCTION(update, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int i = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int j = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    const float dx2 = FLAMEGPU->environment.getProperty<float>("dx2");
    const float dy2 = FLAMEGPU->environment.getProperty<float>("dy2");
    const float old_value = FLAMEGPU->getVariable<float>("value");

    const float left = FLAMEGPU->message_in.at(i == 0 ? FLAMEGPU->message_in.getDimX() - 1 : i - 1, j).getVariable<float>("value");
    const float up = FLAMEGPU->message_in.at(i, j == 0 ? FLAMEGPU->message_in.getDimY() - 1 : j - 1).getVariable<float>("value");
    const float right = FLAMEGPU->message_in.at(i + 1 >= FLAMEGPU->message_in.getDimX() ? 0 : i + 1, j).getVariable<float>("value");
    const float down = FLAMEGPU->message_in.at(i, j + 1 >= FLAMEGPU->message_in.getDimY() ? 0 : j + 1).getVariable<float>("value");

    // Explicit scheme
    float new_value = (left - 2.0 * old_value + right) / dx2 + (up - 2.0 * old_value + down) / dy2;

    const float a = FLAMEGPU->environment.getProperty<float>("a");
    const float dt = FLAMEGPU->environment.getProperty<float>("dt");

    new_value *= a * dt;
    new_value += old_value;

    FLAMEGPU->setVariable<float>("value", new_value);
    return flamegpu::ALIVE;
}
'''

