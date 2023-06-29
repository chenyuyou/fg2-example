output=r'''
FLAMEGPU_AGENT_FUNCTION(output, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<char>("is_alive", FLAMEGPU->getVariable<unsigned int>("is_alive"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return flamegpu::ALIVE;
}
'''

update=r'''
FLAMEGPU_AGENT_FUNCTION(update, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    unsigned int living_neighbours = 0;
    // Iterate 3x3 Moore neighbourhood (this does no include the central cell)
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y)) {
        living_neighbours += message.getVariable<char>("is_alive") ? 1 : 0;
    }
    // Using count, decide and output new value for is_alive
    char is_alive = FLAMEGPU->getVariable<unsigned int>("is_alive");
    if (is_alive) {
        if (living_neighbours < 2)
            is_alive = 0;
        else if (living_neighbours > 3)
            is_alive = 0;
        else  // exactly 2 or 3 living_neighbours
            is_alive = 1;
    } else {
        if (living_neighbours == 3)
            is_alive = 1;
    }
    FLAMEGPU->setVariable<unsigned int>("is_alive", is_alive);
    return flamegpu::ALIVE;
}
'''