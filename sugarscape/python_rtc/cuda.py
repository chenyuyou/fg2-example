metabolise_and_growback= r"""
FLAMEGPU_AGENT_FUNCTION(metabolise_and_growback, flamegpu::MessageNone, flamegpu::MessageNone) {
    int sugar_level = FLAMEGPU->getVariable<int>("sugar_level");
    int env_sugar_level = FLAMEGPU->getVariable<int>("env_sugar_level");
    int env_max_sugar_level = FLAMEGPU->getVariable<int>("env_max_sugar_level");
    int status = FLAMEGPU->getVariable<int>("status");
    // metabolise if occupied
    if (status == 1 || status == 3) {
        // store any sugar present in the cell
        if (env_sugar_level > 0) {
            sugar_level += env_sugar_level;
            // Occupied cells are marked as -1 sugar.
            env_sugar_level = -1;
        }

        // metabolise
        sugar_level -= FLAMEGPU->getVariable<int>("metabolism");

        // check if agent dies
        if (sugar_level == 0) {
            status = 0;
            FLAMEGPU->setVariable<int>("agent_id", -1);
            env_sugar_level = 0;
            FLAMEGPU->setVariable<int>("metabolism", 0);
        }
    }

    // growback if unoccupied
    if (status == 0) {
        env_sugar_level += 1;
        if (env_sugar_level > env_max_sugar_level) {
            env_sugar_level = env_max_sugar_level;
        }
    }

    // set all active agents to unresolved as they may now want to move
    if (status == 1) {
        status = 3;
    }
    FLAMEGPU->setVariable<int>("sugar_level", sugar_level);
    FLAMEGPU->setVariable<int>("env_sugar_level", env_sugar_level);
    FLAMEGPU->setVariable<int>("status", status);

    return flamegpu::ALIVE;
}
"""

output_cell_status = r"""
FLAMEGPU_AGENT_FUNCTION(output_cell_status, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    unsigned int agent_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    unsigned int agent_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);
    FLAMEGPU->message_out.setVariable("location_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable("status", FLAMEGPU->getVariable<int>("status"));
    FLAMEGPU->message_out.setVariable("env_sugar_level", FLAMEGPU->getVariable<int>("env_sugar_level"));
    FLAMEGPU->message_out.setIndex(agent_x, agent_y);
    return flamegpu::ALIVE;
}
"""

movement_request = r"""
FLAMEGPU_AGENT_FUNCTION(movement_request, flamegpu::MessageArray2D, flamegpu::MessageArray2D) {
    int best_sugar_level = -1;
    float best_sugar_random = -1;
    flamegpu::id_t best_location_id = flamegpu::ID_NOT_SET;

    // if occupied then look for empty cells {
    // find the best location to move to (ensure we don't just pick first cell with max value)
    int status = FLAMEGPU->getVariable<int>("status");

    unsigned int agent_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    unsigned int agent_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    // if occupied then look for empty cells
    if (status == 3) {
        for (auto current_message : FLAMEGPU->message_in.wrap(agent_x, agent_y)) {
            // if location is unoccupied then check for empty locations
            if (current_message.getVariable<int>("status") == 0) {
                // if the sugar level at current location is better than currently stored then update
                int message_env_sugar_level = current_message.getVariable<int>("env_sugar_level");
                float message_priority = FLAMEGPU->random.uniform<float>();
                if ((message_env_sugar_level > best_sugar_level) ||
                    (message_env_sugar_level == best_sugar_level && message_priority > best_sugar_random)) {
                    best_sugar_level = message_env_sugar_level;
                    best_sugar_random = message_priority;
                    best_location_id = current_message.getVariable<flamegpu::id_t>("location_id");
                }
            }
        }

        // if the agent has found a better location to move to then update its state
        // if there is a better location to move to then state indicates a movement request
        status = best_location_id != flamegpu::ID_NOT_SET ? 2: 1;
        FLAMEGPU->setVariable<int>("status", status);
    }

    // add a movement request
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getVariable<int>("agent_id"));
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("location_id", best_location_id);
    FLAMEGPU->message_out.setVariable<int>("sugar_level", FLAMEGPU->getVariable<int>("sugar_level"));
    FLAMEGPU->message_out.setVariable<int>("metabolism", FLAMEGPU->getVariable<int>("metabolism"));
    FLAMEGPU->message_out.setIndex(agent_x, agent_y);

    return flamegpu::ALIVE;
}
"""

movement_response = r"""
FLAMEGPU_AGENT_FUNCTION(movement_response, flamegpu::MessageArray2D, flamegpu::MessageArray2D) {
    int best_request_id = -1;
    float best_request_priority = -1;
    int best_request_sugar_level = -1;
    int best_request_metabolism = -1;

    int status = FLAMEGPU->getVariable<int>("status");
    const flamegpu::id_t location_id = FLAMEGPU->getID();
    const unsigned int agent_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int agent_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    for (auto current_message : FLAMEGPU->message_in.wrap(agent_x, agent_y)) {
        // if the location is unoccupied then check for agents requesting to move here
        if (status == 0) {
            // check if request is to move to this location
            if (current_message.getVariable<flamegpu::id_t>("location_id") == location_id) {
                // check the priority and maintain the best ranked agent
                float message_priority = FLAMEGPU->random.uniform<float>();
                if (message_priority > best_request_priority) {
                    best_request_id = current_message.getVariable<int>("agent_id");
                    best_request_priority = message_priority;
                }
            }
        }
    }

    // if the location is unoccupied and an agent wants to move here then do so and send a response
    if ((status == 0) && (best_request_id >= 0))    {
        FLAMEGPU->setVariable<int>("status", 1);
        // move the agent to here and consume the cell's sugar
        best_request_sugar_level += FLAMEGPU->getVariable<int>("env_sugar_level");
        FLAMEGPU->setVariable<int>("agent_id", best_request_id);
        FLAMEGPU->setVariable<int>("sugar_level", best_request_sugar_level);
        FLAMEGPU->setVariable<int>("metabolism", best_request_metabolism);
        FLAMEGPU->setVariable<int>("env_sugar_level", -1);
    }

    // add a movement response
    FLAMEGPU->message_out.setVariable<int>("agent_id", best_request_id);
    FLAMEGPU->message_out.setIndex(agent_x, agent_y);

    return flamegpu::ALIVE;
}
"""

movement_transaction = r"""
FLAMEGPU_AGENT_FUNCTION(movement_transaction, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    int status = FLAMEGPU->getVariable<int>("status");
    int agent_id = FLAMEGPU->getVariable<int>("agent_id");
    unsigned int agent_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    unsigned int agent_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    for (auto current_message : FLAMEGPU->message_in.wrap(agent_x, agent_y)) {
        // if location contains an agent wanting to move then look for responses allowing relocation
        if (status == 2) {  // if the movement response request came from this location
            if (current_message.getVariable<int>("agent_id") == agent_id) {
                // remove the agent and reset agent specific variables as it has now moved
                status = 0;
                FLAMEGPU->setVariable<int>("agent_id", -1);
                FLAMEGPU->setVariable<int>("sugar_level", 0);
                FLAMEGPU->setVariable<int>("metabolism", 0);
                FLAMEGPU->setVariable<int>("env_sugar_level", 0);
            }
        }
    }

    // if request has not been responded to then agent is unresolved
    if (status == 2) {
        status = 3;
    }

    FLAMEGPU->setVariable<int>("status", status);

    return flamegpu::ALIVE;
}
"""