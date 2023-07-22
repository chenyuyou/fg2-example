
output_status = r"""
FLAMEGPU_AGENT_FUNCTION(output_status, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    const flamegpu::id_t id = FLAMEGPU->getID();
    float score = FLAMEGPU->getVariable<float>("score");
    unsigned int move = FLAMEGPU->getVariable<unsigned int>("move");
//    int num_agents = FLAMEGPU->environment.getProperty<unsigned int>("num_agents");
    
//    int selected = FLAMEGPU->random.uniform<int>(1, num_agents);
    FLAMEGPU->message_out.setVariable<int>("id", id);
    FLAMEGPU->message_out.setVariable<float>("score", score);
    FLAMEGPU->message_out.setVariable<unsigned int>("move", move);
    return flamegpu::ALIVE;
}
"""

study= r"""
FLAMEGPU_AGENT_FUNCTION(study, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    float score = FLAMEGPU->getVariable<float>("score");
    unsigned int move = FLAMEGPU->getVariable<unsigned int>("next_move");
    int num_agents = FLAMEGPU->environment.getProperty<unsigned int>("num_agents");
    int selected = FLAMEGPU->random.uniform<int>(1, num_agents);

    for (const auto& msg : FLAMEGPU->message_in) {
        if (msg.getVariable<int>("id") == selected) {
            const float other_score = msg.getVariable<float>("score");
            const unsigned int other_move = msg.getVariable<unsigned int>("move");
            const float intense = FLAMEGPU->environment.getProperty<float>("intense");
            float k = FLAMEGPU->environment.getProperty<float>("k");
            const float b = FLAMEGPU->environment.getProperty<float>("b");
            float c = FLAMEGPU->environment.getProperty<float>("c");
            const float e = FLAMEGPU->environment.getProperty<float>("e");
            float f = FLAMEGPU->environment.getProperty<float>("f");
            const float noise = FLAMEGPU->environment.getProperty<float>("noise");
            if (score < other_score){
                unsigned int next_move = other_move;
            }
            if (move == 0) {
                f = 0.0;
                k = 0.1;

            } else if (move == 1){
                c = 0.0;
                f = 0.0;
                k = 0.0;
            } else {
                c = 0.0;
            }
            
            float pay_off = intense * payoff * (1 - std::exp(-(noise + ((1 - noise) * t * (k * f + c)))));
            score += pay_off
        }
    }

    FLAMEGPU->message_out.setVariable<int>("id", id);
    FLAMEGPU->message_out.setVariable<float>("score", score);
    FLAMEGPU->message_out.setVariable<unsigned int>("move", move);

    return flamegpu::ALIVE;
}
"""




study1 = r"""
FLAMEGPU_AGENT_FUNCTION(study, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const float PRED_PREY_INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("PRED_PREY_INTERACTION_RADIUS");
    // Fetch the predator's position
    const float predator_x = FLAMEGPU->getVariable<float>("x");
    const float predator_y = FLAMEGPU->getVariable<float>("y");

    // Find the closest prey by iterating the prey_location messages
    float closest_prey_x = 0.0f;
    float closest_prey_y = 0.0f;
    float closest_prey_distance = PRED_PREY_INTERACTION_RADIUS;
    int is_a_prey_in_range = 0;

    for (const auto& msg : FLAMEGPU->message_in) {
        // Fetch prey location
        const float prey_x = msg.getVariable<float>("x");
        const float prey_y = msg.getVariable<float>("y");

        // Check if prey is within sight range of predator
        const float dx = predator_x - prey_x;
        const float dy = predator_y - prey_y;
        const float separation = sqrt(dx * dx + dy * dy);

        if (separation < closest_prey_distance) {
            closest_prey_x = prey_x;
            closest_prey_y = prey_y;
            closest_prey_distance = separation;
            is_a_prey_in_range = 1;
        }
    }

    // If there was a prey in range, steer the predator towards it
    if (is_a_prey_in_range) {
        const float steer_x = closest_prey_x - predator_x;
        const float steer_y = closest_prey_y - predator_y;
        FLAMEGPU->setVariable<float>("steer_x", steer_x);
        FLAMEGPU->setVariable<float>("steer_y", steer_y);
    }

    return flamegpu::ALIVE;
}
"""

mutate = r"""
FLAMEGPU_AGENT_FUNCTION(mutate, flamegpu::MessageNone, flamegpu::MessageNone) {
    float rng = FLAMEGPU->random.uniform<float>();
    const float mu = FLAMEGPU->environment.getProperty<float>("mu");
    if (rng < mu) {
        unsigned int selected = FLAMEGPU->random.uniform<unsigned int>(0, 2);    
        FLAMEGPU->setVariable<unsigned int>("move", selected);
    }
    return flamegpu::ALIVE;
}
"""