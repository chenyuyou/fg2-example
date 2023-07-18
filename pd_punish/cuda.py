"""
  
"""
pred_output_location = r"""
FLAMEGPU_AGENT_FUNCTION(pred_output_location, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    const flamegpu::id_t id = FLAMEGPU->getID();
    const float x = FLAMEGPU->getVariable<float>("x");
    const float y = FLAMEGPU->getVariable<float>("y");
    FLAMEGPU->message_out.setVariable<int>("id", id);
    FLAMEGPU->message_out.setVariable<float>("x", x);
    FLAMEGPU->message_out.setVariable<float>("y", y);

    return flamegpu::ALIVE;
}
"""

"""

"""
pred_follow_prey = r"""
FLAMEGPU_AGENT_FUNCTION(pred_follow_prey, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
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

