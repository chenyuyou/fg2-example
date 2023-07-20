"""
    输出捕食者的位置，不需要输入变量。
"""
pred_output_location = r"""
FLAMEGPU_AGENT_FUNCTION(pred_output_location, flamegpu::MessageNone, flamegpu::MessageArray2D) {
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
    捕食者找出最近猎物，计算两者之间的距离的矢量。需要的输入变量为猎物的位置。
"""
pred_follow_prey = r"""
FLAMEGPU_AGENT_FUNCTION(pred_follow_prey, flamegpu::MessageArray2D, flamegpu::MessageNone) {
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

"""
    捕食者找出附近的其他捕猎者，计算出叠加的加速度，并以此改方向适量。需要的输入变量为猎物的位置。
"""
pred_avoid = r"""
FLAMEGPU_AGENT_FUNCTION(pred_avoid, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const float SAME_SPECIES_AVOIDANCE_RADIUS = FLAMEGPU->environment.getProperty<float>("SAME_SPECIES_AVOIDANCE_RADIUS");
    // Fetch this predator's position
    const float predator_x = FLAMEGPU->getVariable<float>("x");
    const float predator_y = FLAMEGPU->getVariable<float>("y");
    float avoid_velocity_x = 0.0f;
    float avoid_velocity_y = 0.0f;

    // Add a steering factor away from each other predator. Strength increases with closeness.
    for (const auto& msg : FLAMEGPU->message_in) {
        // Fetch location of other predator
        const float other_predator_x = msg.getVariable<float>("x");
        const float other_predator_y = msg.getVariable<float>("y");

        // Check if the two predators are within interaction radius
        const float dx = predator_x - other_predator_x;
        const float dy = predator_y - other_predator_y;
        const float separation = sqrt(dx * dx + dy * dy);

        if (separation < SAME_SPECIES_AVOIDANCE_RADIUS && separation > 0.0f) {
            avoid_velocity_x += SAME_SPECIES_AVOIDANCE_RADIUS / separation * dx;
            avoid_velocity_y += SAME_SPECIES_AVOIDANCE_RADIUS / separation * dy;
        }
    }

    float steer_x = FLAMEGPU->getVariable<float>("steer_x");
    float steer_y = FLAMEGPU->getVariable<float>("steer_y");
    steer_x += avoid_velocity_x;
    steer_y += avoid_velocity_y;
    FLAMEGPU->setVariable<float>("steer_x", steer_x);
    FLAMEGPU->setVariable<float>("steer_y", steer_y);

    return flamegpu::ALIVE;
}
"""

"""
    捕食者移动，损失生命。无需信息输入输出。
"""
pred_move = r"""
FLAMEGPU_AGENT_FUNCTION(pred_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float MIN_POSITION = FLAMEGPU->environment.getProperty<float>("MIN_POSITION");
    const float MAX_POSITION = FLAMEGPU->environment.getProperty<float>("MAX_POSITION");
    const float DELTA_TIME = FLAMEGPU->environment.getProperty<float>("DELTA_TIME");
    const float PRED_SPEED_ADVANTAGE = FLAMEGPU->environment.getProperty<float>("PRED_SPEED_ADVANTAGE");
    float predator_x = FLAMEGPU->getVariable<float>("x");
    float predator_y = FLAMEGPU->getVariable<float>("y");
    float predator_vx = FLAMEGPU->getVariable<float>("vx");
    float predator_vy = FLAMEGPU->getVariable<float>("vy");
    const float predator_steer_x = FLAMEGPU->getVariable<float>("steer_x");
    const float predator_steer_y = FLAMEGPU->getVariable<float>("steer_y");
    const float predator_life = FLAMEGPU->getVariable<int>("life");

    // Integrate steering forces and cap velocity
    predator_vx += predator_steer_x;
    predator_vy += predator_steer_y;

    float speed = sqrt(predator_vx * predator_vx + predator_vy * predator_vy);
    if (speed > 1.0f) {
        predator_vx /= speed;
        predator_vy /= speed;
    }

    // Integrate velocity
    predator_x += predator_vx * DELTA_TIME * PRED_SPEED_ADVANTAGE;
    predator_y += predator_vy * DELTA_TIME * PRED_SPEED_ADVANTAGE;

    // Bound the position within the environment 
    predator_x = predator_x < MIN_POSITION ? MIN_POSITION : predator_x;
    predator_x = predator_x > MAX_POSITION ? MAX_POSITION : predator_x;
    predator_y = predator_y < MIN_POSITION ? MIN_POSITION : predator_y;
    predator_y = predator_y > MAX_POSITION ? MAX_POSITION : predator_y;

    // Update agent state
    FLAMEGPU->setVariable<float>("x", predator_x);
    FLAMEGPU->setVariable<float>("y", predator_y);
    FLAMEGPU->setVariable<float>("vx", predator_vx);
    FLAMEGPU->setVariable<float>("vy", predator_vy);

    // Reduce life by one unit of energy
    FLAMEGPU->setVariable<int>("life", predator_life - 1);

    return flamegpu::ALIVE;
}
"""

"""
    捕食者匹配猎物输出的信息，如果匹配，则增加生命，如果不匹配，则减少生命。需要的输入变量为猎物输出的捕食者匹配信息，无输出信息。
"""
pred_eat_or_starve = r"""
FLAMEGPU_AGENT_FUNCTION(pred_eat_or_starve, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const flamegpu::id_t predator_id = FLAMEGPU->getID();
    int predator_life = FLAMEGPU->getVariable<int>("life");
    int isDead = 0;

    // Iterate prey_eaten messages to see if this predator ate a prey
    for (const auto& msg : FLAMEGPU->message_in) {
        if (msg.getVariable<int>("pred_id") == predator_id) {
            predator_life += FLAMEGPU->environment.getProperty<unsigned int>("GAIN_FROM_FOOD_PREDATOR");
        }
    }

    // Update agent state
    FLAMEGPU->setVariable<int>("life", predator_life);

    // Did the predator starve?
    if (predator_life < 1) {
        isDead = 1;
    }

    return isDead ? flamegpu::DEAD : flamegpu::ALIVE;
}
"""

"""
    捕食者繁殖。无需信息输入输出。
"""
pred_reproduction = r"""
FLAMEGPU_AGENT_FUNCTION(pred_reproduction, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float BOUNDS_WIDTH = FLAMEGPU->environment.getProperty<float>("BOUNDS_WIDTH");
    float random = FLAMEGPU->random.uniform<float>();
    const int currentLife = FLAMEGPU->getVariable<int>("life");
    if (random < FLAMEGPU->environment.getProperty<float>("REPRODUCE_PRED_PROB")) {
        int id = FLAMEGPU->random.uniform<float>() * (float)INT_MAX;
        float x = FLAMEGPU->random.uniform<float>() * BOUNDS_WIDTH - BOUNDS_WIDTH / 2.0f;
        float y = FLAMEGPU->random.uniform<float>() * BOUNDS_WIDTH - BOUNDS_WIDTH / 2.0f;
        float vx = FLAMEGPU->random.uniform<float>() * 2 - 1;
        float vy = FLAMEGPU->random.uniform<float>() * 2 - 1;

        FLAMEGPU->setVariable<int>("life", currentLife / 2);
// 一下代码繁殖新的捕食者
        FLAMEGPU->agent_out.setVariable<float>("x", x);
        FLAMEGPU->agent_out.setVariable<float>("y", y);
        FLAMEGPU->agent_out.setVariable<float>("type", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("vx", vx);
        FLAMEGPU->agent_out.setVariable<float>("vy", vy);
        FLAMEGPU->agent_out.setVariable<float>("steer_x", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("steer_y", 0.0f);
        FLAMEGPU->agent_out.setVariable<int>("life", currentLife / 2);

    }
    return flamegpu::ALIVE;
}
"""

"""
    输出猎物的位置，不需要输入变量。
"""
prey_output_location = r"""
FLAMEGPU_AGENT_FUNCTION(prey_output_location, flamegpu::MessageNone, flamegpu::MessageArray2D) {
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
    猎物躲避捕食者。输入为捕食者信息，无输出。
"""
prey_avoid_pred = r"""
FLAMEGPU_AGENT_FUNCTION(prey_avoid_pred, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const float PRED_PREY_INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("PRED_PREY_INTERACTION_RADIUS");
    // Fetch this prey's position
    const float prey_x = FLAMEGPU->getVariable<float>("x");
    const float prey_y = FLAMEGPU->getVariable<float>("y");
    float avoid_velocity_x = 0.0f;
    float avoid_velocity_y = 0.0f;

    // Add a steering factor away from each predator. Strength increases with closeness.
    for (const auto& msg : FLAMEGPU->message_in) {
        // Fetch location of predator
        const float predator_x = msg.getVariable<float>("x");
        const float predator_y = msg.getVariable<float>("y");

        // Check if the two predators are within interaction radius
        const float dx = prey_x - predator_x;
        const float dy = prey_y - predator_y;
        const float distance = sqrt(dx * dx + dy * dy);

        if (distance < PRED_PREY_INTERACTION_RADIUS) {
            // Steer the prey away from the predator
            avoid_velocity_x += (PRED_PREY_INTERACTION_RADIUS / distance) * dx;
            avoid_velocity_y += (PRED_PREY_INTERACTION_RADIUS / distance) * dy;
        }
    }

    // Update agent state 
    FLAMEGPU->setVariable<float>("steer_x", avoid_velocity_x);
    FLAMEGPU->setVariable<float>("steer_y", avoid_velocity_y);

    return flamegpu::ALIVE;
}
"""

"""
    猎物聚集。输入为其他猎物的信息，无输出信息。
"""
prey_flock = r"""
FLAMEGPU_AGENT_FUNCTION(prey_flock, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const float PREY_GROUP_COHESION_RADIUS = FLAMEGPU->environment.getProperty<float>("PREY_GROUP_COHESION_RADIUS");
    const float SAME_SPECIES_AVOIDANCE_RADIUS = FLAMEGPU->environment.getProperty<float>("SAME_SPECIES_AVOIDANCE_RADIUS");
    const flamegpu::id_t prey_id = FLAMEGPU->getID();
    const float prey_x = FLAMEGPU->getVariable<float>("x");
    const float prey_y = FLAMEGPU->getVariable<float>("y");

    float group_centre_x = 0.0f;
    float group_centre_y = 0.0f;
    float group_velocity_x = 0.0f;
    float group_velocity_y = 0.0f;
    float avoid_velocity_x = 0.0f;
    float avoid_velocity_y = 0.0f;
    int group_centre_count = 0;

    for (const auto& msg : FLAMEGPU->message_in) {
        const int   other_prey_id = msg.getVariable<int>("id");
        const float other_prey_x = msg.getVariable<float>("x");
        const float other_prey_y = msg.getVariable<float>("y");
        const float dx = prey_x - other_prey_x;
        const float dy = prey_y - other_prey_y;
        const float separation = sqrt(dx * dx + dy * dy);

        if (separation < PREY_GROUP_COHESION_RADIUS && prey_id != other_prey_id) {
            group_centre_x += other_prey_x;
            group_centre_y += other_prey_y;
            group_centre_count += 1;

            // Avoidance behaviour
            if (separation < SAME_SPECIES_AVOIDANCE_RADIUS) {
                // Was a check for separation > 0 in original - redundant?
                avoid_velocity_x += SAME_SPECIES_AVOIDANCE_RADIUS / separation * dx;
                avoid_velocity_y += SAME_SPECIES_AVOIDANCE_RADIUS / separation * dy;
            }
        }
    }

    // Compute group centre as the average of the nearby prey positions and a velocity to move towards the group centre
    if (group_centre_count > 0) {
        group_centre_x /= group_centre_count;
        group_centre_y /= group_centre_count;
        group_velocity_x = group_centre_x - prey_x;
        group_velocity_y = group_centre_y - prey_y;
    }

    float prey_steer_x = FLAMEGPU->getVariable<float>("steer_x");
    float prey_steer_y = FLAMEGPU->getVariable<float>("steer_y");
    prey_steer_x += group_velocity_x + avoid_velocity_x;
    prey_steer_y += group_velocity_y + avoid_velocity_y;
    FLAMEGPU->setVariable<float>("steer_x", prey_steer_x);
    FLAMEGPU->setVariable<float>("steer_y", prey_steer_y);

    return flamegpu::ALIVE;
}
"""
"""
    猎物移动，损失生命。无需信息输入输出。
"""
prey_move = r"""
FLAMEGPU_AGENT_FUNCTION(prey_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float MIN_POSITION = FLAMEGPU->environment.getProperty<float>("MIN_POSITION");
    const float MAX_POSITION = FLAMEGPU->environment.getProperty<float>("MAX_POSITION");
    const float DELTA_TIME = FLAMEGPU->environment.getProperty<float>("DELTA_TIME");
    float prey_x = FLAMEGPU->getVariable<float>("x");
    float prey_y = FLAMEGPU->getVariable<float>("y");
    float prey_vx = FLAMEGPU->getVariable<float>("vx");
    float prey_vy = FLAMEGPU->getVariable<float>("vy");
    const float prey_steer_x = FLAMEGPU->getVariable<float>("steer_x");
    const float prey_steer_y = FLAMEGPU->getVariable<float>("steer_y");
    const float prey_life = FLAMEGPU->getVariable<int>("life");

    // Integrate steering forces and cap velocity
    prey_vx += prey_steer_x;
    prey_vy += prey_steer_y;

    float speed = sqrt(prey_vx * prey_vx + prey_vy * prey_vy);
    if (speed > 1.0f) {
        prey_vx /= speed;
        prey_vy /= speed;
    }

    // Integrate velocity
    prey_x += prey_vx * DELTA_TIME;
    prey_y += prey_vy * DELTA_TIME;

    // Bound the position within the environment - can this be moved
    prey_x = prey_x < MIN_POSITION ? MIN_POSITION : prey_x;
    prey_x = prey_x > MAX_POSITION ? MAX_POSITION : prey_x;
    prey_y = prey_y < MIN_POSITION ? MIN_POSITION : prey_y;
    prey_y = prey_y > MAX_POSITION ? MAX_POSITION : prey_y;


    // Update agent state
    FLAMEGPU->setVariable<float>("x", prey_x);
    FLAMEGPU->setVariable<float>("y", prey_y);
    FLAMEGPU->setVariable<float>("vx", prey_vx);
    FLAMEGPU->setVariable<float>("vy", prey_vy);

    // Reduce life by one unit of energy
    FLAMEGPU->setVariable<int>("life", prey_life - 1);

    return flamegpu::ALIVE;
}
"""

"""
    猎物找出距离最近的捕食者，且与该捕食者距离小于给定的捕食距离，标记猎物被吃。输入为捕食者信息，输出为特定的距离猎物最近的捕食者。
"""
prey_eaten = r"""
FLAMEGPU_AGENT_FUNCTION(prey_eaten, flamegpu::MessageArray2D, flamegpu::MessageArray2D) {
    const float PRED_KILL_DISTANCE = FLAMEGPU->environment.getProperty<float>("PRED_KILL_DISTANCE");
    const flamegpu::id_t id = FLAMEGPU->getID();
    int eaten = 0;
    int predator_id = -1;
    float closest_pred = PRED_KILL_DISTANCE;
    const float prey_x = FLAMEGPU->getVariable<float>("x");
    const float prey_y = FLAMEGPU->getVariable<float>("y");

    // Iterate predator_location messages to find the closest predator
    for (const auto& msg : FLAMEGPU->message_in) {
        // Fetch location of predator
        const float predator_x = msg.getVariable<float>("x");
        const float predator_y = msg.getVariable<float>("y");

        // Check if the two predators are within interaction radius
        const float dx = prey_x - predator_x;
        const float dy = prey_y - predator_y;
        const float distance = sqrt(dx * dx + dy * dy);

        if (distance < closest_pred) {
            predator_id = msg.getVariable<int>("id");
            closest_pred = distance;
            eaten = 1;
        }
    }

    if (eaten) {
        FLAMEGPU->message_out.setVariable<int>("id", id);
        FLAMEGPU->message_out.setVariable<int>("pred_id", predator_id);
    }

    return eaten ? flamegpu::DEAD : flamegpu::ALIVE;
}
"""

"""
    猎物繁殖。无需信息输入输出。
"""
prey_reproduction = r"""
FLAMEGPU_AGENT_FUNCTION(prey_reproduction, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float REPRODUCE_PREY_PROB = FLAMEGPU->environment.getProperty<float>("REPRODUCE_PREY_PROB");
    const float BOUNDS_WIDTH = FLAMEGPU->environment.getProperty<float>("BOUNDS_WIDTH");
    float random = FLAMEGPU->random.uniform<float>();
    const int currentLife = FLAMEGPU->getVariable<int>("life");
    if (random < FLAMEGPU->environment.getProperty<float>("REPRODUCE_PREY_PROB")) {
        int id = FLAMEGPU->random.uniform<float>() * (float)INT_MAX;
        float x = FLAMEGPU->random.uniform<float>() * BOUNDS_WIDTH - BOUNDS_WIDTH / 2.0f;
        float y = FLAMEGPU->random.uniform<float>() * BOUNDS_WIDTH - BOUNDS_WIDTH / 2.0f;
        float vx = FLAMEGPU->random.uniform<float>() * 2 - 1;
        float vy = FLAMEGPU->random.uniform<float>() * 2 - 1;

        FLAMEGPU->setVariable<int>("life", currentLife / 2);

        FLAMEGPU->agent_out.setVariable<float>("x", x);
        FLAMEGPU->agent_out.setVariable<float>("y", y);
        FLAMEGPU->agent_out.setVariable<float>("type", 1.0f);
        FLAMEGPU->agent_out.setVariable<float>("vx", vx);
        FLAMEGPU->agent_out.setVariable<float>("vy", vy);
        FLAMEGPU->agent_out.setVariable<float>("steer_x", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("steer_y", 0.0f);
        FLAMEGPU->agent_out.setVariable<int>("life", currentLife / 2);

    }
    return flamegpu::ALIVE;
}
"""

"""
    输出草的位置，不需要输入变量。
"""
grass_output_location = r"""
FLAMEGPU_AGENT_FUNCTION(grass_output_location, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    // Exercise 3.1 : Set the variables for the grass_location message
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
    草找出距离最近的猎物（食草），且与该猎物距离小于给定的吃草距离，标记草被吃。输入为猎物的信息，输出为特定的距离草最近的猎物。
"""
grass_eaten = r"""
FLAMEGPU_AGENT_FUNCTION(grass_eaten, flamegpu::MessageArray2D, flamegpu::MessageArray2D) {
    const float grass_x = FLAMEGPU->getVariable<float>("x");
    const float grass_y = FLAMEGPU->getVariable<float>("y");
    int available = FLAMEGPU->getVariable<int>("available");
    if (available) { 

        int prey_id = -1;
        float closest_prey = FLAMEGPU->environment.getProperty<float>("GRASS_EAT_DISTANCE");
        int eaten = 0;

        // Iterate predator_location messages to find the closest predator
        for (const auto& msg : FLAMEGPU->message_in) {
            // Fetch location of prey
            const float prey_x = msg.getVariable<float>("x");
            const float prey_y = msg.getVariable<float>("y");

            // Check if the two preys are within interaction radius
            const float dx = grass_x - prey_x;
            const float dy = grass_y - prey_y;
            const float distance = sqrt(dx*dx + dy*dy);

            if (distance < closest_prey) {
                prey_id = msg.getVariable<int>("id");
                closest_prey= distance;
                eaten = 1;
            }
        }

        if (eaten) {
            // Add grass eaten message
            FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
            FLAMEGPU->message_out.setVariable<int>("prey_id", prey_id);
           
            // Update grass agent variables
            FLAMEGPU->setVariable<int>("dead_cycles", 0);
            FLAMEGPU->setVariable<int>("available", 0);
            FLAMEGPU->setVariable<float>("type", 3.0f);
        }
    }
    return flamegpu::ALIVE;
}
"""

"""
    草匹配猎物（食草）的信息，如果匹配，则猎物增加生命，如果不匹配，则减少生命。需要的输入变量为草输出的猎物的匹配信息，无输出信息。
"""
prey_eat_or_starve = r"""
FLAMEGPU_AGENT_FUNCTION(prey_eat_or_starve, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    int isDead = 0;
    const flamegpu::id_t id = FLAMEGPU->getID();
    const int life = FLAMEGPU->getVariable<int>("life");

    // Iterate the grass eaten messages 
    for (const auto& msg : FLAMEGPU->message_in)
    {
        // If the grass eaten message indicates that this prey ate some grass then increase the preys life by adding energy
        if (id == msg.getVariable<int>("prey_id")) {
            FLAMEGPU->setVariable<int>("life", life + FLAMEGPU->environment.getProperty<unsigned int>("GAIN_FROM_FOOD_PREY"));
        }
    }

    // If the life has reduced to 0 then the prey should die or starvation 
    if (FLAMEGPU->getVariable<int>("life") < 1)
        isDead = 1;

    return isDead ? flamegpu::DEAD : flamegpu::ALIVE;
}
"""

"""
    草繁殖。无需信息输入输出。
"""
grass_growth = r"""
FLAMEGPU_AGENT_FUNCTION(grass_growth, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead_cycles = FLAMEGPU->getVariable<int>("dead_cycles");
    int new_dead_cycles = dead_cycles + 1;
    if (dead_cycles == FLAMEGPU->environment.getProperty<unsigned int>("GRASS_REGROW_CYCLES")) {
        FLAMEGPU->setVariable<int>("dead_cycles", 0);
        FLAMEGPU->setVariable<int>("available", 1);
        FLAMEGPU->setVariable<float>("type", 2.0f);
    } 

    const int available = FLAMEGPU->getVariable<int>("available");
    if (available == 0) {
        FLAMEGPU->setVariable<int>("dead_cycles", new_dead_cycles);
    } 

    return flamegpu::ALIVE;
}
"""