outputdata = r"""
//宏定义了一个名为 outputdata 的代理函数。第一个参数是函数名。
//第二个参数 flamegpu::MessageNone 表示这个函数不读取任何特定类型的消息作为输入。
//第三个参数 flamegpu::MessageSpatial3D 表示这个函数会输出一种名为 location 的、使用 Spatial3D 策略进行通信的消息
FLAMEGPU_AGENT_FUNCTION(outputdata, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // Output each agents publicly visible properties.
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->message_out.setVariable<float>("fz", FLAMEGPU->getVariable<float>("fz"));
    return flamegpu::ALIVE;
}
"""

inputdata = r"""
// 一系列向量工具函数的集合，用于进行三维向量的长度计算、乘法、除法以及位置 clamped。
// 计算一个三维向量 (x, y, z) 的长度（模）
// sqrtf 是 CUDA C++ 中用于浮点数的平方根函数。
FLAMEGPU_HOST_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
    return sqrtf(x * x + y * y + z * z);
}

// 对一个三维向量 (x, y, z) 进行标量乘法
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Mult(float &x, float &y, float &z, const float multiplier) {
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;
}
//  对一个三维向量 (x, y, z) 进行标量除法。
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
    x /= divisor;
    y /= divisor;
    z /= divisor;
}

// 将一个位置向量 (x, y, z) 的每个分量限制在 MIN_POSITION 和 MAX_POSITION 之间。这用于确保代理的位置保持在模拟空间的边界内。使用了三元运算符 (condition ? value_if_true : value_if_false) 进行简单的边界检查和限制。
FLAMEGPU_HOST_DEVICE_FUNCTION void clampPosition(float &x, float &y, float &z, const float MIN_POSITION, const float MAX_POSITION) {
    x = (x < MIN_POSITION)? MIN_POSITION: x;
    x = (x > MAX_POSITION)? MAX_POSITION: x;

    y = (y < MIN_POSITION)? MIN_POSITION: y;
    y = (y > MAX_POSITION)? MAX_POSITION: y;

    z = (z < MIN_POSITION)? MIN_POSITION: z;
    z = (z > MAX_POSITION)? MAX_POSITION: z;
}
// flamegpu::MessageSpatial3D: 指定这个代理函数从哪种类型的消息通道接收消息。这里是 MessageSpatial3D 类型的消息，也就是在 model.py 中定义的 "location" 消息。
// flamegpu::MessageNone: 指定这个代理函数不向任何消息通道输出消息。
FLAMEGPU_AGENT_FUNCTION(inputdata, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
// 获取当前执行这个代理函数的代理的唯一 ID，并存储在常量 id 中。
    const flamegpu::id_t id = FLAMEGPU->getID();
//    printf("%d\n", id);
//  从当前代理的变量中获取名为 "x" 的浮点数变量的值。同样获取 "y", "z", "fx", "fy", "fz" 变量的值，并将它们存储在局部变量中。 
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    float agent_z = FLAMEGPU->getVariable<float>("z");
    // Agent velocity
    float agent_fx = FLAMEGPU->getVariable<float>("fx");
    float agent_fy = FLAMEGPU->getVariable<float>("fy");
    float agent_fz = FLAMEGPU->getVariable<float>("fz");

    // 用于累加邻居代理的位置，以便计算感知到的群体中心。
    float perceived_centre_x = 0.0f;
    float perceived_centre_y = 0.0f;
    float perceived_centre_z = 0.0f;
    // 计数感知到的邻居数量
    int perceived_count = 0;

    // 用于累加邻居代理的速度，以便计算感知到的群体平均速度。
    float global_velocity_x = 0.0f;
    float global_velocity_y = 0.0f;
    float global_velocity_z = 0.0f;

    // 用于累加由各种规则导致的代理速度的变化量。
    float velocity_change_x = 0.f;
    float velocity_change_y = 0.f;
    float velocity_change_z = 0.f;

    // 从环境属性中获取交互半径的值。FLAMEGPU->environment 提供了访问环境属性的方法。
    const float INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("INTERACTION_RADIUS");
    // 从环境属性中获取分离半径的值。
    const float SEPARATION_RADIUS = FLAMEGPU->environment.getProperty<float>("SEPARATION_RADIUS");
    // 从环境属性中获取交互半径的值。FLAMEGPU->environment 提供了访问环境属性的方法。
    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        // 检查消息的发送者 ID 是否与当前代理自身的 ID 不同。这是为了忽略代理自己发送给自己的消息。
        if (message.getVariable<flamegpu::id_t>("id") != id) {
            // Get the message location and velocity.
            const float message_x = message.getVariable<float>("x");
            const float message_y = message.getVariable<float>("y");
            const float message_z = message.getVariable<float>("z");

            // 计算当前代理与发送消息的邻居代理之间的距离（分离）。
            float separation = vec3Length(agent_x - message_x, agent_y - message_y, agent_z - message_z);

            if (separation < INTERACTION_RADIUS) {
                // 如果邻居在交互半径内，将其位置累加到 perceived_centre 变量中，并增加 perceived_count。
                perceived_centre_x += message_x;
                perceived_centre_y += message_y;
                perceived_centre_z += message_z;
                perceived_count++;

                // 获取邻居的速度分量。
                const float message_fx = message.getVariable<float>("fx");
                const float message_fy = message.getVariable<float>("fy");
                const float message_fz = message.getVariable<float>("fz");
                // 如果邻居在交互半径内，将其速度累加到 global_velocity 变量中。
                global_velocity_x += message_fx;
                global_velocity_y += message_fy;
                global_velocity_z += message_fz;

                // 检查距离是否小于分离半径。这是实现分离规则的关键，只有非常近的邻居才会触发分离行为。
                if (separation < (SEPARATION_RADIUS)) {  // dependant on model size
                    // 计算归一化的分离距离，即距离与分离半径的比值。
                    float normalizedSeparation = (separation / SEPARATION_RADIUS);
                    // 计算归一化分离距离的倒数（1 减去归一化分离距离）。距离越近，这个值越大。
                    float invNormSep = (1.0f - normalizedSeparation);
                     //  计算 invNormSep 的平方。这产生一个与距离的平方成反比的权重，使得离得越近的邻居对分离行为的影响越大。
                    float invSqSep = invNormSep * invNormSep;
                    // 获取环境属性中的分离（碰撞）缩放因子。
                    const float collisionScale = FLAMEGPU->environment.getProperty<float>("COLLISION_SCALE");
                    // 计算由当前邻居引起的分离导致的 x 方向速度变化，并累加到 velocity_change_x 中。agent_x - message_x 表示从邻居指向当前代理的向量的 x 分量，乘以 invSqSep 和 collisionScale 后，使得当前代理倾向于远离近距离的邻居，且距离越近，远离的力度越大。y 和 z 方向也类似计算。
                    velocity_change_x += collisionScale * (agent_x - message_x) * invSqSep;
                    velocity_change_y += collisionScale * (agent_y - message_y) * invSqSep;
                    velocity_change_z += collisionScale * (agent_z - message_z) * invSqSep;
                }
            }
        }
    }
// 检查是否感知到了任何邻居。只有当感知到邻居时，才会计算内聚和对齐规则。
    if (perceived_count) {
        // 将累加的邻居位置除以邻居数量，计算出感知到的群体中心坐标。
        vec3Div(perceived_centre_x, perceived_centre_y, perceived_centre_z, perceived_count);
        // 将累加的邻居速度除以邻居数量，计算出感知到的群体平均速度。
        vec3Div(global_velocity_x, global_velocity_y, global_velocity_z, perceived_count);

        // Rule 1) Steer towards perceived centre of flock (Cohesion)
        float steer_velocity_x = 0.f;
        float steer_velocity_y = 0.f;
        float steer_velocity_z = 0.f;
        //  获取环境属性中的内聚缩放因子。
        const float STEER_SCALE = FLAMEGPU->environment.getProperty<float>("STEER_SCALE");
        // 计算指向感知到的群体中心的方向向量，并乘以内聚缩放因子。这将导致代理倾向于飞向群体中心。
        steer_velocity_x = (perceived_centre_x - agent_x) * STEER_SCALE;
        steer_velocity_y = (perceived_centre_y - agent_y) * STEER_SCALE;
        steer_velocity_z = (perceived_centre_z - agent_z) * STEER_SCALE;
        // 将由内聚引起的 velocity 变化累加到 velocity_change 中。
        velocity_change_x += steer_velocity_x;
        velocity_change_y += steer_velocity_y;
        velocity_change_z += steer_velocity_z;

        // Rule 2) Match neighbours speeds (Alignment)
        float match_velocity_x = 0.f;
        float match_velocity_y = 0.f;
        float match_velocity_z = 0.f;
        // 获取环境属性中的对齐缩放因子。
        const float MATCH_SCALE = FLAMEGPU->environment.getProperty<float>("MATCH_SCALE");
        // 计算感知到的群体平均速度，并乘以对齐缩放因子。
        match_velocity_x = global_velocity_x * MATCH_SCALE;
        match_velocity_y = global_velocity_y * MATCH_SCALE;
        match_velocity_z = global_velocity_z * MATCH_SCALE;
        // 计算由对齐引起的 velocity 变化。这里计算的是目标平均速度与当前代理自身速度的差，这个差乘以对齐缩放因子，然后累加到 velocity_change 中。这将导致代理的速度趋向于与邻居的平均速度一致。
        velocity_change_x += match_velocity_x - agent_fx;
        velocity_change_y += match_velocity_y - agent_fy;
        velocity_change_z += match_velocity_z - agent_fz;
    }

    // 将总的速度变化量 velocity_change 乘以全局缩放因子 GLOBAL_SCALE。这允许整体调整 Boids 规则对速度的影响强度。
    vec3Mult(velocity_change_x, velocity_change_y, velocity_change_z, FLAMEGPU->environment.getProperty<float>("GLOBAL_SCALE"));

    // 将计算出的总速度变化量加到代理当前的 velocity 上，更新代理的 velocity。
    agent_fx += velocity_change_x;
    agent_fy += velocity_change_y;
    agent_fz += velocity_change_z;

    // 如果速度大小大于 1 (或某个预设的最大速度，这里是 1)，则将速度向量归一化（使其大小为 1）。这限制了代理的最大速度。
    // 定义一个最小速度。
    float agent_fscale = vec3Length(agent_fx, agent_fy, agent_fz);
    //  如果速度大小小于最小速度，则将速度向量归一化，然后乘以最小速度 minSpeed。这确保了代理的速度不会低于某个阈值。
    if (agent_fscale > 1) {
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale);
    }

    float minSpeed = 0.5f;
    if (agent_fscale < minSpeed) {
        // Normalise
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale);

        // Scale to min
        vec3Mult(agent_fx, agent_fy, agent_fz, minSpeed);
    }

    // 定义代理开始感知到墙壁的距离
    const float wallInteractionDistance = 0.10f;
    // 定义代理避开墙壁的力度。
    const float wallSteerStrength = 0.05f;
    const float minPosition = FLAMEGPU->environment.getProperty<float>("MIN_POSITION");
    const float maxPosition = FLAMEGPU->environment.getProperty<float>("MAX_POSITION");
    // 如果代理的 x 坐标与最小 x 坐标的差小于 wallInteractionDistance，说明代理接近左边的墙壁，就在 x 方向上增加 wallSteerStrength 到 agent_fx，使其向右转向。
    if (agent_x - minPosition < wallInteractionDistance) {
        agent_fx += wallSteerStrength;
    }
    if (agent_y - minPosition < wallInteractionDistance) {
        agent_fy += wallSteerStrength;
    }
    if (agent_z - minPosition < wallInteractionDistance) {
        agent_fz += wallSteerStrength;
    }
    // 如果最大 x 坐标与代理的 x 坐标的差小于 wallInteractionDistance，说明代理接近右边的墙壁，就在 x 方向上减小 wallSteerStrength 到 agent_fx，使其向左转向。
    if (maxPosition - agent_x < wallInteractionDistance) {
        agent_fx -= wallSteerStrength;
    }
    if (maxPosition - agent_y < wallInteractionDistance) {
        agent_fy -= wallSteerStrength;
    }
    if (maxPosition - agent_z < wallInteractionDistance) {
        agent_fz -= wallSteerStrength;
    }

    // 获取环境属性中的时间步长缩放因子。
    const float TIME_SCALE = FLAMEGPU->environment.getProperty<float>("TIME_SCALE");
    // 根据更新后的 velocity 和时间步长，更新代理的位置。这遵循简单的欧拉积分方法：新位置 = 旧位置 + 速度 * 时间步长。
    agent_x += agent_fx * TIME_SCALE;
    agent_y += agent_fy * TIME_SCALE;
    agent_z += agent_fz * TIME_SCALE;

    // 将更新后的代理位置限制在模拟空间的边界内。即使有了墙壁避免行为，这个步骤也提供了额外的保证，防止代理意外地飞出模拟空间。
    clampPosition(agent_x, agent_y, agent_z, FLAMEGPU->environment.getProperty<float>("MIN_POSITION"), FLAMEGPU->environment.getProperty<float>("MAX_POSITION"));

    // 将更新后的代理位置和速度值写回到代理的全局内存中。这些更新后的值将在下一个模拟步骤中被使用，或者被其他代理通过消息读取。
    FLAMEGPU->setVariable<float>("x", agent_x);
    FLAMEGPU->setVariable<float>("y", agent_y);
    FLAMEGPU->setVariable<float>("z", agent_z);

    FLAMEGPU->setVariable<float>("fx", agent_fx);
    FLAMEGPU->setVariable<float>("fy", agent_fy);
    FLAMEGPU->setVariable<float>("fz", agent_fz);

    return flamegpu::ALIVE;
}
"""