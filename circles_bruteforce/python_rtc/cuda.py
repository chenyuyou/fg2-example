output_message = r"""
// r之前的output_message是字符串变量, 用于python代码.
// FLAMEGPU_AGENT_FUNCTION 为C宏
// output_message 为agent函数名, 在c语言各函数中调用
// flamegpu::MessageNone    这个函数不接收任何消息类型作为输入
// flamegpu::MessageBruteForce: 指定这个函数会输出 MessageBruteForce 类型的消息
// 将agent坐标x,y,z以及id作为消息传出
FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    return flamegpu::ALIVE;
}
"""

# agent函数读取位置消息并决定agent应该如何移动
move = r"""
// 这里有两类消息，一类是自身消息，一类是周围符合消息半径的agent的消息
// move 为agent函数名, 在c语言各函数中调用
//  flamegpu::MessageBruteForce 指定这个函数接收 MessageBruteForce 类型的消息作为输入
// flamegpu::MessageNone: 指定这个函数不输出任何消息类型
//  自身的获取消息包括{id,x和y坐标, REPULSE_FACTOR(模型中的斥力), RADIUS(agent自身的变量)}
FLAMEGPU_AGENT_FUNCTION(move, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const flamegpu::id_t ID = FLAMEGPU->getID();
    const float REPULSE_FACTOR = FLAMEGPU->environment.getProperty<float>("repulse");
//  这里因为是全局通信，所以没有消息半径，需要从环境中给出模型的作用半径
    const float RADIUS = FLAMEGPU->environment.getProperty<float>("radius");
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in) {
        if (message.getVariable<flamegpu::id_t>("id") != ID) {
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");
            const float z2 = message.getVariable<float>("z");
            float x21 = x2 - x1;
            float y21 = y2 - y1;
            float z21 = z2 - z1;
            const float separation = sqrtf(x21*x21 + y21*y21 + z21*z21);
            if (separation < RADIUS && separation > 0.0f) {
                float k = sinf((separation / RADIUS)*3.141f*-2)*REPULSE_FACTOR;
                // Normalise without recalculating separation
                x21 /= separation;
                y21 /= separation;
                z21 /= separation;
                fx += k * x21;
                fy += k * y21;
                fz += k * z21;
                count++;
            }
        }
    }
    fx /= count > 0 ? count : 1;
    fy /= count > 0 ? count : 1;
    fz /= count > 0 ? count : 1;
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("z", z1 + fz);
    FLAMEGPU->setVariable<float>("drift", sqrtf(fx*fx + fy*fy + fz*fz));
    return flamegpu::ALIVE;
}
"""