#include "flamegpu/flamegpu.h"

// Agent Function to output the agents ID and position in to a 2D spatial message list
FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));
    return flamegpu::ALIVE;
}

// Agent Function to read the location messages and decide how the agent should move
FLAMEGPU_AGENT_FUNCTION(input_message, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const flamegpu::id_t ID = FLAMEGPU->getID();
    const float REPULSE_FACTOR = FLAMEGPU->environment.getProperty<float>("repulse");
    const float RADIUS = FLAMEGPU->message_in.radius();
    float fx = 0.0;
    float fy = 0.0;
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in(x1, y1)) {
        if (message.getVariable<flamegpu::id_t>("id") != ID) {
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");
            float x21 = x2 - x1;
            float y21 = y2 - y1;
            const float separation = sqrt(x21*x21 + y21*y21);
            if (separation < RADIUS && separation > 0.0f) {
                float k = sinf((separation / RADIUS)*3.141f*-2)*REPULSE_FACTOR;
                // Normalise without recalculating separation
                x21 /= separation;
                y21 /= separation;
                fx += k * x21;
                fy += k * y21;
                count++;
            }
        }
    }
    fx /= count > 0 ? count : 1;
    fy /= count > 0 ? count : 1;
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("drift", sqrt(fx*fx + fy*fy));
    return flamegpu::ALIVE;
}

FLAMEGPU_INIT_FUNCTION(create_agents) {
    // Fetch the desired agent count and environment width
    const unsigned int AGENT_COUNT = FLAMEGPU->environment.getProperty<unsigned int>("AGENT_COUNT");
    const float ENV_WIDTH = FLAMEGPU->environment.getProperty<float>("ENV_WIDTH");
    // Create agents
    flamegpu::HostAgentAPI t_pop = FLAMEGPU->agent("point");
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        auto t = t_pop.newAgent();
        t.setVariable<float>("x", FLAMEGPU->random.uniform<float>() * ENV_WIDTH);
        t.setVariable<float>("y", FLAMEGPU->random.uniform<float>() * ENV_WIDTH);
    }
}

int main(int argc, const char **argv) {
    // Define some useful constants
    const unsigned int AGENT_COUNT = 16384;
    const float ENV_WIDTH = static_cast<float>(floor(cbrt(AGENT_COUNT)));

    // Define the FLAME GPU model
    flamegpu::ModelDescription model("Circles Tutorial");

    {   // (optional local scope block for cleaner grouping)
        // Define a message of type MessageSpatial2D named location
        flamegpu::MessageSpatial2D::Description message = model.newMessage<flamegpu::MessageSpatial2D>("location");
        // Configure the message list
        message.setMin(0, 0);
        message.setMax(ENV_WIDTH, ENV_WIDTH);
        message.setRadius(1.0f);
        // Add extra variables to the message
        // X Y (Z) are implicit for spatial messages
        message.newVariable<flamegpu::id_t>("id");
    }

    // Define an agent named point
    flamegpu::AgentDescription agent = model.newAgent("point");
    // Assign the agent some variables (ID is implicit to agents, so we don't define it ourselves)
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<float>("z");
    agent.newVariable<float>("drift", 0.0f);
    // Setup the two agent functions
    flamegpu::AgentFunctionDescription out_fn = agent.newFunction("output_message", output_message);
    out_fn.setMessageOutput("location");
    flamegpu::AgentFunctionDescription in_fn = agent.newFunction("input_message", input_message);
    in_fn.setMessageInput("location");

    {   // (optional local scope block for cleaner grouping)
        // Define environment properties
        flamegpu::EnvironmentDescription env = model.Environment();
        env.newProperty<unsigned int>("AGENT_COUNT", AGENT_COUNT);
        env.newProperty<float>("ENV_WIDTH", ENV_WIDTH);
        env.newProperty<float>("repulse", 0.05f);
    }

    {   // (optional local scope block for cleaner grouping)
        // Dependency specification
        // Message input depends on output
        in_fn.dependsOn(out_fn);
        // Output is the root of our graph
        model.addExecutionRoot(out_fn);
        model.generateLayers();
    }

    model.addInitFunction(create_agents);

    // Specify the desired StepLoggingConfig
    flamegpu::StepLoggingConfig step_log_cfg(model);
    // Log every step
    step_log_cfg.setFrequency(1);
    // Include the mean of the "point" agent population's variable 'drift'
    step_log_cfg.agent("point").logMean<float>("drift");

    // Create the simulation
    flamegpu::CUDASimulation cuda_model(model, argc, argv);

    // Attach the logging config
    cuda_model.setStepLog(step_log_cfg);

// Only compile this block if being built with visualisation support
#ifdef FLAMEGPU_VISUALISATION
    // Create visualisation
    flamegpu::visualiser::ModelVis m_vis = cuda_model.getVisualisation();
    // Set the initial camera location and speed
    const float INIT_CAM = ENV_WIDTH / 2.0f;
    m_vis.setBeginPaused(true);
    m_vis.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0);
    m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, ENV_WIDTH);
    m_vis.setCameraSpeed(0.01f);
//    m_vis.setSimulationSpeed(25);
    // Add "point" agents to the visualisation
    flamegpu::visualiser::AgentVis point_agt = m_vis.addAgent("point");
    // Location variables have names "x" and "y" so will be used by default
    point_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
    point_agt.setModelScale(1/10.0f);
    // Mark the environment bounds
    flamegpu::visualiser::LineVis pen = m_vis.newPolylineSketch(1, 1, 1, 0.2f);
    pen.addVertex(0, 0, 0);
    pen.addVertex(0, ENV_WIDTH, 0);
    pen.addVertex(ENV_WIDTH, ENV_WIDTH, 0);
    pen.addVertex(ENV_WIDTH, 0, 0);
    pen.addVertex(0, 0, 0);
    // Open the visualiser window
    m_vis.activate();
#endif

    // Run the simulation
    cuda_model.simulate();

#ifdef FLAMEGPU_VISUALISATION
    // Keep the visualisation window active after the simulation has completed
    m_vis.join();
#endif
}