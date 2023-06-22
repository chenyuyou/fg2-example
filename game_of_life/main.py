#! /usr/bin/env python3
from pyflamegpu import *
import sys, random, time
from cuda import *



def main():
    model = pyflamegpu.ModelDescription("game of life")

    SQRT_AGENT_COUNT = 1000
    AGENT_COUNT = SQRT_AGENT_COUNT * SQRT_AGENT_COUNT

    message = model.newMessageSpatial2D("is_alive_message")
    message.newVariableChar("is_alive")
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT)
  
    

    agent = model.newAgent("cell")
    agent.newPropertyArrayInt("pos")
    agent.newVariableInt("foo")
    if pyflamegpu.VISUALISATION:
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
    agent.newRTCFunction("output", outputdata).setMessageOutput("is_alive_message")
    agent.newRTCFunction("update", updatedata).setMessageInput("is_alive_message")
    
    env = model.Environment()
    env.newPropertyFloat("repulse", 0.05)
    env.newPropertyFloat("radius", 1)


    model.newLayer().addAgentFunction("cell", "outputdata")
    model.newLayer().addAgentFunction("cell", "inputdata")

    cudaSimulation = pyflamegpu.CUDASimulation(model)
    cudaSimulation.initialise(sys.argv)

# If no xml model file was is provided, generate a population.
    if not cudaSimulation.SimulationConfig().input_file:
    # Uniformly distribute agents within space, with uniformly distributed initial velocity.
        random.seed(cudaSimulation.SimulationConfig().random_seed)

        init_pop = pyflamegpu.AgentVector(model.Agent("Boid"), AGENT_COUNT)
        for x in range(SQRT_AGENT_COUNT):
            for y in range(SQRT_AGENT_COUNT):
                instance = population[i]
                instance.setVariableArrayInt("pos", 2,[x, y])
                is_alive= 1 if random.random < 0.4 else 0
                instance.setVariableInt("is_alive", is_alive)
                if pyflamegpu.VISUALISATION:
        # Agent position in space
                    instance.setVariableFloat("x", x)
                    instance.setVariableFloat("y", y)
        cudaSimulation.setPopulationData(init_pop)


    if pyflamegpu.VISUALISATION:
        visualisation = cudaSimulation.getVisualisation()
    # Configure vis

        visualisation.setBeginPaused(True)
        visualisation.setSimulationSpeed(5)
        visualisation.setInitialCameraLocation(SQRT_AGENT_COUNT / 2.0, SQRT_AGENT_COUNT / 2.0, 450.0)
        visualisation.setInitialCameraTarget(SQRT_AGENT_COUNT / 2.0, SQRT_AGENT_COUNT / 2.0, 0.0)
        visualisation.setCameraSpeed(0.001 * SQRT_AGENT_COUNT)
        visualisation.setOrthographic(True)
        visualisation.setOrthographicZoomModifier(1.409)
        visualisation.setViewClips(0.01, 2500)
        visualisation.setClearColor(0.6, 0.6, 0.6)

        agt = visualisation.addAgent("cell")
        agt.setModel(pyflamegpu.CUBE)
        agt.setModelScale(1.0)
        agt.setColor(pyflamegpu.DiscreteColor("is_alive", pyflamegpu.DARK2, pyflamegpu.WHITE))
        
        visualisation.activate()
    cudaSimulation.simulate()


# cudaSimulation.exportData("end.xml");

    if pyflamegpu.VISUALISATION:
        visualisation.join()

# Ensure profiling / memcheck work correctly
    pyflamegpu.cleanup()

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(end-start)
    exit()