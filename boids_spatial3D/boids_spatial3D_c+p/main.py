#! /usr/bin/env python3
from pyflamegpu import *
import sys, random, time
from cuda import *
from func import *


def main():
    model = pyflamegpu.ModelDescription("Boids Spatial3D (Python)")


    env = model.Environment()
# Population size to generate, if no agents are loaded from disk
    env.newPropertyUInt("POPULATION_TO_GENERATE", 40000)

# Environment Bounds
    env.newPropertyFloat("MIN_POSITION", -0.5)
    env.newPropertyFloat("MAX_POSITION", +0.5)

# Initialisation parameter(s)
    env.newPropertyFloat("MAX_INITIAL_SPEED", 1.0)
    env.newPropertyFloat("MIN_INITIAL_SPEED", 0.1)

# Interaction radius
    env.newPropertyFloat("INTERACTION_RADIUS", 0.05)
    env.newPropertyFloat("SEPARATION_RADIUS", 0.01)

# Global Scalers
    env.newPropertyFloat("TIME_SCALE", 0.0005)
    env.newPropertyFloat("GLOBAL_SCALE", 0.15)

# Rule scalers
    env.newPropertyFloat("STEER_SCALE", 0.055)
    env.newPropertyFloat("COLLISION_SCALE", 10.0)
    env.newPropertyFloat("MATCH_SCALE", 0.015)


    message = model.newMessageSpatial3D("location")
# Set the range and bounds.
    message.setRadius(env.getPropertyFloat("INTERACTION_RADIUS"))
    message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"))
    message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"))
# A message to hold the location of an agent.
    message.newVariableID("id")
# X Y Z are implicit.
# message.newVariable<float>("x");
# message.newVariable<float>("y");
# message.newVariable<float>("z");
    message.newVariableFloat("fx")
    message.newVariableFloat("fy")
    message.newVariableFloat("fz")
    

    agent = model.newAgent("Boid")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("fx")
    agent.newVariableFloat("fy")
    agent.newVariableFloat("fz")
    agent.newRTCFunction("outputdata", outputdata).setMessageOutput("location")
    agent.newRTCFunction("inputdata", inputdata).setMessageInput("location")


# Layer #1
    model.newLayer().addAgentFunction("Boid", "outputdata")
# Layer #2
    model.newLayer().addAgentFunction("Boid", "inputdata")

    cudaSimulation = pyflamegpu.CUDASimulation(model)


    if pyflamegpu.VISUALISATION:
        visualisation = cudaSimulation.getVisualisation()
    # Configure vis
        envWidth = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION")
        INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 1.25
        visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM)
        visualisation.setCameraSpeed(0.001 * envWidth)
        visualisation.setViewClips(0.00001, 50)
        circ_agt = visualisation.addAgent("Boid")
    # Position vars are named x, y, z; so they are used by default
        circ_agt.setForwardXVariable("fx")
        circ_agt.setForwardYVariable("fy")
        circ_agt.setForwardZVariable("fz")
        circ_agt.setModel(pyflamegpu.STUNTPLANE)
        circ_agt.setModelScale(env.getPropertyFloat("SEPARATION_RADIUS") /3.0)
    # Add a settings UI
        ui = visualisation.newUIPanel("Environment")
        ui.newStaticLabel("Interaction")
        ui.newEnvironmentPropertyDragFloat("INTERACTION_RADIUS", 0.0, 0.05, 0.001)
        ui.newEnvironmentPropertyDragFloat("SEPARATION_RADIUS", 0.0, 0.05, 0.001)
        ui.newStaticLabel("Environment Scalars")
        ui.newEnvironmentPropertyDragFloat("TIME_SCALE", 0.0, 1.0, 0.0001)
        ui.newEnvironmentPropertyDragFloat("GLOBAL_SCALE", 0.0, 0.5, 0.001)
        ui.newStaticLabel("Force Scalars")
        ui.newEnvironmentPropertyDragFloat("STEER_SCALE", 0.0, 10.0, 0.001)
        ui.newEnvironmentPropertyDragFloat("COLLISION_SCALE", 0.0, 10.0, 0.001)
        ui.newEnvironmentPropertyDragFloat("MATCH_SCALE", 0.0, 10.0, 0.001)
        visualisation.activate()



    cudaSimulation.initialise(sys.argv)

# If no xml model file was is provided, generate a population.
    if not cudaSimulation.SimulationConfig().input_file:
    # Uniformly distribute agents within space, with uniformly distributed initial velocity.
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        min_pos = env.getPropertyFloat("MIN_POSITION")
        max_pos = env.getPropertyFloat("MAX_POSITION")
        min_speed = env.getPropertyFloat("MIN_INITIAL_SPEED")
        max_speed = env.getPropertyFloat("MAX_INITIAL_SPEED")
        populationSize = env.getPropertyUInt("POPULATION_TO_GENERATE")
        population = pyflamegpu.AgentVector(model.Agent("Boid"), populationSize)
        for i in range(populationSize):
            instance = population[i]

        # Agent position in space
            instance.setVariableFloat("x", random.uniform(min_pos, max_pos))
            instance.setVariableFloat("y", random.uniform(min_pos, max_pos))
            instance.setVariableFloat("z", random.uniform(min_pos, max_pos))

        # Generate a random velocity direction
            fx = random.uniform(-1, 1)
            fy = random.uniform(-1, 1)
            fz = random.uniform(-1, 1)
            # Generate a random speed between 0 and the maximum initial speed
            fmagnitude = random.uniform(min_speed, max_speed)
        # Use the random speed for the velocity.
            vec3Normalize(fx, fy, fz)
            vec3Mult(fx, fy, fz, fmagnitude)

        # Set these for the agent.
            instance.setVariableFloat("fx", fx)
            instance.setVariableFloat("fy", fy)
            instance.setVariableFloat("fz", fz)

        cudaSimulation.setPopulationData(population)

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