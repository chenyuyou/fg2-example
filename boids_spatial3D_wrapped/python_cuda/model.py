#! /usr/bin/env python3
from pyflamegpu import *
import sys, random, time
from cuda import *
import math, pathlib

def vec3Mult(x, y, z, multiplier):
    x *= multiplier
    y *= multiplier
    z *= multiplier
    
def vec3Div(x, y, z, divisor):
    x /= divisor
    y /= divisor
    z /= divisor

def vec3Normalize(x, y, z):
    length = math.sqrt(x * x + y * y + z * z)
    vec3Div(x, y, z, length)

def create_model():
    model = pyflamegpu.ModelDescription("Boids_BruteForce (RTC)")
    return model

def define_environment(model):
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
    env.newPropertyFloat("TIME_SCALE", 0.001)
    env.newPropertyFloat("GLOBAL_SCALE", 0.25)

# Rule scalers
    env.newPropertyFloat("STEER_SCALE", 0.055)
    env.newPropertyFloat("COLLISION_SCALE", 10.0)
    env.newPropertyFloat("MATCH_SCALE", 0.015)
    return env

def define_messages(model, env):
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
    
def define_agents(model):
    agent = model.newAgent("Boid")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("fx")
    agent.newVariableFloat("fy")
    agent.newVariableFloat("fz")
    agent.newVariableFloat("wing_position")
    agent.newVariableFloat("wing_animation")
    agent.newRTCFunction("outputdata", outputdata).setMessageOutput("location")
    agent.newRTCFunction("inputdata", inputdata).setMessageInput("location")

def define_execution_order(model):
# Layer #1
    model.newLayer().addAgentFunction("Boid", "outputdata")
# Layer #2
    model.newLayer().addAgentFunction("Boid", "inputdata")

def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)

    cudaSimulation = pyflamegpu.CUDASimulation(model)


    if pyflamegpu.VISUALISATION:
        visualisation = cudaSimulation.getVisualisation()
    # Configure vis
        envWidth = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION")
        INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 3
        visualisation.setInitialCameraLocation(0, 0, INIT_CAM)
        visualisation.setCameraSpeed(0.001 * envWidth)
        visualisation.setViewClips(0.00001, 50)
        circ_agt = visualisation.addAgent("Boid")
    # Position vars are named x, y, z; so they are used by default
        circ_agt.setForwardXVariable("fx")
        circ_agt.setForwardYVariable("fy")
        circ_agt.setForwardZVariable("fz")

        script_dir = pathlib.Path(__file__).parent.resolve()
        bird_a = str(script_dir / "model/bird_a.obj")
        bird_b = str(script_dir / "model/bird_b.obj")
        circ_agt.setKeyFrameModel(bird_a, bird_b, "wing_animation")
        circ_agt.setModelScale(env.getPropertyFloat("SEPARATION_RADIUS") /2.0)
    # Add a settings UI
        ui = visualisation.newUIPanel("Settings")
        ui.newSection("Model Parameters")
        ui.newEnvironmentPropertySliderFloat("TIME_SCALE", 0.00001, 0.01)
        ui.newEnvironmentPropertySliderFloat("GLOBAL_SCALE", 0.05, 0.5)
        ui.newEnvironmentPropertySliderFloat("SEPARATION_RADIUS", 0.01, env.getPropertyFloat("INTERACTION_RADIUS"))
        ui.newEnvironmentPropertySliderFloat("STEER_SCALE", 0.00, 1.0)
        ui.newEnvironmentPropertySliderFloat("COLLISION_SCALE", 0.00, 100.0)
        ui.newEnvironmentPropertySliderFloat("MATCH_SCALE", 0.00, 0.10)
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
        # initialise wing speed
            instance.setVariableFloat("wing_position", random.uniform(0, 3.14))
            instance.setVariableFloat("wing_animation", 0)

        cudaSimulation.setPopulationData(population)

    cudaSimulation.simulate()


# cudaSimulation.exportData("end.xml");

    if pyflamegpu.VISUALISATION:
        visualisation.join()

# Ensure profiling / memcheck work correctly
    pyflamegpu.cleanup()

if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()