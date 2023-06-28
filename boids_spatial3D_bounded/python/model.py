from pyflamegpu import *
import pyflamegpu.codegen
import sys, random, math, time

def vec3DivP(x, y, z, divisor):
    x /= divisor
    y /= divisor
    z /= divisor

def vec3NormalizeP(x, y, z):
    length = math.sqrt(x * x + y * y + z * z)
    vec3DivP(x, y, z, length)

def vec3MultP(x, y, z, multiplier):
    x *= multiplier
    y *= multiplier
    z *= multiplier

@pyflamegpu.device_function
def vec3Length(x: float, y: float, z: float) -> float :
    return math.sqrt(x * x + y * y + z * z)

@pyflamegpu.device_function  
def vec3Mult(x: float, y: float, z: float, multiplier: float)-> None:
    x *= multiplier
    y *= multiplier
    z *= multiplier

@pyflamegpu.device_function  
def vec3Div(x: float, y: float, z: float, divisor: float)-> None:
    x /= divisor
    y /= divisor
    z /= divisor

@pyflamegpu.device_function
def clampPosition(x: float, y: float, z: float, MIN_POSITION: float, MAX_POSITION: float)-> None:
    x = MIN_POSITION if (x < MIN_POSITION) else x
    x = MAX_POSITION if (x > MAX_POSITION) else x

    y = MIN_POSITION if (y < MIN_POSITION) else y
    y = MAX_POSITION if (y > MAX_POSITION) else y

    z = MIN_POSITION if (z < MIN_POSITION) else z
    z = MAX_POSITION if (z > MAX_POSITION) else z


@pyflamegpu.agent_function
def outputdata(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageSpatial3D):
    message_out.setVariableInt("id", pyflamegpu.getID())
    message_out.setVariableFloat("x", pyflamegpu.getVariableFloat("x"))
    message_out.setVariableFloat("y", pyflamegpu.getVariableFloat("y"))
    message_out.setVariableFloat("z", pyflamegpu.getVariableFloat("z"))
    message_out.setVariableFloat("fx", pyflamegpu.getVariableFloat("fx"))
    message_out.setVariableFloat("fy", pyflamegpu.getVariableFloat("fy"))
    message_out.setVariableFloat("fz", pyflamegpu.getVariableFloat("fz"))
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def inputdata(message_in: pyflamegpu.MessageSpatial3D, message_out: pyflamegpu.MessageNone):
    id = pyflamegpu.getID()
    # Agent position
    agent_x = pyflamegpu.getVariableFloat("x")
    agent_y = pyflamegpu.getVariableFloat("y")
    agent_z = pyflamegpu.getVariableFloat("z")
    #/ Agent velocity
    agent_fx = pyflamegpu.getVariableFloat("fx")
    agent_fy = pyflamegpu.getVariableFloat("fy")
    agent_fz = pyflamegpu.getVariableFloat("fz")

    # Boids percieved centerexit()
    perceived_centre_x = 0.0
    perceived_centre_y = 0.0
    perceived_centre_z = 0.0
    perceived_count = 0

    # Boids global velocity matching
    global_velocity_x = 0.0
    global_velocity_y = 0.0
    global_velocity_z = 0.0

    # Total change in velocity
    velocity_change_x = 0.0
    velocity_change_y = 0.0
    velocity_change_z = 0.0

    INTERACTION_RADIUS = pyflamegpu.environment.getPropertyFloat("INTERACTION_RADIUS")
    SEPARATION_RADIUS = pyflamegpu.environment.getPropertyFloat("SEPARATION_RADIUS")
    # Iterate location messages, accumulating relevant data and counts.
    for message in message_in.wrap(agent_x, agent_y, agent_z) :
        # Ignore self messages.
        if message.getVariableInt("id") != id :
            # Get the message location and velocity.
            message_x = message.getVirtualX()
            message_y = message.getVirtualY()
            message_z = message.getVirtualZ()

            # Check interaction radius
            separation = vec3Length(agent_x - message_x, agent_y - message_y, agent_z - message_z)

            if separation < INTERACTION_RADIUS :
                # Update the perceived centre
                perceived_centre_x += message_x
                perceived_centre_y += message_y
                perceived_centre_z += message_z
                perceived_count += 1

                # Update perceived velocity matching
                message_fx = message.getVariableFloat("fx")
                message_fy = message.getVariableFloat("fy")
                message_fz = message.getVariableFloat("fz")
                global_velocity_x += message_fx
                global_velocity_y += message_fy
                global_velocity_z += message_fz

                # Update collision centre
                if separation < SEPARATION_RADIUS :  # dependant on model size
                    # Rule 3) Avoid other nearby boids (Separation)
                    normalizedSeparation = (separation / SEPARATION_RADIUS)
                    invNormSep = (float(1.0) - normalizedSeparation)
                    invSqSep = invNormSep * invNormSep

                    collisionScale = pyflamegpu.environment.getPropertyFloat("COLLISION_SCALE")
                    velocity_change_x += collisionScale * (agent_x - message_x) * invSqSep
                    velocity_change_y += collisionScale * (agent_y - message_y) * invSqSep
                    velocity_change_z += collisionScale * (agent_z - message_z) * invSqSep

    if (perceived_count) :
        # Divide positions/velocities by relevant counts.
        vec3Div(perceived_centre_x, perceived_centre_y, perceived_centre_z, perceived_count)
        vec3Div(global_velocity_x, global_velocity_y, global_velocity_z, perceived_count)

        # Rule 1) Steer towards perceived centre of flock (Cohesion)
        steer_velocity_x = 0.0
        steer_velocity_y = 0.0
        steer_velocity_z = 0.0

        STEER_SCALE = pyflamegpu.environment.getPropertyFloat("STEER_SCALE")
        steer_velocity_x = (perceived_centre_x - agent_x) * STEER_SCALE
        steer_velocity_y = (perceived_centre_y - agent_y) * STEER_SCALE
        steer_velocity_z = (perceived_centre_z - agent_z) * STEER_SCALE

        velocity_change_x += steer_velocity_x
        velocity_change_y += steer_velocity_y
        velocity_change_z += steer_velocity_z

        # Rule 2) Match neighbours speeds (Alignment)
        match_velocity_x = 0.0
        match_velocity_y = 0.0
        match_velocity_z = 0.0

        MATCH_SCALE = pyflamegpu.environment.getPropertyFloat("MATCH_SCALE")
        match_velocity_x = global_velocity_x * MATCH_SCALE
        match_velocity_y = global_velocity_y * MATCH_SCALE
        match_velocity_z = global_velocity_z * MATCH_SCALE

        velocity_change_x += match_velocity_x - agent_fx
        velocity_change_y += match_velocity_y - agent_fy
        velocity_change_z += match_velocity_z - agent_fz


    # Global scale of velocity change
    vec3Mult(velocity_change_x, velocity_change_y, velocity_change_z, pyflamegpu.environment.getPropertyFloat("GLOBAL_SCALE"))

    # Update agent velocity
    agent_fx += velocity_change_x
    agent_fy += velocity_change_y
    agent_fz += velocity_change_z

    # Bound velocity
    agent_fscale = vec3Length(agent_fx, agent_fy, agent_fz)
    if agent_fscale > 1 : 
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale)
    
    minSpeed = float(0.5)
    if agent_fscale < minSpeed :
        # Normalise
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale)
        # Scale to min
        vec3Mult(agent_fx, agent_fy, agent_fz, minSpeed)


    # Wrap positions
    wallInteractionDistance = 0.10
    wallSteerStrength = 0.05
    minPosition = pyflamegpu.environment.getPropertyFloat("MIN_POSITION")
    maxPosition = pyflamegpu.environment.getPropertyFloat("MAX_POSITION")

    if (agent_x - minPosition < wallInteractionDistance) :
        agent_fx += wallSteerStrength
    if (agent_y - minPosition < wallInteractionDistance) :
        agent_fy += wallSteerStrength
    if (agent_z - minPosition < wallInteractionDistance) :
        agent_fz += wallSteerStrength

    if (maxPosition - agent_x < wallInteractionDistance) :
        agent_fx -= wallSteerStrength
    if (maxPosition - agent_y < wallInteractionDistance) :
        agent_fy -= wallSteerStrength
    if (maxPosition - agent_z < wallInteractionDistance) :
        agent_fz -= wallSteerStrength

        # Apply the velocity
    TIME_SCALE = pyflamegpu.environment.getPropertyFloat("TIME_SCALE")
    agent_x += agent_fx * TIME_SCALE
    agent_y += agent_fy * TIME_SCALE
    agent_z += agent_fz * TIME_SCALE


    clampPosition(agent_x, agent_y, agent_z, pyflamegpu.environment.getPropertyFloat("MIN_POSITION"), pyflamegpu.environment.getPropertyFloat("MAX_POSITION"))

    # Update global agent memory.
    pyflamegpu.setVariableFloat("x", agent_x)
    pyflamegpu.setVariableFloat("y", agent_y)
    pyflamegpu.setVariableFloat("z", agent_z)

    pyflamegpu.setVariableFloat("fx", agent_fx)
    pyflamegpu.setVariableFloat("fy", agent_fy)
    pyflamegpu.setVariableFloat("fz", agent_fz)

    return pyflamegpu.ALIVE

def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("Boids Spatial3D (Python)")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
    env = model.Environment()
    env.newPropertyUInt("POPULATION_TO_GENERATE", 4000)

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
    return env

def define_messages(model, env):
# Location message
    message = model.newMessageSpatial3D("location")
    
    message.setRadius(env.getPropertyFloat("INTERACTION_RADIUS"))
    message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"))
    message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"))
    message.newVariableID("id")
# message.newVariable<float>("x")
# message.newVariable<float>("y")
# message.newVariable<float>("z")
    message.newVariableFloat("fx")
    message.newVariableFloat("fy")
    message.newVariableFloat("fz")

def define_agents(model):
# Boid agent
    agent = model.newAgent("Boid")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("fx")
    agent.newVariableFloat("fy")
    agent.newVariableFloat("fz")
    outputdata_translated = pyflamegpu.codegen.translate(outputdata)
    inputdata_translated = pyflamegpu.codegen.translate(inputdata)
    agent.newRTCFunction("outputdata", outputdata_translated).setMessageOutput("location")
    agent.newRTCFunction("inputdata", inputdata_translated).setMessageInput("location")

def define_execution_order(model):
    layer = model.newLayer()
    layer.addAgentFunction("Boid", "outputdata")
    layer = model.newLayer()
    layer.addAgentFunction("Boid", "inputdata")

def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)
#   初始化cuda模拟
    cudaSimulation = pyflamegpu.CUDASimulation(model)

    if pyflamegpu.VISUALISATION:
        visualisation = cudaSimulation.getVisualisation()
# Configure vis
#    visualisation.setClearColor(255, 255, 255)
        envWidth = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION")
        INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 1.25
        visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM)
        visualisation.setCameraSpeed(0.001 * envWidth)
        visualisation.setViewClips(0.00001, 50)
        boid_agt = visualisation.addAgent("Boid")
    # Position vars are named x, y, z so they are used by default
        boid_agt.setForwardXVariable("fx")
        boid_agt.setForwardYVariable("fy")
        boid_agt.setForwardZVariable("fz")
    #boid_agt.setModel(pyflamegpu.ARROWHEAD) # Alternative simple 3D model for very large pop sizes
        boid_agt.setModel(pyflamegpu.STUNTPLANE)
        boid_agt.setModelScale(env.getPropertyFloat("SEPARATION_RADIUS") /3.0)
#    boid_agt.setColor(pyflamegpu.RED);
    # Visualisation UI
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
            vec3NormalizeP(fx, fy, fz)
            vec3MultP(fx, fy, fz, fmagnitude)

        # Set these for the agent.
            instance.setVariableFloat("fx", fx)
            instance.setVariableFloat("fy", fy)
            instance.setVariableFloat("fz", fz)

        cudaSimulation.setPopulationData(population)
    # Execution
        cudaSimulation.simulate()

# Export Pop (optional)
# cudaSimulation.exportData("end.xml")

    if pyflamegpu.VISUALISATION:
        visualisation.join()

# Ensure profiling / memcheck work correctly
#    pyflamegpu.cleanup()

if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()