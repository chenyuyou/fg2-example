from pyflamegpu import *
import time, sys, random, math
from cuda import *

def vec3Length(x, y, z):
    return math.sqrt(x * x + y * y + z * z)

def vec3Add(x, y, z, value):
    x += value
    y += value
    z += value

def vec3Sub(x, y, z, value):
    x -= value
    y -= value
    z -= value

def vec3Mult(x, y, z, multiplier):
    x *= multiplier
    y *= multiplier
    z *= multiplier


def vec3Div(x, y, z, divisor):
    x /= divisor
    y /= divisor
    z /= divisor

def vec3Normalize(x, y, z):
    # Get the length
    length = vec3Length(x, y, z)
    vec3Div(x, y, z, length)

def clampPosition(x, y, z, MIN_POSITION, MAX_POSITION):
    x = MIN_POSITION if (x < MIN_POSITION) else x
    x = MAX_POSITION if (x > MAX_POSITION) else x

    y = MIN_POSITION if (y < MIN_POSITION) else y
    y = MAX_POSITION if (y > MAX_POSITION) else y

    z = MIN_POSITION if (z < MIN_POSITION) else z
    z = MAX_POSITION if (z > MAX_POSITION) else z

def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("Boids Spatial3D (Python)")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
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
    return env

def define_messages(model, env):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    message = model.newMessageBruteForce("location")
    message.newVariableID("id")
    message.setRadius(env.getPropertyFloat("INTERACTION_RADIUS"))
    message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"))
    message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"))
#    message.newVariableFloat("x")
#    message.newVariableFloat("y")
#    message.newVariableFloat("z")
    message.newVariableFloat("fx")
    message.newVariableFloat("fy")
    message.newVariableFloat("fz")


def define_agents(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("Boid")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("fx")
    agent.newVariableFloat("fy")
    agent.newVariableFloat("fz")
    fn = agent.newRTCFunction("outputdata", outputdata)
    fn.setMessageOutput("location")
    fn = agent.newRTCFunction("inputdata", inputdata)
    fn.setMessageInput("location")

def define_execution_order(model):
#   引入层主要目的是确定agent行动的顺序。
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


#   设置可视化
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

#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
#   在空间内均匀分布agent，具有均匀分布的初始速度。
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        min_pos = env.getPropertyFloat("MIN_POSITION")
        max_pos = env.getPropertyFloat("MAX_POSITION")
        min_speed = env.getPropertyFloat("MIN_INITIAL_SPEED")
        max_speed = env.getPropertyFloat("MAX_INITIAL_SPEED")
        populationSize = env.getPropertyUInt("POPULATION_TO_GENERATE")
        population = pyflamegpu.AgentVector(model.Agent("Boid"), populationSize)
        for i in range(populationSize):
            instance = population[i]
            instance.setVariableFloat("x",  random.uniform(min_pos, max_pos))
            instance.setVariableFloat("y",  random.uniform(min_pos, max_pos))
            instance.setVariableFloat("z",  random.uniform(min_pos, max_pos))

            fx = random.uniform(-1, 1)
            fy = random.uniform(-1, 1)
            fz = random.uniform(-1, 1)

            fmagnitude = random.uniform(min_speed, max_speed)

            vec3Normalize(fx, fy, fz)
            vec3Mult(fx, fy, fz, fmagnitude)

            instance.setVariableFloat("fx", fx)
            instance.setVariableFloat("fy", fy)
            instance.setVariableFloat("fz", fz)

        cudaSimulation.setPopulationData(population)
    cudaSimulation.simulate()

    if pyflamegpu.VISUALISATION:
    # 模拟完成后保持可视化窗口处于活动状态
        visualisation.join()
    pyflamegpu.cleanup()

if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()