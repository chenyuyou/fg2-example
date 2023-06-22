from pyflamegpu import *
import time, sys, random
from cuda import *



class create_agents(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # Fetch the desired agent count and environment width
        AGENT_COUNT = FLAMEGPU.environment.getPropertyUInt("AGENT_COUNT")
        ENV_WIDTH = FLAMEGPU.environment.getPropertyFloat("ENV_WIDTH")
        # Create agents
        t_pop = FLAMEGPU.agent("point")
        for i in range(AGENT_COUNT):
            t = t_pop.newAgent()
            t.setVariableFloat("x", FLAMEGPU.random.uniformFloat() * ENV_WIDTH)
            t.setVariableFloat("y", FLAMEGPU.random.uniformFloat() * ENV_WIDTH)



def create_model():
    model = pyflamegpu.ModelDescription("Circles Spatial2D")
    return model


def define_environment(model):
    env = model.Environment()
    env.newPropertyUInt("AGENT_COUNT", 16384)
    env.newPropertyFloat("ENV_WIDTH", int(env.getPropertyUInt("AGENT_COUNT")**(1/3)))  
    env.newPropertyFloat("repulse", 0.05)
    return env

def define_messages(model, env):
#   建立一个2D信息“location”，用于管理代理的位置信息
    message = model.newMessageSpatial2D("location")
    message.newVariableID("id")
    message.setRadius(1)
    message.setMin(0, 0)
    message.setMax(env.getPropertyFloat("ENV_WIDTH"), env.getPropertyFloat("ENV_WIDTH"))

def define_agents(model):
    agent = model.newAgent("point")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("drift", 0)
    out_fn = agent.newRTCFunction("output_message", output_message)
    out_fn.setMessageOutput("location")
    in_fn = agent.newRTCFunction("input_message", input_message)
    in_fn.setMessageInput("location")


def define_execution_order(model):
# Layer #1
    layer = model.newLayer()
    layer.addAgentFunction("point", "output_message")
# Layer #2
    layer = model.newLayer()
    layer.addAgentFunction("point", "input_message")


def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)

    cudaSimulation = pyflamegpu.CUDASimulation(model)
    if seed is not None:
        cudaSimulation.SimulationConfig().random_seed = seed
        cudaSimulation.applyConfig()

    if pyflamegpu.VISUALISATION:
        m_vis = cudaSimulation.getVisualisation()
        INIT_CAM = env.getPropertyFloat("ENV_WIDTH") / 2
        m_vis.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0)
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, env.getPropertyFloat("ENV_WIDTH"))
        m_vis.setCameraSpeed(0.01)
        m_vis.setSimulationSpeed(25)
        point_agt = m_vis.addAgent("point")
    # Location variables have names "x" and "y" so will be used by default
        point_agt.setModel(pyflamegpu.ICOSPHERE)
        point_agt.setModelScale(1/10.0)
    # Mark the environment bounds
        pen = m_vis.newPolylineSketch(1, 1, 1, 0.2)
        pen.addVertex(0, 0, 0)
        pen.addVertex(0, env.getPropertyFloat("ENV_WIDTH"), 0)
        pen.addVertex(env.getPropertyFloat("ENV_WIDTH"), env.getPropertyFloat("ENV_WIDTH"), 0)
        pen.addVertex(env.getPropertyFloat("ENV_WIDTH"), 0, 0)
        pen.addVertex(0, 0, 0)
    # Open the visualiser window
        m_vis.activate()
    cudaSimulation.initialise(sys.argv)


    # If no xml model file was is provided, generate a population.
    if not cudaSimulation.SimulationConfig().input_file:
    # Uniformly distribute agents within space, with uniformly distributed initial velocity.
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        population = pyflamegpu.AgentVector(model.Agent("point"), env.getPropertyUInt("AGENT_COUNT"))
        for i in range(env.getPropertyUInt("AGENT_COUNT")):
            instance = population[i]

            instance.setVariableFloat("x",  random.uniform(0.0, env.getPropertyFloat("ENV_WIDTH")))
            instance.setVariableFloat("y",  random.uniform(0.0, env.getPropertyFloat("ENV_WIDTH")))
           

        cudaSimulation.setPopulationData(population)

    cudaSimulation.simulate()

    if pyflamegpu.VISUALISATION:
    # 模拟完成后保持可视化窗口处于活动状态
        m_vis.join()

if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()