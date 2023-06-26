from pyflamegpu import *
from numpy import cbrt, floor
import time, sys, random
from cuda import *

def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("Circles Spatial2D")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
    env = model.Environment()
    env.newPropertyUInt("AGENT_COUNT", 16384)
    env.newPropertyFloat("ENV_MAX", floor(cbrt(env.getPropertyUInt("AGENT_COUNT"))))  
    env.newPropertyFloat("repulse", 0.05)
    env.newPropertyFloat("radius", 2.0)
    return env

def define_messages(model, env):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    message = model.newMessageBruteForce("location")
    message.newVariableID("id")
    message.newVariableFloat("x")
    message.newVariableFloat("y")
    message.newVariableFloat("z")


def define_agents(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("Circle")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("drift")
#   有关信息的描述是FlameGPU2的关键特色，还需要进一步理解。
    out_fn = agent.newRTCFunction("output_message", output_message)
    out_fn.setMessageOutput("location")
    in_fn = agent.newRTCFunction("move", move)
    in_fn.setMessageInput("location")

def define_execution_order(model):
#   引入层主要目的是确定agent行动的顺序。
    layer = model.newLayer()
    layer.addAgentFunction("Circle", "output_message")
    layer = model.newLayer()
    layer.addAgentFunction("Circle", "move")

def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)
#   初始化cuda模拟
    cudaSimulation = pyflamegpu.CUDASimulation(model)
    cudaSimulation.initialise(sys.argv)

#   设置可视化
    if pyflamegpu.VISUALISATION:
        m_vis = cudaSimulation.getVisualisation()
#   设置相机所在位置和速度
        INIT_CAM = env.getPropertyFloat("ENV_MAX") * 1.25
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM)
        m_vis.setCameraSpeed(0.02)
#   将“point” agent添加到可视化中
        point_agt = m_vis.addAgent("Circle")
#   设置“point” agent的形状和大小
        point_agt.setModel(pyflamegpu.ICOSPHERE)
        point_agt.setModelScale(1/10.0)

#   打开可视化窗口
        m_vis.activate()
    
#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
#   在空间内均匀分布agent，具有均匀分布的初始速度。
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        population = pyflamegpu.AgentVector(model.Agent("Circle"), env.getPropertyUInt("AGENT_COUNT"))
        for i in range(env.getPropertyUInt("AGENT_COUNT")):
            instance = population[i]
            instance.setVariableFloat("x",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
            instance.setVariableFloat("y",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
            instance.setVariableFloat("z",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
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