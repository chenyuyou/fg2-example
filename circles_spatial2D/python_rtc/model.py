from pyflamegpu import *
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
    env.newPropertyFloat("ENV_WIDTH", int(env.getPropertyUInt("AGENT_COUNT")**(1/3)))  
    env.newPropertyFloat("repulse", 0.05)
    return env

def define_messages(model, env):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    message = model.newMessageSpatial2D("location")
    message.newVariableID("id")
    message.setRadius(1)
    message.setMin(0, 0)
    message.setMax(env.getPropertyFloat("ENV_WIDTH"), env.getPropertyFloat("ENV_WIDTH"))

def define_agents(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("point")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("drift", 0)
#   有关信息的描述是FlameGPU2的关键特色，还需要进一步理解。
    out_fn = agent.newRTCFunction("output_message", output_message)
    out_fn.setMessageOutput("location")
    in_fn = agent.newRTCFunction("input_message", input_message)
    in_fn.setMessageInput("location")

def define_execution_order(model):
#   引入层主要目的是确定agent行动的顺序。
    layer = model.newLayer()
    layer.addAgentFunction("point", "output_message")
    layer = model.newLayer()
    layer.addAgentFunction("point", "input_message")

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
        m_vis = cudaSimulation.getVisualisation()
#   设置相机所在位置和速度
        INIT_CAM = env.getPropertyFloat("ENV_WIDTH")/2
        m_vis.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0)
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, env.getPropertyFloat("ENV_WIDTH"))
        m_vis.setCameraSpeed(0.01)
        m_vis.setSimulationSpeed(25)
#   将“point” agent添加到可视化中
        point_agt = m_vis.addAgent("point")
#   设置“point” agent的形状和大小
        point_agt.setModel(pyflamegpu.ICOSPHERE)
        point_agt.setModelScale(1/10.0)
#   标记环境边界 
        pen = m_vis.newPolylineSketch(1, 1, 1, 0.2)
        pen.addVertex(0, 0, 0)
        pen.addVertex(0, env.getPropertyFloat("ENV_WIDTH"), 0)
        pen.addVertex(env.getPropertyFloat("ENV_WIDTH"), env.getPropertyFloat("ENV_WIDTH"), 0)
        pen.addVertex(env.getPropertyFloat("ENV_WIDTH"), 0, 0)
        pen.addVertex(0, 0, 0)
#   打开可视化窗口
        m_vis.activate()
    cudaSimulation.initialise(sys.argv) 
    
#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
#   在空间内均匀分布agent，具有均匀分布的初始速度。
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        population = pyflamegpu.AgentVector(model.Agent("point"), env.getPropertyUInt("AGENT_COUNT"))
        for i in range(env.getPropertyUInt("AGENT_COUNT")):
            instance = population[i]
            instance.setVariableFloat("x",  random.uniform(0.0, env.getPropertyFloat("ENV_WIDTH")))
            instance.setVariableFloat("y",  random.uniform(0.0, env.getPropertyFloat("ENV_WIDTH")))
        cudaSimulation.setPopulationData(population)
    cudaSimulation.simulate()

#    if pyflamegpu.VISUALISATION:
    # 模拟完成后保持可视化窗口处于活动状态
#        m_vis.join()

if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()