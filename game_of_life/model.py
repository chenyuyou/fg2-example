#! /usr/bin/env python3
from pyflamegpu import *
import sys, random, time
from cuda import *


def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("Game of life")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
    env = model.Environment()
    env.newPropertyUInt("SQRT_AGENT_COUNT", 1000)
    env.newPropertyUInt("AGENT_COUNT", int(env.getPropertyUInt("SQRT_AGENT_COUNT")**2))  
    env.newPropertyFloat("repulse", 0.05)
    env.newPropertyFloat("radius", 1.0)
    return env

def define_messages(model, env):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    message = model.newMessageArray2D("is_alive_message")
    message.newVariableChar("is_alive")
    message.setDimensions(env.getPropertyUInt("SQRT_AGENT_COUNT"), env.getPropertyUInt("SQRT_AGENT_COUNT"))
    
def define_agents(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("cell")
    agent.newVariableArrayUInt("pos", 2)
    agent.newVariableUInt("is_alive")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
#   有关信息的描述是FlameGPU2的关键特色，还需要进一步理解。
    out_fn = agent.newRTCFunction("output", output)
    out_fn.setMessageOutput("is_alive_message")
    in_fn = agent.newRTCFunction("update", update)
    in_fn.setMessageInput("is_alive_message")

def define_execution_order(model):
#   引入层主要目的是确定agent行动的顺序。
    layer = model.newLayer()
    layer.addAgentFunction("cell", "output")
    layer = model.newLayer()
    layer.addAgentFunction("cell", "update")

def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)
#   初始化cuda模拟
    cudaSimulation = pyflamegpu.CUDASimulation(model)
#   设置随机参数
    if seed is not None:
        cudaSimulation.SimulationConfig().random_seed = seed
        cudaSimulation.applyConfig()
#   设置可视化
    if pyflamegpu.VISUALISATION:
        visualisation = cudaSimulation.getVisualisation()
#        visualisation.setBeginPaused(True)
#   设置相机所在位置和速度
        visualisation.setSimulationSpeed(5)
        visualisation.setInitialCameraLocation(env.getPropertyUInt("SQRT_AGENT_COUNT") / 2.0, env.getPropertyUInt("SQRT_AGENT_COUNT") / 2.0, 450.0)
        visualisation.setInitialCameraTarget(env.getPropertyUInt("SQRT_AGENT_COUNT") / 2.0, env.getPropertyUInt("SQRT_AGENT_COUNT") / 2.0, 0.0)
        visualisation.setCameraSpeed(0.001 * env.getPropertyUInt("SQRT_AGENT_COUNT"))
#        visualisation.setOrthographic(True)
#        visualisation.setOrthographicZoomModifier(1.409)
        visualisation.setViewClips(0.01, 2500)
        visualisation.setClearColor(0.6, 0.6, 0.6)
#   将“cell” agent添加到可视化中
        agt = visualisation.addAgent("cell")
        agt.setModel(pyflamegpu.CUBE)
        agt.setModelScale(1.0)
        cell_colors = pyflamegpu.uDiscreteColor("is_alive", pyflamegpu.Color("#666"))
        cell_colors[0] = pyflamegpu.BLACK
        cell_colors[1] = pyflamegpu.WHITE
        agt.setColor(cell_colors)
#   打开可视化窗口
        visualisation.activate()
    cudaSimulation.initialise(sys.argv)


#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
#   在空间内均匀分布agent，具有均匀分布的初始速度。
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        init_pop = pyflamegpu.AgentVector(model.Agent("cell"), env.getPropertyUInt("AGENT_COUNT"))
        for x in range(env.getPropertyUInt("SQRT_AGENT_COUNT")):
            for y in range(env.getPropertyUInt("SQRT_AGENT_COUNT")):
                init_pop.push_back()
                instance = init_pop.back()
                instance.setVariableArrayUInt("pos", [x, y])
                is_alive= 1 if random.random() < 0.4 else 0
                instance.setVariableUInt("is_alive", is_alive)
                instance.setVariableFloat("x", x)
                instance.setVariableFloat("y", y)
#                if pyflamegpu.VISUALISATION:
        # Agent position in space
#                    instance.setVariableFloat("x", x)
#                    instance.setVariableFloat("y", y)
        cudaSimulation.setPopulationData(init_pop)

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