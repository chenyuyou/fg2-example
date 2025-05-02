#! /usr/bin/env python3
from pyflamegpu import *
from numpy import cbrt, floor
import random, time, math, sys
from cuda import *



def create_model():
    model = pyflamegpu.ModelDescription("Circles Spatial3D")
    return model

def define_environment(model):
    env = model.Environment()
    env.newPropertyUInt("AGENT_COUNT", 27000)
 #   env.newPropertyFloat("ENV_MAX", floor(cbrt(env.getPropertyUInt("AGENT_COUNT"))))  # 两种方式都行
    env.newPropertyFloat("ENV_MAX", int(env.getPropertyUInt("AGENT_COUNT")**(1/3)))  # 环境的边长
    env.newPropertyFloat("RADIUS", 5.0)     # 消息起作用的半径
    env.newPropertyFloat("repulse", 0.5)    # # 模型的斥力
    return env

def define_messages(model, env):    #   FLAME GPU 中将空间划分为格子的方法来实现局部通信
                                    #   用户只需设置环境边界 (setMin, setMax) 和消息的有效半径 (setRadius)，FLAME GPU 会自动处理底层的空间分区和消息组织
    message = model.newMessageSpatial3D("location")
    message.newVariableID("id")
    message.setRadius(env.getPropertyFloat("RADIUS"))
    message.setMin(0, 0, 0)
    message.setMax(env.getPropertyFloat("ENV_MAX"), env.getPropertyFloat("ENV_MAX"), env.getPropertyFloat("ENV_MAX"))

def define_agents(model):
    agent = model.newAgent("Circle")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("drift")
#    用字符串形式提供的 CUDA C++ 代理函数代码注册到 FLAME GPU 模型中
    fn = agent.newRTCFunction("output_message", output_message)
    fn.setMessageOutput("location")
    fn = agent.newRTCFunction("move", move)
    fn.setMessageInput("location")


def define_execution_order(model):
# Layer #1
    layer = model.newLayer()
    layer.addAgentFunction("Circle", "output_message")
# Layer #2
    layer = model.newLayer()
    layer.addAgentFunction("Circle", "move")


def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)

    cudaSimulation = pyflamegpu.CUDASimulation(model)
    cudaSimulation.initialise(sys.argv)

    if pyflamegpu.VISUALISATION:
        m_vis = cudaSimulation.getVisualisation()
    
        INIT_CAM = env.getPropertyFloat("ENV_MAX") * 1.25
        # 设置可视化相机的初始位置。
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM)
        # 设置相机移动的速度。
        m_vis.setCameraSpeed(0.01)
        # 设置模拟的可视化播放速度（例如，每秒显示多少个模拟步长）
        m_vis.setSimulationSpeed(25)
        #   将名为 "Circle" 的代理类型添加到可视化中，以便在窗口中显示它们。
        circ_agt = m_vis.addAgent("Circle")
        #   设置 "Circle" 代理在可视化中使用的 3D 模型形状，这里使用了球体 (ICOSPHERE)
        circ_agt.setModel(pyflamegpu.ICOSPHERE)
        # 代理的可视化模型的大小比例。
        circ_agt.setModelScale(1/10.0)

# 假设 env 已经定义并包含了 ENV_WIDTH, ENV_HEIGHT, ENV_DEPTH 属性
        env_width = env.getPropertyFloat("ENV_MAX")
        env_height = env.getPropertyFloat("ENV_MAX")
        env_depth = env.getPropertyFloat("ENV_MAX")

        # 创建一个新的折线 sketch
        # 第一个参数是线条粗细，接下来的三个参数是颜色 (R, G, B)，最后一个是透明度 (Alpha)
        pen = m_vis.newPolylineSketch(1, 0, 0, 0.5)

        # 绘制立方体底部的四条边
        pen.addVertex(0, 0, 0)
        pen.addVertex(env_width, 0, 0)

        pen.addVertex(env_width, 0, 0)
        pen.addVertex(env_width, env_height, 0)

        pen.addVertex(env_width, env_height, 0)
        pen.addVertex(0, env_height, 0)

        pen.addVertex(0, env_height, 0)
        pen.addVertex(0, 0, 0)

        # 绘制立方体顶部的四条边
        pen.addVertex(0, 0, env_depth)
        pen.addVertex(env_width, 0, env_depth)

        pen.addVertex(env_width, 0, env_depth)
        pen.addVertex(env_width, env_height, env_depth)

        pen.addVertex(env_width, env_height, env_depth)
        pen.addVertex(0, env_height, env_depth)

        pen.addVertex(0, env_height, env_depth)
        pen.addVertex(0, 0, env_depth)

        # 绘制连接底部和顶部的四条垂直边
        pen.addVertex(0, 0, 0)
        pen.addVertex(0, 0, env_depth)

        pen.addVertex(env_width, 0, 0)
        pen.addVertex(env_width, 0, env_depth)

        pen.addVertex(env_width, env_height, 0)
        pen.addVertex(env_width, env_height, env_depth)

        pen.addVertex(0, env_height, 0)
        pen.addVertex(0, env_height, env_depth)

        m_vis.activate()

# If no xml model file was is provided, generate a population.
    if not cudaSimulation.SimulationConfig().input_file:
    # Uniformly distribute agents within space, with uniformly distributed initial velocity.
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        population = pyflamegpu.AgentVector(model.Agent("Circle"), env.getPropertyUInt("AGENT_COUNT"))
        for i in range(env.getPropertyUInt("AGENT_COUNT")):
            instance = population[i]
            instance.setVariableFloat("x",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
            instance.setVariableFloat("y",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
            instance.setVariableFloat("z",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
        cudaSimulation.setPopulationData(population)

    cudaSimulation.simulate()
#    cudaSimulation.exportData("end.xml")


#    if pyflamegpu.VISUALISATION:
#        m_vis.join()

if __name__ == "__main__":
    import os
    start = time.time()
    # 获取 4 个随机字节作为种子
    random_bytes = os.urandom(4)
    # 将字节转换为整数
    random_seed = int.from_bytes(random_bytes, byteorder='big')
    print(f"Using random seed: {random_seed}")
    initialise_simulation(random_seed)
    end = time.time()
    print(end - start)
    exit()
