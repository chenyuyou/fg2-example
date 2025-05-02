from pyflamegpu import *
from numpy import cbrt, floor
import time, sys, random
from cuda import *

def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("Circles Bruteforce")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
    env = model.Environment()
    env.newPropertyUInt("AGENT_COUNT", 27000)   # 代理个数
    env.newPropertyFloat("ENV_MAX", floor(cbrt(env.getPropertyUInt("AGENT_COUNT"))))  
    env.newPropertyFloat("repulse", 0.5)    # 模型的斥力
    env.newPropertyFloat("radius", 5.0)     # 消息起作用的半径
    return env

def define_messages(model, env):        #  全局通信，不需要setMin和setMax
#   Brute Force 消息是全局可访问的，并且不依赖于代理的空间位置进行过滤，所以它不需要知道模拟空间的边界。设定的最大坐标和最小坐标是用于空间消息类型来构建空间数据结构，从而优化基于位置的消息查找。对于 Brute Force 来说，它并不关心消息的物理位置，只关心消息本身的内容
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
#    用字符串形式提供的 CUDA C++ 代理函数代码注册到 FLAME GPU 模型中
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
#   创建一个 CUDASimulation 对象。这是 FLAME GPU 的核心类之一，负责在 CUDA 启用设备（通常是 GPU）上运行模拟。它将之前创建的 model 对象作为输入，以便知道要模拟什么。
    cudaSimulation = pyflamegpu.CUDASimulation(model)
    cudaSimulation.initialise(sys.argv)

#   设置可视化
    if pyflamegpu.VISUALISATION:
        # 获取与 CUDA 模拟相关的可视化对象。
        m_vis = cudaSimulation.getVisualisation()

        INIT_CAM = env.getPropertyFloat("ENV_MAX") * 1.25
        # 设置可视化相机的初始观察目标点。
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM)
        # 设置相机移动的速度。
        m_vis.setCameraSpeed(0.02)
#   将“point” agent添加到可视化中
        point_agt = m_vis.addAgent("Circle")
#   设置“point” agent的形状和大小
        point_agt.setModel(pyflamegpu.ICOSPHERE)
        point_agt.setModelScale(1/10.0)

# 假设 env 已经定义并包含了 ENV_WIDTH, ENV_HEIGHT, ENV_DEPTH 属性
        env_width = env.getPropertyFloat("ENV_MAX")
        env_height = env.getPropertyFloat("ENV_MAX")
        env_depth = env.getPropertyFloat("ENV_MAX")

        # 创建一个新的折线 sketch
        # 第一个参数是线条粗细，接下来的三个参数是颜色 (R, G, B)，最后一个是透明度 (Alpha)
        pen = m_vis.newPolylineSketch(1, 0, 0, 0.2)

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
#    这是运行模拟的核心命令。它根据模型定义、环境、消息、代理和执行顺序，在 GPU 上执行模拟步长。模拟将一直运行，直到达到预定的模拟步长数或者通过命令行参数设置了其他停止条件。
    cudaSimulation.simulate()

#    if pyflamegpu.VISUALISATION:
    # 模拟完成后保持可视化窗口处于活动状态
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