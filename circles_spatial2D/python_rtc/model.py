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
    env.newPropertyUInt("AGENT_COUNT", 900)  # 代理个数
    env.newPropertyFloat("ENV_WIDTH", int(env.getPropertyUInt("AGENT_COUNT")**(1/2)))  # 环境的边长
    env.newPropertyFloat("repulse", 0.5)       # 模型的斥力
    env.newPropertyFloat("RADIUS", 3.0)        # 消息起作用的半径
    return env

def define_messages(model, env):    #   FLAME GPU 中将空间划分为格子的方法来实现局部通信
                                    #   用户只需设置环境边界 (setMin, setMax) 和消息的有效半径 (setRadius)，FLAME GPU 会自动处理底层的空间分区和消息组织                    
    message = model.newMessageSpatial2D("location")
    message.newVariableID("id")
    # 设置消息半径
    message.setRadius(env.getPropertyFloat("RADIUS"))       
    message.setMin(0, 0)
    message.setMax(env.getPropertyFloat("ENV_WIDTH"), env.getPropertyFloat("ENV_WIDTH"))

def define_agents(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("point")
    agent.newVariableFloat("x")     # 坐标x
    agent.newVariableFloat("y")     # 坐标y
    agent.newVariableFloat("drift", 0)  #   漂移量或者速度
#   用字符串形式提供的 CUDA C++ 代理函数代码注册到 FLAME GPU 模型中
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
#   创建一个 CUDASimulation 对象。这是 FLAME GPU 的核心类之一，负责在 CUDA 启用设备（通常是 GPU）上运行模拟。它将之前创建的 model 对象作为输入，以便知道要模拟什么。
    cudaSimulation = pyflamegpu.CUDASimulation(model)
    cudaSimulation.initialise(sys.argv)


#   设置可视化
    if pyflamegpu.VISUALISATION:
        # 获取与 CUDA 模拟相关的可视化对象。
        m_vis = cudaSimulation.getVisualisation()

        INIT_CAM = env.getPropertyFloat("ENV_WIDTH")/2
        # 设置可视化相机的初始观察目标点。
        m_vis.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0)
        # 设置可视化相机的初始位置。
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, env.getPropertyFloat("ENV_WIDTH"))
        # 设置相机移动的速度。
        m_vis.setCameraSpeed(0.01)
        # 设置模拟的可视化播放速度（例如，每秒显示多少个模拟步长）
        m_vis.setSimulationSpeed(50)
#   将名为 "point" 的代理类型添加到可视化中，以便在窗口中显示它们。
        point_agt = m_vis.addAgent("point")
#   设置 "point" 代理在可视化中使用的 3D 模型形状，这里使用了球体 (ICOSPHERE)
        point_agt.setModel(pyflamegpu.ICOSPHERE)
    # 代理的可视化模型的大小比例。
        point_agt.setModelScale(1/10.0)
#   标记环境边界 
# 参数 (1, 1, 1, 0.2) 指定了绘制线条的颜色和透明度。这是一个 RGBA 值：
# 第一个 1 是红色分量 (Red)。
# 第二个 1 是绿色分量 (Green)。
# 第三个 1 是蓝色分量 (Blue)。
# 0.2 是 Alpha 分量，表示透明度。值越大越不透明。在这里，0.2 表示线条是比较透明的。
# 所以，这条线将是白色的，并且具有一定的透明度。
        pen = m_vis.newPolylineSketch(1, 1, 1, 0.5)
        pen.addVertex(0, 0, 0)
        pen.addVertex(0, env.getPropertyFloat("ENV_WIDTH"), 0)
        pen.addVertex(env.getPropertyFloat("ENV_WIDTH"), env.getPropertyFloat("ENV_WIDTH"), 0)
        pen.addVertex(env.getPropertyFloat("ENV_WIDTH"), 0, 0)
        pen.addVertex(0, 0, 0)
#   激活可视化窗口
        m_vis.activate()

    
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