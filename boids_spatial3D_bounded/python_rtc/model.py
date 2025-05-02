from pyflamegpu import *
import time, sys, random, math
from cuda import *


# 可视化部分用到的向量函数
def vec3Length(x, y, z):
    return math.sqrt(x * x + y * y + z * z)
# 可视化部分用到的向量函数
def vec3Mult(x, y, z, multiplier):
    x *= multiplier
    y *= multiplier
    z *= multiplier
    return x, y, z
# 可视化部分用到的向量函数
def vec3Div(x, y, z, divisor):
    x /= divisor
    y /= divisor
    z /= divisor
    return x, y, z
# 可视化部分用到的向量函数
def vec3Normalize(x, y, z):
    # Get the length
    length = vec3Length(x, y, z)
    x, y, z=vec3Div(x, y, z, length)
    return x, y, z

def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("Boids Spatial3D_bounded")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
    env = model.Environment()
# agent个数   
    env.newPropertyUInt("POPULATION_TO_GENERATE", 40000)
# 环境范围
    env.newPropertyFloat("MIN_POSITION", -0.5)
    env.newPropertyFloat("MAX_POSITION", +0.5)
# agent最大和最小速度
    env.newPropertyFloat("MAX_INITIAL_SPEED", 1.0)
    env.newPropertyFloat("MIN_INITIAL_SPEED", 0.1)
# agent交互半径和分离半径
    env.newPropertyFloat("INTERACTION_RADIUS", 0.05)
    env.newPropertyFloat("SEPARATION_RADIUS", 0.01)
# 全局时间和空间尺度
    env.newPropertyFloat("TIME_SCALE", 0.001)
    env.newPropertyFloat("GLOBAL_SCALE", 0.15)
# Rule scalers
    env.newPropertyFloat("STEER_SCALE", 0.055)
    env.newPropertyFloat("COLLISION_SCALE", 10.0)
    env.newPropertyFloat("MATCH_SCALE", 0.015)
    return env

def define_messages(model, env):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    message = model.newMessageSpatial3D("location")
    message.newVariableID("id")
    # 设置空间消息的感知半径。只有在发送代理的这个半径范围内的其他代理才能接收到这条消息。这个半径值取自环境属性
    message.setRadius(env.getPropertyFloat("INTERACTION_RADIUS"))
    #  设置消息的边界。这通常用于在空间消息系统中使用网格或分区时定义消息的查找范围。这里的边界值取自环境属性 "MIN_POSITION" 和 "MAX_POSITION"，与模拟空间的边界一致。
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
    agent.newRTCFunction("outputdata", outputdata).setMessageOutput("location")
    agent.newRTCFunction("inputdata", inputdata).setMessageInput("location")

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
#   创建一个 CUDASimulation 对象，这是在 GPU 上运行模拟的核心类。它需要一个模型描述对象作为参数。
    cudaSimulation = pyflamegpu.CUDASimulation(model)
# 初始化 CUDA 模拟。sys.argv 通常包含命令行参数，可以用来配置模拟（例如，加载代理文件、设置随机种子等）
    cudaSimulation.initialise(sys.argv)

#   设置可视化
    if pyflamegpu.VISUALISATION:
        visualisation = cudaSimulation.getVisualisation()
    # Configure vis
    #   计算模拟空间的边长
        envWidth = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION")
    #   计算初始相机位置的一个坐标值。这里将模拟空间的最大位置乘以 1.25。这意味着相机将被放置在比模拟空间稍微远一些的位置上。
        INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 1.25
    #   设置可视化窗口中相机的初始位置。相机的 x, y, z 坐标都被设置为 INIT_CAM。这意味着相机将位于一个对称的俯视角度，能够看到整个模拟空间。
        visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM)
    #   设置相机在可视化窗口中移动的速度。相机的速度被设置为 envWidth 的 0.001 倍。这样相机移动的速度会根据模拟空间的大小自动调整，确保在不同大小的模拟中都能有合适的操作体验。
        visualisation.setCameraSpeed(0.001 * envWidth)
    #   设置可视化窗口的近裁剪面（near clip plane）和远裁剪面（far clip plane）。裁剪面定义了相机可以看到的深度范围。位于近裁剪面之前的物体和位于远裁剪面之后的物体都不会被渲染。这有助于提高渲染效率，并避免渲染距离过近或过远的物体。这里的设置意味着可以看到从相机前方 0.00001 单位到 50 单位深度范围内的物体。
        visualisation.setViewClips(0.00001, 50)
    #   设置用于表示 "Boid" 代理的可视化模型。
        circ_agt = visualisation.addAgent("Boid")
    # Position vars are named x, y, z; so they are used by default
        circ_agt.setForwardXVariable("fx")
        circ_agt.setForwardYVariable("fy")
        circ_agt.setForwardZVariable("fz")
    #   设置用于表示 "Boid" 代理的可视化模型。
        circ_agt.setModel(pyflamegpu.STUNTPLANE)
    #   设置代理模型在可视化中的缩放比例。这里的比例是根据环境属性 SEPARATION_RADIUS 除以 3.0 计算得出的。这样代理模型的大小会与代理的分离半径相关联，使得在可视化中能够直观地看到代理之间的空间关系。
        circ_agt.setModelScale(env.getPropertyFloat("SEPARATION_RADIUS") /3.0)
    #   在可视化窗口中创建一个新的 UI 面板，面板的标题是 "Environment"。这个面板将用于显示和修改环境属性。
        ui = visualisation.newUIPanel("Environment")
    #    在 UI 面板中添加一个静态文本标签，显示 "Interaction"。这通常用于对面板中的相关设置进行分组或说明。
        ui.newStaticLabel("Interaction")
    #   在 UI 面板中添加一个用于修改环境属性 INTERACTION_RADIUS 的可拖动浮点数控件。
    #    第一个参数 "INTERACTION_RADIUS" 是要关联的环境属性名称。
    #    第二个参数 0.0 是控件的最小值。
    #    第三个参数 0.05 是控件的最大值。
    #    第四个参数 0.001 是控件拖动时的步长。
    #    用户在运行时可以通过拖动这个控件来实时改变代理的交互半径，并观察对模拟结果的影响。
        ui.newEnvironmentPropertyDragFloat("INTERACTION_RADIUS", 0.0, 0.05, 0.001)
    #   添加一个用于修改环境属性 SEPARATION_RADIUS 的可拖动浮点数控件。
        ui.newEnvironmentPropertyDragFloat("SEPARATION_RADIUS", 0.0, 0.05, 0.001)
    #   添加另一个静态文本标签，显示 "Force Scalars"，用于分组力缩放相关的设置。
        ui.newStaticLabel("Environment Scalars")
    #   添加用于修改 STEER_SCALE 环境属性的可拖动浮点数控件。
        ui.newEnvironmentPropertyDragFloat("TIME_SCALE", 0.0, 1.0, 0.0001)
        ui.newEnvironmentPropertyDragFloat("GLOBAL_SCALE", 0.0, 0.5, 0.001)
        ui.newStaticLabel("Force Scalars")
        ui.newEnvironmentPropertyDragFloat("STEER_SCALE", 0.0, 10.0, 0.001)
        ui.newEnvironmentPropertyDragFloat("COLLISION_SCALE", 0.0, 10.0, 0.001)
        ui.newEnvironmentPropertyDragFloat("MATCH_SCALE", 0.0, 10.0, 0.001)



        # 创建一个新的折线 sketch
        # 第一个参数是线条粗细，接下来的三个参数是颜色 (R, G, B)，最后一个是透明度 (Alpha)
        pen = visualisation.newPolylineSketch(1, 0, 0, 0.5)

        # 绘制立方体底部的四条边
        pen.addVertex(-0.5, -0.5, -0.5)
        pen.addVertex(0.5, -0.5, -0.5)

        pen.addVertex(0.5, -0.5, -0.5)
        pen.addVertex(0.5, 0.5, -0.5)

        pen.addVertex(0.5, 0.5, -0.5)
        pen.addVertex(-0.5, 0.5, -0.5)

        pen.addVertex(-0.5, 0.5, -0.5)
        pen.addVertex(-0.5, -0.5, -0.5)

        # 绘制立方体顶部的四条边
        pen.addVertex(-0.5, -0.5, 0.5)
        pen.addVertex(0.5, -0.5, 0.5)

        pen.addVertex(0.5, -0.5, 0.5)
        pen.addVertex(0.5, 0.5, 0.5)

        pen.addVertex(0.5, 0.5, 0.5)
        pen.addVertex(-0.5, 0.5, 0.5)

        pen.addVertex(-0.5, 0.5, 0.5)
        pen.addVertex(-0.5, -0.5, 0.5)

        # 绘制连接底部和顶部的四条垂直边
        pen.addVertex(-0.5, -0.5, -0.5)
        pen.addVertex(-0.5, -0.5, 0.5)

        pen.addVertex(0.5, -0.5, -0.5)
        pen.addVertex(0.5, -0.5, 0.5)

        pen.addVertex(0.5, 0.5, -0.5)
        pen.addVertex(0.5, 0.5, 0.5)

        pen.addVertex(-0.5, 0.5, -0.5)
        pen.addVertex(-0.5, 0.5, 0.5)


        visualisation.activate()



#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
#   在空间内均匀分布agent，具有均匀分布的初始速度。
#   设置 Python 内置 random 模块的随机数生成器的种子。种子值取自模拟配置中的 random_seed。设置种子是为了确保在相同的输入下，模拟的初始代理状态是相同的，从而使模拟结果具有可重复性。
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        min_pos = env.getPropertyFloat("MIN_POSITION")
        max_pos = env.getPropertyFloat("MAX_POSITION")
        min_speed = env.getPropertyFloat("MIN_INITIAL_SPEED")
        max_speed = env.getPropertyFloat("MAX_INITIAL_SPEED")
        populationSize = env.getPropertyUInt("POPULATION_TO_GENERATE")
        # 创建一个 pyflamegpu.AgentVector 对象。AgentVector 是一个用于在 Python 中临时存储和操作代理数据的数据结构，可以在将数据加载到 GPU 之前使用。
        population = pyflamegpu.AgentVector(model.Agent("Boid"), populationSize)
        for i in range(populationSize):
            instance = population[i]
            # 设置agent空间位置
            instance.setVariableFloat("x",  random.uniform(min_pos, max_pos))
            instance.setVariableFloat("y",  random.uniform(min_pos, max_pos))
            instance.setVariableFloat("z",  random.uniform(min_pos, max_pos))

            fx = random.uniform(-1, 1)
            fy = random.uniform(-1, 1)
            fz = random.uniform(-1, 1)

            fmagnitude = random.uniform(min_speed, max_speed)

            fx, fy, fz=vec3Normalize(fx, fy, fz)
            fx, fy, fz=vec3Mult(fx, fy, fz, fmagnitude)
        #   # 设置agent初始速度向量
            instance.setVariableFloat("fx", fx)
            instance.setVariableFloat("fy", fy)
            instance.setVariableFloat("fz", fz)

        cudaSimulation.setPopulationData(population)
    cudaSimulation.simulate()

#    if pyflamegpu.VISUALISATION:
    # 模拟完成后保持可视化窗口处于活动状态
#        visualisation.join()
#    pyflamegpu.cleanup()

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