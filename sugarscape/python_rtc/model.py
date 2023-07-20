from pyflamegpu import *
import time, sys, random, math
from cuda import *

## Grid Size (the product of these is the agent count)
GRID_WIDTH = 256
GRID_HEIGHT = 256

## 代理空闲状态
AGENT_STATUS_UNOCCUPIED = 0
## 代理被占据状态
AGENT_STATUS_OCCUPIED = 1
## 代理需要移动的状态或请求更改代理身份
AGENT_STATUS_MOVEMENT_REQUESTED = 2
## 代理移动未解决或身份未解决
AGENT_STATUS_MOVEMENT_UNRESOLVED = 3

## 糖量自身生长速度
SUGAR_GROWBACK_RATE = 1
## 糖最大量
SUGAR_MAX_CAPACITY  = 7

## Visualisation mode (0=occupied/move status, 1=occupied/sugar/level)
VIS_MODE = 1

class MovementExitCondition(pyflamegpu.HostCondition):        
    def run(self, FLAMEGPU):
        iterations = 0
        iterations += 1
        if iterations < 9:
    # Agent movements still unresolved
            if FLAMEGPU.agent("agent").countInt("status", AGENT_STATUS_MOVEMENT_UNRESOLVED):
                return pyflamegpu.CONTINUE
        iterations = 0
        return pyflamegpu.EXIT


def makeCoreAgent(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("agent")
    agent.newVariableArrayUInt("pos",2)
    agent.newVariableInt("agent_id")
    agent.newVariableInt("status")
    ## agent specific variables
    agent.newVariableInt("sugar_level")
    agent.newVariableInt("metabolism")
    ## environment specific var
    agent.newVariableInt("env_sugar_level")
    agent.newVariableInt("env_max_sugar_level")
    if pyflamegpu.VISUALISATION:
    ## Redundant seperate floating point position vars for vis
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
    return agent

def define_agent_for_sub(agent):
    fn_output_cell_status = agent.newRTCFunction("output_cell_status", output_cell_status)
    fn_output_cell_status.setMessageOutput("cell_status")
    fn_movement_request = agent.newRTCFunction("movement_request", movement_request)
    fn_movement_request.setMessageInput("cell_status")
    fn_movement_request.setMessageOutput("movement_request")
    fn_movement_response = agent.newRTCFunction("movement_response", movement_response)
    fn_movement_response.setMessageInput("movement_request")
    fn_movement_response.setMessageOutput("movement_response")
    fn_movement_transaction = agent.newRTCFunction("movement_transaction", movement_transaction)
    fn_movement_transaction.setMessageInput("movement_response")

def define_agent(model):
    agent = makeCoreAgent(model)
    agent.newRTCFunction("metabolise_and_growback", metabolise_and_growback)

def create_submodel():
#   创建模型，并且起名
    submodel = pyflamegpu.ModelDescription("Movement_model")
    return submodel

def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("Sugarscape")
    return model

def define_sub_messages(model):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    message = model.newMessageArray2D("cell_status")
    message.newVariableID("location_id")
    message.newVariableInt("status")
    message.newVariableInt("env_sugar_level")
    message.setDimensions(GRID_WIDTH, GRID_HEIGHT)
    
    message = model.newMessageArray2D("movement_request")
    message.newVariableInt("agent_id")
    message.newVariableID("location_id")
    message.newVariableInt("sugar_level")
    message.newVariableInt("metabolism")
    message.setDimensions(GRID_WIDTH, GRID_HEIGHT)

    message = model.newMessageArray2D("movement_response")
    message.newVariableID("location_id")
    message.newVariableInt("agent_id")
    message.setDimensions(GRID_WIDTH, GRID_HEIGHT)


def define_sub_execution_order(model):
    layer = model.newLayer()
    layer.addAgentFunction("agent","output_cell_status")
    layer = model.newLayer()
    layer.addAgentFunction("agent","movement_request")
    layer = model.newLayer()
    layer.addAgentFunction("agent","movement_response")
    layer = model.newLayer()
    layer.addAgentFunction("agent","movement_transaction")
    model.addExitCondition(MovementExitCondition())


def define_execution_order(model,movement_sub):
#   引入层主要目的是确定agent行动的顺序。
    layer = model.newLayer()
    layer.addAgentFunction("agent", "metabolise_and_growback")
    layer = model.newLayer()
    layer.addSubModel(movement_sub)



def initialise_simulation(seed):
    submodel = create_submodel()    
    define_sub_messages(submodel)
    agent = makeCoreAgent(submodel)
    define_agent_for_sub(agent)
    define_sub_execution_order(submodel)
    model = create_model()
    define_agent(model)
    movement_sub = model.newSubModel("movement_conflict_resolution_model", submodel)
    movement_sub.bindAgent("agent", "agent", True, True)
    define_execution_order(model,movement_sub)

#   初始化cuda模拟
    cudaSimulation = pyflamegpu.CUDASimulation(model)

#   设置可视化
    if pyflamegpu.VISUALISATION:
        visualisation = cudaSimulation.getVisualisation()

        visualisation.setInitialCameraLocation(GRID_WIDTH / 2.0, GRID_HEIGHT / 2.0, 225.0)
        visualisation.setInitialCameraTarget(GRID_WIDTH / 2.0, GRID_HEIGHT /2.0, 0.0)
        visualisation.setCameraSpeed(0.001 * GRID_WIDTH)
        visualisation.setViewClips(0.1, 5000)
#   将“point” agent添加到可视化中
        agt = visualisation.addAgent("agent")
#   设置“point” agent的形状和大小
        agt.setModel(pyflamegpu.CUBE)
        agt.setModelScale(1.0)
        if VIS_MODE == 0:
            cell_colors = pyflamegpu.iDiscreteColor("status", pyflamegpu.Color("#666"))
            cell_colors[AGENT_STATUS_UNOCCUPIED] = pyflamegpu.RED
            cell_colors[AGENT_STATUS_OCCUPIED] = pyflamegpu.GREEN
            cell_colors[AGENT_STATUS_MOVEMENT_REQUESTED] = pyflamegpu.BLUE
            cell_colors[AGENT_STATUS_MOVEMENT_UNRESOLVED] = pyflamegpu.WHITE
        else:
            cell_colors = pyflamegpu.iDiscreteColor("env_sugar_level", pyflamegpu.Viridis(SUGAR_MAX_CAPACITY + 1), pyflamegpu.Color("#f00"))
            agt.setColor(cell_colors)
        visualisation.activate()



    cudaSimulation.initialise(sys.argv)
#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
#   在空间内均匀分布agent，具有均匀分布的初始速度。
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        sugar_hotspots = []

        ## Hostpot area should cover around 50% of the map
        hotspot_area = 0
        while hotspot_area < GRID_WIDTH * GRID_HEIGHT:
            width_dist = random.randint(0, GRID_WIDTH - 1)
            height_dist = random.randint(0, GRID_HEIGHT - 1)
        ## Each sugar hotspot has a radius of 3-15 blocks
            radius_dist = random.randint(5, 30)
            rad = radius_dist
            hs = [width_dist, height_dist, rad, SUGAR_MAX_CAPACITY]
            sugar_hotspots.append(hs)
#            sugar_hotspots.push_back(hs)
            hotspot_area += math.pi * rad * rad

        CELL_COUNT = GRID_WIDTH * GRID_HEIGHT
        
        agent_sugar_dist = random.randint(0, SUGAR_MAX_CAPACITY * 2)
        poor_env_sugar_dist = random.randint(0, int(SUGAR_MAX_CAPACITY/2))
        i = 0
        agent_id = 0
        init_pop = pyflamegpu.AgentVector(model.Agent("agent"), CELL_COUNT)
#        init_pop.reserve(CELL_COUNT)
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                instance = init_pop[i]
                instance.setVariableArrayUInt("pos", [x, y])
                i += 1
#               初始化地块信息。
                if random.uniform(0, 1)<0.1:
                    ## 有糖的地块为非负数的agent_id，无糖的地块agent_id为-1。
                    instance.setVariableInt("agent_id", agent_id)
                    agent_id += 1
                    instance.setVariableInt("status", AGENT_STATUS_OCCUPIED)
                    instance.setVariableInt("sugar_level", int(agent_sugar_dist / 2))
                    instance.setVariableInt("metabolism", 6)
                else:
                    instance.setVariableInt("agent_id", -1)
                    instance.setVariableInt("status", AGENT_STATUS_UNOCCUPIED)
                    instance.setVariableInt("sugar_level", 0)
                    instance.setVariableInt("metabolism", 0)
                env_sugar_lvl = 0
                hotspot_core_size = 5
                for hs in sugar_hotspots:
             
                    hs_x = int(hs[0])
                    hs_y = int(hs[1])
                    hs_rad = hs[2]
                    hs_level = hs[3]
                    hs_dist = float(((hs_x - x) ** 2 + (hs_y - y) ** 2) ** 0.5)
                    if hs_dist <= hotspot_core_size:
                        t = hs_level
                        env_sugar_lvl = t if t > env_sugar_lvl else env_sugar_lvl
                    elif hs_dist <= hs_rad:
                        non_core_len = hs_rad - hotspot_core_size
                        dist_from_core = hs_dist - hotspot_core_size
                        t = int(hs_level * (non_core_len - dist_from_core) / non_core_len)
                        env_sugar_lvl = t if t > env_sugar_lvl else env_sugar_lvl
                env_sugar_lvl = poor_env_sugar_dist if env_sugar_lvl < (SUGAR_MAX_CAPACITY / 2) else env_sugar_lvl
                instance.setVariableInt("env_max_sugar_level", env_sugar_lvl)
                instance.setVariableInt("env_sugar_level", env_sugar_lvl)
                if pyflamegpu.VISUALISATION:
                    instance.setVariableFloat("x", float(x))
                    instance.setVariableFloat("y", float(y))
        
        cudaSimulation.setPopulationData(init_pop)
    cudaSimulation.simulate()
    if pyflamegpu.VISUALISATION:
        visualisation.join()
    pyflamegpu.cleanup()



if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()