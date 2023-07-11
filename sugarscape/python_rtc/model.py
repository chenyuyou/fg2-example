from pyflamegpu import *
import time, sys, random
from cuda import *

## Grid Size (the product of these is the agent count)
GRID_WIDTH = 256
GRID_HEIGHT = 256

## Agent state variables
AGENT_STATUS_UNOCCUPIED = 0
AGENT_STATUS_OCCUPIED = 1
AGENT_STATUS_MOVEMENT_REQUESTED = 2
AGENT_STATUS_MOVEMENT_UNRESOLVED = 3

## Growback variables
SUGAR_GROWBACK_RATE = 1
SUGAR_MAX_CAPACITY  = 7

## Visualisation mode (0=occupied/move status, 1=occupied/sugar/level)
VIS_MODE = 1

class initfn(pyflamegpu.HostCondition):        
    def run(self, FLAMEGPU):
        iterations = 0
        iterations += 1
        if iterations < 9:
    # Agent movements still unresolved
            if FLAMEGPU.agent("agent").count("status", AGENT_STATUS_MOVEMENT_UNRESOLVED):
                return pyflamegpu.CONTINU
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

#    agent.newRTCFunction("output_message", metabolise_and_growback)
#    agent.newRTCFunction("output_cell_status", output_cell_status).setMessageOutput("output_cell_status_message")
#    agent.newRTCFunction("input_message", input_message).setMessageInput("location")
    return agent

def create_submodel():
#   创建模型，并且起名
    submodel = pyflamegpu.ModelDescription("Movement_model")
    return submodel



def define_sub_messages(model):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    message = model.newMessageSpatial2D("cell_status")
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

def define_messages(model):
    pass

def define_execution_order(model):
    pass

def define_sub_execution_order(model):
#   引入层主要目的是确定agent行动的顺序。
    layer = model.newLayer()
    layer.addAgentFunction("point", "output_message")
    layer = model.newLayer()
    layer.addAgentFunction("point", "input_message")



def initialise_simulation(seed):
    submodel = create_submodel()    
    define_sub_messages(submodel)
    agent = makeCoreAgent(submodel)
    
    define_agents(model)
    define_execution_order(model)





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
            cell_colors = pyflamegpu.iDiscreteColor("env_sugar_level", pyflamegpu.Viridis(SUGAR_MAX_CAPACITY + 1), flamegpu.Color("#f00"))
            agt.setColor(cell_colors)
        visualisation.activate()



    cudaSimulation.initialise(sys.argv)
#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
#   在空间内均匀分布agent，具有均匀分布的初始速度。
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        sugar_hotspots = []
        width_dist = random.randint(0, GRID_WIDTH - 1)
        height_dist = random.randint(0, GRID_HEIGHT - 1)
        ## Each sugar hotspot has a radius of 3-15 blocks
        radius_dist = random.randint(5, 30)
        ## Hostpot area should cover around 50% of the map
        hotspot_area = 0
        while hotspot_area < GRID_WIDTH * GRID_HEIGHT:
            rad = radius_dist
            hs = [width_dist, height_dist, rad, SUGAR_MAX_CAPACITY]
            ugar_hotspots.push_back(hs)
            hotspot_area += math.pi * rad * rad

        CELL_COUNT = GRID_WIDTH * GRID_HEIGHT
        normal = random.uniform(0, 1)
        agent_sugar_dist = random.randint(0, SUGAR_MAX_CAPACITY * 2)
        poor_env_sugar_dist = random.randint(0, SUGAR_MAX_CAPACITY/2)
        i = 0
        agent_id = 0
        init_pop = pyflamegpu.AgentVector(model.Agent("agent"))
        init_pop.reserve(CELL_COUNT)
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                instance = init_pop[i]
                instance.setVariable("pos", [x, y])
                i += 1
                if random.uniform(0, 1)<0.1:
                    instance.setVariableInt("agent_id", agent_id)
                    agent_id += 1
                    instance.setVariableInt("status", AGENT_STATUS_OCCUPIED)
                    instance.setVariableInt("sugar_level", agent_sugar_dist(rng) // 2)
                    instance.setVariableInt("metabolism", 6)
                else:
                    instance.setVariable("agent_id", -1)
                    instance.setVariable("status", AGENT_STATUS_UNOCCUPIED)
                    instance.setVariable("sugar_level", 0)
                    instance.setVariable("metabolism", 0)
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
                        env_sugar_lvl = max(t, env_sugar_lvl)
                    elif hs_dist <= hs_rad:
                        non_core_len = hs_rad - hotspot_core_size
                        dist_from_core = hs_dist - hotspot_core_size
                        t = int(hs_level * (non_core_len - dist_from_core) / non_core_len)
                        env_sugar_lvl = max(t, env_sugar_lvl)
                env_sugar_lvl = poor_env_sugar_dist(rng) if env_sugar_lvl < SUGAR_MAX_CAPACITY / 2 else env_sugar_lvl
                instance.setVariable("env_max_sugar_level", env_sugar_lvl)
                instance.setVariable("env_sugar_level", env_sugar_lvl)
                if pyflamegpu.VISUALISATION:
                    instance.setVariable("x", float(x))
                    instance.setVariable("y", float(y))
        
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