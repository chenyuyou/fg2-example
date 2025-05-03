import pyflamegpu
import sys, random, math
import matplotlib.pyplot as plt
from cuda2D import *

def create_model():
    model = pyflamegpu.ModelDescription("predator_prey2D")
    return model

def define_environment(model):
    """
        Environment
    """
    env = model.Environment()

    # Reproduction
    env.newPropertyFloat("REPRODUCE_PREY_PROB", 0.05)
    env.newPropertyFloat("REPRODUCE_PRED_PROB", 0.03)

    # Cohesion/Avoidance
    env.newPropertyFloat("SAME_SPECIES_AVOIDANCE_RADIUS", 0.035)
    env.newPropertyFloat("PREY_GROUP_COHESION_RADIUS", 0.2)

    # Predator/Prey/Grass interaction
    env.newPropertyFloat("PRED_PREY_INTERACTION_RADIUS", 0.3)
    env.newPropertyFloat("PRED_SPEED_ADVANTAGE", 1.0)
    env.newPropertyFloat("PRED_KILL_DISTANCE", 0.05)
    env.newPropertyFloat("GRASS_EAT_DISTANCE", 0.02)
    env.newPropertyUInt("GAIN_FROM_FOOD_PREY", 80)
    env.newPropertyUInt("GAIN_FROM_FOOD_PREDATOR", 100)
    env.newPropertyUInt("GRASS_REGROW_CYCLES", 100)
    
    # Simulation properties
    env.newPropertyFloat("DELTA_TIME", 0.001)
    env.newPropertyFloat("BOUNDS_WIDTH", 2.0)
    env.newPropertyFloat("MIN_POSITION", -1.0)
    env.newPropertyFloat("MAX_POSITION", 1.0)
    return env

def define_messages(model):
    """
      Location messages
    """      
    #  
    predator_message  = model.newMessageSpatial2D("predator_location_message")
    predator_message .newVariableID("id")
    # 设置消息的范围，这里可以使用环境中定义的相互作用半径
    predator_message .setRadius(model.Environment().getPropertyFloat("PRED_PREY_INTERACTION_RADIUS"))
    # 输出捕食者的位置信息
    predator_message.setMinX(model.Environment().getPropertyFloat("MIN_POSITION"))
    predator_message.setMaxX(model.Environment().getPropertyFloat("MAX_POSITION"))
    predator_message.setMinY(model.Environment().getPropertyFloat("MIN_POSITION"))
    predator_message.setMaxY(model.Environment().getPropertyFloat("MAX_POSITION"))


    prey_message = model.newMessageSpatial2D("prey_location_message")
    prey_message.newVariableID("id")
    prey_message.setRadius(model.Environment().getPropertyFloat("PRED_PREY_INTERACTION_RADIUS"))
    prey_message.setMinX(model.Environment().getPropertyFloat("MIN_POSITION"))
    prey_message.setMaxX(model.Environment().getPropertyFloat("MAX_POSITION"))
    prey_message.setMinY(model.Environment().getPropertyFloat("MIN_POSITION"))
    prey_message.setMaxY(model.Environment().getPropertyFloat("MAX_POSITION"))


    grass_message = model.newMessageSpatial2D("grass_location_message")
    grass_message.newVariableID("id")
    grass_message.setRadius(model.Environment().getPropertyFloat("GRASS_EAT_DISTANCE"))
    grass_message.setMinX(model.Environment().getPropertyFloat("MIN_POSITION"))
    grass_message.setMaxX(model.Environment().getPropertyFloat("MAX_POSITION"))
    grass_message.setMinY(model.Environment().getPropertyFloat("MIN_POSITION"))
    grass_message.setMaxY(model.Environment().getPropertyFloat("MAX_POSITION"))

    """
      Agent eaten messages
    """
        
    prey_eaten_message = model.newMessageSpatial2D("prey_eaten_message")
    prey_eaten_message.newVariableID("id")
    prey_eaten_message.newVariableInt("pred_id")
    prey_eaten_message.setMinX(model.Environment().getPropertyFloat("MIN_POSITION"))
    prey_eaten_message.setMaxX(model.Environment().getPropertyFloat("MAX_POSITION"))
    prey_eaten_message.setMinY(model.Environment().getPropertyFloat("MIN_POSITION"))
    prey_eaten_message.setMaxY(model.Environment().getPropertyFloat("MAX_POSITION"))
    # 设置 grass_eaten_message 的半径
    # 可以使用一个较小的值，或者与草被吃的距离相关的值
    prey_eaten_message.setRadius(2)
#    prey_eaten_message.setRadius(model.Environment().getPropertyFloat("PRED_KILL_DISTANCE")) 



    grass_eaten_message = model.newMessageSpatial2D("grass_eaten_message")
    grass_eaten_message.newVariableID("id")
    grass_eaten_message.newVariableInt("prey_id")
    grass_eaten_message.setMinX(model.Environment().getPropertyFloat("MIN_POSITION"))
    grass_eaten_message.setMaxX(model.Environment().getPropertyFloat("MAX_POSITION"))
    grass_eaten_message.setMinY(model.Environment().getPropertyFloat("MIN_POSITION"))
    grass_eaten_message.setMaxY(model.Environment().getPropertyFloat("MAX_POSITION"))
    # 设置 grass_eaten_message 的半径
    grass_eaten_message.setRadius(0.1)
#    grass_eaten_message.setRadius(model.Environment().getPropertyFloat("GRASS_EAT_DISTANCE")) 


def define_agents(model):
    """
        Prey agent
    """
    # Create the agent
    agent = model.newAgent("prey")

    # Assign its variables
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("vx")
    agent.newVariableFloat("vy")
    agent.newVariableFloat("steer_x")
    agent.newVariableFloat("steer_y")
    agent.newVariableInt("life")
    agent.newVariableFloat("type")
    
    # Assign its functions
    #  输出捕食者的位置信息
    fn = agent.newRTCFunction("prey_output_location", prey_output_location)
    fn.setMessageOutput("prey_location_message")

    fn = agent.newRTCFunction("prey_avoid_pred", prey_avoid_pred)
    fn.setMessageInput("predator_location_message")

    fn = agent.newRTCFunction("prey_flock", prey_flock)
    fn.setMessageInput("prey_location_message")

    fn = agent.newRTCFunction("prey_move", prey_move)

    fn = agent.newRTCFunction("prey_eaten", prey_eaten)
    fn.setMessageInput("predator_location_message")
    fn.setMessageOutput("prey_eaten_message")
    ## setMessageOutputOptional 表示可以选择性输出，如果无数据输出就不输出。 
    fn.setMessageOutputOptional(True)
    fn.setAllowAgentDeath(True)
    
    fn = agent.newRTCFunction("prey_eat_or_starve", prey_eat_or_starve)
    fn.setMessageInput("grass_eaten_message")
    fn.setAllowAgentDeath(True)

    fn = agent.newRTCFunction("prey_reproduction", prey_reproduction)
    fn.setAgentOutput("prey", "default")
    
    """
      Predator agent
    """
    # Create the agent
    agent = model.newAgent("predator")

    # Assign its variables
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("vx")
    agent.newVariableFloat("vy")
    agent.newVariableFloat("steer_x")
    agent.newVariableFloat("steer_y")
    agent.newVariableInt("life")
    agent.newVariableFloat("type")
    
    # Assign its functions
    fn = agent.newRTCFunction("pred_output_location", pred_output_location)
    fn.setMessageOutput("predator_location_message")
    fn = agent.newRTCFunction("pred_follow_prey", pred_follow_prey)
    fn.setMessageInput("prey_location_message")
    fn = agent.newRTCFunction("pred_avoid", pred_avoid)
    fn.setMessageInput("predator_location_message")
    fn = agent.newRTCFunction("pred_move", pred_move)
    fn = agent.newRTCFunction("pred_eat_or_starve", pred_eat_or_starve)
    fn.setMessageInput("prey_eaten_message")
    fn.setAllowAgentDeath(True)
    fn = agent.newRTCFunction("pred_reproduction", pred_reproduction)
    fn.setAgentOutput("predator", "default")

    """
      Grass agent
    """
    # Create the agent
    agent = model.newAgent("grass")

    # Assign its variables
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableInt("dead_cycles")
    agent.newVariableInt("available")
    agent.newVariableFloat("type")

    fn = agent.newRTCFunction("grass_output_location", grass_output_location)
    fn.setMessageOutput("grass_location_message")
    fn.setMessageOutputOptional(True)

    fn = agent.newRTCFunction("grass_eaten", grass_eaten)
    fn.setMessageInput("prey_location_message")
    fn.setMessageOutput("grass_eaten_message")
    fn.setMessageOutputOptional(True)
    fn.setAllowAgentDeath(True)

    fn = agent.newRTCFunction("grass_growth", grass_growth)



class population_tracker(pyflamegpu.HostFunction):
    def __init__(self, has_grass):
        super().__init__();  # Mandatory if we are defining __init__ ourselves
        # Local, so value is maintained between calls to calculate_convergence::run
        self.pred_count = []
        self.prey_count = []
        self.grass_count = []
        self.has_grass = has_grass

    def run(self, FLAMEGPU):
        # Reduce force and overlap
        self.pred_count.append(FLAMEGPU.agent("predator").count())
        self.prey_count.append(FLAMEGPU.agent("prey").count())
        if (self.has_grass):
            self.grass_count.append(FLAMEGPU.agent("grass").countInt("available", 1))
        else:
            self.grass_count.append(0)

    def plot(self):
        plt.figure(figsize=(16,10))
        plt.rcParams.update({'font.size': 18})
        plt.xlabel("Step")
        plt.ylabel("Population")
        plt.plot(range(0, len(self.pred_count)), self.pred_count, 'r', label="Predators")
        plt.plot(range(0, len(self.prey_count)), self.prey_count, 'b', label="Prey")
        plt.plot(range(0, len(self.grass_count)), self.grass_count, 'g', label="Grass")
        plt.legend()
        plt.show()

    def reset(self):
        self.pred_count = []
        self.prey_count = []
        self.grass_count = []
        
def define_execution_order(model):
    """
      Control flow
    """    
    layer = model.newLayer()
    layer.addAgentFunction("prey", "prey_output_location")
    layer.addAgentFunction("predator", "pred_output_location")
    layer.addAgentFunction("grass", "grass_output_location")

    layer = model.newLayer()
    layer.addAgentFunction("predator", "pred_follow_prey")
    layer.addAgentFunction("prey", "prey_avoid_pred")
    
    layer = model.newLayer()
    layer.addAgentFunction("prey", "prey_flock")
    layer.addAgentFunction("predator", "pred_avoid")
    
    layer = model.newLayer()
    layer.addAgentFunction("prey", "prey_move")
    layer.addAgentFunction("predator", "pred_move")
    
    layer = model.newLayer()
    layer.addAgentFunction("grass", "grass_eaten")
    layer.addAgentFunction("prey", "prey_eaten")
    
    layer = model.newLayer()
    layer.addAgentFunction("prey", "prey_eat_or_starve")
    layer.addAgentFunction("predator", "pred_eat_or_starve")
    
    layer = model.newLayer()
    layer.addAgentFunction("predator", "pred_reproduction")
    layer.addAgentFunction("prey", "prey_reproduction")
    layer.addAgentFunction("grass", "grass_growth")

def initialise_simulation(num_prey, num_predators, num_grass, seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model)
    define_agents(model)
    define_execution_order(model)

    # Set up a population tracker for logging/plotting
    pop_tracker = population_tracker(num_grass > 0)
    model.addStepFunction(pop_tracker)
    
    """
      Create Model Runner
    """   
    cudaSimulation = pyflamegpu.CUDASimulation(model)
    
    # Apply a simulation seed
    if seed is not None:
        cudaSimulation.SimulationConfig().random_seed = seed
        cudaSimulation.applyConfig()

    # --- Visualisation Setup (移植自 model.py) ---
    if pyflamegpu.VISUALISATION:
        m_vis = cudaSimulation.getVisualisation()
        min_pos = env.getPropertyFloat("MIN_POSITION")
        max_pos = env.getPropertyFloat("MAX_POSITION")
        bounds_width = env.getPropertyFloat("BOUNDS_WIDTH")
        center = min_pos + bounds_width / 2.0
        m_vis.setInitialCameraTarget(center, center, 0)
        m_vis.setInitialCameraLocation(center, center, bounds_width * 2) # Adjust initial camera distance
        m_vis.setCameraSpeed(0.01)
        m_vis.setSimulationSpeed(50)
        # Add agent visualisations for each type
        predator_agt_vis = m_vis.addAgent("predator")
        predator_agt_vis.setModel(pyflamegpu.ICOSPHERE)
        predator_agt_vis.setModelScale(0.05) # Adjust size
        predator_agt_vis.setColor(pyflamegpu.RED) # Red
        prey_agt_vis = m_vis.addAgent("prey")
        prey_agt_vis.setModel(pyflamegpu.ICOSPHERE)
        prey_agt_vis.setModelScale(0.03) # Adjust size
        prey_agt_vis.setColor(pyflamegpu.BLUE) # Blue
        grass_agt_vis = m_vis.addAgent("grass")
        grass_agt_vis.setModel(pyflamegpu.CUBE) # Use a different shape for grass
        grass_agt_vis.setModelScale(0.02) # Adjust size
        grass_agt_vis.setColor(pyflamegpu.GREEN) # Green
        # Mark environment bounds
        pen = m_vis.newPolylineSketch(1, 1, 1, 0.5) # White, semi-transparent
        pen.addVertex(min_pos, min_pos, 0)
        pen.addVertex(min_pos, max_pos, 0)
        pen.addVertex(max_pos, max_pos, 0)
        pen.addVertex(max_pos, min_pos, 0)
        pen.addVertex(min_pos, min_pos, 0)
        m_vis.activate()
    # --- End Visualisation Setup ---

    """
      Initialise Model
    """
    # If no xml model file was is provided, generate a population programmatically.
    if not cudaSimulation.SimulationConfig().input_file:
        # Uniformly distribute agents within space, with uniformly distributed initial velocity.
        # Using random seed
        random.seed(cudaSimulation.SimulationConfig().random_seed)
    
        # Initialise prey agents
        preyPopulation = pyflamegpu.AgentVector(model.Agent("prey"), num_prey)
        for i in range(0, num_prey):
            prey = preyPopulation[i]
            prey.setVariableFloat("x", random.uniform(-1.0, 1.0))
            prey.setVariableFloat("y", random.uniform(-1.0, 1.0))
            prey.setVariableFloat("vx", random.uniform(-1.0, 1.0))
            prey.setVariableFloat("vy", random.uniform(-1.0, 1.0))
            prey.setVariableFloat("steer_x", 0.0)
            prey.setVariableFloat("steer_y", 0.0)
            prey.setVariableFloat("type", 1.0)
            prey.setVariableInt("life", random.randint(0, 50))
      
        # Initialise predator agents
        predatorPopulation = pyflamegpu.AgentVector(model.Agent("predator"), num_predators)
        for i in range(0, num_predators):
            predator = predatorPopulation[i]
            predator.setVariableFloat("x", random.uniform(-1.0, 1.0))
            predator.setVariableFloat("y", random.uniform(-1.0, 1.0))
            predator.setVariableFloat("vx", random.uniform(-1.0, 1.0))
            predator.setVariableFloat("vy", random.uniform(-1.0, 1.0))
            predator.setVariableFloat("steer_x", 0.0)
            predator.setVariableFloat("steer_y", 0.0)
            predator.setVariableFloat("type", 0.0)
            predator.setVariableInt("life", random.randint(0, 50))

        grassPopulation = pyflamegpu.AgentVector(model.Agent("grass"), num_grass)
        for i in range(0, num_grass):
            grass = grassPopulation[i]
            grass.setVariableFloat("x", random.uniform(-1.0, 1.0))
            grass.setVariableFloat("y", random.uniform(-1.0, 1.0))
            grass.setVariableInt("dead_cycles", 0)
            grass.setVariableInt("available", 1)
            grass.setVariableFloat("type", 2.0)
	

    cudaSimulation.setPopulationData(predatorPopulation)
    cudaSimulation.setPopulationData(preyPopulation)
    cudaSimulation.setPopulationData(grassPopulation)        
    return [cudaSimulation, pop_tracker]

def run_simulation():
    """
      Execution
    """
    # Initialise the simulation
    [cudaSimulation, pop_tracker] = initialise_simulation(num_prey = 500, num_predators = 250, num_grass = 1000, seed = 64)
    cudaSimulation.SimulationConfig().steps = 200

    # Run the simulation
    pop_tracker.reset()
    cudaSimulation.simulate()
    pop_tracker.plot()

run_simulation()  