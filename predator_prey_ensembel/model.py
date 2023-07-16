import pyflamegpu
import sys, random, math, time
import matplotlib.pyplot as plt
from cuda import *

def create_model():
    model = pyflamegpu.ModelDescription("python-tutorial")
    return model

def define_environment(model):
    """
        Environment
    """
    env = model.Environment()

    env.newPropertyUInt("num_prey", 200)
    env.newPropertyUInt("num_predators", 50)
    env.newPropertyUInt("num_grass", 0)
    # Reproduction
    env.newPropertyFloat("REPRODUCE_PREY_PROB", 0.05)
    env.newPropertyFloat("REPRODUCE_PRED_PROB", 0.03)

    # Cohesion/Avoidance
    env.newPropertyFloat("SAME_SPECIES_AVOIDANCE_RADIUS", 0.035)
    env.newPropertyFloat("PREY_GROUP_COHESION_RADIUS", 0.2)

    # Predator/Prey/Grass interaction
    env.newPropertyFloat("PRED_PREY_INTERACTION_RADIUS", 0.3)
    env.newPropertyFloat("PRED_SPEED_ADVANTAGE", 3.0)
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


def define_messages(model):
    """
      Location messages
    """      
    message = model.newMessageBruteForce("predator_location_message")
    message.newVariableID("id")
    message.newVariableFloat("x")
    message.newVariableFloat("y")
        
    message = model.newMessageBruteForce("prey_location_message")
    message.newVariableID("id")
    message.newVariableFloat("x")
    message.newVariableFloat("y")
    
    message = model.newMessageBruteForce("grass_location_message")
    message.newVariableID("id")
    message.newVariableFloat("x")
    message.newVariableFloat("y")


    """
      Agent eaten messages
    """
        
    message = model.newMessageBruteForce("prey_eaten_message")
    message.newVariableID("id")
    message.newVariableInt("pred_id")

    message = model.newMessageBruteForce("grass_eaten_message")
    message.newVariableID("id")
    message.newVariableInt("prey_id")


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

    model.addInitFunction(initfn())

def define_runs(model):
    ## 设置为要测试的参数。
    runs = pyflamegpu.RunPlanVector(model, 5)
    runs.setSteps(100)
    runs.setRandomSimulationSeed(12, 1)

#    runs.setPropertyLerpRangeFloat("REPRODUCE_PREY_PROB", 0.05, 1.05)
#    runs.setPropertyLerpRangeFloat("REPRODUCE_PRED_PROB", 0.03, 1.03)
#    runs.setPropertyLerpRangeFloat("SAME_SPECIES_AVOIDANCE_RADIUS", 0.035, 0.135)
#    runs.setPropertyLerpRangeFloat("PREY_GROUP_COHESION_RADIUS", 0.2, 20.2)
#    runs.setPropertyLerpRangeFloat("PRED_PREY_INTERACTION_RADIUS", 0.3, 30.3)
#    runs.setPropertyLerpRangeFloat("PRED_SPEED_ADVANTAGE", 3, 303)
#    runs.setPropertyLerpRangeFloat("PRED_KILL_DISTANCE", 0.05, 5.05)
#    runs.setPropertyLerpRangeFloat("GRASS_EAT_DISTANCE", 0.02, 2.02)
#    runs.setPropertyLerpRangeFloat("DELTA_TIME", 0.001, 0.101)

#    runs.setPropertyLerpRangeUInt("GAIN_FROM_FOOD_PREY", 80, 90)
#    runs.setPropertyLerpRangeUInt("GAIN_FROM_FOOD_PREDATOR", 100, 110)
#    runs.setPropertyLerpRangeUInt("GRASS_REGROW_CYCLES", 100, 110)
    return runs

def define_logs(model):
    log = pyflamegpu.StepLoggingConfig(model)
    log.setFrequency(1)
#    log.logEnvironment("REPRODUCE_PREY_PROB")
#    log.logEnvironment("REPRODUCE_PRED_PROB")
#    log.logEnvironment("SAME_SPECIES_AVOIDANCE_RADIUS")
#    log.logEnvironment("PREY_GROUP_COHESION_RADIUS")
#    log.logEnvironment("PRED_PREY_INTERACTION_RADIUS")
#    log.logEnvironment("PRED_SPEED_ADVANTAGE")
#    log.logEnvironment("PRED_KILL_DISTANCE")
#    log.logEnvironment("GRASS_EAT_DISTANCE")
#    log.logEnvironment("DELTA_TIME")
#    log.logEnvironment("GAIN_FROM_FOOD_PREY")
#    log.logEnvironment("GAIN_FROM_FOOD_PREDATOR")
#    log.logEnvironment("GRASS_REGROW_CYCLES")
    log.agent("prey").logCount()
    log.agent("predator").logCount()
    log.agent("grass").logCount()
    return log

class initfn(pyflamegpu.HostFunction):        
    def run(self, FLAMEGPU):
        num_prey = FLAMEGPU.environment.getPropertyUInt("num_prey")
        num_predators = FLAMEGPU.environment.getPropertyUInt("num_predators")
        num_grass = FLAMEGPU.environment.getPropertyUInt("num_grass")

        prey = FLAMEGPU.agent("prey")
        for i in range(num_prey):            
            prey.newAgent().setVariableFloat("x",  random.uniform(-1.0, 1.0))
            prey.newAgent().setVariableFloat("y",  random.uniform(-1.0, 1.0))
            prey.newAgent().setVariableFloat("vx",  random.uniform(-1.0, 1.0))
            prey.newAgent().setVariableFloat("vy",  random.uniform(-1.0, 1.0))
            prey.newAgent().setVariableFloat("steer_x",  0.0)
            prey.newAgent().setVariableFloat("steer_y", 0.0)
            prey.newAgent().setVariableFloat("type", 1.0)
            prey.newAgent().setVariableInt("life", random.randint(0, 50))

        predator = FLAMEGPU.agent("predator")
        for i in range(num_predators):            
            predator.newAgent().setVariableFloat("x",  random.uniform(-1.0, 1.0))
            predator.newAgent().setVariableFloat("y",  random.uniform(-1.0, 1.0))
            predator.newAgent().setVariableFloat("vx",  random.uniform(-1.0, 1.0))
            predator.newAgent().setVariableFloat("vy",  random.uniform(-1.0, 1.0))
            predator.newAgent().setVariableFloat("steer_x",  0.0)
            predator.newAgent().setVariableFloat("steer_y", 0.0)
            predator.newAgent().setVariableFloat("type", 0.0)
            predator.newAgent().setVariableInt("life", random.randint(0, 5))

        grass = FLAMEGPU.agent("grass")
        for i in range(num_grass):            
            grass.newAgent().setVariableFloat("x",  random.uniform(-1.0, 1.0))
            grass.newAgent().setVariableFloat("y",  random.uniform(-1.0, 1.0))
            grass.newAgent().setVariableInt("dead_cycles", 0)
            grass.newAgent().setVariableInt("available", 1)
            grass.newAgent().setVariableFloat("type", 2.0)



def define_output(ensemble):
    ensemble.Config().out_directory = "results"
    ensemble.Config().out_format = "json"
    ensemble.Config().concurrent_runs = 1
    ensemble.Config().timing = True
    ensemble.Config().truncate_log_files = True
    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
    ensemble.Config().devices = pyflamegpu.IntSet([0])

def initialise_simulation(seed):
    model = create_model()
    define_messages(model)
    define_agents(model)
    define_environment(model)
    define_execution_order(model)
    runs = define_runs(model)
    logs = define_logs(model)
    ensembleSimulation = pyflamegpu.CUDAEnsemble(model)
    define_output(ensembleSimulation)
    ensembleSimulation.setStepLog(logs)
    ensembleSimulation.simulate(runs)




	

   




if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()