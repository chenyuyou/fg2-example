import pyflamegpu
import sys, random, math, time
import matplotlib.pyplot as plt
from cuda import *

def create_model():
    model = pyflamegpu.ModelDescription("punish")
    return model

def define_environment(model):
    """
        Environment
    """
    env = model.Environment()

    env.newPropertyUInt("num_agents", 100)

    env.newPropertyFloat("intense", 0.1)

    env.newPropertyFloat("k", 1.5)

    env.newPropertyFloat("b", 2.0)
    env.newPropertyFloat("c", 2.0)
    env.newPropertyFloat("e", 5.0)
    env.newPropertyFloat("f", 1.0)

    env.newPropertyFloat("noise", 0.7)
    env.newPropertyFloat("mu", 0.7)
    env.newMacroProperty<float, 3, 3>("payoff")



def define_messages(model):
    message = model.newMessageBruteForce("agent_punish_message")
    message.newVariableID("id")
    message.newVariableFloat("score")
    message.newVariableUInt("move")   

def define_agents(model):
    # Create the agent
    agent = model.newAgent("agent")
    # Assign its variables
    agent.newVariableFloat("score")
    agent.newVariableUInt("move")
    agent.newVariableUInt("next_move")    
    agent.newState('cooperation')
    agent.newState('defect')    
    agent.newState('punishment')    
    # Assign its functions
#    fn = agent.newRTCFunction("prey_eaten", prey_eaten)
#    fn.setMessageInput("predator_location_message")
#    fn.setMessageOutput("prey_eaten_message")
#    fn.setMessageOutputOptional(True)
#    fn.setAllowAgentDeath(True)
        
def define_execution_order(model):
    """
      Control flow
    """    
    layer = model.newLayer()
    layer.addAgentFunction("agent", "prey_output_location")
    layer.addAgentFunction("predator", "pred_output_location")
    layer.addAgentFunction("grass", "grass_output_location")

    layer = model.newLayer()
    layer.addAgentFunction("predator", "pred_follow_prey")
    layer.addAgentFunction("prey", "prey_avoid_pred")
    

    model.addInitFunction(initfn())

def define_runs(model):
    ## 设置为要测试的参数。
    runs = pyflamegpu.RunPlanVector(model, 5)
    runs.setSteps(100000)
    runs.setRandomSimulationSeed(12, 1)
#    runs.setPropertyLerpRangeFloat("REPRODUCE_PREY_PROB", 0.05, 1.05)
    return runs

def define_logs(model):
    log = pyflamegpu.StepLoggingConfig(model)
    log.setFrequency(1)
#    log.logEnvironment("REPRODUCE_PREY_PROB")
    log.agent("agent","cooperation").logSumUInt("move")
    log.agent("agent","defect").logSumUInt("move")
    log.agent("agent","punishment").logSumUInt("move")

    return log

class initfn(pyflamegpu.HostFunction):        
    def run(self, FLAMEGPU):
        num_agents = FLAMEGPU.environment.getPropertyUInt("num_agents")
        agents = FLAMEGPU.agent("agent")
        for i in range(num_agents):            
            agents.newAgent().setVariableFloat("score", 0)
            agents.newAgent().setVariableUInt("move", random.choice([0,1,2]))
            agents.newAgent().setVariableUInt("next_move", random.choice([0,1,2]))


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