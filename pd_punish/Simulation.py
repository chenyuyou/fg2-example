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

    env.newMacroPropertyFloat("payoff",3,3)


def define_messages(model):
    message = model.newMessageBruteForce("status_message")
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
    agent.newState("cooperation")
    agent.newState('defect')    
    agent.newState('punishment')    
    # Assign its functions
    fn = agent.newRTCFunction("output_status", output_status)
    fn.setMessageOutput("status_message")

    fn = agent.newRTCFunction("study", study)
    fn.setMessageInput("status_message")

    fn = agent.newRTCFunction("mutate", mutate)
#    fn.setMessageInput("predator_location_message")
#    fn.setMessageOutput("prey_eaten_message")
#    fn.setMessageOutputOptional(True)
#    fn.setAllowAgentDeath(True)
        
def define_execution_order(model):
    """
      Control flow
    """    

    layer = model.newLayer()
    layer.addAgentFunction("agent", "output_status")

    layer = model.newLayer()
    layer.addAgentFunction("agent", "study")

    layer = model.newLayer()
    layer.addAgentFunction("agent", "mutate")
    model.addInitFunction(initfn())






def define_logs(model):
    log = pyflamegpu.StepLoggingConfig(model)
    log.setFrequency(5)
    log.agent("agent","cooperation").logCount()
#    log.logEnvironment("REPRODUCE_PREY_PROB")

#    log.agent("agent").logSumUInt("move")
#    log.agent("agent","defect").logSumUInt("move")
#    log.agent("agent","punishment").logSumUInt("move")

    return log

class initfn(pyflamegpu.HostFunction):        
    def run(self, FLAMEGPU):
        payoff = FLAMEGPU.environment.getMacroPropertyFloat("payoff")
        b = FLAMEGPU.environment.getPropertyFloat("b")
        c = FLAMEGPU.environment.getPropertyFloat("c")
        e = FLAMEGPU.environment.getPropertyFloat("e")
        f = FLAMEGPU.environment.getPropertyFloat("f")
        payoff[0][0]=b-c
        payoff[0][1]=-c
        payoff[0][2]=-c-e
        payoff[1][0]=b
        payoff[1][1]=0
        payoff[1][2]=-e
        payoff[2][0]=b-f
        payoff[2][1]=-f
        payoff[2][2]=-f-e

        num_agents = FLAMEGPU.environment.getPropertyUInt("num_agents")
#        agents = FLAMEGPU.agent("agent","cooperation")
        agents = FLAMEGPU.agent("agent","cooperation")
        for i in range(num_agents):            
            agents.newAgent().setVariableFloat("score", 0)
            agents.newAgent().setVariableUInt("move", 0)
            agents.newAgent().setVariableUInt("next_move", random.choice([0,1,2]))
#            agents.newAgent().setInitialState("cooperation")
        agents = FLAMEGPU.agent("agent","defect")
        for i in range(num_agents):            
            agents.newAgent().setVariableFloat("score", 0)
            agents.newAgent().setVariableUInt("move", 1)
            agents.newAgent().setVariableUInt("next_move", random.choice([0,1,2]))
        agents = FLAMEGPU.agent("agent","punishment")
        for i in range(num_agents):            
            agents.newAgent().setVariableFloat("score", 0)
            agents.newAgent().setVariableUInt("move", 2)
            agents.newAgent().setVariableUInt("next_move", random.choice([0,1,2]))
#def define_output(ensemble):
#    ensemble.Config().out_directory = "results"
#    ensemble.Config().out_format = "json"
#    ensemble.Config().concurrent_runs = 1
#    ensemble.Config().timing = True
#    ensemble.Config().truncate_log_files = True
#    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
#    ensemble.Config().devices = pyflamegpu.IntSet([0])

def define_runs(model):
    ## 设置为要测试的参数。
    runs = pyflamegpu.RunPlan(model)
    runs.setRandomSimulationSeed(123456)
    runs.setSteps(10000)
#    runs.setPropertyLerpRangeFloat("REPRODUCE_PREY_PROB", 0.05, 1.05)
    return runs

def initialise_simulation(seed):
    model = create_model()
    define_messages(model)
    define_agents(model)
    define_environment(model)
    define_execution_order(model)
    logs = define_logs(model)
    run=define_runs(model)
    cuda_sim = pyflamegpu.CUDASimulation(model)
    
#    cuda_sim.SimulationConfig().steps = 100
#    cuda_sim.SimulationConfig().random_seed=12
    cuda_sim.setStepLog(logs)
#    cuda_sim.initialise(sys.argv)
    cuda_sim.simulate(run)
#    cuda_sim.exportData('end.json')
#    runs.setPopulationData(predatorPopulation)
    cuda_sim.exportLog(
  "log.json", # The file to output (must end '.json' or '.xml')
  True,       # Whether the step log should be included in the log file
  False,       # Whether the exit log should be included in the log file
  False,       # Whether the step time should be included in the log file (treated as false if step log not included)
  False,       # Whether the simulation time should be included in the log file (treated as false if exit log not included)
  False) 


#    logs = define_logs(model)
#    ensembleSimulation = pyflamegpu.CUDAEnsemble(model)
#    define_output(ensembleSimulation)
#    ensembleSimulation.setStepLog(logs)
#    ensembleSimulation.simulate(runs)


if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()