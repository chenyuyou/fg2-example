from pyflamegpu import *
import time, sys, random
from cuda import *


class create_agents(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # Fetch the desired agent count and environment width
        AGENT_COUNT = FLAMEGPU.environment.getPropertyUInt("AGENT_COUNT")
#        ENV_WIDTH = FLAMEGPU.environment.getPropertyFloat("ENV_WIDTH")
        # Create agents
        agent = FLAMEGPU.agent("agent")
        min_x = agent.minFloat("x")
        max_x = agent.maxFlout("x")        
        print("Init Function! (AgentCount: {}, Min: {}, Max: {})".format(FLAMEGPU.agent("agent").count(), min_x, max_x))
        for i in range(AGENT_COUNT/):
            t = agent.newAgent()
            t.setVariableFloat("x",  float(i))
            t.setVariableInt("y", 1 if i % 2 == 0 else 0)




def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("host_functions_example")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
    env = model.Environment()
    env.newPropertyUInt("AGENT_COUNT", 16384)
    env.newPropertyFloat("ENV_WIDTH", int(env.getPropertyUInt("AGENT_COUNT")**(1/3)))  
    env.newPropertyFloat("repulse", 0.05)
    return env

def define_messages(model, env):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    pass

def define_agents(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("agent")
    agent.newVariableFloat("x")
#   有关信息的描述是FlameGPU2的关键特色，还需要进一步理解。
    agent.newRTCFunction("device_function", device_function)


def define_execution_order(model):
    model.addInitFunction(init_function)
    model.addStepFunction(step_function)
    model.addExitFunction(exit_function)
    model.addExitCondition(exit_condition)
#   引入层主要目的是确定agent行动的顺序。
    devicefn_layer = model.newLayer("devicefn_layer")
    devicefn_layer.addAgentFunction("agent","device_function")
    hostfn_layer = model.newLayer("hostfn_layer")
    hostfn_layer.addHostFunction("agent","host_function")

def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)
#   初始化cuda模拟
    cudaSimulation = pyflamegpu.CUDASimulation(model)
    cudaSimulation.initialise(sys.argv)
    
#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
#   在空间内均匀分布agent，具有均匀分布的初始速度。
        cudaSimulation.SimulationConfig().steps = 0
        population = pyflamegpu.AgentVector(model.Agent("agent"), env.getPropertyUInt("AGENT_COUNT")/2)
        for i in range(env.getPropertyUInt("AGENT_COUNT")/2):
            instance = population[i]
            instance.setVariableFloat("x",  float(i))
            instance.setVariableInt("a", 1 if i % 2 == 0 else 0 )
        cudaSimulation.setPopulationData(population)
    cudaSimulation.simulate()

    pyflamegpu.cleanup()


if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()