from pyflamegpu import *
import time, sys, random
from cuda import *


class initfn(pyflamegpu.HostFunction):        
    def run(self, FLAMEGPU):
#        # Fetch the desired agent count and environment width
        POPULATION_TO_GENERATE = FLAMEGPU.environment.getPropertyUInt("POPULATION_TO_GENERATE")
        Init = FLAMEGPU.environment.getPropertyInt("init")
        Init_offset = FLAMEGPU.environment.getPropertyInt("init_offset")
        # Create agents
        agent = FLAMEGPU.agent("Agent")
        for i in range(POPULATION_TO_GENERATE):            
            agent.newAgent().setVariableInt("x", Init + i * Init_offset)

atomic_init=0
atomic_result=0

class exitfn(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()
        self.atomic_init = 0
        self.atomic_result = 0

    def run(self, FLAMEGPU):
        self.atomic_init +=FLAMEGPU.environment.getPropertyInt("init")
        self.atomic_result += FLAMEGPU.agent("Agent").sumInt("x")
        atomic_init =self.atomic_init
        atomic_result=self.atomic_result

def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("boids_spatial3D")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
    env = model.Environment()
    env.newPropertyUInt("POPULATION_TO_GENERATE", 100000, True)
    env.newPropertyUInt("STEPS", 10)
    env.newPropertyInt("init", 0)
    env.newPropertyInt("init_offset", 0)
    env.newPropertyInt("offset", 1)
    return env

def define_messages(model, env):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    pass

def define_agents(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("Agent")
    agent.newVariableInt("x")
#   有关信息的描述是FlameGPU2的关键特色，还需要进一步理解。
    agent.newRTCFunction("AddOffset", AddOffset)


def define_execution_order(model):
#   引入层主要目的是确定agent行动的顺序。
    layer = model.newLayer()
    layer.addAgentFunction("Agent","AddOffset")
    model.addInitFunction(initfn())
    model.addExitFunction(exitfn())

def define_runs(model, env):
    runs = pyflamegpu.RunPlanVector(model, 100)
    runs.setSteps(env.getPropertyUInt("STEPS"))
    runs.setRandomSimulationSeed(12, 1)
    runs.setPropertyLerpRangeInt("init", 0, 9)
    runs.setPropertyLerpRangeInt("init_offset", 1, 0)
    runs.setPropertyLerpRangeInt("offset", 0, 99)
    return runs

def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)
    runs = define_runs(model,env)
#   初始化cuda模拟
    cuda_ensemble = pyflamegpu.CUDAEnsemble(model)
    cuda_ensemble.simulate(runs)

    init_sum = 0
    result_sum = 0
    for i in range(100):
        init = i/10
        init_offset = 1 - i/50
        init_sum += init
        result_sum += env.getPropertyUInt("POPULATION_TO_GENERATE") * init + init_offset * ((env.getPropertyUInt("POPULATION_TO_GENERATE")-1)*env.getPropertyUInt("POPULATION_TO_GENERATE")/2)
        result_sum += env.getPropertyUInt("POPULATION_TO_GENERATE") * env.getPropertyUInt("STEPS") * i
    print("Ensemble init: {}, calculated init {}".format(atomic_init, init_sum))
    print("Ensemble result: {}, calculated result {}".format(atomic_result, result_sum))

    pyflamegpu.cleanup()


if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()