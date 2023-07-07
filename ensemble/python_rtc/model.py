from pyflamegpu import *
import time, sys, random
from cuda import *

AddOffset = r"""
FLAMEGPU_AGENT_FUNCTION(AddOffset, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Output each agents publicly visible properties.
    FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") + FLAMEGPU->environment.getProperty<int>("offset"));
    return flamegpu::ALIVE;
}
"""

class initfn(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # Fetch the desired agent count and environment width
        POPULATION_TO_GENERATE = FLAMEGPU.environment.getPropertyInt("POPULATION_TO_GENERATE")
        Init = FLAMEGPU.environment.getPropertyInt("init")
        Init_offset = FLAMEGPU.environment.getPropertyInt("init_offset")
        # Create agents
        agent = FLAMEGPU.agent("Agent")
        for i in range(POPULATION_TO_GENERATE):
            agent.newAgent().setVariableInt("x", Init + i * Init_offset)

class exitfn(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # Fetch the desired agent count and environment width
        atomic_init += FLAMEGPU.environment.getPropertyInt("init")
        atomic_result += FLAMEGPU.agent("Agent").sumInt("x")


def create_model():
#   创建模型，并且起名
    model = pyflamegpu.ModelDescription("boids_spatial3D")
    return model

def define_environment(model):
#   创建环境，给出一些不受模型影响的外生变量
    env = model.Environment()
    env.newPropertyUInt("POPULATION_TO_GENERATE", 100000, True)
    env.newPropertyUInt("STEPS", 10)
    env.newPropertyUInt("init", 0)
    env.newPropertyUInt("init_offset", 1)
    env.newPropertyUInt("offset", 10)
    return env

def define_messages(model, env):
#   创建信息，名为location，为agent之间传递的信息变量，还没太明白信息的作用，还需要琢磨下
    pass

def define_agents(model):
#   创建agent，名为point，是agent自己的变量和函数。
    agent = model.newAgent("Agent")
    agent.newVariableFloat("x")
#   有关信息的描述是FlameGPU2的关键特色，还需要进一步理解。
    agent.newRTCFunction("AddOffset", AddOffset)


def define_execution_order(model):
#   引入层主要目的是确定agent行动的顺序。
    layer = model.newLayer()
    layer.addAgentFunction("AddOffset", "AddOffset")
    model.addInitFunction(initfn())
    model.addExitFunction(exitfn())

def define_runs(model, env):
    runs = pyflamegpu.RunPlanvector(model, 100)
    runs.setSteps(env.getPropertyUInt("STEPS"))
    runs.setRandomSimulationSeed(12, 1)
    runs.setPropertyLerpRangeInt("init", 0, 9)
    runs.setPropertyLerpRangeInt("init_offset", 1, 0)
    runs.setPropertyLerpRangeInt("offset", 0, 99)

def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)
    define_runs(model,env)
#   初始化cuda模拟
    cuda_ensemble = pyflamegpu.CUDAEnsemble(model, argc, argv)
    cuda_ensemble.simulate(runs)

<<<<<<< HEAD
    init_sum = 0
    result_sum = 0
    for i in range(100):
        init = i/10
        init_offset = 1 - i/50
        init_sum += init
        result_sum += env.getPropertyUInt("POPULATION_TO_GENERATE") * init + init_offset * ((env.getPropertyUInt("POPULATION_TO_GENERATE")-1)*env.getPropertyUInt("POPULATION_TO_GENERATE")/2)
        result_sum += env.getPropertyUInt("POPULATION_TO_GENERATE") * env.getPropertyUInt("STEPS") * i
    print("Ensemble init: {}, calculated init {}".format(atomic_init.load(), init_sum))
    print("Ensemble result: {}, calculated result {}".format(atomic_result.load(), result_sum))

    pyflamegpu.cleanup()

=======
    
#   如果未提供 xml 模型文件，则生成一个填充。
    if not cudaSimulation.SimulationConfig().input_file:
        init_sum = 0
        result_sum = 0
        for i in range(100):
            init = i/10
            init_offset=1-1/50
            init_sum +=init
            result_sum += env.getPropertyUint("POPULATION_TO_GENERATE") * init + init_offset * ((env.getPropertyUint("POPULATION_TO_GENERATE")-1)*env.getPropertyUint("POPULATION_TO_GENERATE")/2)
            result_sum += env.getPropertyUint("POPULATION_TO_GENERATE") * env.getPropertyUint("STEPS") * i
        print("Ensemble init: {}, calculated init {}".format(atomic_init.load(), init_sum))
        print("Ensemble result: {}, calculated result {}".format(atomic_result.load(), result_sum))
#   在空间内均匀分布agent，具有均匀分布的初始速度。
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        population = pyflamegpu.AgentVector(model.Agent("point"), env.getPropertyUInt("AGENT_COUNT"))
        for i in range(env.getPropertyUInt("AGENT_COUNT")):
            instance = population[i]
            instance.setVariableFloat("x",  random.uniform(0.0, env.getPropertyFloat("ENV_WIDTH")))
            instance.setVariableFloat("y",  random.uniform(0.0, env.getPropertyFloat("ENV_WIDTH")))
        cudaSimulation.setPopulationData(population)
    cudaSimulation.simulate()


>>>>>>> bf8cf516b832034c249ae8edb3ee69fe0416e3e9

if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()