#! /usr/bin/env python3
from pyflamegpu import *
from numpy import cbrt, floor
import random, time, math, sys
from cuda import *

def create_model():
    model = pyflamegpu.ModelDescription("Circles Spatial3D")
    return model

def define_environment(model):
    """
        环境设置
    """
    env = model.Environment()
#   代理个数
    env.newPropertyUInt("AGENT_COUNT", 16384)
#   环境界限
    env.newPropertyFloat("ENV_MAX", floor(cbrt(env.getPropertyUInt("AGENT_COUNT"))))
    env.newPropertyFloat("RADIUS", 2.0)
#
    env.newPropertyFloat("repulse", 0.05)
    return env

def define_messages(model, env):
#   建立一个3D信息“location”，用于管理代理的位置信息
    message = model.newMessageSpatial3D("location")
    message.newVariableID("id")
    message.setRadius(env.getPropertyFloat("RADIUS"))
    message.setMin(0, 0, 0)
    message.setMax(env.getPropertyFloat("ENV_MAX"), env.getPropertyFloat("ENV_MAX"), env.getPropertyFloat("ENV_MAX"))

def define_agents(model):
    agent = model.newAgent("Circle")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("drift")
    fn = agent.newRTCFunction("output_message", output_message)
    fn.setMessageOutput("location")
    fn = agent.newRTCFunction("move", move)
    fn.setMessageInput("location")


def define_execution_order(model):
# Layer #1
    layer = model.newLayer()
    layer.addAgentFunction("Circle", "output_message")
# Layer #2
    layer = model.newLayer()
    layer.addAgentFunction("Circle", "move")


def initialise_simulation(seed):
    model = create_model()
    env = define_environment(model)
    define_messages(model, env)
    define_agents(model)
    define_execution_order(model)

    cudaSimulation = pyflamegpu.CUDASimulation(model)
    cudaSimulation.initialise(sys.argv)

    if pyflamegpu.VISUALISATION:
        m_vis = cudaSimulation.getVisualisation()
    
        INIT_CAM = env.getPropertyFloat("ENV_MAX") * 1.25
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM)
        m_vis.setCameraSpeed(0.01)
#        m_vis.setSimulationSpeed(25)
        circ_agt = m_vis.addAgent("Circle")
        circ_agt.setModel(pyflamegpu.ICOSPHERE)
        circ_agt.setModelScale(1/10.0)
        m_vis.activate()
    
# If no xml model file was is provided, generate a population.
    if not cudaSimulation.SimulationConfig().input_file:
    # Uniformly distribute agents within space, with uniformly distributed initial velocity.
        random.seed(cudaSimulation.SimulationConfig().random_seed)
        population = pyflamegpu.AgentVector(model.Agent("Circle"), env.getPropertyUInt("AGENT_COUNT"))
        for i in range(env.getPropertyUInt("AGENT_COUNT")):
            instance = population[i]
            instance.setVariableFloat("x",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
            instance.setVariableFloat("y",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
            instance.setVariableFloat("z",  random.uniform(0.0, env.getPropertyFloat("ENV_MAX")))
        cudaSimulation.setPopulationData(population)

    cudaSimulation.simulate()
#    cudaSimulation.exportData("end.xml")


#    if pyflamegpu.VISUALISATION:
#        m_vis.join()

if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()
