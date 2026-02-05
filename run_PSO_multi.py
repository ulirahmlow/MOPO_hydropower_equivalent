# -----------------------------------------------------------------------------
# ---            FUNCTION CONTAINING THE PSO ALGORITHM,
# ---            Together with Multiprocession                          - - ----
# ---            EQ PARAMETERS AND FINAL FITNESSVALUE                      -----
# ------------------------------------------------------------------------------
import numpy as np
from fitnessPSO import fitness_pso
#from fitnessPSO_test import fitness_pso
import pandas as pd
from copy import deepcopy
import logging
from dataclasses import dataclass
from typing import Union
from naive_aggregation import EqModel
from set_config import PSOparameters
from solve_equivalent_compact import solve_equivalent
from set_pso_problem import PSOproblem

logger = logging.getLogger(__name__)


@dataclass
class Particle:
        position : pd.DataFrame
        velocity: pd.DataFrame
        fitness: Union[float,np.ndarray]
        best_position : pd.DataFrame
        best_fitness : Union[float,np.ndarray]


def initialize_pso(problem:PSOproblem, params:PSOparameters, eq_init:EqModel,train_scenario):
    logger.info(("Start PSO"))
    dim = problem.dim                                   # dimensions
    # search space limits
    varMin = problem.varMin
    varMax = problem.varMax
    # added res for loop at line ~40
    nPop = params.nPop                      # swarm size
    # dynamic inertia parameters
    # params.wDelta = 0.1
    # params.wMin = 0.3
    # params.wMax = 0.9
    init_df = pd.DataFrame(np.zeros((nPop, dim)), columns=problem.varMax.columns)
    multi_index = pd.MultiIndex.from_product([init_df.index, varMax.index])

    # Initialization -----------------------------------------------------------
    particles = Particle(
        position=pd.DataFrame(index=multi_index, columns=init_df.columns, dtype=float),
        velocity=pd.DataFrame(index=multi_index, columns=init_df.columns, dtype=float),
        fitness=np.ones(nPop) * np.inf,
        best_position=pd.DataFrame(index=multi_index, columns=init_df.columns),
        best_fitness=np.ones(nPop)* np.inf )     # init particle swarm

    # init global best
    global_best = {'fitness': np.inf,
                   'position': particles.position.loc[0, :][:]}
    problem.vMax = varMax - varMin                       # maximum velocity, array
    problem.vStart = ((abs(varMax - varMin)).sum(axis=1)/dim).iloc[0]    # start velocity (used for ideal velocity)
    iteration_parameters = pd.DataFrame(
        index=range(params.maxIter),
        columns=['avgChange', 'bestFitness_dev', 'idealChange', 'inertiaChange'])

    random_position = np.random.uniform(
        low=problem.varMin, high = problem.varMax, size = (nPop,len(problem.varMin.index),dim))
    random_position = pd.DataFrame(
        random_position.reshape(-1,random_position.shape[-1]),
        columns=particles.position.columns,index=particles.position.index)
    particles.position = deepcopy(random_position)
    radom_velocity = np.random.uniform(
            low= 0, high= 1, 
            size = (nPop,len(problem.varMin.index),dim))*problem.vMax.values-problem.vMax.values/2
    radom_velocity = pd.DataFrame(
        radom_velocity.reshape(-1,random_position.shape[-1]),
        columns=particles.velocity.columns,index=particles.velocity.index)
    particles.velocity = deepcopy(radom_velocity)
    eq_init.dwnAd = eq_init.dwnAd.T.to_dict()
    eq_init.upAq = eq_init.upAq.T.to_dict()
    if isinstance(eq_init.mu, pd.DataFrame):
        eq_init.mu = eq_init.mu.to_dict()
    if isinstance(eq_init.mu_pump, pd.DataFrame):
        eq_init.mu_pump = eq_init.mu_pump.to_dict()
        
    if 'Qmax' in problem.parameter.varaibles:
        eq_init.Qmax = pd.DataFrame(
            np.zeros((eq_init.nbr_stations, eq_init.segments)),
            columns=[f'Qmax_{seg}' for seg in range(eq_init.segments)], 
            index = [f'res_{i_res}' for i_res in range(eq_init.nbr_stations)])
    
    # If not delay then it should be said to none
    if not 'delay' in problem.varMax.columns:
        eq_init.delay = None
    
    if not 'Smin' in problem.varMax.columns:
        eq_init.Smin.loc[:] = 0

    if not 'inflow_multiplier' in problem.varMax.columns and isinstance(eq_init.inflow, pd.DataFrame):
        eq_init.inflow = eq_init.inflow.to_dict()    
    if not 'ramp_1' in problem.varMax.columns:
        eq_init.ramp1h = {'res_0':0}
        eq_init.ramp3h = {'res_0':0}
        eq_init.ramp4h = {'res_0':0}
    train_scenario.price = train_scenario.price.to_dict()


    return problem, global_best, particles, iteration_parameters, eq_init, train_scenario

def initialize_pso_normalized(problem, params:PSOparameters, eq_init:EqModel,train_scenario):
    logger.info(("Start PSO"))
    dim = problem.dim                                   # dimensions
    # search space limits
    varMin = problem.varMin
    varMax = problem.varMax
    # added res for loop at line ~40
    nPop = params.nPop                      # swarm size
    # dynamic inertia parameters
    # params.wDelta = 0.1
    # params.wMin = 0.3
    # params.wMax = 0.9
    init_df = pd.DataFrame(np.zeros((nPop, dim)), columns=problem.varMax.columns)
    multi_index = pd.MultiIndex.from_product([init_df.index, varMax.index])

    # Initialization -----------------------------------------------------------
    particles = Particle(
        position=pd.DataFrame(index=multi_index, columns=init_df.columns, dtype=float),
        velocity=pd.DataFrame(index=multi_index, columns=init_df.columns, dtype=float),
        fitness=np.ones(nPop) * np.inf,
        best_position=pd.DataFrame(index=multi_index, columns=init_df.columns),
        best_fitness=np.ones(nPop)* np.inf )     # init particle swarm

    # init global best
    global_best = {'fitness': np.inf,
                   'position': particles.position.loc[0, :][:]}
    problem.vMax = deepcopy(varMax)                      # maximum velocity, array
    problem.vMax.loc[:] = 1 
    problem.vStart = (1/dim)    # start velocity (used for ideal velocity)
    iteration_parameters = pd.DataFrame(
        index=range(params.maxIter),
        columns=['avgChange', 'bestFitness_dev', 'idealChange', 'inertiaChange'])

    random_position = np.random.uniform(
        low=0, high = 1, size = (nPop,len(problem.varMin.index),dim))
    random_position = pd.DataFrame(
        random_position.reshape(-1,random_position.shape[-1]),
        columns=particles.position.columns, index=particles.position.index)
    particles.position = deepcopy(random_position)
    radom_velocity = np.random.uniform(
            low= 0, high= 1, 
            size = (nPop,len(problem.varMin.index),dim))*problem.vMax.values-problem.vMax.values/2
    radom_velocity = pd.DataFrame(
        radom_velocity.reshape(-1,random_position.shape[-1]),
        columns=particles.velocity.columns,index=particles.velocity.index)
    particles.velocity = deepcopy(radom_velocity)
    eq_init.dwnAd = eq_init.dwnAd.T.to_dict()
    eq_init.upAq = eq_init.upAq.T.to_dict()
    eq_init.mu = eq_init.mu.to_dict()
    

    eq_init.Qmax = pd.DataFrame(
        np.zeros((eq_init.nbr_stations, eq_init.segments)),
        columns=[f'Qmax_{seg}' for seg in range(eq_init.segments)], 
        index = [f'res_{i_res}' for i_res in range(eq_init.nbr_stations)])
    
    # If not delay then it should be said to none
    if not 'delay' in problem.varMax.columns:
        eq_init.delay = None
    
    if not 'Smin' in problem.varMax.columns:
        eq_init.Smin.loc[:] = 0

    if not 'inflow_multiplier' in problem.varMax.columns:
        eq_init.inflow = eq_init.inflow.to_dict()    
    
    train_scenario.price = train_scenario.price.to_dict()

    if not 'without' in problem.parameter.thermal:
        train_scenario.demand = train_scenario.demand.to_dict()

    return problem, global_best, particles, iteration_parameters, eq_init, train_scenario

def limit_particle_to_search_space(position, velocity, problem):
    # Limit particle position to search space
    for d in problem.varMin.columns:
        for eq_res in problem.varMin.index:
            if position.loc[eq_res,d] <= problem.varMin[d][eq_res]:
                position.loc[eq_res,d] = problem.varMin[d][eq_res]
                velocity.loc[eq_res,d] = 0

            elif position.loc[eq_res,d] >= problem.varMax[d][eq_res]:
                position.loc[eq_res,d] = problem.varMax[d][eq_res]
                velocity.loc[eq_res,d] = 0
    return position, velocity

def check_specific_limts(position, problem, eqModel):
    for d in problem.varMin.index:
        # if Mmax < Mmin
        if position.loc[d,'Mmax'] < position.loc[d,'Mmin']: 
            position.loc[d,'Mmax'] = position.loc[d,'Mmin']*1.05
            position.loc[d,'Mmin'] = position.loc[d,'Mmin'] * 0.95

        if 'Qmax' in position:
            if position.loc[d,'Qmax'] <= position.loc[d,'Qmin']:  
                position.loc[d,'Qmax'] = position.loc[d,'Qmin']*1.1
                position.loc[d,'Qmin'] *= 0.9
            for seg in range(1,eqModel.segments):
                if f'mu_{seg-1}' in position:
                    if position.loc[d,f'mu_{seg}'] > position.loc[d,f'mu_{seg-1}']: 
                        position.loc[d,f'mu_{seg}'] = position.loc[d,f'mu_{seg-1}'] - 0.000001 # Make it slitly lower

        else: 
            if position.loc[d,'Qmax_0'] < position.loc[d,'Qmin']:  
                position.loc[d,'Qmax_0'] = position.loc[d,'Qmin']*1.1
                position.loc[d,'Qmin'] *= 0.9

            for seg in range(1,eqModel.segments):
                if position.loc[d,f'Qmax_{seg}'] > position.loc[d,f'Qmax_{seg-1}']: 
                    position.loc[d,f'Qmax_{seg-1}'] = position.loc[d,f'Qmax_{seg}']*1.1
                    position.loc[d,f'Qmax_{seg}'] *= 0.9
                    
                if f'mu_{seg-1}' in position:
                    if position.loc[d,f'mu_{seg}'] > position.loc[d,f'mu_{seg-1}']: 
                        position.loc[d,f'mu_{seg}'] = position.loc[d,f'mu_{seg-1}'] - 0.000001 # Make it slitly lower

    return position

def check_feasable_position(position, velocity, problem:PSOproblem, eqModel):
    """_summary_

    Args:
        position (_type_): _description_
        velocity (_type_): _description_
        problem (_type_): _description_

    Returns:
        _type_: _description_
    """
    if problem.nominal_positions:
        position = position * (problem.varMax - problem.varMin) + problem.varMin
    if problem.parameter.first_level_objectiv != 'pump':
        position = check_specific_limts(position, problem, eqModel)

    position, velocity = limit_particle_to_search_space(position, velocity, problem)

    # Reesitamte alpha to get in total one
    if 'alpha' in position:
        position['alpha'] = (position['alpha']  / position['alpha'].sum())

    if problem.nominal_positions:
        position = (position - problem.varMin) /(problem.varMax - problem.varMin)

    return position, velocity

def update_particle_position(particle, global_best, problem, test_seed = 0):
    if test_seed:
        np.random.seed(42)
        caziness_number = np.random.uniform(0, 1)
        np.random.seed(42)
        c1_random = np.random.uniform(0, 1, size=problem.dim)
        np.random.seed(42)
        c2_random = np.random.uniform(0, 1, size=problem.dim)
    else:
        caziness_number = np.random.uniform(0, 1)
        c1_random = np.random.uniform(0, 1, size=problem.dim)
        c2_random = np.random.uniform(0, 1, size=problem.dim)
    # If a particle was not feasible try another postion first before updating velocity
    if particle.best_position.isna().any().any():
        if global_best['position'].isna().any().any():
            particle.position=  pd.DataFrame(np.random.uniform(problem.varMin,problem.varMax),
                                             columns=problem.varMax.columns,index=particle.position.index)
        else:
            particle.position = global_best['position']
            particle.velocity= (
                problem.parameter.inertia * particle.velocity.loc[:] +
                    problem.parameter.c2 * c2_random * (
                        global_best['position'] - particle.position.loc[:])
                    )

    elif caziness_number > 0.1:  # not caziness
        particle.velocity = (
        problem.parameter.inertia * particle.velocity.loc[:] +
            problem.parameter.c1 *c1_random * (
                particle.best_position.loc[:] - particle.position.loc[:]) +
            problem.parameter.c2 * c2_random * (
                global_best['position'] - particle.position.loc[:])
            )

        # Limit velocity
        maximum_velocity = pd.concat(
            [-1*problem.vMax, 
                particle.velocity.loc[:]])
        maximum_velocity = maximum_velocity.groupby(maximum_velocity.index).max()
        particle.velocity = maximum_velocity

        minimumm_velocity = pd.concat(
            [problem.vMax,particle.velocity.loc[:]])
        minimumm_velocity = minimumm_velocity.groupby(minimumm_velocity.index).min()
        particle.velocity= minimumm_velocity

    else:   # craziness
        particle.velocity = (-1)*problem.vMax + 2*np.random.uniform(0,1,size=problem.dim)*problem.vMax
    # Compute new position -----------------------------------
    particle.position += particle.velocity

    return particle


def update_globals(global_best_fitness, it, particles, params, problem, iteration_parameters):
    # Store the best fitness value
    iteration_parameters.loc[it, 'bestFitness_dev'] = global_best_fitness
    # Display iteration information
    if (params.showIter) & ((it == 1) | (it % 5 == 0)):
        logger.info(f"-- Iteration {it}: Best fitness value = {global_best_fitness}")

    # Update inertia ---------------------------------------------
    vAvg = (abs(particles.velocity)).sum().sum() * (1/(problem.dim*params.nPop))
    iteration_parameters.loc[it, 'avgChange'] = vAvg

    angle = (it*np.pi)/(params.maxIter*0.95)
    vIdeal = problem.vStart * (1 + np.cos(angle))/2
    iteration_parameters.loc[it, 'idealChange'] = vIdeal

    if vAvg >= vIdeal:
        params.inertia = max(params.inertia - params.wDelta, params.wMin)
    else:
        params.inertia = min(params.inertia + params.wDelta, params.wMax)

    iteration_parameters.loc[it, 'inertiaChange'] = params.inertia

    return iteration_parameters, params


def initilaze_particle(particles, n_particle):
    particle = Particle(
        position=deepcopy(particles.position.loc[n_particle, :][:]),
        velocity=deepcopy(particles.velocity.loc[n_particle, :][:]),
        fitness=deepcopy(particles.fitness[n_particle]),
        best_fitness=deepcopy(particles.best_fitness[n_particle]),
        best_position=deepcopy(particles.best_position.loc[n_particle, :][:])
    )

    return particle

def update_particles(particles: Particle, mulit_results: list, global_best: dict):
    for n_particle, particle in enumerate(mulit_results):
        
        particles.position.loc[n_particle, :] = particle.position.values
        particles.velocity.loc[n_particle, :] = particle.velocity.values
        particles.fitness[n_particle] = particle.fitness
        if particle.fitness < particle.best_fitness:
            particles.best_position.loc[n_particle, :] = particle.position.values
            particles.best_fitness[n_particle] = particle.fitness

        if particle.fitness < global_best['fitness']:
            global_best['position'] = particle.position
            global_best['fitness'] = particle.fitness
            logger.info(f" best fittness :  {global_best['fitness']}")
            logger.info(f" best position \n {global_best['position']}")

    return particles, global_best

def write_to_output_model(eqModel:EqModel, global_best,conf,problem):
    # Convert the best particle position --> Eq parameters ---------------------
    eqModel_output =EqModel()
    eqModel_output.nbr_stations = eqModel.nbr_stations    
    eqModel_output.segments = conf['para_general']['segments']
    eqModel_output.segments_pump = eqModel.segments_pump 
    eqModel_output.alpha =eqModel.alpha
    eqModel_output.mu = eqModel.mu
    eqModel_output.mu_pump = eqModel.mu_pump
    eqModel_output.Smin = eqModel.Smin
    eqModel_output.dwnAd = eqModel.dwnAd 
    eqModel_output.upAq = eqModel.upAq
    eqModel_output.indices = eqModel.indices
    eqModel_output.indices_with_segments = eqModel.indices_with_segments
    eqModel_output.indices_with_segments_pump = eqModel.indices_with_segments_pump
    eqModel_output.indices_thermal = eqModel.indices_thermal
    eqModel_output.scenarios = eqModel.scenarios
    eqModel_output.Mmax = global_best['position']['Mmax'].to_dict()    # Max content in each reservoirs
    if 'Mmin' in global_best['position']:
        eqModel_output.Mmin = global_best['position']['Mmin'].to_dict()    # Min content in each reservoirs

    if 'Qmax' in  global_best['position']:
        Qmax = global_best['position']['Qmax'].to_frame()            # Max discharge in each plant
        eqModel_output.Qmax = pd.DataFrame(
            np.zeros((
                eqModel_output.nbr_stations, eqModel_output.segments)),
                columns=[f'Qmax_{seg}' for seg in range(eqModel_output.segments)],
                index=Qmax.index)
        
        for i in range(eqModel_output.nbr_stations):
            for k in range(eqModel_output.segments):
                eqModel_output.Qmax.iloc[i,k] = Qmax.iloc[i] * conf['para_PSO']['Q_break'][k]

        eqModel_output.Qmax = eqModel_output.Qmax.to_dict()
    elif 'Qmax_0' in  global_best['position']:
        eqModel_output.Qmax =  global_best['position'][(f'Qmax_{seg}' for seg in range(eqModel_output.segments))].to_dict()
    if 'Qmin' in global_best['position']:
        eqModel_output.Qmin = global_best['position']['Qmin'].to_dict()  # Minimum discharge in each plant

    if 'ramp1h' in global_best['position']:
        eqModel_output.ramp1h = global_best['position']['ramp1h'].to_dict()     # Limit on ramping for 1 hour
    if 'ramp3h' in global_best['position']:
        eqModel_output.ramp3h = global_best['position']['ramp3h'].to_dict()     # Limit on ramping for 3 hours

    if 'mu_1' in  global_best['position']:
         for res in range(eqModel_output.nbr_stations):
            for seg in range(1,eqModel_output.segments):
                eqModel_output.mu[f'mu_{seg}'][f'res_{res}'] = global_best['position'].loc[f'res_{res}',f'mu_{seg}']

    if 'delay' in global_best['position']:
        eqModel_output.delay=global_best['position']['delay']
    
    if 'Smin' in global_best['position']:
         eqModel_output.Smin=global_best['position']['Smin'].to_dict()
    else:
        eqModel_output.Smin.index = [f'res_{i_res}' for i_res in range(eqModel_output.nbr_stations)]
        eqModel_output.Smin = eqModel_output.Smin['Smin'].to_dict()
    

    if 'inflow_multiplier' in global_best['position']:
        eqModel_output.inflow_multiplier = global_best['position']['inflow_multiplier'].to_dict()['res_0']
        eqModel_output.inflow = eqModel.inflow * global_best['position']['inflow_multiplier'].to_dict()['res_0']
        eqModel_output.inflow = eqModel_output.inflow.to_dict()
    else:
        eqModel_output.inflow = eqModel.inflow

    if 'alpha' in  global_best['position']:
        eqModel_output.alpha=global_best['position']['alpha']
    if 'Pmax' in  global_best['position']:
        eqModel_output.Pmax=global_best['position']['Pmax']
    elif 'Pmax_0' in  global_best['position']:
        eqModel_output.Pmax =  global_best['position'][(f'Pmax_{seg}' for seg in range(eqModel_output.segments))].to_dict()
    if 'Cmax' in  global_best['position']:
        eqModel_output.Cmax=global_best['position']['Cmax']
    elif 'Cmax_0' in  global_best['position']:
        eqModel_output.Cmax =  global_best['position'][(f'Cmax_{seg}' for seg in range(eqModel_output.segments_pump))].to_dict()
    if 'mu_pump_1' in global_best['position']:
         for res in range(eqModel_output.nbr_stations):
            for seg in range(1,eqModel_output.segments_pump):
                eqModel_output.mu_pump[f'mu_pump_{seg}'][f'res_{res}'] = global_best['position'].loc[f'res_{res}',f'mu_pump_{seg}']
    if 'pump_loss' in  global_best['position']:
        eqModel_output.pump_loss=global_best['position']['pump_loss']

    return eqModel_output


def run_iteration_pso(arg):

    # n_particle, particles, problem, original, eqModel, scenario_data = arg
    particle, problem, eqModel, scenario_data, global_best = arg

    particle = update_particle_position(particle, global_best, problem)

    particle.position.loc[:], particle.velocity.loc[:] = check_feasable_position(
        particle.position.loc[:],
        particle.velocity.loc[:],
        problem,
        eqModel)

    # Current fitness value
    # particles.fitness[n] = fitness_pso(particles.position[n,:], original, eqModel, scenario_data.price)
    particle.fitness = fitness_pso(
        position=particle.position.loc[:], 
        eqModel=eqModel, 
        scenario_data=scenario_data,
        problem = problem)

    return particle


def run_initial_iteration(arg):
    n_particle, particles, problem, eqModel, scenario_data = arg

    particle = initilaze_particle(particles, n_particle)

    # for d=1:res # limit min relative max M and Q
    particle.position.loc[:], particle.velocity.loc[:] = check_feasable_position(
        particle.position.loc[:],
        particle.velocity.loc[:],
        problem,
        eqModel)

    # particles.fitness[n] = fitnessPSO(particles.position[n,:], original, eqModel, scenario_data.price)
    # fitnes_pso
    particle.fitness = fitness_pso(
        position=particle.position.loc[:],
        eqModel=eqModel,
        scenario_data=scenario_data,
        problem=problem)

    return particle
    # return particle, global_best