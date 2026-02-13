from itertools import accumulate
import math
import numpy as np
from fitnessPSO import *
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
        position : dict
        velocity: dict
        fitness: Union[float,np.ndarray]
        best_position : dict
        best_fitness : Union[float,np.ndarray]


def initialize_pso(problem:PSOproblem, params:PSOparameters, eq_init:EqModel,train_scenario):
    logger.info(("Start PSO"))
    dim = problem.dim                                   # dimensions
    # search space limits
    varMin = problem.varMin
    varMax = problem.varMax
    # added res for loop at line ~40
    nPop = params.nPop                      # swarm size
    # Initialization -----------------------------------------------------------
    particles = Particle(
        position={n:{} for n in range(nPop)},
        velocity={n:{} for n in range(nPop)},
        fitness=np.ones(nPop) * np.inf,
        best_position={n:{} for n in range(nPop)},
        best_fitness=np.ones(nPop)* np.inf)

    global_best = {'fitness': np.inf,
                   'position': {}}
    problem.vMax = {var:{res: varMax[var][res] - varMin[var][res]  for res in varMin[var].keys() } for var in varMin.keys()}                 # maximum velocity, array
    problem.vStart = (sum(abs(varMax[var][res] - varMin[var][res]) for var in varMin.keys() for res in varMin[var].keys()))/dim    # start velocity (used for ideal velocity)
    iteration_parameters = {iter:{'avgChange':0, 'bestFitness_dev':0, 'idealChange':0, 'inertiaChange':0} for iter in range(params.maxIter)}
    random_position = {n:{var:{res: np.random.uniform(varMin[var][res], varMax[var][res]) for res in varMin[var].keys() } for var in varMin.keys()} for n in range(nPop)}
    radom_velocity = {n:{var:{res: np.random.uniform(0, 1) * problem.vMax[var][res] - problem.vMax[var][res]/2 for res in varMin[var].keys() } for var in varMin.keys()} for n in range(nPop)}
    particles.position = deepcopy(random_position)
    particles.velocity = deepcopy(radom_velocity)
    eq_init.dwnAd = eq_init.dwnAd.T.to_dict()
    eq_init.upAq = eq_init.upAq.T.to_dict()
    if isinstance(eq_init.mu, pd.DataFrame):
        eq_init.mu = eq_init.mu.to_dict()

    eq_init.Qmax = pd.DataFrame(
        np.zeros((eq_init.nbr_stations, eq_init.segments)),
        columns=[f'Qmax_{seg}' for seg in range(eq_init.segments)], 
        index = [f'res_{i_res}' for i_res in range(eq_init.nbr_stations)])
    eq_init.Qmax = eq_init.Qmax.to_dict()
    # If not delay then it should be said to none
    if not 'delay' in problem.varMax.keys():
        eq_init.delay = np.zeros(eq_init.nbr_stations)
        eq_init.delay_hours = np.zeros(eq_init.nbr_stations) 
        eq_init.delay_minutes = np.zeros(eq_init.nbr_stations) 

    if not 'Smin' in problem.varMax.keys():
        eq_init.Smin = {f'res_{res}': 0 for res in range(eq_init.nbr_stations)} 

    if not 'inflow_multiplier' in problem.varMax.keys() and not 'inflow_multiplier_per_res' in problem.varMax.keys() and not eq_init.inflow is None :
        eq_init.inflow = eq_init.inflow.to_dict()    

    train_scenario.price = train_scenario.price.to_dict()

    return problem, global_best, particles, iteration_parameters, eq_init, train_scenario

def limit_particle_to_search_space(position, velocity, problem):
    # Limit particle position to search space
    for (var,eq_res) in problem.zip_array:
        if position[var][eq_res] <= problem.varMin[var][eq_res]:
            velocity[var][eq_res] = 0 #problem.varMin[var][eq_res] - (position[var][eq_res] - velocity[var][eq_res])   
            position[var][eq_res] = problem.varMin[var][eq_res]

        elif position[var][eq_res] >= problem.varMax[var][eq_res]:
            velocity[var][eq_res] = 0  # problem.varMax[var][eq_res] - (position[var][eq_res] - velocity[var][eq_res])
            position[var][eq_res] = problem.varMax[var][eq_res]

    return position, velocity

def check_specific_limts(position:dict, velocity:dict, problem:PSOproblem, eqModel:EqModel):
    for res in problem.res_array:
        # if Mmax < Mmin
        if position['Mmax'][res] < position['Mmin'][res] : 
            position['Mmax'][res] = position['Mmin'][res] * 1.05
            position['Mmin'][res] = position['Mmin'][res] * 0.95
            velocity['Mmax'][res] = 0
            #velocity['Mmin'][res] = position['Mmin'][res] - (position['Mmin'][res] /0.95 - velocity['Mmin'][res])

        if 'Qmax' in position:
            if position['Qmax'][res] <= position['Qmin'][res]:  
                position['Qmax'][res] = position['Qmin'][res]*1.1
                position['Qmin'][res] *= 0.9
            for seg in range(1,eqModel.segments):
                if f'mu_{seg-1}' in position:
                    if position[f'mu_{seg}'][res] > position[f'mu_{seg-1}'][res]: 
                        position[f'mu_{seg}'][res] = position[f'mu_{seg-1}'][res] - 0.000001 # Make it slitly lower

        else: 
            if position['Qmax_0'][res] < position['Qmin'][res]:  
                position['Qmax_0'][res] = position['Qmin'][res]*1.1
                position['Qmin'][res] *= 0.9

            for seg in range(1,eqModel.segments):
                if position[f'Qmax_{seg}'][res] > position[f'Qmax_{seg-1}'][res]: 
                    position[f'Qmax_{seg-1}'][res] = position[f'Qmax_{seg}'][res]*1.1
                    position[f'Qmax_{seg}'][res] *= 0.9
                    
                if f'mu_{seg-1}' in position:
                    if position[f'mu_{seg}'][res] > position[f'mu_{seg-1}'][res]: 
                        position[f'mu_{seg}'][res] = position[f'mu_{seg-1}'][res] - 0.000001 # Make it slitly lower

    return position, velocity

def check_feasable_position(position, velocity, problem:PSOproblem, eqModel):
    if problem.parameter.first_level_objectiv != 'pump':
        position, velocity = check_specific_limts(position, velocity, problem, eqModel)

    position, velocity = limit_particle_to_search_space(position, velocity, problem)

    # Reesitamte alpha to get in total one
    if 'alpha' in position:
        position['alpha'] = (position['alpha']  / position['alpha'].sum())

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
        keys = list(particle.velocity.keys())
        lengths = [len(particle.velocity[k]) for k in keys]
        starts = [0] + list(accumulate(lengths))[:-1]
        velocity = {}
    # If a particle was not feasible try another postion first before updating velocity
    if len(particle.best_position)==0:
        if len(global_best['position'])==0:
            particle.position= {var:{res: np.random.uniform(problem.varMin[var][res], problem.varMax[var][res]) for res in problem.varMin[var].keys() } for var in problem.varMin.keys()}
        else:
            particle.position = global_best['position']
            for key, start in zip(keys, starts):
                inner_keys = list(particle.velocity[key].keys())
                # Convert values of the inner dict for vel, po, and g_po to NumPy arrays
                v_vals    = np.array([particle.velocity[key][k] for k in inner_keys])
                po_vals   = np.array([particle.position[key][k] for k in inner_keys])
                g_po_vals = np.array([global_best['position'][key][k] for k in inner_keys])
                # Select the appropriate slice from the c2 array.
                c2_slice = c2_random[start: start + len(particle.velocity[key])]
                # Compute the corresponding velocity values.
                velocity_vals = problem.parameter.inertia * v_vals + problem.parameter.c2*c2_slice * (g_po_vals - po_vals)
                # Pair the inner keys with their computed values.
                inner_keys = list(particle.velocity[key].keys())
                velocity[key] = dict(zip(inner_keys, velocity_vals))     
            particle.velocity = velocity
        return particle

    elif caziness_number > 0.1:  # not caziness
        new_position = deepcopy(particle.position)
        for key, start in zip(keys, starts):
            inner_keys = list(particle.velocity[key].keys())
            
            v_vals    = np.array([particle.velocity[key][k] for k in inner_keys])
            po_vals   = np.array([particle.position[key][k] for k in inner_keys])
            po_best_vals = np.array([particle.best_position[key][k] for k in inner_keys])
            g_po_vals = np.array([global_best['position'][key][k] for k in inner_keys])
            v_max_vals = np.array([problem.vMax[key][k] for k in inner_keys])

            c2_slice = c2_random[start: start + len(particle.velocity[key])]
            c1_slice = c1_random[start: start + len(particle.velocity[key])]
            
            velocity_vals  = (problem.parameter.inertia * v_vals 
                              + problem.parameter.c1 * c1_slice * (po_best_vals - po_vals)
                              + problem.parameter.c2 * c2_slice * (g_po_vals - po_vals))

            neg_vmax_vals = -1 * v_max_vals
            max_vals = np.maximum(velocity_vals, neg_vmax_vals)
            max_min_velocity = np.minimum(max_vals,v_max_vals)
            velocity[key] = dict(zip(inner_keys, max_min_velocity))
            po_vals = po_vals + max_min_velocity
            new_position[key] = dict(zip(inner_keys, po_vals))

        particle.position = new_position
        particle.velocity = velocity

    else:   # craziness
        for outer_key, inner_dict in problem.vMax.items():
            keys = list(inner_dict.keys())
            po_vals = np.array([particle.position[outer_key][k] for k in keys])
            values = np.array([inner_dict[k] for k in keys])
            random_weights = -1 + 2 * np.random.uniform(0, 1, size=len(values))
            scaled_values = values * random_weights
            velocity[outer_key] = dict(zip(keys, scaled_values))
            po_vals = po_vals + scaled_values
            particle.position[outer_key] = dict(zip(keys, po_vals))
            particle.velocity[outer_key] = velocity[outer_key]

    return particle

def deep_sum(d):
    stack = [d]
    total = 0
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())  # Process values in the dict
        elif isinstance(current, (int, float)) and not math.isnan(current):  
            total += abs(current)  # Sum numeric values
    return total

def update_globals(global_best_fitness, it, particles, params, problem, iteration_parameters):
    # Store the best fitness value
    iteration_parameters[it]['bestFitness_dev'] = global_best_fitness
    # Display iteration information
    if (params.showIter) & ((it == 1) | (it % 5 == 0)):
        logger.info(f"-- Iteration {it}: Best fitness value = {global_best_fitness}")

    # Update inertia ---------------------------------------------
    vAvg = deep_sum(particles.velocity) * (1/(problem.dim*params.nPop))
    iteration_parameters[it]['avgChange'] = vAvg

    angle = (it*np.pi)/(params.maxIter*0.95)
    vIdeal = problem.vStart * (1 + np.cos(angle))/2
    iteration_parameters[it]['idealChange'] = vIdeal

    if vAvg >= vIdeal:
        params.inertia = max(params.inertia - params.wDelta, params.wMin)
    else:
        params.inertia = min(params.inertia + params.wDelta, params.wMax)

    iteration_parameters[it]['inertiaChange'] = params.inertia

    return iteration_parameters, params


def initilaze_particle(particles, n_particle):
    particle = Particle(
        position=deepcopy(particles.position[n_particle]),
        velocity=deepcopy(particles.velocity[n_particle]),
        fitness=deepcopy(particles.fitness[n_particle]),
        best_fitness=deepcopy(particles.best_fitness[n_particle]),
        best_position=deepcopy(particles.best_position[n_particle]),
        )

    return particle

def update_particles(particles: Particle, mulit_results: list, global_best: dict):
    for n_particle, particle in enumerate(mulit_results):
        particles.position[n_particle] = particle.position
        particles.velocity[n_particle] = particle.velocity
        particles.fitness[n_particle] = particle.fitness

        if particle.fitness < particle.best_fitness:
            particles.best_position[n_particle] = particle.position
            particles.best_fitness[n_particle] = particle.fitness

        if particle.fitness < global_best['fitness']:
            global_best['position'] = particle.position
            global_best['fitness'] = particle.fitness
            logger.info(f" best fittness :  {global_best['fitness']}")
            logger.info(f" best position \n {pd.DataFrame.from_dict(global_best['position'])}")

    return particles, global_best

def run_iteration_pso(arg):

    # n_particle, particles, problem, original, eqModel, scenario_data = arg
    particle, problem, eqModel, scenario_data, global_best = arg

    particle = update_particle_position(particle, global_best, problem)

    particle.position, particle.velocity = check_feasable_position(
        particle.position,
        particle.velocity,
        problem,
        eqModel)

    # Current fitness value
    # particles.fitness[n] = fitness_pso(particles.position[n,:], original, eqModel, scenario_data.price)
    particle.fitness = fitness_pso(
        position=particle.position, 
        eqModel=eqModel, 
        scenario_data=scenario_data,
        problem = problem)

    return particle


def run_initial_iteration(arg):
    n_particle, particles, problem, eqModel, scenario_data = arg

    particle = initilaze_particle(particles, n_particle)

    # for d=1:res # limit min relative max M and Q
    particle.position, particle.velocity = check_feasable_position(
        particle.position,
        particle.velocity,
        problem,
        eqModel)

    # particles.fitness[n] = fitnessPSO(particles.position[n,:], original, eqModel, scenario_data.price)
    # fitnes_pso
    particle.fitness = fitness_pso(
        position=particle.position,
        eqModel=eqModel,
        scenario_data=scenario_data,
        problem=problem)

    return particle
    # return particle, global_best