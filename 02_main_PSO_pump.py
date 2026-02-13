import logging
import multiprocessing
import time
import os

import sys
from pathlib import Path

# Make the src/ folder importable
sys.path.append(str(Path(__file__).parent / "src"))

from get_period_data import get_period_data
from naive_aggregation import create_eq_model
from read_scenario_data import read_scenario_data
from run_PSO_multi import *
from set_config import set_config
from set_pso_problem import set_pso_problem_pump
from save_output import save_output_pump

sys.stdout = open(os.devnull, 'w')
mpl_handler = logging.getLogger('matplotlib')
mpl_handler.setLevel(logging.ERROR)
gurobi_logger = logging.getLogger('gurobipy')
gurobi_logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    logger = logging.getLogger('main_PSO_pump')

    # Specify the data folder in .env file 
    folder_path =  os.getenv("SECRET_FOLDER_PATH")

    config_file_name = 'Pump' 
    areas = ['UK','PL'] # area list-> 2018 
    years = '2018' # Select a year or a set of years like from_till: 2018_2020
    for area in areas:
        if area in ['ES','DE']:
            years = '2023'
        conf, param_pso = set_config(
            area=area, # Area 
            run_setup='Ext_inflow', # folder in the data
            category=1,
            config_file_name=config_file_name,
            folder=folder_path,
            year=years)
        
        train_scenario= read_scenario_data(conf=conf)
        
        train_scenario = get_period_data(
            train_scenario, conf)
        
        train_scenario.scenarios = len(train_scenario.price.keys())
        eq_init = create_eq_model(train_scenario, conf)
        
        problem = set_pso_problem_pump(
            eq_init=eq_init,
            params_pso=param_pso,
            scenario=train_scenario)
        start_pso = time.time()
        ## PSO: Initial loop
        problem, global_best, particles, iteration_parameters, eq_init, train_scenario = initialize_pso(problem, param_pso, eq_init, train_scenario)

        arguments = [(
            n_particle,
            particles,
            problem, eq_init, train_scenario) for n_particle in range(param_pso.nPop)]
        
        if conf['para_PSO']['use_multiprocessing']: 
            with multiprocessing.Pool(processes=param_pso.nPop) as pool:  
                results_mulit = pool.map(run_initial_iteration, arguments)
        else:
            results_mulit = [run_initial_iteration(arg) for arg in arguments]
        particles, global_best = update_particles(deepcopy(particles), results_mulit, global_best)
        stored_particles = [deepcopy(particles)]
        stored_global_best = [deepcopy(global_best)]

        # PSO MAIN LOOP:
        for iter in range(param_pso.maxIter):
        
            arguments = [(
                initilaze_particle(particles, n_particle),
                problem, 
                eq_init, train_scenario, global_best) for n_particle in range(param_pso.nPop)]
            if conf['para_PSO']['use_multiprocessing']:
                with multiprocessing.Pool(processes=param_pso.nPop) as pool: # processes=param_pso.nPop
                    results_mulit = pool.map(run_iteration_pso, arguments)
            else:
                results_mulit = [run_iteration_pso(arg) for arg in arguments]

            particles, global_best = update_particles(deepcopy(particles), results_mulit, global_best)
            stored_particles.append(deepcopy(particles))
            stored_global_best.append(deepcopy(global_best))
            iteration_parameters, param_pso =  update_globals(
                global_best['fitness'], iter, particles, param_pso, problem, iteration_parameters)
        
        eqModel_output = inititale_eq_model(global_best['position'], deepcopy(eq_init))
        elapsed_pso = time.time()-start_pso

        logger.info(
            f"--- Time for {param_pso.maxIter} iterations, "
            f"{param_pso.nPop} population, in PSO for "
            f"with j= {conf['para_general']['segments']}: {elapsed_pso}"
        )
        conf['Time_needed_s'] = elapsed_pso # Save time needed 
        save_output_pump(
            config=conf,
            train_scenario=train_scenario,
            eq_updated=eqModel_output,
            iteration_parameters=iteration_parameters,
            problem=problem)
