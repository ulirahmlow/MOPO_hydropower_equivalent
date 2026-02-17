import numpy as np
# from solve_equivalent import solve_equivalent
from solve_equivalent_compact import solve_equivalent,solve_equivalent_pump
from set_pso_problem import PSOproblem
from naive_aggregation import EqModel, ScenarioData
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

def inititale_eq_model(position, eqModel:EqModel):
    segments = eqModel.segments
    # Particle positions --> eqParameters
    eqModel.Mmax = position['Mmax']    # Max content in each reservoirs
    if 'Mmin' in position:
        eqModel.Mmin = position['Mmin']   # Min content in each reservoirs

    Q_break = eqModel.Q_break
    if 'Qmax' in position:
        Qmax_temp = position['Qmax']           # Max discharge in each plant
        for res in range(eqModel.nbr_stations):
            if f'res_{res}' not in Qmax_temp:
                continue
            for seg in range(eqModel.segments):
                eqModel.Qmax[f'Qmax_{seg}'][f'res_{res}'] = Qmax_temp[f'res_{res}'] * Q_break[seg]  
                if f'mu_{seg}' in position.keys():
                    eqModel.mu[f'mu_{seg}'][f'res_{res}'] = position[f'mu_{seg}'][f'res_{res}']

    elif 'Qmax_0' in position:
        eqModel.Qmax = {f'Qmax_{seg}': position[f'Qmax_{seg}'] for seg in range(segments)}
        for seg in range(segments):
            if f'mu_{seg}' in position.keys():
                eqModel.mu[f'mu_{seg}'] = position[f'mu_{seg}']

    if 'Qmin' in position:
        eqModel.Qmin = position['Qmin']

    if 'ramp3h' in position:
        eqModel.ramp3h = position['ramp3h']

    if 'ramp4h' in position:
        eqModel.ramp3h = position['ramp4h']

    if 'ramp1h' in position:
        eqModel.ramp1h = position['ramp1h']
    
    if 'delay' in position:
        eqModel.delay = position['delay']
        
    if 'Smin' in position:
        eqModel.Smin = position['Smin']
    
    if 'inflow_multiplier' in position:
        eqModel.inflow_multiplier = position['inflow_multiplier']
        for scen in range(eqModel.scenarios):
            for res in range(eqModel.nbr_stations):
                eqModel.inflow.loc[:,f'scen_{scen}_res_{res}'] = eqModel.inflow.loc[:,f'scen_{scen}_res_{res}'] * position['inflow_multiplier'][f'res_{res}']
        eqModel.inflow = eqModel.inflow.to_dict()

    if 'alpha' in position:
        eqModel.alpha = position['alpha'].to_frame()

    if 'Cmax' in position:
        eqModel.Cmax = position['Cmax']

    elif 'Cmax_0' in position:
        eqModel.Cmax = {f'Cmax_{seg}': position[f'Cmax_{seg}'] for seg in range(segments)}
        for k in range(eqModel.segments_pump):
            if f'mu_pump_{k}' in position:
                eqModel.mu_pump[f'mu_pump_{k}'] = position[f'mu_pump_{k}']

    if 'Pmax' in position:
        eqModel.Pmax = position['Pmax']
    elif 'Pmax_0' in position:
        eqModel.Pmax = {f'Pmax_{seg}': position[f'Pmax_{seg}'] for seg in range(segments)}
        for seg in range(eqModel.segments):
            if f'mu_{seg}' in position:
                eqModel.mu[f'mu_{seg}'] = position[f'mu_{seg}']

    if 'pump_loss' in position:
        eqModel.pump_loss = position['pump_loss']

    return eqModel

def fitness_pso(position, eqModel, scenario_data, problem):
    if problem.nominal_positions:
        position = position * (problem.varMax - problem.varMin) + problem.varMin
    eqModel_init = inititale_eq_model(position, deepcopy(eqModel))

    if problem.parameter.first_level_objectiv == 'pump':
        result_eqModel = solve_equivalent_pump(
            scenario_data=scenario_data, 
            eqModel = eqModel_init,
            problem = problem)
    else:   
        result_eqModel = solve_equivalent(
            scenario_data=scenario_data, 
            eqModel = eqModel_init,
            problem = problem)
    
    #end = time.perf_counter()
    #print(f"One opt call was taken {end-start}")
    if result_eqModel.optimal:
        if problem.parameter.first_level_objectiv == 'against_real_production':
            diff = np.sqrt(((result_eqModel.power - scenario_data.real_production)**2).sum().sum() / sum(scenario_data.hours))
        elif problem.parameter.first_level_objectiv == 'pump':
            diff = np.sqrt(((result_eqModel.power - scenario_data.real_production)**2).sum().sum() / sum(scenario_data.hours))

            if scenario_data.pump_consumption.sum().sum() > 0:
                diff = diff + np.sqrt(((result_eqModel.consumption - scenario_data.pump_consumption)**2).sum().sum() / sum(scenario_data.hours))

    else:
        diff = np.inf
    
    return diff
