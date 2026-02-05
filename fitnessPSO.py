import numpy as np
# from solve_equivalent import solve_equivalent
from solve_equivalent_compact import solve_equivalent,solve_equivalent_pump
from naive_aggregation import EqModel
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

def inititale_eq_model(position, eqModel:EqModel):

    segments = eqModel.segments
    # Particle positions --> eqParameters
    eqModel.Mmax = position['Mmax'].to_dict()    # Max content in each reservoirs
    if 'Mmin' in position:
        eqModel.Mmin = position['Mmin'].to_dict()    # Min content in each reservoirs

    Q_break = eqModel.Q_break
    if 'Qmax' in position:
        Qmax_temp = position['Qmax']           # Max discharge in each plant
        for i in range(eqModel.nbr_stations):
            for k in range(eqModel.segments):
                eqModel.Qmax.iloc[i,k] = Qmax_temp.iloc[i] * Q_break[k]                
                if f'mu_{k}' in position.columns:
                    eqModel.mu[f'mu_{k}'][f'res_{i}'] = position.loc[f'res_{i}', f'mu_{k}']
        eqModel.Qmax.index = position['Qmax'].index
        eqModel.Qmax = eqModel.Qmax.to_dict()
    elif 'Qmax_0' in position:
        eqModel.Qmax = position[(f'Qmax_{seg}' for seg in range(segments))].to_dict()
        for i in range(eqModel.nbr_stations):
            for k in range(eqModel.segments):
                if f'mu_{k}' in position.columns:
                    eqModel.mu[f'mu_{k}'][f'res_{i}'] = position.loc[f'res_{i}', f'mu_{k}']
    if 'Qmin' in position:
        eqModel.Qmin = position['Qmin'].to_dict()   # Limit on ramping for 1 hour

    if 'ramp3h' in position:
        eqModel.ramp3h = position['ramp3h'].to_dict()

    if 'ramp4h' in position:
        eqModel.ramp3h = position['ramp4h'].to_dict()

    if 'ramp1h' in position:
        eqModel.ramp1h = position['ramp1h'].to_dict()
    
    if 'delay' in position:
        eqModel.delay = position['delay']

    if 'prod_spill_mul' in position:
        eqModel.prod_spill_mul = position['prod_spill_mul'].to_dict()
        
    if 'Smin' in position:
        eqModel.Smin = position['Smin'].to_dict()
    else:
        eqModel.Smin.index = position['Mmax'].index
        eqModel.Smin = eqModel.Smin['Smin'].to_dict()
    
    if 'inflow_multiplier' in position:
        eqModel.inflow = eqModel.inflow * position['inflow_multiplier'].to_dict()['res_0']
        eqModel.inflow = eqModel.inflow.to_dict()

    if 'alpha' in position:
        eqModel.alpha = position['alpha'].to_frame()

    if 'rolling_prod_hours' in position:
        eqModel.rolling_prod_hours = position['rolling_prod_hours'].to_dict()['res_0']
    if 'Cmax' in position:
        eqModel.Cmax = position['Cmax'].to_dict()
    elif 'Cmax_0' in position:
        eqModel.Cmax = position[(f'Cmax_{seg}' for seg in range(eqModel.segments_pump))].to_dict()
        for i in range(eqModel.nbr_stations):
            for k in range(eqModel.segments_pump):
                if f'mu_pump_{k}' in position.columns:
                    eqModel.mu_pump[f'mu_pump_{k}'][f'res_{i}'] = position.loc[f'res_{i}', f'mu_pump_{k}']

    if 'Pmax' in position:
        eqModel.Pmax = position['Pmax'].to_dict()
    elif 'Pmax_0' in position:
        eqModel.Pmax = position[(f'Pmax_{seg}' for seg in range(eqModel.segments))].to_dict()
        for i in range(eqModel.nbr_stations):
            for k in range(eqModel.segments):
                if f'mu_{k}' in position.columns:
                    eqModel.mu[f'mu_{k}'][f'res_{i}'] = position.loc[f'res_{i}', f'mu_{k}']

    if 'pump_loss' in position:
        eqModel.pump_loss = position['pump_loss'].to_dict()
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
            diff = ((result_eqModel.power - scenario_data.real_production)**2).sum().sum()
            diff = diff + ((result_eqModel.consumption - scenario_data.pump_consumption)**2).sum().sum()

    else:
        diff = np.inf
    
    return diff
