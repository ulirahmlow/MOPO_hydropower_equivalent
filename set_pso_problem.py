import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from set_config import PSOparameters
from naive_aggregation import EqModel
from read_scenario_data import ScenarioData
logger = logging.getLogger(__name__)

@dataclass
class PSOproblem():
    dim : int  # dimension of search space
    varMin : pd.DataFrame  # minimum variable limits
    varMax : pd.DataFrame  # maximum variable limits
    parameter : dict # PSO specific parameters
    gurobi_variables: dict = None
    nominal_positions: bool = False


def calculate_initial_content(eq_init, scenario:ScenarioData):
    res = eq_init.nbr_stations
    days, scen = scenario.days.shape
    M0 = np.zeros((res, scen)) # initial content in reservoirs
    M_max = np.zeros(res) # initial content in reservoirs
    for i in range(res):
        M_max[i] = (
            scenario.max_content_energy * eq_init.alpha.iloc[i,0]) / (
                sum(eq_init.dwnAd.iloc[i, down] * eq_init.mu.iloc[down, 0] for down in range(res)) ) # 0.9993*
        if isinstance(scenario.initialEnergy,int) or isinstance(scenario.initialEnergy,np.ndarray):
            for w in range(scen):
                M0[i,w] = (
                scenario.initialEnergy[w] * eq_init.alpha.iloc[i,0]) / (
                        sum(eq_init.dwnAd.iloc[i, down] * eq_init.mu.iloc[down, 0] for down in range(res)) ) # 0.9993*

    # TODO: WEIRD CALCULATION OF M0. Why is the second reservoir more content even if this should be much lower...
    return M0, M_max


def initialze_one_reservoir(var_min, var_max, initial_res_content, max_res_content, eq_init:EqModel, scenario, params_pso):
    var_lim_min = params_pso.var_lim_min
    var_lim_max = params_pso.var_lim_max
    var_min.loc[0,'Mmax'] = np.max(max_res_content) * var_lim_min['Mmax']
    var_min.loc[0,'Mmin'] = np.max(max_res_content) * var_lim_min['Mmin']
    var_min.loc[0,'Qmin'] = var_lim_min['Qmin']
    #var_min.loc[0,'ramp1h'] = np.max(eq_init.Qmax) * var_lim_min['ramp1h']
    #var_min.loc[0,'ramp3h'] = np.max(eq_init.Qmax) * var_lim_min['ramp3h']
    #var_min.loc[0,'Smin'] = 0

    # var_max.loc[0,'Mmax'] = (np.max(initial_res_content) * np.max([1, np.max(scenario.MTminShare)])) * var_lim_max['Mmax']
    #if var_max.loc[0,'Mmax']> max_res_content[0]:
    var_max.loc[0,'Mmax'] = np.max(max_res_content) * var_lim_max['Mmax']

    var_max.loc[0,'Mmin'] = np.max(max_res_content) * var_lim_max['Mmin']
    var_max.loc[0,'Qmin'] =  scenario.min_historical_production * var_lim_max['Qmin']
    #var_max.loc[0,'ramp1h'] =  np.max(eq_init.Qmax) * var_lim_max['ramp1h']
    #var_max.loc[0,'ramp3h'] = np.max(eq_init.Qmax) * var_lim_max['ramp3h']

    if eq_init.Q_break:
        var_max.loc[0,'Qmax'] = scenario.max_historical_production *var_lim_max['Qmax'] 
        var_min.loc[0,'Qmax'] = scenario.max_historical_production * var_lim_min['Qmax'] 
    else: 
        for seg in range(eq_init.segments):
            var_max.loc[0,f'Qmax_{seg}'] = scenario.max_historical_production * var_lim_max['Qmax'] / (seg+1)
            var_min.loc[0,f'Qmax_{seg}'] = scenario.max_historical_production * var_lim_min['Qmax'] / (seg+1)

        var_max = var_max.drop('Qmax',axis=1)
        var_min = var_min.drop('Qmax',axis=1)
    
    if 'ramp1h' in var_lim_min:
        var_min.loc[0,'ramp1h'] = scenario.max_historical_production * var_lim_min['ramp1h']
        var_max.loc[0,'ramp1h'] = scenario.max_historical_production * var_lim_max['ramp1h']
        
    if 'ramp3h' in var_lim_min:
        var_min.loc[0,'ramp3h'] = scenario.max_historical_production * var_lim_min['ramp3h']
        var_max.loc[0,'ramp3h'] = scenario.max_historical_production * var_lim_max['ramp3h']

    change_var = ['Smin', 'inflow_multiplier']

    for var in change_var:
        if var in var_lim_min:
            var_min.loc[0,var] = var_lim_min[var] 
            var_max.loc[0,var] = var_lim_max[var]

    
    return var_min, var_max


def initialze_two_reservoirs(var_min, var_max, initial_res_content,max_res_content, eq_init:EqModel, scenario,params_pso):
    var_lim_min = params_pso.var_lim_min
    var_lim_max = params_pso.var_lim_max
    # First Reservoir
    var_min.loc[0,'Mmax'] = np.max(initial_res_content) * var_lim_min['Mmax']
    var_min.loc[0,'Mmin'] = np.min(initial_res_content) * np.min([1, np.min(scenario.MTminShare)]) * var_lim_min['Mmin']
    var_min.loc[0,'Qmin'] =  np.max(eq_init.Qmax) * var_lim_min['Qmin']

    #var_min.loc[0,'Smin'] = 0

    var_max.loc[0,'Mmax'] = (np.max(initial_res_content) * np.max([1, np.max(scenario.MTminShare)])) * var_lim_max['Mmax']
    var_max.loc[0,'Mmin'] = np.min(initial_res_content) * np.min(np.array(scenario.MTminShare)) * var_lim_max['Mmin']
    var_max.loc[0,'Qmin'] =  np.max(eq_init.Qmax ) * var_lim_max['Qmin']

    #var_max.loc[0,'Smin'] = np.max(eq_init.Smin)*2
    
    # Second Reservoir
    var_min.loc[1,'Mmax'] = np.max(initial_res_content) * var_lim_min['Mmax']
    var_min.loc[1,'Mmin'] = np.min(initial_res_content) * np.min([1, np.min(scenario.MTminShare)]) * var_lim_min['Mmin']
    
    var_min.loc[1,'Qmin'] =  np.max(eq_init.Qmax) * var_lim_min['Qmin']

    #var_min.loc[0,'Smin'] = 0

    var_max.loc[1,'Mmax'] = (np.max(initial_res_content) * np.max([1, np.max(scenario.MTminShare)])) * var_lim_max['Mmax']
    var_max.loc[1,'Mmin'] = np.min(initial_res_content) * np.min(np.array(scenario.MTminShare)) * var_lim_max['Mmin']
    var_max.loc[1,'Qmin'] =  np.max(eq_init.Qmax) * var_lim_max['Qmin']
    
    if var_max.loc[0,'Mmax']> max_res_content[0]:
        var_max.loc[0,'Mmax'] = max_res_content[0]
    if var_max.loc[1,'Mmax']> max_res_content[1]:
        var_max.loc[1,'Mmax'] = max_res_content[1]
    # Varaibles varaibles that are not always part of the PSO
    change_var = [
        'delay','ramp1h','ramp3h','alpha', 'Smin', 'inflow_multiplier']
    # combination between reservoir one and two:
    for var in change_var:
        if var in var_lim_min:
            var_min.loc[0,var] = var_lim_min[var] * np.max(getattr(eq_init,var))
            var_max.loc[0,var] = var_lim_max[var] * np.max(getattr(eq_init,var)) 
            var_min.loc[1,var] = var_lim_min[var] * np.max(getattr(eq_init,var))
            var_max.loc[1,var] = var_lim_max[var] * np.max(getattr(eq_init,var)) 
            if ('delay' == var) | ('inflow_multiplier' == var) | ('ramp' in var) :
                
                var_min.loc[1,var] = 0
                var_max.loc[1,var] = 0
                if ('delay' == var) :
                    var_min.loc[0,var] = 60
                    
            if 'rolling_prod_hours' == var:
                var_min.loc[0,var] = var_lim_min[var]
                var_max.loc[0,var] = var_lim_max[var]
    # Q_max per segments or predefined:

    if eq_init.Q_break:
        var_max.loc[0,'Qmax'] = np.max(eq_init.Qmax)* var_lim_max['Qmax']
        var_min.loc[0,'Qmax'] = np.max(eq_init.Qmax)* var_lim_min['Qmax']
        var_min.loc[1,'Qmax'] = np.max(eq_init.Qmax)* var_lim_min['Qmax'] 
        var_max.loc[1,'Qmax'] = np.max(eq_init.Qmax) * var_lim_max['Qmax']
    else: 
        for seg in range(eq_init.segments):
            var_min.loc[0,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_min['Qmax']/(seg+1)
            var_min.loc[1,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_min['Qmax']/(seg+1)
            var_max.loc[0,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_max['Qmax']/((1.5*seg)**2+1)
            var_max.loc[1,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_max['Qmax']/((1.5*seg)**2+1)
        var_max = var_max.drop('Qmax',axis=1)
        var_min = var_min.drop('Qmax',axis=1)

    return var_min, var_max

def initialze_three_reservoirs(var_min, var_max, initial_res_content, max_res_content, eq_init:EqModel, scenario,params_pso):
    
    var_lim_min = params_pso.var_lim_min
    var_lim_max = params_pso.var_lim_max

    # First Reservoir
    var_min.loc[0,'Mmax'] = np.max(initial_res_content) * var_lim_min['Mmax']
    var_min.loc[0,'Mmin'] = np.min(initial_res_content) * np.min([1, np.min(scenario.MTminShare)]) * var_lim_min['Mmin']
    var_min.loc[0,'Qmin'] =  np.max(eq_init.Qmax) * var_lim_min['Qmin']
    var_max.loc[0,'Mmax'] = (np.max(initial_res_content) * np.max([1, np.max(scenario.MTminShare)])) * var_lim_max['Mmax']
    var_max.loc[0,'Mmin'] = np.min(initial_res_content) * np.min(np.array(scenario.MTminShare)) * var_lim_max['Mmin']

    # Second Reservoir
    var_min.loc[1,'Mmax'] = np.max(initial_res_content) * var_lim_min['Mmax']
    var_min.loc[1,'Mmin'] = np.min(initial_res_content) * np.min([1, np.min(scenario.MTminShare)]) * var_lim_min['Mmin']  
    var_min.loc[1,'Qmin'] =  np.max(eq_init.Qmax) * var_lim_min['Qmin']
    var_max.loc[1,'Mmax'] = (np.max(initial_res_content) * np.max([1, np.max(scenario.MTminShare)])) * var_lim_max['Mmax']
    var_max.loc[1,'Mmin'] = np.min(initial_res_content) * np.min(np.array(scenario.MTminShare)) * var_lim_max['Mmin']
    
    # Third Reservoir
    var_min.loc[2,'Mmax'] = np.max(initial_res_content) * var_lim_min['Mmax']
    var_min.loc[2,'Mmin'] = np.min(initial_res_content) * np.min([1, np.min(scenario.MTminShare)]) * var_lim_min['Mmin']  
    var_min.loc[2,'Qmin'] =  np.max(eq_init.Qmax) * var_lim_min['Qmin']
    var_max.loc[2,'Mmax'] = (np.max(initial_res_content) * np.max([1, np.max(scenario.MTminShare)])) * var_lim_max['Mmax']
    var_max.loc[2,'Mmin'] = np.min(initial_res_content) * np.min(np.array(scenario.MTminShare)) * var_lim_max['Mmin']

    # Varaibles varaibles that are not always part of the PSO
    change_var = [
        'delay','ramp1h','ramp3h','alpha', 'Smin', 'inflow_multiplier' ]
    
    # combination between reservoir one, two, three:

    for var in change_var:
        if var in var_lim_min:
            var_min.loc[0,var] = var_lim_min[var] * np.max(getattr(eq_init,var))
            var_max.loc[0,var] = var_lim_max[var] * np.max(getattr(eq_init,var)) 
            var_min.loc[1,var] = var_lim_min[var] * np.max(getattr(eq_init,var))
            var_max.loc[1,var] = var_lim_max[var] * np.max(getattr(eq_init,var)) 
            var_min.loc[2,var] = var_lim_min[var] * np.max(getattr(eq_init,var))
            var_max.loc[2,var] = var_lim_max[var] * np.max(getattr(eq_init,var)) 
            if ('delay' == var) | ('inflow_multiplier' == var):
                var_min.loc[1,var] = 0
                var_max.loc[1,var] = 0
                var_min.loc[2,var] = 0
                var_max.loc[2,var] = 0


    # Q_max per segments or predefined:
    if eq_init.Q_break:
        var_max.loc[0,'Qmax'] = np.max(eq_init.Qmax)* var_lim_max['Qmax']
        var_min.loc[0,'Qmax'] = np.max(eq_init.Qmax)* var_lim_min['Qmax']
        var_min.loc[1,'Qmax'] = np.max(eq_init.Qmax)* var_lim_min['Qmax'] 
        var_max.loc[1,'Qmax'] = np.max(eq_init.Qmax) * var_lim_max['Qmax']
        var_min.loc[2,'Qmax'] = np.max(eq_init.Qmax)* var_lim_min['Qmax'] 
        var_max.loc[2,'Qmax'] = np.max(eq_init.Qmax) * var_lim_max['Qmax']
        
    else: 
        for seg in range(eq_init.segments):
            var_min.loc[0,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_min['Qmax']/(seg+1)
            var_min.loc[1,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_min['Qmax']/(seg+1)
            var_min.loc[2,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_min['Qmax']/(seg+1)

            var_max.loc[0,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_max['Qmax']/((1.5*seg)**2+1)
            var_max.loc[1,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_max['Qmax']/((1.5*seg)**2+1)
            var_max.loc[2,f'Qmax_{seg}'] = np.max(eq_init.Qmax)* var_lim_max['Qmax']/((1.5*seg)**2+1)

        var_max = var_max.drop('Qmax',axis=1)
        var_min = var_min.drop('Qmax',axis=1)

    return var_min, var_max

def initialze_additional_parameter(var_min:pd.DataFrame,var_max:pd.DataFrame, eq_init:EqModel,params_pso:PSOparameters):

    var_lim_min = params_pso.var_lim_min
    var_lim_max = params_pso.var_lim_max
    for res in range (eq_init.nbr_stations):
        for seg in range(1,eq_init.segments):
            var_min.loc[res,f'mu_{seg}'] = var_lim_min['mu_add']
            var_max.loc[res,f'mu_{seg}'] = var_lim_max['mu_add']
    if eq_init.segments_pump>1:
        for res in range (eq_init.nbr_stations):
            for seg in range(1,eq_init.segments_pump):
                var_min.loc[res,f'mu_pump_{seg}'] = var_lim_min['mu_add']
                var_max.loc[res,f'mu_pump_{seg}'] = var_lim_max['mu_add']      

    var_max = var_max.drop('mu_add',axis=1)
    var_min = var_min.drop('mu_add',axis=1)
    
    return var_min, var_max

def set_pso_problem(scenario:ScenarioData, eq_init:EqModel, params_pso:PSOparameters):

    initial_res_content, max_res_content = calculate_initial_content(eq_init, scenario)

    var_min =  pd.DataFrame(
        np.zeros((eq_init.nbr_stations, params_pso.num_varaibles)),
        columns=params_pso.varaibles)
    var_max =  pd.DataFrame(
        np.zeros((eq_init.nbr_stations, params_pso.num_varaibles)),
        columns=params_pso.varaibles)

    if eq_init.nbr_stations == 1:
        var_min, var_max = initialze_one_reservoir(var_min, var_max, initial_res_content,max_res_content, eq_init, scenario,params_pso)

    elif eq_init.nbr_stations == 2:
        var_min, var_max = initialze_two_reservoirs(var_min, var_max, initial_res_content,max_res_content, eq_init, scenario, params_pso)

    elif eq_init.nbr_stations == 3: #res ==3
        var_min, var_max = initialze_three_reservoirs(var_min, var_max, initial_res_content,max_res_content, eq_init, scenario, params_pso)

    if "mu_add" in params_pso.var_lim_max.keys():
        var_min, var_max = initialze_additional_parameter(var_min, var_max, eq_init, params_pso)
        
    var_min.index = ['res_' +str(res_int) for res_int in var_min.index]
    var_max.index = var_min.index
    problem = PSOproblem(dim=len(var_max.columns), varMin=var_min, varMax=var_max, parameter=params_pso)
    logger.info("Lower limit:\n%s", var_min.to_string(index=False))
    logger.info("Upper limit:\n%s", var_max.to_string(index=False))

    return problem

def initialze_pump_reservoir(var_min, var_max, eq_init:EqModel, params_pso:PSOparameters):
    var_lim_min = params_pso.var_lim_min
    var_lim_max = params_pso.var_lim_max
    per_res_varaible =['Mmax', 'Mmin', 'Pmax', 'ramp1h', 'ramp3h', 'Cmax','pump_loss']
    for var in per_res_varaible:
        if var in params_pso.varaibles:
            for i_res in range(eq_init.nbr_stations):
                var_min.loc[i_res,var] = var_lim_min[var] 
                var_max.loc[i_res,var] = var_lim_max[var] 

    if eq_init.segments>1:
        for i_res in range(eq_init.nbr_stations):
            for seg in range(eq_init.segments):
                var_min.loc[0,f'Pmax_{seg}'] = var_lim_min['Pmax']/(seg+1)
                var_max.loc[0,f'Pmax_{seg}'] = var_lim_max['Pmax']/((1.5*seg)**2+1)
        
        var_max = var_max.drop('Pmax',axis=1)
        var_min = var_min.drop('Pmax',axis=1)

    if eq_init.segments_pump>1:
        for i_res in range(eq_init.nbr_stations):
            for seg in range(eq_init.segments_pump):
                var_min.loc[0,f'Cmax_{seg}'] = var_lim_min['Cmax']/(seg+1)
                var_max.loc[0,f'Cmax_{seg}'] = var_lim_max['Cmax']/((1.5*seg)**2+1)
        
        var_max = var_max.drop('Cmax',axis=1)
        var_min = var_min.drop('Cmax',axis=1)

    return var_min, var_max

def set_pso_problem_pump(eq_init:EqModel, params_pso:PSOparameters):

    var_min =  pd.DataFrame(
        np.zeros((eq_init.nbr_stations, params_pso.num_varaibles)),
        columns=params_pso.varaibles)
    var_max =  pd.DataFrame(
        np.zeros((eq_init.nbr_stations, params_pso.num_varaibles)),
        columns=params_pso.varaibles)    
    var_min, var_max = initialze_pump_reservoir(var_min, var_max, eq_init,params_pso)
    if "mu_add" in params_pso.var_lim_max.keys():
        var_min, var_max = initialze_additional_parameter(var_min, var_max, eq_init, params_pso)

    var_min.index = ['res_' +str(res_int) for res_int in var_min.index]
    var_max.index = var_min.index
    problem = PSOproblem(dim=len(var_max.columns), varMin=var_min, varMax=var_max, parameter=params_pso)
    logger.info("Lower limit:\n%s", var_min.to_string(index=False))
    logger.info("Upper limit:\n%s", var_max.to_string(index=False))

    return problem