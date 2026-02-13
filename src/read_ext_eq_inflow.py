import pandas as pd
import numpy as np
import logging
from get_period_data import *

logger = logging.getLogger(__name__)

def create_mu_eq_model(mu_opt,conf):
    mu_0_df  = pd.DataFrame(
        mu_opt.values(),columns=['mu_0'],index=['res_' +str(res_int) for res_int in range(conf['para_general']['stations'])])
    
    if conf['para_general']['segments'] >1:
        for res_int in range(conf['para_general']['stations']):
            for seg in range(conf['para_general']['segments']):
                if seg == 0:
                    mu_0_df.loc[f'res_{res_int}', f'mu_{seg}'] = conf['para_PSO']['mu_percent'][seg] * mu_opt[res_int]
                else:
                    mu_0_df.loc[f'res_{res_int}', f'mu_{seg}'] = conf['para_PSO']['mu_percent'][seg] * mu_opt[res_int]
    
    return mu_0_df

def create_hourly_data(inflow):

    last_day = inflow.index[-1] + pd.DateOffset(days=1)
    last_day_df = pd.DataFrame({'inflow':inflow.iloc[-1,0]}, index=[last_day])
    inflow = pd.concat([inflow,last_day_df], axis=0)
    inflow = inflow.resample('h').ffill()
    inflow = inflow.drop(inflow.index[-1])
    inflow = inflow /24 #  Divide by 24 to convert from daily total value to hourly
    
    return inflow


def create_inflow_scenarios(inflow, conf, scenario_data, dwnAd):
    cluster_data = read_cluster_data(conf,scenario_data)
    cluster_data.periods = sum(cluster_data.categories[i] == cluster_data.category for i in range(
        len(cluster_data.categories)))
    cluster_data.periods_ids, cluster_data.hours_ids = create_periods_ids(cluster_data)
    inflow.reset_index(inplace=True, drop=True)
    inflow_periods = peridos_hourly_data(inflow, cluster_data)
    tot_scen = scenario_data.scenarios
    eqStations = np.size(dwnAd,0) # size(dwnAd,1)
    hours = len(scenario_data.price)
    scenario_names = [f"scen_{i}_res_{res}" for i in range(tot_scen) for res in range(eqStations)]
    inflow_periods.columns = scenario_names

    return inflow_periods

def get_inflow_and_mu(conf:dict,avg_mu:pd.DataFrame,scenario_data:ScenarioData,dwnAd:np.ndarray):

    inflow = pd.read_csv(conf['files']['ext_inflow_file'], sep=';')
    if len(inflow.columns) <= 1:
        # If the file is empty, we change the delimiter and try again
        inflow = pd.read_csv(conf['files']['ext_inflow_file'], sep=',')
    if not 'time' in inflow.columns:
        inflow.rename(mapper={'valid_time': 'time'}, inplace=True, axis=1)
    inflow.set_index('time', inplace=True)
    inflow.columns = ['inflow']

    # recreate mu to get the same shape as in the calcution mode
    avg_mu = {0:avg_mu.values[0][0]}

    mu = create_mu_eq_model(avg_mu,conf)

    # Calcaulte water inflow

    inflow = inflow / mu.loc['res_0','mu_0']

    if len(inflow) < 367:
        inflow = create_hourly_data(inflow)

    # create scenario data
    inflow = create_inflow_scenarios(inflow, conf, scenario_data,dwnAd)


    return inflow, mu