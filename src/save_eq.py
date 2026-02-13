import pandas as pd
from naive_aggregation import EqModel
import numpy as np

def create_output_framework(conf):
    basic_output_varaibels = np.array(conf['para_PSO']['variables'])
    basic_output_varaibels = basic_output_varaibels[basic_output_varaibels!='Mmax']
    basic_output_varaibels = basic_output_varaibels[basic_output_varaibels!='Mmin']
    basic_output_varaibels = basic_output_varaibels[basic_output_varaibels!='Qmax']
    basic_output_varaibels = basic_output_varaibels[basic_output_varaibels!='Qmin']
    basic_output_varaibels = basic_output_varaibels[basic_output_varaibels!='mu_add']
    basic_dict = {}
    for varaible in basic_output_varaibels:
        basic_dict.update({varaible:[]})

    basic_df = pd.DataFrame(basic_dict) 

    return basic_df

def save_eq(eqModel:EqModel,conf):
    res = eqModel.nbr_stations
    seg = eqModel.segments
    # Make all read things to put into basics 

    basic_df = create_output_framework(conf=conf)

    #%%
    if 'pump' in conf['para_PSO']['first_level_objectiv']:
        for i in range(res):
            basic_df.loc['res_0','pump_loss'] = eqModel.pump_loss['res_0']
            basic_df = pd.concat([basic_df,pd.DataFrame(eqModel.Cmax).loc[f'res_{i}',:].to_frame().T],axis=1)
            basic_df = pd.concat([basic_df,pd.DataFrame(eqModel.Pmax).loc[f'res_{i}',:].to_frame().T],axis=1)
            #basic_df.loc['res_0','ramp1h'] = eqModel.ramp1h['res_0']
            #basic_df.loc['res_0','ramp3h'] = eqModel.ramp3h['res_0'] 
        if eqModel.segments_pump > 1:
            basic_df = pd.concat([basic_df,pd.DataFrame(eqModel.mu_pump)],axis=1)
        if eqModel.segments > 1:
            basic_df = pd.concat([basic_df,pd.DataFrame(eqModel.mu)],axis=1)
        basic_df.dropna(inplace=True,axis=1)     
        Mmax_df = pd.DataFrame(eqModel.Mmax,index = ['Mmax']).T
        eq_df = pd.concat([basic_df,Mmax_df],axis=1)
        inflow_df = None
    else:
        mu_df = pd.DataFrame(eqModel.mu)
        mu_df.index=[f'res_{i}' for i in range(res)]
        mu_df.columns = [f"mu_{k}" for k in range(seg)]
        Mmax_df = pd.DataFrame(eqModel.Mmax,index = ['Mmax']).T
        Mmax_energy_df = (Mmax_df*mu_df.loc[:,'mu_0'].to_frame().values / 1000).rename(columns={'Mmax':'Mmax_GWh'})
        # Mmax_df.columns = [f"Mmax_{i}" for i in range(res)]
        Mmin_df = pd.DataFrame(eqModel.Mmin,index = ['Mmin']).T
        Mmin_energy_df = (Mmin_df*mu_df.loc[:,'mu_0'].to_frame().values / 1000).rename(columns={'Mmin':'Mmin_GWh'})
        #Mmin_df.columns = [f"Mmin_{i}" for i in range(res)]   

        Qmax_df = pd.DataFrame(eqModel.Qmax)
        #Qmax_df.columns = [f"Qmax_{k}" for k in range(seg)]
        Qmin_df = pd.DataFrame(eqModel.Qmin,index = ['Qmin']).T
        #Qmin_df.columns = [f"Qmin_{i}" for i in range(res)]
        Smin_df = pd.DataFrame(eqModel.Smin,index = ['Smin']).T
        Smin_df.columns = ["Smin"]
        Smin_energy_df = (Smin_df*mu_df.loc[:,'mu_0'].to_frame().values / 1000).rename(columns={"Smin":'Smin_GW'})
        # Smin_df.columns = [f"Smin_{i}" for i in range(res)]

        Pmax_df = (Qmax_df*mu_df.values) / 1000 # Power in GW
        Pmax_df.columns=[f"P_max_{k}" for k in range(seg)]

        basic_df.loc['res_0','segments'] = seg
        basic_df.loc['res_0','alpha'] = 1 #eqModel.alpha['res_0']
        if isinstance(eqModel.ramp1h, dict):
            basic_df.loc['res_0','ramp1h'] = eqModel.ramp1h['res_0']
            basic_df.loc['res_0','ramp3h'] = eqModel.ramp3h['res_0']
            # basic_df.loc['res_0','ramp1h_GW'] = eqModel.ramp1h['res_0'] * mu_df.loc[:,'mu_0'].values[0] / 1000
            # basic_df.loc['res_0','ramp3h_GW'] = eqModel.ramp3h['res_0'] * mu_df.loc[:,'mu_0'].values[0] / 1000
        if 'delay' in conf['para_PSO']['variables']:
            for i in range(res):
                if i < res-1:
                    basic_df.loc[f'res_{i}','delay'] = eqModel.delay[f'res_{i}']

        inflow_df = pd.DataFrame(eqModel.inflow)

        if isinstance(eqModel.inflow_multiplier, pd.Series): 
            basic_df.loc['res_0','inflow_multiplier'] = eqModel.inflow_multiplier.iloc[0]        
        elif eqModel.inflow_multiplier and not isinstance(eqModel.inflow_multiplier, int):
            basic_df.loc['res_0','inflow_multiplier'] = eqModel.inflow_multiplier['res_0']

        eq_df = pd.concat(
            [basic_df,mu_df,
            Mmax_df,Mmax_energy_df,Mmin_df,Mmin_energy_df,Qmax_df,Qmin_df,Smin_df,Smin_energy_df,Pmax_df],axis=1)
        
    return inflow_df, eq_df

