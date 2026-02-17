import os
import pandas as pd
from dataclasses import dataclass
import logging
import numpy as np
from pathlib import Path
logger = logging.getLogger(__name__)

@dataclass
class ScenarioData:
    days : int = None
    dwnCapacity : int = None            # Required down-regulation capacity
    hours : np.ndarray = None                  # Number of hours in each scenario
    initialEnergy : float = None        # Initial energy in the system
    M0share : pd.DataFrame = None       # Relative energy content at the start (row 1) and end (row 2) of the planning period
    MTminShare : float = None           # Min relative energy content at the end of the planning period
    price : pd.Series = None            # Price matrix
    scenarios : int = None              # Total number of scenarios
    upCapacity : int = None             # Required up-regulation capacity
    max_content_energy: float = None    # Max reservoar capacity total, if available
    max_historical_production: float = None # Max historical production, if available
    min_historical_production: float = None # Min historical production, if available
    pump_consumption: pd.DataFrame = None # Pump consumption data, if available
    real_production: pd.DataFrame = None # Real production data, if available

def read_entsoe_data(file_path: str):

    API_KEY = os.getenv("ENTSOE_API_KEY")
    SE1_EIC = "10Y1001A1001A44P"   # SE1 bidding zone





def read_scenario_data(conf:dict) -> ScenarioData:
    """ Read the data that is needed for the scenarios and create the ScenarioData dataclass

    Args:
        conf (dict): The configuration dictionary

    Returns:
        ScenarioData: The scenario dataclass with all the data for the scenarios
    """
    scenario_data = ScenarioData()
    scenario_data.price = pd.read_csv(conf['files']['train_price_file'],delimiter=';',index_col=[0]).reset_index(drop=True)
    scenario_data.days = pd.read_csv(conf['files']['day_file'],delimiter=';')

    if len(scenario_data.days)*24 < len(scenario_data.price):
        scenario_data.price = scenario_data.price.groupby(scenario_data.price.index//4).mean()
        logger.warning(f"Number of hours in days file is less than number of price rows. Adjusting price data.")
        
    hours, scen = scenario_data.price.shape
    scenario_data.hours = hours # hours in each scenario
    scenario_data.scenarios = scen # nbr scenarios
    if Path(conf['files']['M_file']).is_file():
        M_temp = pd.read_csv(conf['files']['M_file'],delimiter=';')
        scenario_data.M0share = M_temp.iloc[:-1, 1:]
        scenario_data.MTminShare = M_temp.iloc[-1,1:] #§ only one value for all
    else:
        scenario_data.M0share = pd.DataFrame(np.full((1, scenario_data.scenarios), 0.5), columns=[f'scen_{i}' for i in range(scenario_data.scenarios)])
        scenario_data.MTminShare = 0.1
        logger.warning(f"M file not found. Setting M0share to 0.5 and MTminShare to 0.1 for all scenarios.")

    if 'real_production' in conf['files']:
        real_production = pd.read_csv(conf['files']['real_production'],delimiter=';',index_col=[0]).reset_index(drop=True)
        if real_production.empty:
            # If the file is empty, we change the delimiter and try again
            real_production = pd.read_csv(
                conf['files']['real_production'],delimiter=',',index_col=[0]).reset_index(drop=True)
        if len(real_production)>len(scenario_data.price):
            real_production = real_production.groupby(np.arange(len(real_production)) // 4).mean().reset_index(drop=True)
        if len(real_production)!=len(scenario_data.price):
            raise KeyError(f'Real production does not match the price matrix. Real production has {len(real_production)} and prices has {len(scenario_data.price)}')        
        if real_production.isna().any().any():
            logger.warning(f"Real production has missing values. Filling missing values with linear interpolation.")
            real_production = real_production.interpolate(method='linear', limit_direction='forward', axis=0)
        scenario_data.real_production = real_production

    
    if 'pump_data' in conf['files']:
        pump_data = pd.read_csv(conf['files']['pump_data'],delimiter=';',index_col=[0]).reset_index(drop=True)
        pump_consumption = pump_data.loc[:, 'Actual Consumption']
        real_production = pump_data.loc[:, 'Actual Aggregated']
        if pump_consumption.isna().all():
            pump_consumption = pump_consumption.fillna(0)
            logger.warning(f"Pump consumption file is empty. Setting pump consumption to None.")
        if real_production.isna().all():
            real_production = real_production.fillna(0)
            logger.warning(f"Real production file is empty. Setting real production to None.")

        if len(pump_consumption)>=len(scenario_data.price)*4-1:
            pump_consumption = pump_consumption.groupby(np.arange(len(pump_consumption)) // 4).mean().reset_index(drop=True) 
            real_production = real_production.groupby(np.arange(len(real_production)) // 4).mean().reset_index(drop=True)
        elif len(pump_consumption)>=len(scenario_data.price)*2-1:
            pump_consumption = pump_consumption.groupby(np.arange(len(pump_consumption)) // 2).mean().reset_index(drop=True) 
            real_production = real_production.groupby(np.arange(len(real_production)) // 2).mean().reset_index(drop=True)
        scenario_data.pump_consumption = pump_consumption
        scenario_data.real_production = real_production

        if len(scenario_data.pump_consumption)!=len(scenario_data.price):
            raise KeyError(f'Pump consumption does not match the price matrix. Pump consumption has {len(scenario_data.pump_consumption)} and prices has {len(scenario_data.price)}')
        
    # If we have a file with the max content, we read it
    if Path(conf['files']['basic_files']).is_file():
        initial_parameter = pd.read_csv(conf['files']['basic_files'],index_col=[0],delimiter=';')
        scenario_data.max_content_energy = initial_parameter.loc[conf['area'],'max_M(MWh)']
        scenario_data.initialEnergy = scenario_data.max_content_energy * scenario_data.M0share.iloc[0,0]
        scenario_data.max_historical_production = initial_parameter.loc[conf['area'],'max_P(MWh)']
        scenario_data.min_historical_production = initial_parameter.loc[conf['area'],'min_P(MWh)']
        logger.info(f"Initial Parameters read from file for area:{conf['area']}")

    else:
        scenario_data.max_content_energy = 10000000000 #initlize value 
        scenario_data.initialEnergy = None
        scenario_data.max_historical_production = real_production.max().max()
        scenario_data.min_historical_production = real_production.min().min()
        logger.warning(f"Max content set to max :{scenario_data.max_content_energy}")       

    return scenario_data