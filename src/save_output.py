import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import logging
from solve_equivalent_compact import *
from save_eq import save_eq
import yaml

logger = logging.getLogger(__name__)

def save_all_particles(output_folder, filename, stored_particles,global_best,config):
    temp_particle_position= [
        pd.DataFrame() for n_pop in range(config['para_PSO']['population']*config['para_general']['stations'])]
    temp_particle_velocity= [
        pd.DataFrame() for n_pop in range(config['para_PSO']['population']*config['para_general']['stations'])]
    temp_particle_best_postion= [
        pd.DataFrame() for n_pop in range(config['para_PSO']['population']*config['para_general']['stations'])]
    excel_file_position = output_folder + filename + "all_particels_position.xlsx"
    excel_file_velocity = output_folder + filename + "all_particels_velocity.xlsx"
    excel_file_best_postion = output_folder + filename + "all_particels_best_position.xlsx"
    index_names = stored_particles[0].position.index
    for scen in stored_particles:
        scen.position.reset_index(inplace=True,drop=True)
        for index, row in scen.position.iterrows():                
            temp_particle_position[index] = pd.concat([temp_particle_position[index],row],axis=1)

        scen.velocity.reset_index(inplace=True,drop=True)
        for index, row in scen.velocity.iterrows():
            temp_particle_velocity[index] = pd.concat([temp_particle_velocity[index],row],axis=1)

        scen.best_position.reset_index(inplace=True,drop=True)
        for index, row in scen.best_position.iterrows():
            temp_particle_best_postion[index] = pd.concat([temp_particle_best_postion[index],row],axis=1)

    with pd.ExcelWriter(excel_file_position) as writer:
        for temp_index,df in enumerate(temp_particle_position):
            sheet_name = index_names[temp_index][1] + "_p_" + str(index_names[temp_index][0])
            df.to_excel(writer,sheet_name=sheet_name)

    with pd.ExcelWriter(excel_file_velocity) as writer:
        for temp_index,df in enumerate(temp_particle_velocity):
            sheet_name = index_names[temp_index][1] + "_p_" + str(index_names[temp_index][0])
            df.to_excel(writer,sheet_name=sheet_name)

    with pd.ExcelWriter(excel_file_best_postion) as writer:
        for temp_index,df in enumerate(temp_particle_best_postion):
            sheet_name = index_names[temp_index][1] + "_p_" + str(index_names[temp_index][0])
            df.to_excel(writer,sheet_name=sheet_name)

    global_best_df = pd.DataFrame()
    for iteration in global_best:
        global_best_df = pd.concat([global_best_df,iteration['position']],axis=0)

    global_best_df.reset_index(drop=True).to_excel(output_folder + filename + "global_best_iteration.xlsx")

def save_config_file(config):
    filename = config['output_filename_suffix']
    output_folder = config['file_location']['output']
    # Save all config in one file:
    config_file_csv = output_folder+ filename + "config.yaml"

    def custom_dump(data, stream=None, indent=2):
        def represent_dict(dumper, data):
            return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

        def represent_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        def represent_scalar(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.Dumper.add_representer(dict, represent_dict)
        yaml.Dumper.add_representer(list, represent_list)
        yaml.Dumper.add_representer(str, represent_scalar)

        if stream is None:
            return yaml.dump(data, default_flow_style=False, indent=indent,width=float("inf"))
        else:
            yaml.dump(data, stream, default_flow_style=False, indent=indent,width=float("inf"))

    # Write the dictionary to a YAML file
    with open(config_file_csv, 'w') as yaml_file:
        custom_dump(config, yaml_file)


def save_output(config,train_scenario,eq_updated, iteration_parameters,problem, stored_particles = None, stored_global_best = None):
    # Label for PSO output
    filename = config['output_filename_suffix']
    output_folder = config['file_location']['output']

    # ------------------------------------------------------------------------------
    # Save Equivalent and updated inflow in csv-files:
    inflow_df, eq_df = save_eq(
        eqModel=eq_updated,
        conf = config)
    # Save the rest
    eq_df.to_csv(output_folder + filename + "eqModelPSO.csv", sep=';')
    iteration_parameters = pd.DataFrame.from_dict(iteration_parameters).T
    iteration_parameters.to_csv(output_folder + filename + "iteration_parameters.csv", sep=';')

    output_file_data = pd.DataFrame()
    eqModel_trainResults = solve_equivalent(train_scenario, eq_updated, problem, end_simulation=1)
    
    save_config_file(config)
    # Remove initial equivalent power data which is equal to -1
    eqModel_trainResults.power[ eqModel_trainResults.power == -1] = np.nan 
    # Save equivalent model data
    if config['para_general']['stations']>1:
        eq_flow = eqModel_trainResults.q_flow.add_prefix('flow')
        output_file_data = pd.concat([output_file_data, eq_flow])
    eqPower = pd.DataFrame(eqModel_trainResults.power).rename(
        columns={f"scen{k}" :f"power_scen{k}" for k in range(1,eq_updated.scenarios+1)})
    eqContent = pd.DataFrame(eqModel_trainResults.content).add_prefix('content_scen')
    eq_spill = pd.DataFrame(eqModel_trainResults.spill).add_prefix('spill')
    
    inflow_df = inflow_df.add_prefix('inflow_')
    output_file_data = pd.concat([output_file_data, eqPower, eqContent, eq_spill, inflow_df],axis=1)
    output_file_data.to_csv(output_folder + filename + "data.csv", sep=";", index=False)

    create_plot(train_scenario,config,filename, eqModel_trainResults)
    
    if stored_particles:
        save_all_particles(output_folder,filename, stored_particles,stored_global_best,config)

def save_output_pump(config,train_scenario,eq_updated, iteration_parameters,problem, stored_particles = None, stored_global_best = None):
    filename = config['output_filename_suffix']
    output_folder = config['file_location']['output']    
    _, eq_df = save_eq(
        eqModel=eq_updated,
        conf = config)
    
    eq_df.to_csv(output_folder + filename + "eqModelPSO.csv", sep=';')
    iteration_parameters = pd.DataFrame.from_dict(iteration_parameters).T
    iteration_parameters.to_csv(output_folder + filename + "iteration_parameters.csv", sep=';')

    output_file_data = pd.DataFrame()
    eqModel_trainResults = solve_equivalent_pump(train_scenario, eq_updated, problem, end_simulation=1)

    save_config_file(config)

    eqPower = pd.DataFrame(eqModel_trainResults.power).rename(
        columns={f"scen{k}" :f"power_scen{k}" for k in range(1,eq_updated.scenarios+1)})
    eqContent = pd.DataFrame(eqModel_trainResults.content).add_prefix('content_scen')
    eqConsumption = pd.DataFrame(eqModel_trainResults.consumption).rename(
        columns={f"scen{k}" :f"consumption_scen{k}" for k in range(1,eq_updated.scenarios+1)})
    output_file_data = pd.concat([output_file_data, eqPower, eqContent, eqConsumption],axis=1)
    output_file_data.to_csv(output_folder + filename + "data.csv", sep=";", index=False)
    create_plot(train_scenario,config,filename, eqModel_trainResults)
    if train_scenario.pump_consumption.sum().sum() > 0:
        create_consumption_plot(train_scenario,config,filename, eqModel_trainResults)

def create_consumption_plot(train_scenario,config,filename, eqModel_trainResults):
    tim = train_scenario.hours
    t = np.arange(sum(train_scenario.hours))
    original_power = pd.DataFrame()
    eqModel_power = pd.DataFrame()
    plt.figure(figsize=(15, 6))

    for scen in range(1,train_scenario.scenarios+1):
        original_power = pd.concat([original_power, train_scenario.pump_consumption['scen'+str(scen)].dropna().rename(0)]).reset_index(drop=True)
        eqModel_power = pd.concat([eqModel_power, eqModel_trainResults.consumption['scen'+str(scen)].dropna().rename(0)]).reset_index(drop=True)
        if scen < train_scenario.scenarios:
            plt.axvline(x=train_scenario.hours[scen-1])

def create_plot(train_scenario,config,filename, eqModel_trainResults, original_system=None):
    tim = train_scenario.hours
    t = np.arange(sum(train_scenario.hours))
    # Plot it
    original_power = pd.DataFrame()
    eqModel_power = pd.DataFrame()
    plt.figure(figsize=(15, 6))

    for scen in range(1,train_scenario.scenarios+1):
        original_power = pd.concat([original_power, train_scenario.real_production['scen'+str(scen)].dropna().rename(0)]).reset_index(drop=True)
        eqModel_power = pd.concat([eqModel_power, eqModel_trainResults.power['scen'+str(scen)].dropna().rename(0)]).reset_index(drop=True)
        if scen < train_scenario.scenarios:
            plt.axvline(x=train_scenario.hours[scen-1])
    
    total_power_orig = round(original_power.sum().sum() / 1000 /1000,2)
    total_power_eq = round(eqModel_power.sum().sum() / 1000 /1000,2)
    avg_error_pso = round(np.nansum(
        np.abs(np.array(original_power) - np.array(eqModel_power))) / (
            np.round(np.sum(train_scenario.hours)).astype(int)),2)
    # original_power = original_system.power['scen1']
    # eqModel_power = eqModel_trainResults.power['scen1']
    # Plot
    
    plt.plot(t, original_power,
             label=f"Detailed Power orig:{total_power_orig}",
             linestyle='-', color="black", linewidth=0.5)
    plt.plot(
        t, eqModel_power,
        label=f"PSO %:{avg_error_pso} tot_P:{total_power_eq}", linestyle='--', color="blue", linewidth=0.5)
    
    # Plot settings
    plt.title(
        f"Power generation , j={config['para_general']['segments']}, scenarios = {train_scenario.hours}")
    
    plt.xlabel("Hour")
    plt.ylabel("Power [MWh/h]")
    plt.xlim(0, sum(train_scenario.hours))
    plt.ylim(0, (np.ceil(original_power.max().max()/500)) * 500+100)
    plt.xticks(np.arange(0, sum(train_scenario.hours), round(sum(train_scenario.hours)/10)))
    plt.yticks(np.arange(0, (np.ceil(original_power.max().max()/500)+1) * 500, 500))
    plt.yticks(np.arange(0, np.round(np.max(eqModel_power) * 1.3), np.round(np.max(eqModel_power) * 1.3 / 10)))

    plt.legend(loc='lower center')

    # Show the plot

    plt.savefig(config['file_location']['output'] + filename + 'power_comparission.pdf')
    # plt.show()
    # Calculate mean hourly power production difference
    avg_error = np.sum(
        np.abs(np.array(original_power) - np.array(eqModel_power))) / (
            np.round(np.sum(train_scenario.hours)).astype(int))
    logger.info("(SE2) The average error for the power output is {:.2f}% MWh/h after PSO".format(avg_error / 7653.9 * 100))
    
    # Power duration curve
    f_duration = plt.figure(figsize=(15, 6))
    p_duration = eqModel_power.sort_values(by=0,ascending=False).reset_index(drop=True)
    plt.plot(p_duration)

    f_duration.savefig(config['file_location']['output'] + filename + 'power_duration_curve.pdf')

