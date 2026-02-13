import numpy as np
import highspy
import logging
import pandas as pd
import time
import itertools
from copy import deepcopy
from naive_aggregation import EqModel
from solve_equivalent_compact import OptResult

logger = logging.getLogger(__name__)

def solve_equivalent_HIGHS(scenario_data, eqModel:EqModel,problem, end_simulation=0)->OptResult:
    # Read data ------------------------------------------
    alpha = eqModel.alpha  # alpha - share of M0 in each res
    hrs = scenario_data.hours.astype(np.int32)  # Set of hours in each scenario
    tim = np.max(hrs)
    scen = scenario_data.scenarios  # Total number of scenarios
    Mmax = eqModel.Mmax  # Max content in reservoirs
    Mmin = eqModel.Mmin  # Min content in reservoirs
    Qmax = eqModel.Qmax  # Max discharge
    Qmin = eqModel.Qmin  # Min discharge
    Smin = eqModel.Smin  # Max spill
    ad = eqModel.dwnAd  # Matrix declaring downstream hydropower plants
    aq = eqModel.upAq  # Matrix declaring upstream hydropower plants
    price = scenario_data.price  # Electricity price per hour
    V = eqModel.inflow  # Inflow per hour
    M0_share = scenario_data.M0share  # Initial content in reservoirs
    mu = eqModel.mu  # Production equivalent
    prob = scenario_data.probability  # Probability for each scenario
    EndRes = scenario_data.MTminShare # Reservoir content at the end of the planning period
    res = eqModel.nbr_stations  # Number of reservoirs in the river
    seg = eqModel.segments  # Number of segments in Qmax and mu
    TiE = scenario_data.initialEnergy  # Initial energy
    ramp1h = eqModel.ramp1h  # Ramping limit for 1h
    ramp3h = eqModel.ramp3h  # Ramping limit for 3h
    ramp1h_time = problem.parameter.ramp_short
    ramp3h_time = problem.parameter.ramp_long
    # res_array = ['res_0','res_1']
    res_array = [f'res_{i_res}' for i_res in range(res)]
    seg_array = [f'{i_seg}' for i_seg in range(seg)]
    P_max = [(sum(mu['mu_'+ seg_array[k]][res_array[i]] * Qmax.get('Qmax_'+str(k)).get('res_'+ str(i), None) for k in range(seg))) for i in range(res)]
    
    delay = False
    if isinstance(eqModel.delay,pd.Series):
        delay= True
        delay_Hour = np.floor(eqModel.delay/60).astype(int).to_dict()  # Flow time in whole hours for the discharge
        delay_Min = (eqModel.delay.astype(float)%60).astype(int).to_dict()   # Flow time in remaining minutes for the discharge    

    M0 = np.zeros((res, scen))  # initial content in reservoirs
    for i in range(res):  # reservoirs/stations
        for w in range(scen):  # scenarios
            M0[i, w] = (TiE[w] * alpha.iloc[i,0]) / (
                sum(ad[i][down]* mu['mu_0'][res_array[down]] for down in range(res))
            ) 

    model_equivalent = highspy.Highs()
    model_equivalent.setOptionValue('log_to_console',False)
    inf = highspy.kHighsInf
    M = {}
    Qexp = {}
    S = {}
    P={}
    Q = {}
    Sflow = {}
    Qflow = {}
    energy_end = {}
    # Ucap = {}
    # Dcap = {}
    for w in range(scen):
        energy_end[w] = model_equivalent.addVariable(lb=0, name=f"energy_end_{w}")  # Energy at the end (dummy variable)

    indices = eqModel.indices
    indices_with_segments = eqModel.indices_with_segments
    M_names = {ind: 'M_' + str(ind[0]) + '_' + str(ind[2]) +'_' + str(ind[1]) for ind in indices}
    Mmax_values = {ind: Mmax.get('res_'+ str(ind[1]), None) for ind in indices}
    Smin_values = {ind: Smin.get('res_'+ str(ind[1]), None) for ind in indices}
    Qmax_values = {ind: Qmax.get('Qmax_'+str(ind[3])).get('res_'+ str(ind[1]), None) for ind in indices_with_segments}
    
    
    Mmin_values = list( Mmin.get('res_'+ str(ind[1]), None) for ind in indices)
    Mmax_values = list(Mmax.get('res_'+ str(ind[1]), None) for ind in indices)
    Smin_values = list(Smin.get('res_'+ str(ind[1]), None) for ind in indices)
    Qmax_values = list(Qmax.get('Qmax_'+str(ind[3])).get('res_'+ str(ind[1]), None) for ind in indices_with_segments)

    M = model_equivalent.addVariables(indices,lb=Mmin_values,ub=Mmax_values,name_prefix='M')
    Q = model_equivalent.addVariables(indices_with_segments,ub=Qmax_values, name_prefix='Q')
    Sflow = model_equivalent.addVariables(indices,lb=0,name_prefix='Sflow')
    Qflow = model_equivalent.addVariables(indices,name_prefix='Qflow')
    P = model_equivalent.addVariables(indices,lb=0,name_prefix='P')
    S = model_equivalent.addVariables(indices,lb=0,ub=Smin_values,name_prefix='S')
    Qexp = model_equivalent.addVariables(indices,name_prefix='Qexp')
    
    z = model_equivalent.addVariable(name="z", obj=-1)  # Profit (objective)
    model_equivalent.addConstr(
        (sum (prob[w] * sum(price['scen'+str(w+1)][t]
            * sum(P[t,i,w] for i in range(res)) for t in range(tim) if hrs[w]>t) for w in range(scen))) == z,
        name="obj")
    for w in range(scen):
        model_equivalent.addConstr(
            energy_end[w] == sum(sum(
                    M[hrs[w]-1, i, w] * ad[i][down] * mu['mu_0'][res_array[down]]
                    for down in range(res) )for i in range(res)) ,name =f"limit_start_res_{w}")
        model_equivalent.addConstr(energy_end[w] >= EndRes.iloc[0,w] * TiE[w],name =f"limit_end_res_{w}")

        for t in range(tim):
            if hrs[w]>t: # When Scenarios are not equal long then only use it when the scenarios is 
                if (t>ramp1h_time-1) & (ramp1h_time != -1):
                    model_equivalent.addConstr(sum(
                            P[t-ramp1h_time,i,w] - P[t,i,w]
                            for i in range(res))  
                            - ramp1h['res_0']
                        <= 0, name =f"Ramping1_0_lim_{t}_{i}_{w}")
                    
                    model_equivalent.addConstr(sum(
                            P[t,i,w] - P[t-ramp1h_time,i,w]
                            for i in range(res)
                        ) - ramp1h['res_0']
                        <= 0, name =f"Ramping1_1_lim_{t}_{i}_{w}")        
                    
                if (t > ramp3h_time-1) & (ramp3h_time != -1):
                    model_equivalent.addConstr(
                    sum(P[t - ramp3h_time,i,w] - P[t,i,w]
                            for i in range(res)
                        ) - ramp3h['res_0']
                        <= 0 , name =f"Ramping3_0_lim_{t}_{i}_{w}")

                    model_equivalent.addConstr(
                        sum(P[t,i,w] - P[t-ramp3h_time,i,w]
                            for i in range(res) 
                        ) - ramp3h['res_0']
                        <= 0, name =f"Ramping3_1_lim_{t}_{i}_{w}") 

                for i in range(res):
                    model_equivalent.addConstr(
                            sum(Q[t, i, w, k] for k in range(seg)) == Qexp[t, i, w], name=f"expo_{t}_{i}_{w}")

                    if (not delay) & (res==1): # (res ==1)
                        model_equivalent.addConstr(
                            0 == M[t, i, w]
                            - (M[t - 1, i, w] if t > 0 else M0[i, w])
                            - V[f'scen_{w}_res_{i}'][t]
                            + Qexp[t, i, w]
                            + S[t, i, w]
                            ,name =f"hydbal_{t}_{i}_{w}")
                        
                    elif (res>1) & (not delay):                        
                        model_equivalent.addConstr(
                            0 == M[t, i, w]
                            - (M[t - 1, i, w] if t > 0 else M0[i, w])
                            - V[f'scen_{w}_res_{i}'][t]
                            + Qexp[t, i, w]
                            + S[t, i, w]
                            - (sum(aq[i][above] * (Qexp[t, above, w] + S[t, above, w]) for above in range(res)))
                            ,name =f"hydbal_{t}_{i}_{w}")
                    else:

                        model_equivalent.addConstr(
                            0 == M[t, i, w]
                            - (M[t - 1, i, w] if t > 0 else M0[i, w])
                            - V[f'scen_{w}_res_{i}'][t]
                            + Qexp[t, i, w]
                            + S[t, i, w]
                            - sum(aq[i][above] * (Qflow[t, above,w] + Sflow[t, above, w]) for above in range(res))
                            ,name =f"hydbal_{t}_{i}_{w}")
                        i_res = res_array[i]
                        model_equivalent.addConstr(
                            Qflow[t, i, w] ==
                                (delay_Min[i_res]/60) * (Qexp[t-(delay_Hour[i_res]+1), i, w] if t-delay_Hour[i_res]-1 >= 0 else 0) +
                                ((60-delay_Min[i_res])/60) * (Qexp[t-delay_Hour[i_res], i,w] if t-delay_Hour[i_res] >= 0 else 0)
                                , name =f"flowdelay_{t}_{i}_{w}")
                        
                        model_equivalent.addConstr(
                            Sflow[t, i, w] ==
                                (delay_Min[i_res]/60) * (S[t-(delay_Hour[i_res]+1), i,w] if t-delay_Hour[i_res]-1 >= 0 else 0) +
                                ((60-delay_Min[i_res])/60) * (S[t-delay_Hour[i_res], i,w] if t-delay_Hour[i_res] >= 0 else 0)
                                , name =f"spilldelay1_{t}_{i}_{w}") 
                        
                    model_equivalent.addConstr(
                            P[t,i,w] ==  sum(mu['mu_'+ seg_array[k]][res_array[i]]*Q[t, i, w, k] for k in range(seg)),name=f"Power_gen_{t}_{i}_{w}")

                    model_equivalent.addConstr(
                        Qmin[res_array[i]] - sum(Q[t, i, w, k] for k in range(seg)) <= 0 ,name =f"Q_limits_{t}_{i}_{w}")
    model_equivalent.changeObjectiveSense(highspy.ObjSense.kMinimize)
    if end_simulation:
        start_time = time.perf_counter()
    model_equivalent.run()
    if end_simulation:
        end_time = time.perf_counter()
        Time_needed_s = end_time - start_time
    #model_equivalent.writeModel(r"C:\Users\rahmlow\OneDrive - KTH\PhD\Adiitional_analysis\Gurobi model\LP_problem\\"  + "Compare_highs.lp")#file_path="C:\\Users\\rahmlow\OneDrive - KTH\\PhD\\Code")
    #model_equivalent.changeColsBounds()
    # count = len(Q.values())
    # Q_index = np.array([var.index for _, var in Q.items()], dtype=np.int32)
    # lower_bound = 0*np.ones(len(Q_index),dtype=np.float64)
    # upper_bound = 1000*np.ones(len(Q_index),dtype=np.float64)
    # model_equivalent.changeColsBounds(count,Q_index,lower_bound,upper_bound)

    if model_equivalent.getModelStatus() == highspy.HighsModelStatus.kOptimal:

        opt_results = OptResult()
        solution = model_equivalent.getSolution()
        
        optimal_values = solution.col_value 
        opt_results.profit = optimal_values[z.index]
        discharg_temp = {var_name: optimal_values[var.index] for var_name, var in Q.items()}
        p_result = np.array([[sum(discharg_temp[t, i_res, w, k] * mu['mu_'+ seg_array[k]][res_array[i_res]] 
                                      if hrs[w]>t else 0 
                                      for k in range(seg)) 
                                      for w in range(scen)
                                      for i_res in range(res)]
                                      for t in range(tim)])
        p_result = pd.DataFrame(p_result,columns=[
            f'scen_{w}_res_{i_res}' for w in range(scen) for i_res in range(res)])
        p_result_per_scen = pd.DataFrame(np.zeros((tim,scen)),columns=[f'scen{w+1}' for w in range(scen)])
        for w in range(scen):
            p_result_per_scen.loc[:,f'scen{w+1}'] = p_result_per_scen.loc[:,f'scen{w+1}'] + sum(
                p_result[f'scen_{w}_res_{i_res}']  for i_res in range(res))
        
        opt_results.power = p_result_per_scen
        # Make more output for the last simulation
        if end_simulation:

            opt_results.run_time = Time_needed_s
            M_t = {var_name: optimal_values[var.index] for var_name, var in M.items()}
            S_temp = {var_name: optimal_values[var.index] for var_name, var in S.items()}
            Q_flow_temp = {var_name: optimal_values[var.index] for var_name, var in Qflow.items()}
            #M_t = M.X
            content = np.zeros([tim, scen])
            S_t = np.zeros((tim, scen))
            p_result_per_seg = {}
            discharg = pd.DataFrame(index=range(tim),columns=[f'{res}_scen_{w}' for res in res_array for w in range(scen)])
            if res >1:
                S_t_per_res = pd.DataFrame(index=range(tim),columns=[f'{res}_scen_{w}' for res in res_array for w in range(scen)])
                Qflow_total = pd.DataFrame(index=range(tim),columns=[f'{res}_scen_{w}' for res in res_array for w in range(scen)])
            for t in range(tim):
                for w in range(scen):
                    if hrs[w]>t: # When Scenarios are not equal long then only use it when the scenarios is 
                        content[t, w] = sum( M_t[t, i, w] * sum( ad[i][down] * mu['mu_0'][res_array[down]]
                                        for down in range(res)) for i in range(res))
                        # Total spill in all stations:
                        S_t[t,w] = sum(S_temp[t,i,w] for i in range(res))
                        # if ('rolling_spill' in problem.parameter.eq_additional_constraints):
                        #     S_week_t[t,w] = (S_week[t,w].x)
                                                                            
                        for i in range(res):
                            discharg.loc[t,f'res_{i}_scen_{w}'] = sum(discharg_temp[t,i,w,k] for k in range(seg))
                            if res >1:
                                S_t_per_res.loc[t,f'res_{i}_scen_{w}'] = S_temp[t,i,w]
                                Qflow_total.loc[t,f'res_{i}_scen_{w}'] = Q_flow_temp[t,i,w]
                            for i_seg in range(seg):
                                p_result_per_seg[t,i,w,i_seg] = discharg_temp[t,i,w,i_seg] * mu['mu_'+ seg_array[i_seg]][res_array[i]]
            
            # Make it faster wit comprishension
            # S_t_2 = [[np.sum(S[t, i, w].x for i in range(res)) for w in range(scen)] for t in range(tim)]
            discharg_detailed_per_scen = np.array([[sum(discharg_temp[t, i, w, k] if hrs[w]>t else 0 for k in range(seg)) for w in range(scen)] for t in range(tim)])                   
            discharg_detailed_per_scen = pd.DataFrame(discharg_detailed_per_scen,columns=[f'scen_{w}' for w in range(scen)])
            # p_result_per_seg = np.array([[(p_result_per_seg[t, i, w, k] if hrs[w]>t else 0 ) for k in range(seg) for w in range(scen)] for t in range(tim)])  
            # p_result_per_seg = pd.DataFrame(p_result_per_seg, columns=[f'res_{i_res}_scen_{i_scen}_seg_{i_seg}' for i_seg in range(seg) for i_scen in range(scen)  for i_res in range(res)] )  
            discharg[discharg.isna()]=0
            opt_results.discharge = discharg
            opt_results.discharg_detailed = discharg_detailed_per_scen
            opt_results.content = content
            opt_results.final_energy = {var_name: optimal_values[var.index] for var_name, var in energy_end.items()}[0]
            opt_results.initial_energy = TiE
            opt_results.spill = S_t
            # if ('rolling_spill' in problem.parameter.eq_additional_constraints ) & (t>=rolling_spill_hours):
            #     opt_results.spill_week = S_week_t
            if res >1:
                opt_results.spill_per_res = S_t_per_res
                opt_results.q_flow = Qflow_total
            # opt_results.power_per_seg = p_result_per_seg
            logger.info("Total inflow after: %s", pd.DataFrame(V).sum().sum())
            
    else:
        opt_results = OptResult()
        opt_results.optimal = False
    model_equivalent.clearModel()
    
    return opt_results
