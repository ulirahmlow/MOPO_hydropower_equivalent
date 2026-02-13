import numpy as np
import gurobipy as gp
import logging
import pandas as pd
from naive_aggregation import EqModel
#from copy import deepcopy

#logging.Logger()
logger = logging.getLogger(__name__)

class OptResult:
    def __init__(self):
        self.content = None
        self.discharge = None
        self.final_energy = None
        self.initial_energy = None
        self.power = None
        self.consumption = None # Only if Pumping is used
        self.profit = None
        self.spill = None
        self.optimal = True
        self.discharg_detailed = None
        self.power_per_seg = None
        self.total_spill = None
        self.loss_high_p_temp = None
        self.spill_week = None
        self.q_flow = None
        self.spill_per_res = None

def solve_equivalent(scenario_data, eqModel:EqModel,problem, end_simulation=0):
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
    mu = eqModel.mu  # Production equivalent
    EndRes = scenario_data.MTminShare # Reservoir content at the end of the planning period
    res = eqModel.nbr_stations  # Number of reservoirs in the river
    seg = eqModel.segments  # Number of segments in Qmax and mu
    TiE = scenario_data.initialEnergy  # Initial energy
    ramp1h = eqModel.ramp1h  # Ramping limit for 1h
    ramp3h = eqModel.ramp3h  # Ramping limit for 3h
    ramp1h_time = problem.parameter.ramp_short
    ramp3h_time = problem.parameter.ramp_long
    res_array = [f'res_{i_res}' for i_res in range(res)]
    seg_array = [f'{i_seg}' for i_seg in range(seg)]
    
    delay = False
    if isinstance(eqModel.delay,pd.Series):
        delay= True
        delay_Hour = np.floor(eqModel.delay/60).astype(int).to_dict()  # Flow time in whole hours for the discharge
        delay_Min = (eqModel.delay.astype(float)%60).astype(int).to_dict()   # Flow time in remaining minutes for the discharge
    # Calulate initial reservoir content -----------------------
    M0 = np.zeros((res, scen))  # initial content in reservoirs
    for i in range(res):  # reservoirs/stations
        for w in range(scen):  # scenarios
            M0[i, w] = (TiE[w] * alpha.iloc[i,0]) / (
                sum(ad[i][down]* mu['mu_0'][res_array[down]] for down in range(res))
            )  

    with gp.Env() as env, gp.Model(env=env) as model_equivalent:
        model_equivalent.setParam("OutputFlag", 0)
        model_equivalent.setParam("LogToConsole", 0)

        M = {}
        Q = {}
        S = {}
        energy_end = {}
        for w in range(scen):
            energy_end[w] = model_equivalent.addVar(lb=0, name=f"energy_end_{w}")  # Energy at the end (dummy variable)
        indices = eqModel.indices
        indices_with_segments = eqModel.indices_with_segments

        # Extract values using dictionary comprehension
        Mmin_values = {ind: Mmin.get('res_'+ str(ind[1]), None) for ind in indices}
        Mmax_values = {ind: Mmax.get('res_'+ str(ind[1]), None) for ind in indices}
        Smin_values = {ind: Smin.get('res_'+ str(ind[1]), None) for ind in indices}
        Qmax_values = {ind: Qmax.get('Qmax_'+str(ind[3])).get('res_'+ str(ind[1]), None) for ind in indices_with_segments}
        
        M = model_equivalent.addVars(indices,
                                    lb=Mmin_values,
                                    ub=Mmax_values,
                                    name="M")

        S = model_equivalent.addVars(indices,
                                lb =Smin_values,
                                name="S")        
            
        if res>1:
            Sflow = model_equivalent.addVars(indices,
                                            lb=0,
                                            name="Sflow")
            Qflow = model_equivalent.addVars(indices,
                                lb=0,
                                name="Qflow")
        Qexp = model_equivalent.addVars(indices,
                                lb=0,
                                name="Q_tot")
            
        P = model_equivalent.addVars(indices,
                                    lb =0,
                                    name="P")
    

        Q = model_equivalent.addVars(indices_with_segments,
            lb=0, ub=Qmax_values,
            vtype=gp.GRB.CONTINUOUS,
            name="Q"
        )
        
        z = model_equivalent.addVar(vtype=gp.GRB.CONTINUOUS, name="z", obj=-1)  # Profit (objective)
        model_equivalent.addLConstr(
            (gp.quicksum( 1 *
                gp.quicksum(price['scen'+str(w+1)][t]
                * gp.quicksum(
                        P[t,i,w]
                                    for i in range(res)) 
                                    for t in range(tim) if hrs[w]>t)
                                    for w in range(scen))) == z,
            name="obj",
        )

        # Hydrological balance: row 2:25
        for w in range(scen):
            # 3 Limit end reservoir content: (only for last hour)
            model_equivalent.addLConstr(
                energy_end[w]
                == gp.quicksum(
                    gp.quicksum(
                        M[hrs[w]-1, i, w] * ad[i][down] * mu['mu_0'][res_array[down]]
                        for down in range(res) )for i in range(res))
            ,name =f"limit_end_res1_{w}")
            model_equivalent.addLConstr(energy_end[w] >= EndRes.iloc[0,w] * TiE[w],name =f"limit_end_res1_{w}")

            for t in range(tim):
                if hrs[w]>t: # When Scenarios are not equal long then only use it when the scenarios is 
                        
                    if (t>ramp1h_time-1) & (ramp1h_time != -1):
                        model_equivalent.addLConstr(
                            gp.quicksum(
                                P[t-ramp1h_time,i,w] - P[t,i,w]
                                    for i in range(res) )  
                                    - ramp1h['res_0']
                            <= 0, name =f"Ramping1_0_lim_{t}_{i}_{w}"
                        )
                        
                        model_equivalent.addLConstr(
                            gp.quicksum(
                                P[t,i,w] - P[t-ramp1h_time,i,w]
                                for i in range(res)
                            ) - ramp1h['res_0']
                            
                            <= 0, name =f"Ramping1_1_lim_{t}_{i}_{w}"
                        )

                    # 14-15 Ramping limits for 4 hours:
                    if  (t > ramp3h_time-1) & (ramp3h_time != -1):
                        model_equivalent.addLConstr(
                            gp.quicksum(
                                P[t - ramp3h_time,i,w] - P[t,i,w]
                                for i in range(res)
                            ) - ramp3h['res_0']
                            
                            <= 0 , name =f"Ramping3_0_lim_{t}_{i}_{w}"
                        )

                        model_equivalent.addLConstr(
                            gp.quicksum(
                                P[t,i,w] - P[t-ramp3h_time,i,w]
                                for i in range(res) 
                            ) - ramp3h['res_0']
                            
                            <= 0, name =f"Ramping3_1_lim_{t}_{i}_{w}"
                        )
                        
                                
                    for i in range(res):
                        model_equivalent.addLConstr(
                                gp.quicksum(Q[t, i, w, k] for k in range(seg)) == Qexp[t, i, w], name=f"expo_{t}_{i}")
                                
                            
                        if (not delay) & (res==1): # (res ==1)
                            model_equivalent.addLConstr(
                                0 == M[t, i, w]
                                - (M[t - 1, i, w] if t > 0 else M0[i, w])
                                - V[f'scen_{w}_res_{i}'][t]
                                + Qexp[t, i, w]
                                + S[t, i, w]
                                ,name =f"hydbal_{t}_{i}_{w}")
                        
                        elif (res>1) & (not delay):
                                
                            model_equivalent.addLConstr(
                                0 == M[t, i, w]
                                - (M[t - 1, i, w] if t > 0 else M0[i, w])
                                - V[f'scen_{w}_res_{i}'][t]
                                + Qexp[t, i, w]
                                + S[t, i, w]
                                - (gp.quicksum(aq[i][above] * (Qexp[t, above, w] + S[t, above, w]) for above in range(res)))
                                ,name =f"hydbal_{t}_{i}_{w}")
                            
                        else:
                            # Export of Q = total output from all segments in Q:
                            # model_equivalent.addLConstr(
                            #     gp.quicksum(Q[t, i, w, k] for k in range(seg)) == Qexp[t, i, w], name=f"expo_{t}_{i}")
                            
                            model_equivalent.addLConstr(
                                0 == M[t, i, w]
                                - (M[t - 1, i, w] if t > 0 else M0[i, w])
                                - V[f'scen_{w}_res_{i}'][t]
                                + Qexp[t, i, w]
                                + S[t, i, w]
                                - gp.quicksum(aq[i][above] * (Qflow[t, above,w] + Sflow[t, above, w]) for above in range(res))
                                ,name =f"hydbal_{t}_{i}_{w}")
                            i_res = res_array[i]
                            model_equivalent.addLConstr(
                                Qflow[t, i, w] ==
                                    (delay_Min[i_res]/60) * (Qexp[t-(delay_Hour[i_res]+1), i, w] if t-delay_Hour[i_res]-1 >= 0 else 0) +
                                    ((60-delay_Min[i_res])/60) * (Qexp[t-delay_Hour[i_res], i,w] if t-delay_Hour[i_res] >= 0 else 0)
                                    , name =f"flowdelay_{t}_{i}")
                            
                            model_equivalent.addLConstr(
                                Sflow[t, i, w] ==
                                    (delay_Min[i_res]/60) * (S[t-(delay_Hour[i_res]+1), i,w] if t-delay_Hour[i_res]-1 >= 0 else 0) +
                                    ((60-delay_Min[i_res])/60) * (S[t-delay_Hour[i_res], i,w] if t-delay_Hour[i_res] >= 0 else 0)
                                    , name =f"spilldelay1_{t}_{i}") 
                                               
                        # Imput Help varaible P 
                        model_equivalent.addLConstr(
                            P[t,i,w] ==  gp.quicksum(mu['mu_'+ seg_array[k]][res_array[i]]*Q[t, i, w, k] for k in range(seg)),name=f"Power_gen_{t}")

                        # Discharge lower limit
                        model_equivalent.addLConstr(
                            Qmin[res_array[i]] - gp.quicksum(Q[t, i, w, k] for k in range(seg)) <= 0 
                            ,name =f"Q_limits_{t}_{i}_{w}")
                        

        # Optimize
        model_equivalent.update()
        model_equivalent.optimize()

        if model_equivalent.Status == 2:  # checks if optimisatzion has found an optimimum
            # Initiaizie theoutput
            opt_results = OptResult()
            # opt_results.profit = z.X
            discharg_temp = model_equivalent.getAttr('x',Q)

            p_result = np.array([[sum(discharg_temp[t, i_res, w, k] * mu['mu_'+ seg_array[k]][res_array[i_res]] 
                                      if hrs[w]>t else 0 
                                      for k in range(seg)) 
                                      for w in range(scen)
                                      for i_res in range(res)]
                                      for t in range(tim)])
            p_result = pd.DataFrame(p_result,columns=[
                f'scen_{w}_res_{i_res}' for w in range(scen) for i_res in range(res)])
            p_result_per_scen = pd.DataFrame(np.zeros((tim,scen)),columns=[f'scen{w+1}' for w in range(scen) ])
            for w in range(scen):
                p_result_per_scen.loc[:,f'scen{w+1}'] = p_result_per_scen.loc[:,f'scen{w+1}'] + sum(
                    p_result[f'scen_{w}_res_{i_res}']  for i_res in range(res))
            
            spill_temp = model_equivalent.getAttr('x', S)
            total_spill = sum(sum(np.array([[spill_temp[t, i_res, w] * mu['mu_0'][res_array[i_res]] 
                                      if hrs[w]>t else 0
                                      for w in range(scen)
                                      for i_res in range(res)]
                                      for t in range(tim)])))        
            opt_results.total_spill = total_spill
            opt_results.power = p_result_per_scen

            if end_simulation:
                # For the last simulation, get more outputs and initialze the data differently
                p_result_per_scen = pd.DataFrame(np.ones((tim,scen))*-1,columns=[f'scen{w+1}' for w in range(scen) ])
                for w in range(scen):
                    p_result_per_scen.loc[:,f'scen{w+1}'] = p_result_per_scen.loc[:,f'scen{w+1}'] + sum(
                        p_result[f'scen_{w}_res_{i_res}']  for i_res in range(res))
                opt_results.power = p_result_per_scen
                M_t = model_equivalent.getAttr('x',M)
                content = np.zeros([tim, scen])
                S_t = np.zeros((tim, scen))
                p_result_per_seg = {}
                discharg = {}
                if res >1:
                    S_t_per_res = pd.DataFrame(index=range(tim),columns=[f'{res}_scen_{w}' for res in res_array for w in range(scen)])
                    Qflow_total = pd.DataFrame(index=range(tim),columns=[f'{res}_scen_{w}' for res in res_array for w in range(scen)])
                for t in range(tim):
                    for w in range(scen):
                        if hrs[w]>t: # When Scenarios are not equal long then only use it when the scenarios is 
                            content[t, w] = sum( M_t[t, i, w] * sum( ad[i][down] * mu['mu_0'][res_array[down]]
                                            for down in range(res)) for i in range(res))
                            # Total spill in all stations:
                            S_t[t,w] = sum(S[t,i,w].x for i in range(res))
                                                                              
                            for i in range(res):
                                discharg[t,i,w] = sum(discharg_temp[t,i,w,k] for k in range(seg))
                                if res >1:
                                    S_t_per_res.loc[t,f'res_{i}_scen_{w}'] = S[t,i,w].x
                                    Qflow_total.loc[t,f'res_{i}_scen_{w}'] = Qflow[t,i,w].x
                                for i_seg in range(seg):
                                    p_result_per_seg[t,i,w,i_seg] = discharg_temp[t,i,w,i_seg] * mu['mu_'+ seg_array[i_seg]][res_array[i]]

                discharg_detailed = np.array([[sum(discharg_temp[t, i, w, k] if hrs[w]>t else 0 for k in range(seg)) for w in range(scen)] for t in range(tim)])                   
                discharg_detailed = pd.DataFrame(discharg_detailed,columns=opt_results.power.columns)
                
                opt_results.discharge = discharg
                opt_results.discharg_detailed = discharg_detailed
                opt_results.content = content
                opt_results.final_energy = model_equivalent.getAttr('x',energy_end)[0]
                opt_results.initial_energy = TiE
                opt_results.spill = S_t
                if res >1:
                    opt_results.spill_per_res = S_t_per_res
                    opt_results.q_flow = Qflow_total
                logger.info("Total inflow after: %s", pd.DataFrame(V).sum().sum())
        else:
            opt_results = OptResult()
            opt_results.optimal = False

        model_equivalent.dispose()
    return opt_results 
        
def solve_equivalent_pump(scenario_data,eqModel:EqModel,problem,end_simulation=0):
    hrs = scenario_data.hours.astype(np.int32)  # Set of hours in each scenario
    tim = np.max(hrs)
    scen = scenario_data.scenarios  # Total number of scenarios
    Mmax = eqModel.Mmax  # Max content in reservoirs
    Mmin = eqModel.Mmin  # Min content in reservoirs
    price = scenario_data.price  # Electricity price per hour
    res = eqModel.nbr_stations  # Number of reservoirs in the river
    seg = eqModel.segments  # Number of segments in Qmax and mu
    seg_pump = eqModel.segments_pump  # Number of segments in Qmax and mu
    ramp1h = eqModel.ramp1h  # Ramping limit for 1h
    ramp3h = eqModel.ramp3h  # Ramping limit for 3h
    pump_loss = eqModel.pump_loss

    ramp1h_time = problem.parameter.ramp_short
    ramp3h_time = problem.parameter.ramp_long
    res_array = [f'res_{i_res}' for i_res in range(res)]

    with gp.Env() as env, gp.Model(env=env) as model_equivalent:
        model_equivalent.setParam("OutputFlag", 0)
        model_equivalent.setParam("LogToConsole", 0)
    
        M = {}
        S = {}

        indices = eqModel.indices
        indices_with_segments = eqModel.indices_with_segments
        indices_with_segments_pump = eqModel.indices_with_segments_pump
        # Extract values using dictionary comprehension
        Mmax_values = {ind: Mmax.get(f'res_{ind[1]}', None) for ind in indices}
        #Smin_values = {ind: Smin.get('res_'+ str(ind[1]), None) for ind in indices}
        M = model_equivalent.addVars(indices,
                                    lb=0,
                                    ub=Mmax_values,
                                    name="M")

        S = model_equivalent.addVars(indices,
                                lb =0,
                                name="S")        
        if eqModel.segments > 1:
            mu = eqModel.mu
            P_max_values = {ind: eqModel.Pmax.get('Pmax_'+str(ind[3])).get('res_'+ str(ind[1]), None) for ind in indices_with_segments}
            P = model_equivalent.addVars(indices,
                                    lb =0,
                                    name="P")  
            P_seg = model_equivalent.addVars(indices_with_segments,
                                    lb =0,
                                    ub=P_max_values,
                                    name="P_seg")      
        else:    
            P_max_values = {ind: eqModel.Pmax.get('Pmax_0').get('res_'+ str(ind[1]), None) for ind in indices}
            P = model_equivalent.addVars(indices,
                                    lb =0,
                                    ub=P_max_values,
                                    name="P")
        
    
        if eqModel.segments_pump > 1:
            mu_pump = eqModel.mu_pump
            
            C_max_values = {ind: eqModel.Cmax.get('Cmax_'+str(ind[3])).get('res_'+ str(ind[1]), None) for ind in indices_with_segments_pump}
            C = model_equivalent.addVars(indices,
                                    lb =0,
                                    name="C")
            C_seg = model_equivalent.addVars(indices_with_segments_pump,
                                    lb =0,
                                    ub=C_max_values,
                                    name="C_seg")
            # Caclte the the maximum consumption for the helper function to not produce and cosume max at the same time
            C_max = sum(C_max_values[0,0,0,i] for i in range(res))
            
        else:     
            #C_max = [(sum(mu[f'mu_pump_{k}'][res_array[i]] * eqModel.Cmax.get('Cmax_{k}').get(f'res_{i}', None) for k in range(seg))) for i in range(res)]
            C_max_values = {ind: eqModel.Cmax.get('Cmax_0').get('res_'+ str(ind[1]), None) for ind in indices}
            C = model_equivalent.addVars(indices,
                                        lb =0,
                                        ub=C_max_values,
                                        name="C")
            # Caclte the the maximum consumption for the helper function to not produce and cosume max at the same time
            C_max = sum(C_max_values[0,0,i] for i in range(res))
        
        z = model_equivalent.addVar(vtype=gp.GRB.CONTINUOUS, name="z", obj=-1)  # Profit (objective)
        model_equivalent.addLConstr(
            (gp.quicksum( 1 *
                gp.quicksum(price['scen'+str(w+1)][t]
                * gp.quicksum(
                        P[t,i,w] - C[t,i,w]
                                    for i in range(res)) 
                                    for t in range(tim) if hrs[w]>t)
                                    for w in range(scen))) == z,
            name="obj",
        )
        # Start and End constraint
        for i in range(res):
            for w in range(scen):
                model_equivalent.addLConstr(
                        M[0,i,0] == M[hrs[w]-1,i,w],
                        name=f"Start_{i}"
                    )

        # Hydrological balance: row 2:25
        for w in range(scen):
            for t in range(tim):
                if hrs[w]>t: # When Scenarios are not equal long then only use it when the scenarios is 
                    # Helper function to not produce and cosume at the same time. this works as P and C are defined as positive variables:
                    model_equivalent.addLConstr(
                        gp.quicksum(P[t,i,w] + C[t,i,w] for i in range(res)) <= C_max
                    )
                    if (t>ramp1h_time-1) & (ramp1h_time != -1):
                        model_equivalent.addLConstr(
                            gp.quicksum(
                                P[t-ramp1h_time,i,w] - P[t,i,w]
                                    for i in range(res) )  
                                    - ramp1h['res_0']
                            <= 0, name =f"Ramping1_0_lim_{t}_{i}_{w}"
                        )
                        
                        model_equivalent.addLConstr(
                            gp.quicksum(
                                P[t,i,w] - P[t-ramp1h_time,i,w]
                                for i in range(res)
                            ) - ramp1h['res_0']
                            <= 0, name =f"Ramping1_1_lim_{t}_{i}_{w}"
                        )

                    # 14-15 Ramping limits for 4 hours:
                    if  (t > ramp3h_time-1) & (ramp3h_time != -1):
                        model_equivalent.addLConstr(
                            gp.quicksum(
                                P[t - ramp3h_time,i,w] - P[t,i,w]
                                for i in range(res)
                            ) - ramp3h['res_0']
                            
                            <= 0 , name =f"Ramping3_0_lim_{t}_{i}_{w}"
                        )

                        model_equivalent.addLConstr(
                            gp.quicksum(
                                P[t,i,w] - P[t-ramp3h_time,i,w]
                                for i in range(res) 
                            ) - ramp3h['res_0']
                            <= 0, name =f"Ramping3_1_lim_{t}_{i}_{w}"
                        )
                        

                    for i in range(res):
                        model_equivalent.addLConstr(
                                0 == M[t, i, w]
                                - (M[t - 1, i, w] if t > 0 else 0)
                                + P[t, i, w]
                                - pump_loss[f'res_{i}']*C[t, i, w]
                                ,name =f"hydbal_{t}_{i}_{w}")
                        
                        if eqModel.segments > 1:
                            # Helper function to calculate total power per station
                            model_equivalent.addLConstr(
                                gp.quicksum(P_seg[t, i, w, k] for k in range(seg)) == P[t, i, w],
                            )
                        if eqModel.segments_pump > 1:
                            # Helper function to calculate total consumption per station
                            model_equivalent.addLConstr(
                                gp.quicksum(C_seg[t, i, w, k] * mu_pump[f'mu_pump_{k}'][f'res_{i}'] for k in range(seg_pump)) == C[t, i, w],
                            )
                                            
                    
        # Optimize
        model_equivalent.update()
        model_equivalent.optimize()

        if model_equivalent.Status == 2:  # checks if optimisatzion has found an optimimum
            # Initiaizie theoutput
            opt_results = OptResult()
            # opt_results.profit = z.X
            p_temp = model_equivalent.getAttr('x',P)
            c_temp = model_equivalent.getAttr('x',C)

            p_result = np.array([p_temp[t, i_res, w] if hrs[w]>t else 0 for w in range(scen) for i_res in range(res) for t in range(tim)])
            c_result = np.array([c_temp[t, i_res, w] if hrs[w]>t else 0 for w in range(scen) for i_res in range(res) for t in range(tim)])

            p_result = pd.DataFrame(p_result,columns=[
                f'scen_{w}_res_{i_res}' for w in range(scen) for i_res in range(res)])
            p_result_per_scen = pd.DataFrame(np.zeros((tim,scen)),columns=[f'scen{w+1}' for w in range(scen) ])
            c_result = pd.DataFrame(c_result,columns=[
                f'scen_{w}_res_{i_res}' for w in range(scen) for i_res in range(res)
            ])
            c_result_per_scen = pd.DataFrame(np.zeros((tim,scen)),columns=[f'scen{w+1}' for w in range(scen) ])
            for w in range(scen):
                p_result_per_scen.loc[:,f'scen{w+1}'] = p_result_per_scen.loc[:,f'scen{w+1}'] + sum(
                    p_result[f'scen_{w}_res_{i_res}']  for i_res in range(res))
                c_result_per_scen.loc[:,f'scen{w+1}'] = c_result_per_scen.loc[:,f'scen{w+1}'] + sum(
                    c_result[f'scen_{w}_res_{i_res}']  for i_res in range(res))
            
            spill_temp = model_equivalent.getAttr('x', S)
            total_spill = sum(sum(np.array([[spill_temp[t, i_res, w]
                                      if hrs[w]>t else 0
                                      for w in range(scen)
                                      for i_res in range(res)]
                                      for t in range(tim)])))        
            opt_results.total_spill = total_spill
            opt_results.power = p_result_per_scen
            opt_results.consumption = c_result_per_scen

            if end_simulation:
                M_t = model_equivalent.getAttr('x',M)
                content = np.zeros([tim, scen])
                S_t = np.zeros((tim, scen))
                if res >1:
                    S_t_per_res = pd.DataFrame(index=range(tim),columns=[f'{res}_scen_{w}' for res in res_array for w in range(scen)])
                for t in range(tim):
                    for w in range(scen):
                        if hrs[w]>t: # When Scenarios are not equal long then only use it when the scenarios is 
                            content[t, w] = sum(M_t[t, i, w] for i in range(res))
                            # Total spill in all stations:
                            S_t[t,w] = sum(S[t,i,w].x for i in range(res))                           
                            for i in range(res):
                                if res >1:
                                    S_t_per_res.loc[t,f'res_{i}_scen_{w}'] = S[t,i,w].x
                
                opt_results.content = content
                opt_results.spill = S_t
                if res >1:
                    opt_results.spill_per_res = S_t_per_res
        else:
            opt_results = OptResult()
            opt_results.optimal = False

        model_equivalent.dispose()
    return opt_results 