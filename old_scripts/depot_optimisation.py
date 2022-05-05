global tmp
import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory
from scipy.ndimage.filters import minimum_filter1d


def feas_optim(optim_data, bus_battery_cap, max_charging_rate, deadtime, repeat_days=7, Q=100, R=10, safety_factor=1.0,
               SC=0.9, FC=0.9, resolution=10):
    """
    Runs the feasibility optimisation for part 1 outputs
    :param optim_data: data created about enery requirements for trips and busses at dept
    :param bus_battery_cap: battery capacity of the busses (kWh)
    :param max_charging_rate: maximum charging rate per bus per charger (kW)
    :param deadtime: time factor for departing or returning to depot
    :param repeat_days: data is based on a single day at present and so this is the number of times to stack that day up
    :param R: cost on number of chargers for aggregating costs
    :param Q: cost on connection capacity for aggregating costs
    :param safety_factor increase above 1 to allow for higher energy usage
    :param SC: starting charge ratio for the batteries
    :param FC: required finished charge ratio for the batteries
    :param resolution: the time resolution that was used when creating the optim_data
    :return:
    """

    li = []
    for i in range(repeat_days):
        tmp = optim_data.copy(deep=True)
        if li:
            tmp.index = tmp.index + li[i - 1].index.max()
        li.append(tmp)
    optim_data = pd.concat(li, axis=0)

    CB = bus_battery_cap
    U = (max_charging_rate)*(resolution/60) # convert from power to kWh in interval

    M = optim_data['Nt'].max()
    Nt = optim_data['Nt'].to_numpy()
    Nt_avail = minimum_filter1d(Nt, int(np.ceil(deadtime*2/resolution)))
    times = optim_data.index
    num_times = len(times)

    ER = optim_data['ER'].to_numpy() * safety_factor
    ED = optim_data['ED'].to_numpy() * safety_factor


    model = pyo.ConcreteModel()
    model.T = range(num_times)
    # model.x = pyo.Var(model.T, domain=pyo.Reals, bounds=(0,min(U*num_chargers,depot_limit)))
    model.x = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.Nc = pyo.Var(domain=pyo.Integers, bounds=(1, None))
    model.G = pyo.Var(domain=pyo.Reals, bounds=(1, None))
    model.slack = pyo.Var(domain=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=model.G * Q + model.Nc * R + model.slack * 1e10, sense=pyo.minimize)
    # maximum charging
    model.max_charge = pyo.ConstraintList()
    for t in range(num_times):
        model.max_charge.add(model.x[t] <= Nt_avail[t] * U)
        model.max_charge.add(model.x[t] <= model.Nc * U)
        model.max_charge.add(model.x[t] <= model.G)
    # minimum charge limit
    model.max_charged = pyo.ConstraintList()
    for t in range(0, num_times):
        if t == 0:
            model.max_charged.add(model.x[0] <= ((1 - SC) * M * CB))
        else:
            model.max_charged.add(sum(model.x[i] for i in range(t + 1)) <= sum(ER[k] for k in range(t)))
    model.min_charged = pyo.ConstraintList()
    for t in range(1, num_times):
        #  will be infeasible if starting charge not enough
        model.min_charged.add(SC * M * CB - sum(ED[k] for k in range(t + 1)) + sum(model.x[i] for i in range(t)) >= 0.)
    # enforce battery finishes at least 80% charged
    model.end_constraint = pyo.Constraint(
        expr=M * CB * SC + sum(model.x[t] - ED[t] for t in model.T) + model.slack >= FC * M * CB)



    # get results
    opt = SolverFactory('cbc')
    status = opt.solve(model)
    x_array = np.zeros((num_times,))
    for t in model.T:
        x_array[t] = model.x[t].value
    min_depot_limit = model.G.value
    min_number_chargers = model.Nc.value
    end_period_charge_slack = model.slack.value
    energy_available = SC * CB * M + np.cumsum(x_array) - np.cumsum(ER)

    out = {'charging_rate':x_array, "min_connection_point":min_depot_limit,"min_chargers":min_number_chargers,
           'end_slack':end_period_charge_slack,"energy_available":energy_available}
    return out

def economic_optimisation(optim_data, bus_battery_cap, max_charging_rate, deadtime, num_chargers, depot_limit, price_signal, extra_buses,
                          repeat_days=7, safety_factor=1.0, SC=0.9, FC=0.9, resolution=10):

    li=[]
    for i in range(repeat_days):
        tmp = optim_data.copy(deep=True)
        if li:
            tmp.index = tmp.index + li[i - 1].index.max()
        li.append(tmp)

    optim_data = pd.concat(li, axis=0)
    ###

    ## define some things
    CB = bus_battery_cap  # max battery capacity per bus
    U = (max_charging_rate)*(resolution/60) # convert from power to kWh in interval

    M = optim_data['Nt'].max()
    Nt = optim_data['Nt'].to_numpy() + extra_buses
    Nt_avail = minimum_filter1d(Nt, int(np.ceil(deadtime * 2 / 10)))
    times = optim_data.index
    num_times = len(times)
    depot_limit = depot_limit *(resolution/60)  # convert from power to kWh per interval

    ER = optim_data['ER'].to_numpy() * safety_factor
    ED = optim_data['ED'].to_numpy() * safety_factor


    p = np.repeat(price_signal, repeat_days)

    model = pyo.ConcreteModel()
    model.T = range(num_times)
    model.x = pyo.Var(model.T, domain=pyo.Reals, bounds=(0, min(U * num_chargers, depot_limit)))
    model.slack = pyo.Var(domain=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=sum(p[t] * model.x[t] for t in model.T)+model.slack * 1e10,
                              sense=pyo.minimize)

    # maximum charging
    model.max_charge = pyo.ConstraintList()
    for t in range(num_times):
        model.max_charge.add(model.x[t] <= Nt_avail[t] * U)

    # minimum charge limit
    model.max_charged = pyo.ConstraintList()
    for t in range(0, num_times):
        if t == 0:
            model.max_charged.add(
                model.x[0] <= ((1 - SC) * M * CB))
        else:
            model.max_charged.add(sum(model.x[i] for i in range(t + 1)) <= sum(ER[k] for k in range(t)))

    model.min_charged = pyo.ConstraintList()
    for t in range(1, num_times):
        #  will be infeasible if starting charge not enough
        model.min_charged.add(SC * M * CB - sum(ED[k] for k in range(t + 1)) + sum(model.x[i] for i in range(t)) >= 0.)

    # enforce battery finishes at least 80% charged
    model.end_constraint = pyo.Constraint(expr=M * CB * SC + sum(model.x[t] - ED[t] for t in model.T)+ model.slack >= FC * M * CB)


    # get out results
    opt = SolverFactory('cbc')
    status = opt.solve(model)
    x_array = np.zeros((num_times,))
    for t in model.T:
        x_array[t] = model.x[t].value
    energy_available = SC * CB * M + np.cumsum(x_array) - np.cumsum(ER)
    end_period_charge_slack = model.slack.value
    results = {'charging_rate':x_array,'energy_available':energy_available, 'end_slack':end_period_charge_slack}
    return results
