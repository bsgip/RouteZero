import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory

from scipy.ndimage.filters import minimum_filter1d

# load saved data
optim_data = pd.read_csv('./data/optim_data.csv', index_col=0)
optim_data['ER_cum'] = optim_data['ER'].cumsum().shift(1, fill_value=0.)


## the following makes repeat of the one days worth of data taht was saved
repeat_days = 7
li = []
for i in range(repeat_days):
    tmp = optim_data.copy(deep=True)
    if li:
        tmp.index = tmp.index + li[i-1].index.max()
    li.append(tmp)

optim_data = pd.concat(li,axis=0)
###


## define some things
Q = 100         # cost on connection capacity
R = 10          # cost on number of chargers

CB = 368         # max battery capacity per bus
# CB = 100         # max battery capacity per bus
# U = 10         # maximum amount charged per charger (kwh), equiv to 60kW
U = 150/6        # equiv to 150 kw
# num_chargers = 80     # in this problem we are determining the min number for this
SC = 0.9      # initial charge
FC = 0.9       # required end charge

dead_time = 15      # time either side of bus returning that can't be used to charge
safety_factor = 1.0     # increase above 1 to consider higher energy usage


M = optim_data['Nt'].max()
Nt = optim_data['Nt'].to_numpy()
Nt_avail = minimum_filter1d(Nt, int(np.ceil(dead_time*2/10)))
times = optim_data.index
end_time = times.max()
num_times = len(times)
# we are optimising for depot limit in this one
# depot_limit = 2*1500/6      # (kwh), # roughly converted from 1MVA to kwh in a 10 minute interval

ER = optim_data['ER'].to_numpy() * safety_factor
ED = optim_data['ED'].to_numpy() * safety_factor

out = minimum_filter1d(Nt, 10)
plt.plot(Nt[0:50])
plt.plot(Nt_avail[0:50])
plt.show()

# a price signal that is cheapest in the middle of the day when solar would be available
# p = np.cos(2*np.pi*times/end_time*repeat_days)+1.1      # this works for one day worth
# p = 0 * times/end_time*repeat_days + 1

model = pyo.ConcreteModel()
model.T = range(num_times)
# model.x = pyo.Var(model.T, domain=pyo.Reals, bounds=(0,min(U*num_chargers,depot_limit)))
model.x = pyo.Var(model.T, domain=pyo.NonNegativeReals)
model.Nc = pyo.Var(domain=pyo.Integers,bounds=(1,None))
model.G = pyo.Var(domain=pyo.Reals,bounds=(1,None))
model.slack = pyo.Var(domain=pyo.NonNegativeReals)

model.obj = pyo.Objective(expr=model.G*Q + model.Nc*R + model.slack*1e10,sense=pyo.minimize)

# maximum charging
model.max_charge = pyo.ConstraintList()
for t in range(num_times):
    model.max_charge.add(model.x[t] <= Nt_avail[t]*U)
    model.max_charge.add(model.x[t] <= model.Nc*U)
    model.max_charge.add(model.x[t] <= model.G)

# minimum charge limit
model.max_charged = pyo.ConstraintList()
for t in range(0, num_times):
    if t==0:
        model.max_charged.add(model.x[0]<=((1-SC)*M*CB))
    else:
        model.max_charged.add(sum(model.x[i] for i in range(t+1)) <= sum(ER[k] for k in range(t)))

model.min_charged = pyo.ConstraintList()
for t in range(1, num_times):
   #  will be infeasible if starting charge not enough
    model.min_charged.add(SC*M*CB -sum(ED[k] for k in range(t+1)) + sum(model.x[i] for i in range(t)) >= 0.)

# enforce battery finishes at least 80% charged
model.end_constraint = pyo.Constraint(expr=M*CB*SC + sum(model.x[t]-ED[t] for t in model.T) + model.slack>=FC*M*CB)

opt = SolverFactory('cbc')
results = opt.solve(model)
# model.display()


x_array = np.zeros((num_times,))
for t in model.T:
    x_array[t] = model.x[t].value


min_depot_limit = model.G.value
min_number_chargers = model.Nc.value
end_period_charge_slack = model.slack.value

print('Minimum number of chargers required is ', min_number_chargers)
print('Min depot connection rating is ', min_depot_limit, ' in kWh per interval')
print('Min depot connection power rating is ', min_depot_limit*6, 'in kW')
print('Failed to meet desired end aggregate charge by ', end_period_charge_slack, ' kWh')
print('which amounts to', end_period_charge_slack/(M*CB)*100, ' (%)')

energy_available = SC * CB * M + np.cumsum(x_array) - np.cumsum(ER)

plt.subplot(2,1,1)
plt.plot(times/60,x_array, label='charging')
plt.plot(times/60,Nt_avail, label='busses available to charge')
plt.title('Bus charging (x)')
plt.xlabel('time of day (hours)')
plt.legend()
plt.ylabel('kwh')
plt.subplot(2,1,2)
plt.plot(times/60,energy_available)
plt.xlabel('time of day (hours)')
plt.title('Energy available at in busses at depot')
plt.ylabel('kwh')
plt.tight_layout()
plt.show()

print('Done')