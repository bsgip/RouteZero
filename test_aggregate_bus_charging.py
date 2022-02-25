import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory

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
CB = 368         # max battery capacity per bus
# CB = 100         # max battery capacity per bus
U = 10         # maximum amount charged per charger (kwh)
num_chargers = 30
SC = 0.9      # initial charge
FC = 0.8       # required end charge


M = optim_data['Nt'].max()
Nt = optim_data['Nt'].to_numpy()
times = optim_data.index
end_time = times.max()
num_times = len(times)
depot_limit = 1000*2/6      # (kwh), # roughly converted from 1MVA to kwh in a 10 minute interval

ER = optim_data['ER'].to_numpy()
ED = optim_data['ED'].to_numpy()

ER_cum = optim_data['ER_cum']       # shifted cumulative

# a price signal that is cheapest in the middle of the day when solar would be available
p = np.cos(2*np.pi*times/end_time*repeat_days)+1.1      # this works for one day worth


model = pyo.ConcreteModel()
model.T = range(num_times)
model.x = pyo.Var(model.T, domain=pyo.Reals, bounds=(0,min(U*num_chargers,depot_limit)))

model.obj = pyo.Objective(expr=sum(p[t]*model.x[t] for t in model.T),
                          sense=pyo.minimize)

# maximum charging
model.max_charge = pyo.ConstraintList()
for t in range(num_times):
    model.max_charge.add(model.x[t] <= Nt[t]*U)

# minimum charge limit
model.max_charged = pyo.ConstraintList()
for t in range(0, num_times):
    if t==0:
        model.max_charged.add(model.x[0]==0.) ## todo: currently assuming battery starts full, change this?
    else:
        model.max_charged.add(sum(model.x[i] for i in range(t+1)) <= sum(ER[k] for k in range(t)))

model.min_charged = pyo.ConstraintList()
for t in range(1, num_times):
    # if t==0:
        # model.min_charged.add(M * CB - sum(ER[k] for k in range(t + 1)) >=0.)    # todo: assumes starts fully charged, will be infeasible if starting charge not enough
    # else:
    model.min_charged.add(SC*M*CB -sum(ER[k] for k in range(t+1)) + sum(model.x[i] for i in range(t)) >= 0.)

# enforce battery finishes at least 80% charged
model.end_constraint = pyo.Constraint(expr=M*CB*SC + sum(model.x[t]-ER[t] for t in model.T)>=FC*M*CB)

opt = SolverFactory('cbc')
results = opt.solve(model)
# model.display()


x_array = np.zeros((num_times,))
for t in model.T:
    x_array[t] = model.x[t].value

energy_available = SC * CB * M + np.cumsum(x_array) - np.cumsum(ER)

plt.subplot(3,1,1)
plt.plot(times/60, p)
plt.xlabel('time of day (hours)')
plt.ylabel('dollars')
plt.title('price signal')
plt.subplot(3,1,2)
plt.plot(times/60,x_array, label='charging')
plt.plot(times/60,Nt, label='busses at depot')
plt.title('Bus charging (x)')
plt.xlabel('time of day (hours)')
plt.legend()
plt.ylabel('kwh')
plt.subplot(3,1,3)
plt.plot(times/60,energy_available)
plt.xlabel('time of day (hours)')
plt.title('Energy available at depot')
plt.ylabel('kwh')
plt.tight_layout()
plt.show()

print('Done')