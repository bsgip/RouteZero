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
CB = 368                # max battery capacity per bus
U = 150/6               # equiv to 150 kw
num_chargers = 80
SC = 0.9                # initial charge
FC = 0.8                # required end charge
depot_limit = 1000/6      # (kwh), # roughly converted from 1MVA to kwh in a 10 minute interval

dead_time = 15      # time either side of bus returning that can't be used to charge
safety_factor = 1.0     # increase above 1 to consider higher energy usage

# define a battery
battery_cap = 1000           #   kWh
battery_discharge = -200/6   #   equivalent to a max dishcarge of 30kW
battery_charge = 200/6       #   equivalent to a max charge of 30kW
battery_init_soc = 0.5       #   0% initial charge


M = optim_data['Nt'].max()
Nt = optim_data['Nt'].to_numpy()
Nt_avail = minimum_filter1d(Nt, int(np.ceil(dead_time*2/10)))
times = optim_data.index
end_time = times.max()
num_times = len(times)


ER = optim_data['ER'].to_numpy() * safety_factor
ED = optim_data['ED'].to_numpy() * safety_factor

ER_cum = optim_data['ER_cum']       # shifted cumulative

import_charge = np.cos(2*np.pi*times/end_time*repeat_days)+1.1      # this works for one day worth
export_charge = import_charge * 0.1

model = pyo.ConcreteModel()
model.T = range(num_times)
model.x = pyo.Var(model.T, domain=pyo.Reals, bounds=(0,U*num_chargers))
model.slack = pyo.Var(domain=pyo.NonNegativeReals)
model.bx = pyo.Var(model.T, domain=pyo.Reals, bounds=(battery_discharge,battery_charge))
model.a = pyo.Var(model.T, domain=pyo.Reals,bounds=(-1000,1000), initialize=100)   # auxilliary variables for splitting cost of import vs export


model.obj = pyo.Objective(expr=sum(import_charge[t]*model.x[t] for t in model.T)+model.slack * 1e10 +
                          sum(model.a[t] for t in model.T),
                          sense=pyo.minimize)

# maximum charging
model.max_charge = pyo.ConstraintList()
for t in range(num_times):
    model.max_charge.add(model.x[t] <= Nt_avail[t] * U)
    model.max_charge.add(model.x[t] + model.bx[t] <= depot_limit)



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

# enforce bys battery finishes at least some percent charged
model.end_constraint = pyo.Constraint(expr=M*CB*SC + sum(model.x[t]-ED[t] for t in model.T) + model.slack>=FC*M*CB)

# depot battery stuff
model.soc_min = pyo.ConstraintList()
model.soc_max = pyo.ConstraintList()
for t in range(1, num_times):
   #  will be infeasible if starting charge not enough
    model.soc_min.add(battery_cap*battery_init_soc + sum(model.bx[i] for i in range(t)) >= 0.)
    model.soc_max.add(battery_cap*battery_init_soc + sum(model.bx[i] for i in range(t)) <= battery_cap)

# auxilliary battery variable constraints for import/export pricing
model.a_con = pyo.ConstraintList()

for t in model.T:
    model.a_con.add(model.a[t] - model.bx[t] * import_charge[t] >= 0.)
    model.a_con.add(model.a[t] >= model.bx[t] * export_charge[t])



opt = SolverFactory('cbc')
results = opt.solve(model)
# model.display()


x_array = np.zeros((num_times,))
bx_array = np.zeros((num_times,))
a_array = np.zeros((num_times,))

for t in model.T:
    x_array[t] = model.x[t].value
    bx_array[t] = model.bx[t].value
    a_array[t] = model.a[t].value



energy_available = SC * CB * M + np.cumsum(x_array) - np.cumsum(ER)
battery_soc = battery_cap*battery_init_soc + np.cumsum(bx_array)

plt.subplot(3,1,1)
plt.plot(times/60, import_charge, label='import')
plt.plot(times/60, export_charge, label='export')
plt.legend()
plt.xlabel('time of day (hours)')
plt.ylabel('dollars')
plt.title('price signal')
plt.subplot(3,1,2)
plt.plot(times/60,x_array, label='charging')
plt.plot(times/60,Nt_avail, label='busses available to charge')
plt.title('Bus charging (x)')
plt.xlabel('time of day (hours)')
plt.legend()
plt.ylabel('kwh')
plt.subplot(3,1,3)
plt.plot(times/60,energy_available)
plt.xlabel('time of day (hours)')
plt.title('Energy available at in busses at depot')
plt.ylabel('kwh')
plt.tight_layout()
plt.show()



plt.subplot(3,1,1)
plt.plot(times/60,bx_array*6)
plt.xlabel('time of day (hours)')
plt.title('Battery charge/discharge power')
plt.ylabel('kW')

plt.subplot(3,1,2)
plt.plot(times/60,battery_soc)
plt.xlabel('time of day (hours)')
plt.title('battery state of charge')
plt.ylabel('kWh')

plt.subplot(3,1,3)
plt.plot(times/60,(bx_array)*6, label='battery')
plt.plot(times/60,(x_array)*6, label='bus_charging')
plt.plot(times/60,(bx_array+x_array)*6, label='aggregate',linestyle='-.')
plt.axhline(depot_limit*6, linestyle='--',color='r', label='limit')
plt.xlabel('time of day (hours)')
plt.title('all loads (power)')
plt.legend()
plt.ylabel('kW')

plt.tight_layout()
plt.show()

print('Done')

