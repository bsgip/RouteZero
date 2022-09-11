import numpy as np

import RouteZero.bus as ebus
from RouteZero.models import PredictionPipe
import matplotlib.pyplot as plt
import pandas as pd
from RouteZero.route import calc_buses_in_traffic
from RouteZero.optim import Extended_feas_problem, plot_results
import time

NUM_ROUTES = 40
RESOLUTION = 10  # mins
CHARGER_MAX_POWER = 150  # kW
MIN_CHARGE_TIME = 1 * 60  # mins
RESERVE = 0.2  # percent of all battery to keep in reserve [0-1]
DEADHEAD = 0.1


trips_data = pd.read_csv('../data/gtfs/greater_sydney/trip_data.csv', index_col='Unnamed: 0')
trips_data = trips_data[trips_data['agency_name']=='Newcastle Transport']

# print("total routes: ", len(trips_data["route_short_name"].unique()))


test = trips_data.groupby(by=['route_short_name'])["route_short_name"].count().sort_values()

route_names = test.index[-NUM_ROUTES:]

trips_data = trips_data[trips_data["route_short_name"].isin(route_names)]

print("number of trips is",len(trips_data))


trips_data['passengers'] = 30
bus = ebus.BYD()
prediction_pipe = PredictionPipe()

ec_km, ec_total = prediction_pipe.predict_worst_case(trips_data, bus)

times, buses_in_traffic, depart_trip_energy_reqs, return_trip_enery_consumed = calc_buses_in_traffic(trips_data,
                                                                                                     deadhead=DEADHEAD,
                                                                                                     resolution=RESOLUTION,
                                                                                                     trip_ec=ec_total)
t = np.cumsum(depart_trip_energy_reqs) - np.cumsum(return_trip_enery_consumed)
print("max requried buses = ", buses_in_traffic.max())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(times / 60, buses_in_traffic)
plt.xlabel('Hour of week')
plt.ylabel('# buses')

plt.subplot(1, 2, 2)
# plt.plot(times / 60, depart_trip_energy_reqs)
plt.plot(times/60, t)
# plt.plot(times/60, return_trip_enery_consumed, label="returning trips")
plt.xlabel("Hour of week")
plt.ylabel("Energy required by departing trips (kWh)")

plt.tight_layout()
plt.show()

chargers = {'power': [150], 'number': ['optim'], 'cost':[10]}
bus.charging_rate=300
grid_limit='optim'
battery = {'power':2500, 'capacity':10000, 'efficiency':0.95}
# battery = None
problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.9,
                              deadhead=DEADHEAD,resolution=RESOLUTION, min_charge_time=MIN_CHARGE_TIME, reserve=RESERVE,
                              battery=battery)

t1 = time.time()
results = problem.solve()
t2 = time.time()

print('Solve took {} seconds'.format(t2 - t1))

plot_results(results, problem)