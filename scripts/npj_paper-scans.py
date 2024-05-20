import numpy as np

import RouteZero.bus as ebus
from RouteZero.models import PredictionPipe
import matplotlib.pyplot as plt
import pandas as pd
from RouteZero.route import calc_buses_in_traffic
from RouteZero.optim import Extended_feas_problem, plot_results, determine_charger_use
import time

NUM_ROUTES = 40
RESOLUTION = 10  # mins
CHARGER_MAX_POWER = 150  # kW
MIN_CHARGE_TIME = 1 * 60  # mins
RESERVE = 0.2  # percent of all battery to keep in reserve [0-1]
DEADHEAD = 0.1

scenario = 4



def plot_study_results(results, problem):
    grid_limit = results['grid_limit']
    # optim_chargers = results['chargers']
    battery_power = results['battery_action']
    charging_power = results['charging_power']
    total_energy_avail = results['total_energy_available']
    # battery_soc = results['battery_soc']
    aggregate_power = results['aggregate_power']
    chargers = results['chargers']
    battery_soc = results["battery_soc"]
    # battery_spec = results['battery_spec']
    times = problem.times

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(times / 60, charging_power, label='combined bus charging')
    plt.plot(times / 60, aggregate_power, label='power from grid', linestyle='-.')
    plt.plot(times / 60, battery_power, label='battery power')
    plt.axhline(grid_limit, linestyle='--', color='r', label='peak grid power')
    plt.title('Depot power')
    plt.xlabel('Hour of week')
    plt.ylabel('Power (kW)')
    plt.xlim([0, times[-1] / 60])
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times / 60, total_energy_avail, label="combined bus battery")
    plt.plot(times / 60, battery_soc, label = "on-site battery")
    plt.axhline(problem.final_charge * problem.num_buses * problem.bus_capacity, linestyle='--', color='k',
                label='required end bus SOC')
    plt.axhline(problem.reserve_energy, linestyle='--', color='r', label='bus reserve')
    plt.xlabel('Hour of week')
    plt.ylabel('Energy available (kWh)')
    plt.title('Battery capacities')
    plt.xlim([0, times[-1] / 60])
    plt.ylim([0, problem.num_buses * problem.bus_capacity])
    plt.legend()

    chargers_in_use = determine_charger_use(chargers,problem.Nt_avail, charging_power, problem.windows)
    r, c = chargers_in_use.shape
    plt.subplot(3, 1, 3)
    plt.plot(times / 60, chargers_in_use[:, 0])
    plt.xlabel('Hour of week')
    plt.ylabel('# chargers'.format(chargers['power'][0]))
    plt.title("Number of bus chargers in use")
    plt.xlim([0, times[-1] / 60])

    plt.tight_layout()
    plt.show()





trips_data = pd.read_csv('../data/gtfs/greater_sydney/trip_data.csv', index_col='Unnamed: 0')
trips_data = trips_data[trips_data['agency_name']=='Newcastle Transport']

# print("total routes: ", len(trips_data["route_short_name"].unique()))


test = trips_data.groupby(by=['route_short_name'])["route_short_name"].count().sort_values()

route_names = test.index[-NUM_ROUTES:]

print(route_names)

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
print("min requried buses = ", buses_in_traffic.max())

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

chargers = {'power': [150], 'number': ['optim'], 'cost':[1]}
bus.charging_rate=300
grid_limit='optim'

if scenario==1: # SCENARIO 1: all day charging and No battery
    battery = None
    problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.9,
                                  deadhead=DEADHEAD,resolution=RESOLUTION, min_charge_time=MIN_CHARGE_TIME, reserve=RESERVE,
                                  battery=battery)

    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2 - t1))


elif scenario==2: # SCENARIO 2: all day charging and 10MWh battery
    battery = {'power':2500, 'capacity':10000, 'efficiency':0.95}
    problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.9,
                                  deadhead=DEADHEAD,resolution=RESOLUTION, min_charge_time=MIN_CHARGE_TIME, reserve=RESERVE,
                                  battery=battery)

    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2 - t1))




elif scenario==3: # SCENARIO 3:  night charging only and no battery
    windows = [1]*6 + [0]*12 + [1]*6  # allowed charging windows
    battery = None
    problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.9,
                                  deadhead=DEADHEAD,resolution=RESOLUTION, min_charge_time=MIN_CHARGE_TIME, reserve=RESERVE,
                                  battery=battery, windows=windows, num_buses=150)

    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2 - t1))


elif scenario==4: # SCENARIO 4:  night charging only and 10MWh battery
    windows = [1]*6 + [0]*12 + [1]*6  # allowed charging windows
    # windows = [1]*7 + [0]*3 + [1]*5 + [0]*5 + [1]*4 # allowed charging windows
    battery = {'power':2500, 'capacity':10000, 'efficiency':0.95}
    problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.9,
                                  deadhead=DEADHEAD,resolution=RESOLUTION, min_charge_time=MIN_CHARGE_TIME, reserve=RESERVE,
                                  battery=battery, windows=windows, num_buses=150)

    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2 - t1))


elif scenario==5: # SCENARIO 5:  night charging only and no battery
    windows = [1]*7 + [0]*3 + [1]*5 + [0]*5 + [1]*4 # allowed charging windows
    battery = None
    problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.9,
                                  deadhead=DEADHEAD,resolution=RESOLUTION, min_charge_time=MIN_CHARGE_TIME, reserve=RESERVE,
                                  battery=battery, windows=windows, num_buses=93)

    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2 - t1))


elif scenario==6: # SCENARIO 6:  offpeak charging only and 10MWh battery
    windows = [1]*7 + [0]*3 + [1]*5 + [0]*5 + [1]*4 # allowed charging windows
    battery = {'power':2500, 'capacity':10000, 'efficiency':0.95}
    problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.9,
                                  deadhead=DEADHEAD,resolution=RESOLUTION, min_charge_time=MIN_CHARGE_TIME, reserve=RESERVE,
                                  battery=battery, windows=windows, num_buses=93)

    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2 - t1))




plot_study_results(results, problem)
print("peak grid power ", results['grid_limit'])
chargers = results['chargers']
print("number of chargers ", chargers["number"][0])