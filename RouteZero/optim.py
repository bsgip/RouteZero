import numpy as np
from scipy.ndimage import minimum_filter1d
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from RouteZero.route import calc_buses_in_traffic

"""

            Functions for performing the depot charging feasibility calculations

"""


# todo charging constraitns when multiple sets of batteries
# check battery charge is correct

class base_problem():
    def __init__(self, trips_data, trips_ec, bus, chargers, grid_limit, deadhead=0.1, resolution=10, num_buses=None,
                 min_charge_time=60, start_charge=0.9, final_charge=0.8, reserve=0.2, battery=None):
        times, buses_in_traffic, depart_trip_energy_req, return_trip_energy_cons = calc_buses_in_traffic(trips_data,
                                                                                                         deadhead,
                                                                                                         resolution,
                                                                                                         trips_ec)
        # if num_buses not specified calculate minimum feasible number
        if num_buses is None:
            num_buses = buses_in_traffic.max()
        else:
            assert num_buses > buses_in_traffic, 'num_buses must be greater than max(buses_in_traffic)'
        # calculate number of buses at the depot
        buses_at_depot = num_buses - buses_in_traffic

        # calculate number of buses available to charge
        Nt_avail = minimum_filter1d(buses_at_depot, int(np.ceil(min_charge_time / resolution)))

        if chargers is None:
            chargers = {'power': 300, 'number': 'optim'}


        self.deadhead = deadhead
        self.resolution = resolution
        self.trips_data = trips_data
        self.bus = bus
        self.min_charge_time = min_charge_time
        self.grid_limit = grid_limit

        self.num_buses = num_buses
        self.buses_at_depot = buses_at_depot
        self.depart_trip_energy_req = depart_trip_energy_req
        self.return_trip_energy_cons = return_trip_energy_cons
        self.bus_capacity = bus.battery_capacity
        self.kw_to_kwh = resolution / 60
        self.chargers = chargers
        self.Nt_avail = Nt_avail
        self.end_time = times.max()
        self.num_times = len(times)
        self.times = times
        self.start_charge = start_charge
        self.final_charge = final_charge
        self.reserve = reserve
        self.reserve_energy = reserve * num_buses * bus.battery_capacity
        self.bus_eta = bus.charging_efficiency
        self.battery = battery

    def _solve(self, model):
        """
        Solve pyomo model
        """
        opt = SolverFactory('cbc')
        status = opt.solve(model)

        if model.reserve_slack.value > 0:
            print('Reserve not achieved')

        x_array = self.get_array('x')

        energy_available = self.start_charge * self.bus_capacity * self.num_buses \
                           + np.cumsum(self._p2e(x_array)) - np.cumsum(self.return_trip_energy_cons)

        bx_array = self.get_array('bx')

        agg_power = bx_array + x_array

        bv_array = self.get_array('bv')
        battery_soc = np.cumsum(self.kw_to_kwh * bv_array)
        if self.battery is not None:
            beff = self.battery['efficiency']
            battery_soc_test = np.cumsum(np.maximum(bx_array, 0) * beff * self.kw_to_kwh) + np.cumsum(
                np.minimum(0, bx_array) / beff * self.kw_to_kwh)
            diff = battery_soc - battery_soc_test
            if np.abs(diff).max() > 1e-1:
                print('Warning: battery SOC might be innacurate')
                battery_soc = battery_soc_test

        infeasibilty = np.max(np.abs(np.minimum(0, energy_available)))

        if infeasibilty > 0:
            print('charging not feasible')

        self.chargers['number'] = self.get_array('Nc')

        if self.battery is not None:
            self.battery['capacity'] = self.pyo_model.bcap.value

        results = {"charging_power": x_array,
                   "total_energy_available": energy_available,
                   "final_soc_infeas": self.pyo_model.end_slack.value,
                   "final_soc_infeas_%": self.pyo_model.end_slack.value / (self.num_buses * self.bus_capacity) * 100,
                   "grid_limit": self.pyo_model.G.value,
                   "status": status,
                   "reserve_infeas": self.pyo_model.reserve_slack.value,
                   "reserve_infease_%": self.pyo_model.reserve_slack.value / (self.num_buses * self.bus_capacity) * 100,
                   "infeasibility": infeasibilty,
                   "infeasibility_%": infeasibilty / (self.num_buses * self.bus_capacity) * 100,
                   "chargers": self.chargers,
                   "battery_action": bx_array,
                   "aggregate_power": agg_power,
                   "battery_soc": battery_soc,
                   "battery_soc_delta": bv_array,
                   "battery_spec":self.battery}

        return results

    def _p2e(self, x):
        """
        Converts charging power (kW) into battery energy (kWh) for the time period defined by resolution
        """
        return x * self.kw_to_kwh * self.bus_eta

    def get_array(self, variable):
        """
        gets an array result from the solved pyomo model, variable should be the name of the pyomo model var
        """
        pyo_var = getattr(self.pyo_model, variable)
        values = np.zeros((len(pyo_var),))
        for i in range(len(pyo_var)):
            if isinstance(pyo_var[i], float):
                values[i] = pyo_var[i]
            else:
                values[i] = pyo_var[i].value
        return values

    def summarise_daily(self):
        """
        returns a summary of the daily amount of energy charged and used
        """
        energy_used = self.depart_trip_energy_req
        used_daily = np.histogram(self.times / 60, bins=np.arange(0, 24 * 8, 24), weights=energy_used)[0]
        charged_daily = \
        np.histogram(self.times / 60, bins=np.arange(0, 24 * 8, 24), weights=self._p2e(self.get_array('x')))[0]
        return used_daily, charged_daily

    def base_objective(self, model):
        return model.end_slack * 1e10 + model.reserve_slack * 1e10 \
               + sum(model.reg[t] for t in model.Tminus) + sum(model.x[t] for t in model.T) \
               + sum(model.bx[t]*0.1 for t in model.T)

    def add_chargers(self, model):
        number = self.chargers['number']
        power = self.chargers['power']

        if not isinstance(number, list):
            number = [number]
            self.chargers['number'] = number
        if not isinstance(power, list):
            power = [power, ]
            self.chargers['power'] = power


        assert len(number) == len(power), 'must specify the same number of powers and numbers'
        sets = len(number)
        inds = np.argsort(-np.array(power))  # sort descending
        power = np.array(power)[inds].tolist()
        number = np.array(number)[inds].tolist()
        self.chargers['power'] = power
        self.chargers['number'] = number
        if 'cost' in self.chargers:
            if not isinstance(self.chargers['cost'],list):
                self.chargers['cost'] = [self.chargers['cost']]
            self.chargers['cost'] = np.array(self.chargers['cost'])[inds].tolist()
        model.charger_sets = range(sets)
        model.Nc = pyo.Var(model.charger_sets, domain=pyo.Integers, bounds=(0, None))
        for i in range(sets):
            if number[i] != 'optim':
                model.Nc[i].fix(int(number[i]))

    def _build_base_pyo(self):
        """
        Builds the common parts of the pyomo model
        """
        model = pyo.ConcreteModel()
        model.T = range(self.num_times)
        model.Tminus = range(self.num_times - 1)
        model.x = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # depot charging power (kW)
        model.end_slack = pyo.Var(domain=pyo.NonNegativeReals)
        model.reserve_slack = pyo.Var(domain=pyo.NonNegativeReals)
        model.reg = pyo.Var(model.Tminus, domain=pyo.NonNegativeReals)
        if self.grid_limit == 'optim':
            model.G = pyo.Var(domain=pyo.Reals, bounds=(1, None))  # depot connection power (kW)
        else:
            model.G = pyo.Param(initialize=self.grid_limit)
        if self.battery is None:
            model.bx = pyo.Param(model.T, initialize=0.)
            model.bv = pyo.Param(model.T, initialize=0.)
        else:
            model.bx = pyo.Var(model.T, domain=pyo.Reals, bounds=(-self.battery['power'], self.battery['power']))
            if self.battery['capacity']=='optim':
                model.bcap = pyo.Var(domain=pyo.NonNegativeReals)
            else:
                model.bcap = pyo.Param(initialize=self.battery['capacity'])

            model.bv = pyo.Var(model.T, domain=pyo.Reals)  # auxilliary variable for after charging efficiencies are applied
            model.beff = pyo.Param(initialize=self.battery['efficiency'])

        # self.add_chargers(model, self.chargers)
        self.add_chargers(model)

        # maximum charging
        model.max_charge = pyo.ConstraintList()
        for t in range(self.num_times):
            model.max_charge.add(
                model.x[t] <= self.Nt_avail[t] * np.minimum(self.chargers['power'][0], self.bus.charging_rate))
            model.max_charge.add(
                model.x[t] <= sum(model.Nc[i] * np.minimum(self.chargers['power'][i], self.bus.charging_rate)
                                  for i in model.charger_sets))
            model.max_charge.add(model.x[t] + model.bx[t] <= model.G)
            # combined available buses and chargers constraints
            for i in model.charger_sets:
                model.max_charge.add(model.x[t] <= self.Nt_avail[t] * self.chargers['power'][i]
                    + sum(self.chargers['power'][j] * model.Nc[j] for j in range(i))
                    - self.chargers['power'][i] * sum(model.Nc[j] for j in range(i)))

            # model.max_charge

        model.reg_con = pyo.ConstraintList()
        for t in range(self.num_times - 1):
            model.reg_con.add(model.reg[t] >= model.x[t + 1] - model.x[t])
            model.reg_con.add(model.reg[t] >= -(model.x[t + 1] - model.x[t]))

        # maximim charge limit
        model.max_charged = pyo.ConstraintList()
        for t in range(0, self.num_times):
            if t == 0:
                model.max_charged.add(
                    self._p2e(model.x[0]) <= ((1 - self.start_charge) * self.num_buses * self.bus_capacity))
            else:
                model.max_charged.add(sum(self._p2e(model.x[i]) for i in range(t + 1)) <=
                                      ((1 - self.start_charge) * self.num_buses * self.bus_capacity) + sum(
                    self.return_trip_energy_cons[k] for k in range(t)))

        # minimum charge limit
        model.min_charged = pyo.ConstraintList()
        for t in range(1, self.num_times):
            #  will be infeasible if starting charge not enough
            model.min_charged.add(self.start_charge * self.num_buses * self.bus_capacity - sum(
                self.depart_trip_energy_req[k] for k in range(t + 1))
                                  + sum(
                self._p2e(model.x[i]) for i in range(t)) + model.reserve_slack >= self.reserve_energy)


        # enforce aggregate bus battery finishes at least 80% charged
        model.end_constraint = pyo.Constraint(expr=self.start_charge * self.num_buses * self.bus_capacity
                                                   + sum(
            self._p2e(model.x[t]) - self.depart_trip_energy_req[t] for t in model.T)
                                                   + model.end_slack >= self.final_charge * self.num_buses * self.bus_capacity)

        # depot battery stuff
        if self.battery is not None:
            model.v_discharge_con = pyo.ConstraintList()
            model.v_charge_con = pyo.ConstraintList()

            for t in model.T:
                model.v_charge_con.add(model.bv[t] <= model.beff * model.bx[t])
                model.v_discharge_con.add(model.bv[t] <= model.bx[t] / model.beff)

            model.bsoc_min = pyo.ConstraintList()
            model.bsoc_max = pyo.ConstraintList()
            for t in range(1, self.num_times + 1):  # assumes battery starts empty
                model.bsoc_min.add(0.0 + sum(model.bv[i] for i in range(t)) * self.kw_to_kwh >= 0.)
                model.bsoc_max.add(0.0 + sum(model.bv[i] for i in range(t)) * self.kw_to_kwh <= model.bcap)

        return model

    def solve(self):
        """
        Solve pyomo model and return the results
        """
        self.pyo_model = self._build_pyomo()
        results = self._solve(self.pyo_model)

        return results

    def _build_pyomo(self):
        " to be overriden by each child class"
        return None


class Immediate_charge_problem(base_problem):
    def __init__(self, trips_data, trips_ec, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.8,
                 deadhead=0.1, resolution=10, min_charge_time=60, reserve=0.2, battery=None):
        """
        Solves the charging problem where we would always like to charge as much as possible as soon as possible
        :param trips_data: the summarised trips dataframe
        :param trips_ec: the energy consumed on each trip, must have same length as trips_data
        :param bus: bus object specifying bus parameters
        :param chargers: dictionary containing powers and numbers
        :param deadhead: (default=0.1) ratio [0->1] of route time that is spent between routes
        :param resolution: (default=10) resolution of binning in minutes
        :param min_charge_time: (default=60) minimum time to charge a bus for in minutes
        :param grid_limit: max power from grid (kW)
        :param start_charge: percentage starting battery capacity [0-1]
        :param final_charge: required final state of charge percentage [0-1]
        """
        super().__init__(trips_data, trips_ec, bus, chargers, grid_limit, deadhead=deadhead, resolution=resolution,
                         reserve=reserve,min_charge_time=min_charge_time, start_charge=start_charge,
                         final_charge=final_charge, battery=battery)

    def _build_pyomo(self):
        """
            Builds the pyomo model
        """

        model = self._build_base_pyo()

        model.obj = pyo.Objective(
            expr=self.base_objective(model) - 100 * sum(sum(model.x[i] for i in range(t)) for t in model.T),
            sense=pyo.minimize)

        return model


class Feasibility_problem(base_problem):
    def __init__(self, trips_data, trips_ec, bus, charger_max_power, start_charge=0.9, final_charge=0.8, deadhead=0.1,
                 resolution=10, min_charge_time=60, Q=1000, R=100, reserve=0.2, battery=None):
        """

        :param trips_data: the summarised trips dataframe
        :param trips_ec: the energy consumed on each trip, must have same length as trips_data
        :param bus: bus object specifying bus parameters
        :param charger_max_power: maximum charger power (kW)
        :param deadhead: (default=0.1) ratio [0->1] of route time that is spent between routes
        :param resolution: (default=10) resolution of binning in minutes
        :param min_charge_time: (default=60) minimum time to charge a bus for in minutes
        :param Q: (default=100) cost on connection capacity (should be higher than cost on chargers)
        :param R: (default=10) cost on number of chargers
        :param start_charge: percentage starting battery capacity [0-1]
        :param final_charge: required final state of charge percentage [0-1]
        :param battery: dictionary containing battery specifications: (capacity, efficiency, power)
        """
        chargers = {'power': charger_max_power, 'number': 'optim', 'cost':R}
        super().__init__(trips_data, trips_ec, bus, chargers, grid_limit='optim', deadhead=deadhead,
                         resolution=resolution, reserve=reserve,
                         min_charge_time=min_charge_time, start_charge=start_charge, final_charge=final_charge,
                         battery=battery)
        self.Q = Q


    def _build_pyomo(self):
        """
            Builds the pyomo model
        """
        model = self._build_base_pyo()

        model.obj = pyo.Objective(expr=model.G * self.Q + model.Nc[0] * self.chargers['cost'][0] + self.base_objective(model),
                                  sense=pyo.minimize)

        return model

class Extended_feas_problem(base_problem):
    def __init__(self, trips_data, trips_ec, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.8, deadhead=0.1,
                 resolution=10, min_charge_time=60, Q=5000, reserve=0.2, battery=None):
        """
        Minimise the grid power limit, and the number of chargers in each set.
        :param trips_data: the summarised trips dataframe
        :param grid_limit: max power available from grid or 'optim'
        :param trips_ec: the energy consumed on each trip, must have same length as trips_data
        :param bus: bus object specifying bus parameters
        :param charger_max_power: dictionary with fields ('power', 'number') which are lists of the same length,
                                number[i] is either the number of chargers with power[i] or 'optim' to optimise
        :param deadhead: (default=0.1) ratio [0->1] of route time that is spent between routes
        :param resolution: (default=10) resolution of binning in minutes
        :param min_charge_time: (default=60) minimum time to charge a bus for in minutes
        :param Q: (default=100) cost on connection capacity (should be higher than cost on chargers)
        :param start_charge: percentage starting battery capacity [0-1]
        :param final_charge: required final state of charge percentage [0-1]
        :param battery: dictionary containing battery specifications: (capacity, efficiency, power)
        """
        super().__init__(trips_data, trips_ec, bus, chargers, grid_limit=grid_limit, deadhead=deadhead,
                         resolution=resolution, reserve=reserve,
                         min_charge_time=min_charge_time, start_charge=start_charge, final_charge=final_charge,
                         battery=battery)
        self.Q = Q

    def _build_pyomo(self):
        """
            Builds the pyomo model
        """
        model = self._build_base_pyo()
        model.obj = pyo.Objective(expr=model.G * self.Q
                                       + sum(model.Nc[i] * self.chargers['cost'][i] for i in model.charger_sets)
                                       + self.base_objective(model), sense=pyo.minimize)


        return model


class Battery_spec_problem(base_problem):
    def __init__(self, trips_data, trips_ec, bus, chargers, battery_power, battery_efficiency=0.95, start_charge=0.9, final_charge=0.8, deadhead=0.1,
                 resolution=10, min_charge_time=60, Q=1000, reserve=0.2):
        """
        Attempts to find the battery capacity that allows the grid limit to be reduced
        :param trips_data: the summarised trips dataframe
        :param trips_ec: the energy consumed on each trip, must have same length as trips_data
        :param bus: bus object specifying bus parameters
        :param chargers: chargers dictionary
        :param deadhead: (default=0.1) ratio [0->1] of route time that is spent between routes
        :param resolution: (default=10) resolution of binning in minutes
        :param min_charge_time: (default=60) minimum time to charge a bus for in minutes
        :param Q: (default=100) cost on connection capacity (should be higher than cost on chargers)
        :param R: (default=10) cost on number of chargers
        :param start_charge: percentage starting battery capacity [0-1]
        :param final_charge: required final state of charge percentage [0-1]
        :param battery_power: battery power rating (kW)
        :param battery_efficiency: one way battery efficiency applied to charge and discharge (default 0.95)
        """

        battery = {'capacity':'optim', 'power':battery_power, 'efficiency':battery_efficiency}
        super().__init__(trips_data, trips_ec, bus, chargers, grid_limit='optim', deadhead=deadhead,
                         resolution=resolution, reserve=reserve,
                         min_charge_time=min_charge_time, start_charge=start_charge, final_charge=final_charge,
                         battery=battery)
        self.Q = Q


    def _build_pyomo(self):
        """
            Builds the pyomo model
        """
        model = self._build_base_pyo()

        model.obj = pyo.Objective(expr=model.G * self.Q + model.bcap*10 + self.base_objective(model),
                                  sense=pyo.minimize)

        return model



if __name__ == "__main__":
    import pandas as pd
    import RouteZero.bus as ebus
    import matplotlib.pyplot as plt
    from RouteZero.models import LinearRegressionAbdelatyModel
    import time

    # load saved leichhardt summary data
    # trips_data = pd.read_csv('../data/test_trip_summary.csv')
    trips_data = pd.read_csv('../data/trip_data_leichhardt.csv')
    trips_data['passengers'] = 38
    bus = ebus.BYD()
    model = LinearRegressionAbdelatyModel()
    ec_km, ec_total = model.predict_worst_temp(trips_data, bus)

    deadhead = 0.1  # percent [0-1]
    resolution = 10  # mins
    charger_max_power = 150  # kW
    min_charge_time = 1 * 60  # mins
    reserve = 0.2  # percent of all battery to keep in reserve [0-1]

    # battery = {'power':1000, 'capacity':4000, 'efficiency':0.95}
    battery = None
    # problem = Feasibility_problem(trips_data, ec_total, bus, charger_max_power, start_charge=0.9, final_charge=0.8,
    #                               deadhead=deadhead,resolution=resolution, min_charge_time=min_charge_time, reserve=reserve,
    #                               battery=battery)

    # chargers = {'power': [50, 150], 'number': [20, 40]}
    # bus.charging_rate = 200
    # grid_power = 5000
    # battery = {'power': 1000, 'capacity': 4000, 'efficiency': 0.95}
    # problem = Immediate_charge_problem(trips_data, ec_total, bus, chargers, grid_power, start_charge=0.9,
    #                                    final_charge=0.8,
    #                                    deadhead=deadhead, resolution=resolution, min_charge_time=min_charge_time,
    #                                    reserve=reserve, battery=battery)

    chargers = {'power': [50, 150], 'number': [20, 60]}
    problem = Battery_spec_problem(trips_data, ec_total, bus, chargers, battery_power=5000, start_charge=0.9, final_charge=0.8,
                                  deadhead=deadhead,resolution=resolution, min_charge_time=min_charge_time, reserve=reserve)

    # chargers = {'power': [40, 80, 150], 'number': ['optim', 'optim', 'optim'], 'cost':[10, 50, 100]}
    # bus.charging_rate=300
    # grid_limit=4000
    # battery = {'power':1000, 'capacity':4000, 'efficiency':0.95}
    # problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.8,
    #                               deadhead=deadhead,resolution=resolution, min_charge_time=min_charge_time, reserve=reserve,
    #                               battery=battery)

    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2 - t1))

    grid_limit = results['grid_limit']
    optim_chargers = results['chargers']
    battery_power = results['battery_action']
    charging_power = results['charging_power']
    total_energy_avail = results['total_energy_available']
    battery_soc = results['battery_soc']
    aggregate_power = results['aggregate_power']
    battery_spec = results['battery_spec']
    times = problem.times

    plt.subplot(3, 1, 1)
    plt.plot(times / 60, problem.buses_at_depot)
    plt.title('Number of buses at depot')
    plt.xlabel('Hour of week')
    plt.ylabel('# buses')
    plt.xlim([0, times[-1] / 60])

    plt.subplot(3, 1, 2)
    plt.plot(times / 60, charging_power, label='bus')
    plt.plot(times / 60, aggregate_power, label='aggregate')
    plt.plot(times / 60, battery_power, label='battery')
    plt.axhline(grid_limit, linestyle='--', color='r', label='max grid power')
    plt.title('Grid power needed for charging')
    plt.xlabel('Hour of week')
    plt.ylabel('Power (kW)')
    plt.xlim([0, times[-1] / 60])
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times / 60, total_energy_avail)
    plt.axhline(problem.final_charge * problem.num_buses * problem.bus_capacity, linestyle='--', color='k',
                label='required end energy')
    plt.axhline(problem.reserve_energy, linestyle='--', color='r', label='reserve')
    plt.xlabel('Hour of week')
    plt.ylabel('Energy available (kWh)')
    plt.title('Total battery energy available at depot')
    plt.xlim([0, times[-1] / 60])
    plt.ylim([0, problem.num_buses * problem.bus_capacity])
    plt.legend()

    plt.tight_layout()
    plt.show()

    used_daily, charged_daily = problem.summarise_daily()

    fig = plt.figure()
    X = np.arange(1, 8)
    plt.bar(X + 0.00, used_daily / 1000, color='orange', width=0.3, label='Used')
    plt.bar(X + 0.3, charged_daily / 1000, color='g', width=0.3, label='Charged')
    plt.ylabel('Energy (MWh)')
    plt.title('Daily summary')
    plt.xlabel('day')
    plt.legend()
    plt.show()
