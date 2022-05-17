import numpy as np
from scipy.ndimage import minimum_filter1d
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from RouteZero.route import calc_buses_in_traffic

"""

            Functions for performing the depot charging feasibility calculations

"""

class base_problem():
    def __init__(self, trips_data, trips_ec, bus, chargers, deadhead=0.1, resolution=10, num_buses=None,
                 min_charge_time=60, start_charge=0.9, final_charge=0.8, reserve=0.2):
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
            chargers = {'power':300,'number':'optim'}

        self.deadhead = deadhead
        self.resolution = resolution
        self.trips_data = trips_data
        self.bus = bus
        self.min_charge_time = min_charge_time

        self.num_buses = num_buses
        self.buses_at_depot = buses_at_depot
        self.depart_trip_energy_req = depart_trip_energy_req
        self.return_trip_energy_cons = return_trip_energy_cons
        self.bus_capacity = bus.battery_capacity
        self.kw_to_kwh = resolution/60
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

    def _solve(self, model):
        """
        Solve pyomo model
        """
        opt = SolverFactory('cbc')
        results = opt.solve(model)
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
            values[i] = pyo_var[i].value
        return values



class Feasibility_problem(base_problem):
    def __init__(self, trips_data, trips_ec, bus, charger_max_power,start_charge=0.9, final_charge=0.8, deadhead=0.1,
                 resolution=10, min_charge_time=60, Q=1000, R=100, reserve=0.2):
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
        """
        chargers = {'power':charger_max_power,'number':'optim'}
        super().__init__(trips_data, trips_ec, bus, chargers, deadhead=deadhead, resolution=resolution,reserve=reserve,
                         min_charge_time=min_charge_time, start_charge=start_charge, final_charge=final_charge)
        self.Q = Q
        self.R = R
        ## define some things

    def solve(self):
        """
        Solve pyomo model and return the results
        """
        self.pyo_model = self._build_pyomo()
        status = self._solve(self.pyo_model)

        x_array = self.get_array('x')

        energy_available = self.start_charge * self.bus_capacity * self.num_buses \
                           + np.cumsum(self._p2e(x_array)) - np.cumsum(self.return_trip_energy_cons)

        results = {"min_depot_connection":self.pyo_model.G.value, "min_number_chargers":self.pyo_model.Nc.value,
                   "final_charge_infeasibility":self.pyo_model.slack.value, "charging_power":x_array,
                   "total_energy_available":energy_available,
                   "final_charge_infeas_percent": self.pyo_model.slack.value/(self.num_buses*self.bus_capacity)*100}

        return results


    def _build_pyomo(self):
        """
            Builds the pyomo model
        """
        ER = self.return_trip_energy_cons
        ED = self.depart_trip_energy_req

        model = pyo.ConcreteModel()
        model.T = range(self.num_times)
        model.Tminus = range(self.num_times-1)
        model.x = pyo.Var(model.T, domain=pyo.NonNegativeReals)         # depot charging power (kW)
        model.Nc = pyo.Var(domain=pyo.Integers, bounds=(1, None))       # number of chargers
        model.G = pyo.Var(domain=pyo.Reals, bounds=(1, None))           # depot connection power (kW)
        model.slack = pyo.Var(domain=pyo.NonNegativeReals)
        model.reg = pyo.Var(model.Tminus, domain=pyo.NonNegativeReals)

        # model.obj = pyo.Objective(expr=model.G * self.Q + model.Nc * self.R + model.slack * 1e10, sense=pyo.minimize)
        # model.obj = pyo.Objective(expr=model.G * self.Q + model.Nc * self.R + model.slack * 1e10
        #                           + sum(model.reg[t] for t in model.Tminus)/100
        #                           + sum(model.reg2[t]*100 for t in model.T), sense=pyo.minimize)
        model.obj = pyo.Objective(expr=model.G * self.Q + model.Nc * self.R + model.slack * 1e10
                                  + sum(model.reg[t] for t in model.Tminus)
                                  + sum(model.x[t] for t in model.T), sense=pyo.minimize)


        # maximum charging
        model.max_charge = pyo.ConstraintList()
        for t in range(self.num_times):
            model.max_charge.add(model.x[t] <= self.Nt_avail[t]*np.minimum(self.chargers['power'],self.bus.charging_rate))
            model.max_charge.add(model.x[t] <= model.Nc*np.minimum(self.chargers['power'],self.bus.charging_rate))
            model.max_charge.add(model.x[t] <= model.G)

        model.reg_con = pyo.ConstraintList()
        for t in range(self.num_times-1):
            model.reg_con.add(model.reg[t] >= model.x[t+1] - model.x[t])
            model.reg_con.add(model.reg[t] >= -(model.x[t + 1] - model.x[t]))


        # maximim charge limit
        model.max_charged = pyo.ConstraintList()
        for t in range(0, self.num_times):
            if t==0:
                model.max_charged.add(self._p2e(model.x[0]) <= ((1-self.start_charge)*self.num_buses*self.bus_capacity))
            else:
                model.max_charged.add(sum(self._p2e(model.x[i]) for i in range(t+1)) <=
                                      ((1-self.start_charge)*self.num_buses*self.bus_capacity) + sum(ER[k] for k in range(t)))

        # minimum charge limit
        model.min_charged = pyo.ConstraintList()
        for t in range(1, self.num_times):
           #  will be infeasible if starting charge not enough
            model.min_charged.add(self.start_charge*self.num_buses*self.bus_capacity - sum(ED[k] for k in range(t+1))
                                  + sum(self._p2e(model.x[i]) for i in range(t)) >= self.reserve_energy)

        # enforce battery finishes at least 80% charged
        model.end_constraint = pyo.Constraint(expr=self.start_charge*self.num_buses*self.bus_capacity
                                                   + sum(self._p2e(model.x[t])-ED[t] for t in model.T)
                                                   + model.slack>=self.final_charge*self.num_buses*self.bus_capacity)

        return model


if __name__=="__main__":
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
    ec_km, ec_total = model.predict_hottest(trips_data, bus)

    deadhead = 0.1              # percent [0-1]
    resolution = 10             # mins
    charger_max_power = 150     # kW
    min_charge_time = 60     # mins
    reserve = 0.2          # percent of all battery to keep in reserve [0-1]

    problem = Feasibility_problem(trips_data, ec_total, bus, charger_max_power, start_charge=0.9, final_charge=0.8,
                                  deadhead=deadhead,resolution=resolution, min_charge_time=min_charge_time, reserve=reserve)

    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2-t1))

    grid_con = results['min_depot_connection']
    num_chargers = results['min_number_chargers']
    charging_power = results['charging_power']
    total_energy_avail = results['total_energy_available']
    times = problem.times

    plt.subplot(3, 1, 1)
    plt.plot(times/60, problem.buses_at_depot)
    plt.title('Number of buses at depot')
    # plt.axhline(grid_con, linestyle='--', color='r')
    plt.xlabel('Hour of week')
    plt.ylabel('# buses')
    plt.xlim([0, times[-1]/60])

    plt.subplot(3, 1, 2)
    plt.plot(times/60, charging_power)
    plt.axhline(grid_con, linestyle='--', color='r',label='max grid power')
    plt.title('Grid power needed for charging')
    plt.xlabel('Hour of week')
    plt.ylabel('Power (kW)')
    plt.xlim([0, times[-1] / 60])
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times/60, total_energy_avail)
    plt.axhline(problem.final_charge*problem.num_buses*problem.bus_capacity, linestyle='--',color='k',label='required end energy')
    # plt.axhline(problem.start_charge*problem.num_buses*problem.bus_capacity, linestyle='--',color='k')
    plt.axhline(problem.reserve_energy, linestyle='--', color='r', label='reserve')
    plt.xlabel('Hour of week')
    plt.ylabel('Energy available (kWh)')
    plt.title('Total battery energy available at depot')
    plt.xlim([0, times[-1] / 60])
    plt.ylim([0, problem.num_buses*problem.bus_capacity])
    plt.legend()

    plt.tight_layout()
    plt.show()
