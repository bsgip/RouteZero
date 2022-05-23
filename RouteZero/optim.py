import numpy as np
from scipy.ndimage import minimum_filter1d
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from RouteZero.route import calc_buses_in_traffic

"""

            Functions for performing the depot charging feasibility calculations

"""

# todo: add in battery efficiency
# todo: add in battery regularisation

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
            chargers = {'power':300,'number':'optim'}

        # battery = {'power':500, "capacity":2000,"efficiency":0.9}

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

        infeasibilty = np.max(np.abs(np.minimum(0, energy_available)))

        if infeasibilty > 0:
            print('charging not feasible')


        results = {"charging_power":x_array,
                   "total_energy_available":energy_available,
                   "final_soc_infeas": self.pyo_model.end_slack.value,
                   "final_soc_infeas_%": self.pyo_model.end_slack.value/(self.num_buses*self.bus_capacity)*100,
                   "grid_limit":self.pyo_model.G.value,
                   "status":status,
                   "reserve_infeas":self.pyo_model.reserve_slack.value,
                   "reserve_infease_%":self.pyo_model.reserve_slack.value/(self.num_buses*self.bus_capacity)*100,
                   "infeasibility":infeasibilty,
                   "infeasibility_%":infeasibilty/(self.num_buses*self.bus_capacity)*100}



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

    def summarise_daily(self):
        """
        returns a summary of the daily amount of energy charged and used
        """
        energy_used = self.depart_trip_energy_req
        used_daily = np.histogram(self.times / 60, bins=np.arange(0, 24 * 8, 24), weights=energy_used)[0]
        charged_daily = np.histogram(self.times / 60, bins=np.arange(0, 24 * 8, 24), weights=self._p2e(self.get_array('x')))[0]
        return used_daily, charged_daily

    def base_objective(self, model):
        return model.end_slack * 1e10 + model.reserve_slack * 1e10  \
               + sum(model.reg[t] for t in model.Tminus)  + sum(model.x[t] for t in model.T) \
               + sum(model.bx[t] for t in model.T)

    def add_chargers(self, model, chargers):
        if chargers['number']=='optim':
            model.Nc = pyo.Var(domain=pyo.Integers, bounds=(1, None))       # number of chargers
        else:
            model.Nc = pyo.Param(initialize=chargers['number'])

    def _build_base_pyo(self):
        """
        Builds the common parts of the pyomo model
        """
        model = pyo.ConcreteModel()
        model.T = range(self.num_times)
        model.Tminus = range(self.num_times-1)
        model.x = pyo.Var(model.T, domain=pyo.NonNegativeReals)         # depot charging power (kW)
        model.end_slack = pyo.Var(domain=pyo.NonNegativeReals)
        # model.empty_slack = pyo.Var(domain=pyo.NonNegativeReals)
        model.reserve_slack = pyo.Var(domain=pyo.NonNegativeReals)
        model.reg = pyo.Var(model.Tminus, domain=pyo.NonNegativeReals)
        if self.grid_limit=='optim':
            model.G = pyo.Var(domain=pyo.Reals, bounds=(1, None))           # depot connection power (kW)
        else:
            model.G = pyo.Param(initialize=self.grid_limit)
        if self.battery is None:
            model.bx = pyo.Param(model.T, initialize=0.)
            model.bcap = pyo.Param(initialize=0.)
            model.bpower = pyo.Param(initialize=0.)

        self.add_chargers(model, self.chargers)


        # maximum charging
        model.max_charge = pyo.ConstraintList()
        for t in range(self.num_times):
            model.max_charge.add(model.x[t] <= self.Nt_avail[t]*np.minimum(self.chargers['power'],self.bus.charging_rate))
            model.max_charge.add(model.x[t] <= model.Nc*np.minimum(self.chargers['power'],self.bus.charging_rate))
            model.max_charge.add(model.x[t]+model.bx[t] <= model.G)
            if self.battery is not None:
                model.max_charge.add(model.bx[t] <= model.bpower)

        model.reg_con = pyo.ConstraintList()
        for t in range(self.num_times-1):
            model.reg_con.add(model.reg[t] >= model.x[t+1] - model.x[t])
            model.reg_con.add(model.reg[t] >= -(model.x[t + 1] - model.x[t]))


        if self.battery is not None:
            model.b_cap_con = pyo.ConstraintList()
            for t in range(self.num_times):
                model.b_cap_con.add(sum(model.bx[k] for k in range(t)) >= 0)
                model.b_cap_con.add(sum(model.bx[k] for k in range(t)) <= 0)


        # maximim charge limit
        model.max_charged = pyo.ConstraintList()
        for t in range(0, self.num_times):
            if t==0:
                model.max_charged.add(self._p2e(model.x[0]) <= ((1-self.start_charge)*self.num_buses*self.bus_capacity))
            else:
                model.max_charged.add(sum(self._p2e(model.x[i]) for i in range(t+1)) <=
                                      ((1-self.start_charge)*self.num_buses*self.bus_capacity) + sum(self.return_trip_energy_cons[k] for k in range(t)))

        # minimum charge limit
        model.min_charged = pyo.ConstraintList()
        for t in range(1, self.num_times):
           #  will be infeasible if starting charge not enough
            model.min_charged.add(self.start_charge*self.num_buses*self.bus_capacity - sum(self.depart_trip_energy_req[k] for k in range(t+1))
                                  + sum(self._p2e(model.x[i]) for i in range(t)) + model.reserve_slack >= self.reserve_energy)
            # model.min_charged.add(self.start_charge * self.num_buses * self.bus_capacity
            #                       - sum(self.depart_trip_energy_req[k] for k in range(t + 1)) + sum(self._p2e(model.x[i]) for i in range(t))
            #                       + model.empty_slack >= 0.)

        # enforce battery finishes at least 80% charged
        model.end_constraint = pyo.Constraint(expr=self.start_charge*self.num_buses*self.bus_capacity
                                                   + sum(self._p2e(model.x[t])-self.depart_trip_energy_req[t] for t in model.T)
                                                   + model.end_slack>=self.final_charge*self.num_buses*self.bus_capacity)

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
    def __init__(self, trips_data, trips_ec, bus, chargers, grid_limit, start_charge=0.9, final_charge=0.8, deadhead=0.1,
                 resolution=10, min_charge_time=60, reserve=0.2):
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
        super().__init__(trips_data, trips_ec, bus, chargers, grid_limit, deadhead=deadhead, resolution=resolution,reserve=reserve,
                         min_charge_time=min_charge_time, start_charge=start_charge, final_charge=final_charge)


    def _build_pyomo(self):
        """
            Builds the pyomo model
        """

        model = self._build_base_pyo()

        model.obj = pyo.Objective(expr=self.base_objective(model)-100*sum(sum(model.x[i] for i in range(t)) for t in model.T), sense=pyo.minimize)


        return model



class Feasibility_problem(base_problem):
    def __init__(self, trips_data, trips_ec, bus, charger_max_power, start_charge=0.9, final_charge=0.8, deadhead=0.1,
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
        super().__init__(trips_data, trips_ec, bus, chargers, grid_limit='optim', deadhead=deadhead, resolution=resolution,reserve=reserve,
                         min_charge_time=min_charge_time, start_charge=start_charge, final_charge=final_charge)
        self.Q = Q
        self.R = R



    def _build_pyomo(self):
        """
            Builds the pyomo model
        """
        model = self._build_base_pyo()

        model.obj = pyo.Objective(expr=model.G * self.Q + model.Nc * self.R + self.base_objective(model), sense=pyo.minimize)


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
    ec_km, ec_total = model.predict_worst_temp(trips_data, bus)

    deadhead = 0.1              # percent [0-1]
    resolution = 10             # mins
    charger_max_power = 150     # kW
    min_charge_time = 2*60     # mins
    reserve = 0.2          # percent of all battery to keep in reserve [0-1]

    problem = Feasibility_problem(trips_data, ec_total, bus, charger_max_power, start_charge=0.9, final_charge=0.8,
                                  deadhead=deadhead,resolution=resolution, min_charge_time=min_charge_time, reserve=reserve)

    # chargers = {'power':300, 'number':80}
    # grid_power = 5000
    # problem = Immediate_charge_problem(trips_data, ec_total, bus, chargers, grid_power, start_charge=0.9, final_charge=0.8,
    #                               deadhead=deadhead,resolution=resolution, min_charge_time=min_charge_time, reserve=reserve)


    t1 = time.time()
    results = problem.solve()
    t2 = time.time()

    print('Solve took {} seconds'.format(t2-t1))

    grid_limit = results['grid_limit']
    # num_chargers = results['min_number_chargers']
    charging_power = results['charging_power']
    total_energy_avail = results['total_energy_available']
    times = problem.times

    plt.subplot(3, 1, 1)
    plt.plot(times/60, problem.buses_at_depot)
    plt.title('Number of buses at depot')
    plt.xlabel('Hour of week')
    plt.ylabel('# buses')
    plt.xlim([0, times[-1]/60])

    plt.subplot(3, 1, 2)
    plt.plot(times/60, charging_power)
    plt.axhline(grid_limit, linestyle='--', color='r',label='max grid power')
    plt.title('Grid power needed for charging')
    plt.xlabel('Hour of week')
    plt.ylabel('Power (kW)')
    plt.xlim([0, times[-1] / 60])
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times/60, total_energy_avail)
    plt.axhline(problem.final_charge*problem.num_buses*problem.bus_capacity, linestyle='--',color='k',label='required end energy')
    plt.axhline(problem.reserve_energy, linestyle='--', color='r', label='reserve')
    plt.xlabel('Hour of week')
    plt.ylabel('Energy available (kWh)')
    plt.title('Total battery energy available at depot')
    plt.xlim([0, times[-1] / 60])
    plt.ylim([0, problem.num_buses*problem.bus_capacity])
    plt.legend()

    plt.tight_layout()
    plt.show()

    used_daily, charged_daily = problem.summarise_daily()

    fig = plt.figure()
    X = np.arange(1, 8)
    plt.bar(X + 0.00, used_daily/1000, color='orange', width=0.3, label='Used')
    plt.bar(X + 0.3, charged_daily/1000, color='g', width=0.3, label='Charged')
    plt.ylabel('Energy (MWh)')
    plt.title('Daily summary')
    plt.xlabel('day')
    plt.legend()
    plt.show()










