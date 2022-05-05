import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from scipy.interpolate import interp1d
from weather import location_design_temp
import geopy.distance
# custom functions
from gtfs_routes import process_gtfs_routes
from ebus_energy_models import LinearRegressionAbdelatyModel
import pyomo.environ as pyEnv
# for plotting the demo example
import matplotlib.pyplot as plt



# todo: deadhead time should go on either side of a route not just at the end!!



if __name__=='__main__':
    # choose some things
    busiest = 'day'
    cutoffs = [0, 6, 9, 15, 19, 22, 24,
               30]  # optional input for splitting up the route summary information into time windows
    wrap_time = 24*60


    passenger_loading = 38

    gtfs_file = "./data/full_greater_sydney_gtfs_static.zip"        # location of the gtfs zip file
    # route_short_names = ["305", "320"]      # the short names of the routes we want to get summaries of
    route_short_names = ["305", "320", '389', '406']
    # route_short_names = ["305", "320", '389', '406',
    #                      '428', '430', '431', '433']
    # route_short_names = ["305", "320", '389', '406',
    #                      '428', '430', '431', '433',
    #                      '437', '438N', '438X', '440',
    #                      '441', '442', '445', '469',
    #                      '470', '502', '503', '504']      # the short names of the routes we want to get summaries of
    route_desc = 'Sydney Buses Network'     # optional input if we also want to filter by particular types of routes


    print('Processing routes '+", ".join(route_short_names))
    route_data, subset_shapes, elevation_profiles, trip_totals, _ = process_gtfs_routes(gtfs_file, route_short_names, cutoffs=cutoffs, busiest=busiest, route_desc=route_desc)

    ts_tmp = trip_totals.trip_start_time.to_numpy()/60
    tf_tmp = trip_totals.trip_end_time.to_numpy()/60

    start_locations = []
    end_locations = []
    for i, r in trip_totals.iterrows():
        shape_id = r['shape_id']
        start_location = route_data[route_data['shape_id']==shape_id]['start_location'].values[0]
        end_location = route_data[route_data['shape_id']==shape_id]['end_location'].values[0]
        start_locations.append(start_location)
        end_locations.append(end_location)
    num_trips = len(trip_totals)

    # work out the connecting time
    tc_tmp = np.zeros((num_trips,num_trips))
    av_speed_connecting = 25/3.6
    for i in range(num_trips):
        for j in range(num_trips):
            coord1 = (end_locations[i][1], end_locations[i][0])
            coord2 = (start_locations[j][1], start_locations[j][0])
            d = geopy.distance.geodesic(coord1,coord2).m
            tc_tmp[i,j] = d/av_speed_connecting/60  # time in minutes

    N = num_trips + 1

    # define some required trip start times (null trip included as nan)?
    ts = np.hstack([np.nan, ts_tmp])

    # define trip finish times
    tf = np.hstack([np.nan, tf_tmp])
    # define connecting travel times (this needs to include to and from the null route
    tc = tc_tmp.mean()*np.ones((N, N))      # using average of other connecting times as return time to depot
    tc[1:,1:] = tc_tmp
    for i in range(1,N):
        tc[i,i] = 1000

    N = 30
    tc = tc[:N,:N]
    ts = ts[:N]
    tf = tf[:N]

    # Model
    model = pyEnv.ConcreteModel()

    # indices for the trips including the null trip
    model.N = pyEnv.RangeSet(N)

    # Index for the dummy variable u
    model.U = pyEnv.RangeSet(2, N)

    # Decision variables xij
    model.x = pyEnv.Var(model.N, model.N, within=pyEnv.Binary)

    # Dummy variable ui
    model.u = pyEnv.Var(model.N, within=pyEnv.NonNegativeIntegers, bounds=(0, N - 1))

    # Cost Matrix cij
    model.tc = pyEnv.Param(model.N, model.N, initialize=lambda model, i, j: tc[i - 1][j - 1])


    # objective function
    def obj_func(model):
        return sum(sum(model.x[i, j] * model.tc[i, j] for i in model.N) for j in model.N)


    model.objective = pyEnv.Objective(rule=obj_func, sense=pyEnv.minimize)


    def rule_const1(model, N):
        if N == 1:
            return pyEnv.Constraint.Skip
        else:
            return sum(model.x[i, N] for i in model.N if i != N) == 1


    model.const1 = pyEnv.Constraint(model.N, rule=rule_const1)


    def rule_const2(model, N):
        if N == 1:
            return pyEnv.Constraint.Skip
        else:
            return sum(model.x[N, j] for j in model.N if j != N) == 1


    model.rest2 = pyEnv.Constraint(model.N, rule=rule_const2)


    def rule_const3(model, i, j):
        if i != j:
            return model.u[i] - model.u[j] + model.x[i, j] * N <= N - 1
        else:
            # Yeah, this else doesn't say anything
            return model.u[i] - model.u[i] == 0


    model.rest3 = pyEnv.Constraint(model.U, model.N, rule=rule_const3)


    # now include the cant start trip before we have travelled from previous trip constraint
    def rule_const4(model, i, j):
        if (i == 1) or (j == 1):
            return pyEnv.Constraint.Skip
        else:  # the minus ones are to convert to zero based indexing for numpy stuff
            return ts[j - 1] - tf[i - 1] - tc[i - 1, j - 1] + (1 - model.x[i, j]) * 100. >= 0.


    # model.rest = pyEnv.Constraint(model.N, model.N, rule=rule_const4)

    # Solves
    solver = pyEnv.SolverFactory('cplex')
    result = solver.solve(model, tee=False)

    # Prints the results
    print(result)

    l = list(model.x.keys())
    for i in l:
        if model.x[i]() != 0:
            print(i, '--', model.x[i]())