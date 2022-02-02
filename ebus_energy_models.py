import numpy as np
import gtfs_functions as gtfs
import pandas as pd
pd.options.display.max_columns = None
import matplotlib.pyplot as plt

# todo: elevation data from routes - > road grade as a percentage
# todo: temperature minimum and maximum for period



class linearRegressionAbdelatyModel:
    """
    A linear regression model for eBus energy consumption taken from

    Abdelaty, H.; Mohamed, M. APrediction Model for Battery Electric Bus Energy Consumption in Transit.
    Energies 2021, 14, 2824. https://doi.org/10.3390/en14102824

    model:

    EC = B0 + B1*GR + B2*SoCi + B3*RC + B4*HVAC + B5*PL + B6*Dagg + B7*SD + B8*Va + e

    where:
    EC: energy consumption (KWh/km)
    GR: road grade average (%) [-100, 100]
    SoCi: initial battery state of charge (%) [0, 100]
    RC: road condition in three levels (three levels)
    HVAC: auxilliary systems (heating, cooling etc) (KW)
    PL: passenger loading (passengers)
    Dagg: driver aggressiveness (three levels)
    SD: stop density (stops/km)
    Va: average velocity (km/h)

    Parameter values are given as
    B0 = -0.782
    B1 = 0.38
    B2 = 0.0124
    B3 = 0.26
    B4 = 0.036
    B5 = 0.005
    B6 = 0.065
    B7 = 0.128
    B8 = 0.007
    """
    def __init__(self):
        self.B = np.array([-0.782, 0.38, 0.0124, 0.26, 0.036, 0.005, 0.065, 0.128, 0.007])

    def _predict(self, X):
        return np.dot(X, self.B)

    def predict(self, GR, SoCi, RC, HVAC, PL, Dagg, SD, Va):
        X = np.array([GR, SoCi, RC, HVAC, PL, Dagg, SD, Va])
        self._predict(X)

if __name__ == "__main__":
    print('Testing ebus model')

    routes, stops, stop_times, trips, shapes = gtfs.import_gtfs("./data/full_greater_sydney_gtfs_static.zip",
                                                                busiest_date=True)
    # bus services have route types in teh 700 so >=700 and < 800
    bus_routes = routes.loc[(routes.route_type >= 700) & (routes.route_type < 800)].reset_index()

    # cut down to sydney bus network routes
    sydney_bus_routes = routes.loc[routes.route_desc == 'Sydney Buses Network']
    sydney_bus_route_ids = sydney_bus_routes['route_id'].unique()

    # we can cut down the trips based on the route ids
    sydney_bus_trips = trips.loc[trips.route_id.isin(sydney_bus_route_ids)]  # this has shape id

    # can cut down stop_times based on trip id
    sydney_bus_stop_times = stop_times.loc[stop_times.trip_id.isin(sydney_bus_trips['trip_id'])]

    # can cut down stops based on stop id from stop times
    sydney_bus_stops = stops.loc[stops.stop_id.isin(sydney_bus_stop_times['stop_id'])]

    # cut down shapes based on shape id from above
    sydney_bus_shapes = shapes.loc[shapes.shape_id.isin(sydney_bus_trips['shape_id'])]

    # test on two routes
    route_short_names = ["305", "320"]

    subset_routes = sydney_bus_routes.loc[sydney_bus_routes['route_short_name'].isin(route_short_names)]
    subset_trips = sydney_bus_trips.loc[sydney_bus_trips.route_id.isin(subset_routes['route_id'])]
    subset_stop_times = stop_times.loc[stop_times.trip_id.isin(subset_trips['trip_id'])]
    subset_stops = stops.loc[stops.stop_id.isin(subset_stop_times['stop_id'])]
    subset_shapes = shapes.loc[shapes.shape_id.isin(subset_trips['shape_id'])]

    subset_stop_times.sort_values(by=['trip_id', 'direction_id', 'stop_sequence'], ascending=True, inplace=True)

    # routes need to be split up by route_id, direction_id, shape_id
    # shape_id needs to be included because same route can have a different shape and therefore follow a different path (not sure why)
    trip_start_stops = subset_stop_times.groupby(by='trip_id').head(1).reset_index()
    trip_end_stops = subset_stop_times.groupby(by='trip_id').tail(1).reset_index()

    # average speed, distance, time
    trip_totals = trip_start_stops[['route_id','direction_id','shape_id','trip_id']]
    trip_totals['trip_distance'] = trip_end_stops['shape_dist_traveled'] - trip_start_stops['shape_dist_traveled']
    trip_totals['trip_start_time'] = trip_start_stops['departure_time']
    trip_totals['trip_end_time'] = trip_end_stops['arrival_time']
    trip_totals['trip_duration'] =  trip_totals['trip_end_time'] - trip_totals['trip_start_time']
    trip_totals['average_speed_mps'] = trip_totals['trip_distance'] / trip_totals['trip_duration']

    # get average over route, dir, shape
    route_averages = trip_totals.groupby(by=['route_id','direction_id','shape_id']).mean().reset_index()



    # number of stop along route ?? count up stops and convert to stops/km
    df_tmp = subset_stop_times.groupby(by=['route_id','direction_id','shape_id','trip_id'])['stop_id'].count().reset_index().rename(columns={'stop_id':'num_stops'})
    num_stops_by_route_dir_shape = df_tmp.groupby(by=['route_id','direction_id','shape_id'])['num_stops'].mean().reset_index()

    # merge
    route_summaries = pd.merge(route_averages, num_stops_by_route_dir_shape, on=['route_id','direction_id','shape_id'])
    route_summaries['stops_km'] = route_summaries['num_stops'] / (route_summaries['trip_distance']/1000)
    # # merge

    # todo: add in a break up into time windows
    cutoffs = [0, 6, 9, 15, 19, 22, 24]
    mid_window = (np.array(cutoffs[:-1]) + np.array(cutoffs[1:]))/2
    labels = [str(cutoffs[i]) + '-' + str(cutoffs[i + 1]) for i in range(0, len(cutoffs) - 1)]
    trip_totals_w = trip_totals.copy(deep=True)
    trip_totals_w['window'] = pd.cut(trip_totals['trip_start_time']/3600, bins=cutoffs,right=False, labels=mid_window)

    route_averages_window = trip_totals_w.groupby(by=['route_id','direction_id','shape_id','window']).mean().reset_index()
    route_averages_window.dropna(inplace=True)

    # number of stop along route ?? count up stops and convert to stops/km
    # df_tmp = subset_stop_times.copy(deep=True)
    # df_tmp['trip_start_time'] = trip_totals['trip_start_time']
    # # df_tmp['window'] = pd.cut(trip_totals['trip_start_time']/3600, bins=cutoffs,right=False, labels=mid_window)
    # df_tmp = subset_stop_times.groupby(by=['route_id','direction_id','shape_id','trip_id'])['stop_id'].count().reset_index().rename(columns={'stop_id':'num_stops'})
    # num_stops_by_route_dir_shape = df_tmp.groupby(by=['route_id','direction_id','shape_id'])['num_stops'].mean().reset_index()

    # merge
    route_averages_window['num_stops'] = np.nan
    for i, r in route_averages_window.iterrows():
        route_averages_window.loc[i,'num_stops'] = num_stops_by_route_dir_shape[num_stops_by_route_dir_shape.shape_id==r['shape_id']]['num_stops'].to_numpy()[0]


    # route_summaries['stops_km'] = route_summaries['num_stops'] / (route_summaries['trip_distance']/1000)



    t = route_averages_window[route_averages_window.shape_id=='74-320-sj2-1.4.H']
    plt.plot(t['window'],t['average_speed_mps']*3.6)
    plt.xlabel('Time of day')
    plt.ylabel('average speed (km/h)')
    plt.title('speed vs time for route id {}'.format(t['route_id'].unique()[0]))
    plt.show()



    # elevation along route (possibly from the shape file)
    # for shape_id in route_summaries['shape_id']:
    shape_id = '74-305-sj2-1.1.R'
    coordsxy = subset_shapes[subset_shapes.shape_id==shape_id]['geometry'].values[0].xy






