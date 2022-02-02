import numpy as np
import gtfs_functions as gtfs
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = None

# todo: do we also need to keep either agency id information or route_long_name information?
def process_gtfs_routes(gtfs_file, route_short_names, cutoffs=None, busiest_day=True, route_desc=None):
    # import all gtfs route data from file
    routes, stops, stop_times, trips, shapes = gtfs.import_gtfs(gtfs_file, busiest_date=busiest_day)

    # cut down to only bus routes
    # bus services have route types in teh 700 so >=700 and < 800
    if route_desc:
        bus_routes = routes.loc[(routes.route_type >= 700) & (routes.route_type < 800) & (routes.route_desc==route_desc)].reset_index()
    else:
        bus_routes = routes.loc[(routes.route_type >= 700) & (routes.route_type < 800)].reset_index()
    bus_route_ids = bus_routes['route_id'].unique()
    # we can cut down the trips based on the route ids
    bus_trips = trips.loc[trips.route_id.isin(bus_route_ids)]  # this has shape id
    # can cut down stop_times based on trip id
    bus_stop_times = stop_times.loc[stop_times.trip_id.isin(bus_trips['trip_id'])]
    # can cut down stops based on stop id from stop times
    bus_stops = stops.loc[stops.stop_id.isin(bus_stop_times['stop_id'])]
    # cut down shapes based on shape id from above
    bus_shapes = shapes.loc[shapes.shape_id.isin(bus_trips['shape_id'])]

    # cut down to selected routes

    subset_routes = bus_routes.loc[bus_routes['route_short_name'].isin(route_short_names)]
    subset_trips = bus_trips.loc[bus_trips.route_id.isin(subset_routes['route_id'])]
    subset_stop_times = bus_stop_times.loc[bus_stop_times.trip_id.isin(subset_trips['trip_id'])]
    subset_stops = bus_stops.loc[bus_stops.stop_id.isin(subset_stop_times['stop_id'])]
    subset_shapes = bus_shapes.loc[bus_shapes.shape_id.isin(subset_trips['shape_id'])]

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

    # number of stop along route ?? count up stops and convert to stops/km
    df_tmp = subset_stop_times.groupby(by=['route_id', 'direction_id', 'shape_id', 'trip_id'])[
        'stop_id'].count().reset_index().rename(columns={'stop_id': 'num_stops'})
    num_stops_by_route_dir_shape = df_tmp.groupby(by=['route_id', 'direction_id', 'shape_id'])[
        'num_stops'].mean().reset_index()

    if cutoffs:
        cutoffs = [0, 6, 9, 15, 19, 22, 24]
        mid_window = (np.array(cutoffs[:-1]) + np.array(cutoffs[1:])) / 2
        labels = [str(cutoffs[i]) + '-' + str(cutoffs[i + 1]) for i in range(0, len(cutoffs) - 1)]
        trip_totals['window'] = pd.cut(trip_totals['trip_start_time'] / 3600, bins=cutoffs, right=False,
                                         labels=mid_window)

        route_averages = trip_totals.groupby(by=['route_id', 'direction_id',
                                                 'shape_id', 'window']).mean().reset_index()
        route_averages.dropna(inplace=True)

        # merge
        route_summaries = route_averages.copy(deep=True)
        route_summaries['num_stops'] = np.nan
        for i, r in route_summaries.iterrows():
            route_summaries.loc[i, 'num_stops'] = \
            num_stops_by_route_dir_shape[num_stops_by_route_dir_shape.shape_id == r['shape_id']][
                'num_stops'].to_numpy()[0]
    else:
        # get average over route, dir, shape
        route_averages = trip_totals.groupby(by=['route_id','direction_id','shape_id']).mean().reset_index()

        # merge
        route_summaries = pd.merge(route_averages, num_stops_by_route_dir_shape, on=['route_id','direction_id','shape_id'])

    route_summaries['stops_km'] = route_summaries['num_stops'] / (route_summaries['trip_distance']/1000)

    return route_summaries

# def route_elevation_data(shape_ids):

if __name__ == "__main__":
    gtfs_file = "./data/full_greater_sydney_gtfs_static.zip"        # location of the gtfs zip file
    route_short_names = ["305", "320"]      # the short names of the routes we want to get summaries of
    route_desc = 'Sydney Buses Network'     # optional input if we also want to filter by particular types of routes
    cutoffs = [0, 6, 9, 15, 19, 22, 24]     # optional input for splitting up the route summary information into time windows

    route_summaries = process_gtfs_routes(gtfs_file, route_short_names, cutoffs=cutoffs, busiest_day=True, route_desc=route_desc)

    if cutoffs:
        plt.subplot(2,1,1)
        t = route_summaries[route_summaries.shape_id=='74-320-sj2-1.4.H']
        plt.plot(t['window'],t['average_speed_mps']*3.6)
        plt.xlabel('Time of day')
        plt.ylabel('average speed (km/h)')
        plt.title('speed vs time for route id {}'.format(t['route_id'].unique()[0]))
        plt.subplot(2,1,2)
        t = route_summaries[route_summaries.shape_id=='74-320-sj2-1.3.H']
        plt.plot(t['window'],t['average_speed_mps']*3.6)
        plt.xlabel('Time of day')
        plt.ylabel('average speed (km/h)')
        plt.title('speed vs time for route id {}'.format(t['route_id'].unique()[0]))
        plt.tight_layout()
        plt.show()


