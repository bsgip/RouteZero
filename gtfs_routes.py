import numpy as np
import gtfs_functions as gtfs
import matplotlib.pyplot as plt
import srtm
import pandas as pd
pd.options.display.max_columns = None
import geopy.distance
import os

# todo: add a location tag to route data


def process_gtfs_routes(gtfs_file, route_short_names, cutoffs=None, busiest='day', route_desc=None):
    # import all gtfs route data from file
    # if cache:
    #     base_name = os.path.splitext(gtfs_file)[0]
    #     if busiest is None:
    #         busiest = ''
    #     if os.path.exists(base_name+'_routes_'+busiest+'.csv') and (os.path.getmtime(base_name+'_routes_'+busiest+'.csv') > os.path.getmtime(gtfs_file)):    # cache was done since gtfs modifications
    #         routes = pd.read_csv(base_name+'_routes_'+busiest+'.csv')
    #         stops = pd.read_csv(base_name + '_stops_' + busiest + '.csv')
    #         stop_times = pd.read_csv(base_name + '_stop_times_' + busiest + '.csv', low_memory=False)
    #         trips = pd.read_csv(base_name + '_trips_' + busiest + '.csv')
    #         shapes = pd.read_csv(base_name + '_shapes_' + busiest + '.csv')
    #     else:
    #         if busiest == 'week':
    #             routes, stops, stop_times, trips, shapes = gtfs.import_gtfs_busiest_week(gtfs_file)
    #         elif busiest == 'day':
    #             routes, stops, stop_times, trips, shapes = gtfs.import_gtfs(gtfs_file, busiest_date=True)
    #         else:
    #             routes, stops, stop_times, trips, shapes = gtfs.import_gtfs(gtfs_file, busiest_date=False)
    #
    #         routes.to_csv(base_name+'_routes_'+busiest+'.csv')
    #         stops.to_csv(base_name + '_stops_'+busiest+'.csv')
    #         trips.to_csv(base_name + '_trips_'+busiest+'.csv')
    #         stop_times.to_csv(base_name + '_stop_times_'+busiest+'.csv')
    #         shapes.to_csv(base_name + '_shapes_'+busiest+'.csv')
    # else:
    if busiest=='week':
        routes, stops, stop_times, trips, shapes = gtfs.import_gtfs_busiest_week(gtfs_file)
    elif busiest=='day':
        routes, stops, stop_times, trips, shapes = gtfs.import_gtfs(gtfs_file, busiest_date=True)
    else:
        routes, stops, stop_times, trips, shapes = gtfs.import_gtfs(gtfs_file, busiest_date=False)

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

    if cutoffs is not None:
        max_stop_time = stop_times['departure_time'].max()/3600
        if cutoffs[-1] < max_stop_time:       # do something about this
            print('Warning: last cutoff ({}) is less than last stop time ({})'.format(cutoffs[-1],stop_times['departure_time'].max()/3600))
        mid_window = (np.array(cutoffs[:-1]) + np.array(cutoffs[1:])) / 2
        # labels = [str(cutoffs[i]) + '-' + str(cutoffs[i + 1]) for i in range(0, len(cutoffs) - 1)]
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
        # drop some left over data
        route_summaries.drop(['trip_start_time', 'trip_end_time'], axis=1, inplace=True)

    # compute stops per km
    route_summaries['stops_km'] = route_summaries['num_stops'] / (route_summaries['trip_distance']/1000)

    # get elevation information
    elevation_profiles = elevation_from_shape(subset_shapes)

    # get average grade and add to route summaries and also add route start location
    route_summaries['av_grade_%'] = np.nan
    start_locations = []
    end_locations = []
    for i, r in route_summaries.iterrows():
        els = elevation_profiles[r['shape_id']].values
        dists = elevation_profiles[r['shape_id']].index.to_numpy()
        av_grade = (els[-1] - els[0]) / (dists[-1] - dists[0]) * 100    # average grade as percent
        route_summaries.loc[i, 'av_grade_%'] = av_grade

        start_locations.append(subset_shapes[subset_shapes.shape_id==r['shape_id']]['geometry'].values[0].coords[0])
        end_locations.append(subset_shapes[subset_shapes.shape_id==r['shape_id']]['geometry'].values[0].coords[-1])

    route_summaries['start_location'] = start_locations
    route_summaries['end_location'] = end_locations


    # add location data


    # below is unused code to get grade along the route rather than just average
    # shape_id = '74-320-sj2-1.4.H'
    # els = elevation_profiles[shape_id].values
    # dists = elevation_profiles[shape_id].index.to_numpy()
    # dist_diff = dists[1:]-dists[:-1]
    # els_diff = (els[1:]-els[:-1])[dist_diff!=0]
    # dist_diff = dist_diff[dist_diff!=0]
    # remember when averaging individual grades that they need to be weighted by segment length

    return route_summaries, subset_shapes, elevation_profiles, trip_totals, subset_stops

def elevation_from_shape(shapes):
    elevation_data = srtm.get_data()
    elevation_profiles = {}
    for i, r in shapes.iterrows():
        el = []
        dist = []
        last_coord = None
        for coord in r['geometry'].coords:
            el.append(elevation_data.get_elevation(coord[1], coord[0]))
            if last_coord:
                dist.append(geopy.distance.distance((coord[1], coord[0]), (last_coord[1], last_coord[0])).m +dist[-1])
            else:
                dist.append(0)
            last_coord = coord + ()
        elevation_profiles[r['shape_id']] = pd.Series(np.array(el), index=dist)

    return elevation_profiles


if __name__ == "__main__":
    gtfs_file = "./data/full_greater_sydney_gtfs_static.zip"        # location of the gtfs zip file
    route_short_names = ["305", "320"]      # the short names of the routes we want to get summaries of
    route_desc = 'Sydney Buses Network'     # optional input if we also want to filter by particular types of routes
    cutoffs = [0, 6, 9, 15, 19, 22, 24]     # optional input for splitting up the route summary information into time windows

    route_summaries,subset_shapes, elevation_profiles, trip_totals = process_gtfs_routes(gtfs_file, route_short_names, cutoffs=cutoffs, busiest_day=True, route_desc=route_desc)

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

    plt.plot(elevation_profiles[list(elevation_profiles.keys())[3]])
    plt.show()

    el = elevation_profiles[list(elevation_profiles.keys())[0]]

    # get average grade
    # shape_id = '74-320-sj2-1.4.H'
    # els = elevation_profiles[shape_id].values
    # dists = elevation_profiles[shape_id].index.to_numpy()
    # dist_diff = dists[1:]-dists[:-1]
    # els_diff = (els[1:]-els[:-1])[dist_diff!=0]
    # dist_diff = dist_diff[dist_diff!=0]
    # grade = els_diff/dist_diff



    # for i, r in route_summaries.iterrows():
    # elevation_profiles[r['shape_id'][]
    # """
    # Work out busses in traffic
    # """
    # buffer = 0.1        # add a 10% buffer to trip times
    # resolution = 10      # resolution in 5 mins
    # time_slot_edges = np.arange(0, 24*60, resolution)      # time slot edges in minutes
    # c_starts = np.histogram(trip_totals.trip_start_time/60, bins=time_slot_edges)[0]
    #
    # buffered_time_ends = trip_totals.trip_start_time/60 + trip_totals.trip_duration/60*(1+buffer)
    # c_ends = np.histogram(buffered_time_ends, bins=time_slot_edges)[0]
    #
    # busses_in_traffic = np.cumsum(c_starts) - np.cumsum(c_ends)
    #
    # plt.step(time_slot_edges[:-1]/60, busses_in_traffic)
    # plt.title('Busses in traffic on routes 305 and 320')
    # plt.xlabel('Time of day (hour)')
    # plt.ylabel('Number of busses')
    # plt.show()


