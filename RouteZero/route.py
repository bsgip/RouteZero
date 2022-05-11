import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import srtm
import geopy.distance

## RouteZero modules
import RouteZero.gtfs as gtfs
import RouteZero.weather as weather

"""
                Functions for processing route/trip data and appending information to this
"""

def process(trips, stop_times, stops):
    """
    Processes the gtfs data frames and summarises all relevant model input information in one data frame
    :param trips: gtfs trips data frame
    :param stop_times: gtfs stop_times data frame
    :param stops: gtfs stops data frame
    :return: data frame summarising relevant model input information
    """
    # add elevatoin info to stops
    stops = _stop_elevations(stops)
    # summarise all model input data into one data frame
    trip_summary = _summarise_trip_data(trips, stop_times, stops)

    return trip_summary


def _summarise_trip_data(trips, stop_times, stops):
    """
    Summarises all pertinent information contained in trips and stop_times relevant to each unique trip
    :param trips: the trips data frame loaded from a gtfs feed and augmented with unique_id
    :param stop_times: the stop_times data frame loaded from the gtfs feed and augmented with unique_id
    :param stops: gtfs stop data frame with elevation information added
    :return: trip_summary data frame
    """
    stop_times = stop_times.merge(stops[['stop_id', 'elevation']], how='left')
    stop_times = pd.merge(stop_times, trips[['unique_id','route_id','direction_id','shape_id','trip_id']], how='left')
    stop_times.sort_values(by=['unique_id', 'direction_id', 'stop_sequence'], ascending=True, inplace=True)
    trip_start_stops = stop_times.groupby(by='unique_id').head(1).reset_index(drop=True)
    trip_end_stops = stop_times.groupby(by='unique_id').tail(1).reset_index(drop=True)

    trip_summary = trip_start_stops[['unique_id','route_id', 'direction_id', 'shape_id', 'trip_id','date']].copy(deep=True)
    trip_summary['trip_distance'] = (trip_end_stops['shape_dist_traveled'] - trip_start_stops['shape_dist_traveled']).to_numpy()
    trip_summary['trip_start_time'] = trip_start_stops['departure_time']
    trip_summary['trip_end_time'] = trip_end_stops['arrival_time']
    trip_summary['trip_duration'] =  trip_summary['trip_end_time'] - trip_summary['trip_start_time']
    trip_summary['average_speed_mps'] = trip_summary['trip_distance'] / trip_summary['trip_duration']
    trip_summary['average_gradient_%'] = (trip_end_stops['elevation']-trip_start_stops['elevation'])/trip_summary['trip_distance'] * 100

    tmp = trip_summary[trip_summary['route_id'] == '74-320-sj2-1']
    plt.plot(tmp['trip_start_time']/3600,tmp['average_speed_mps'],'x')
    plt.title('Average speed on route (both directions)')
    plt.ylabel('speed (mps)')
    plt.xlabel('hour of week')
    plt.show()

    # number of stop along route, count up stops and convert to stops/km
    df_tmp = stop_times.groupby(by=['route_id', 'direction_id', 'shape_id', 'unique_id'])[
        'stop_id'].count().reset_index().rename(columns={'stop_id': 'num_stops'})

    trip_summary = trip_summary.merge(df_tmp[['unique_id','num_stops']], how='left')
    trip_summary['stops_per_km'] = trip_summary['num_stops']/trip_summary['trip_distance']*1000

    # average elevation on trip
    trip_summary['av_elevation'] = stop_times.groupby(by='unique_id')['elevation'].mean().reset_index(drop=True)

    return trip_summary

def calc_buses_in_traffic(trip_summary, deadhead=0.1, resolution=10):
    """
    calculated the number of buses in traffic as a function of the hour of the week
    :param trip_summary: trip summary data frame
    :param deadhead: (default=0.1) ratio [0->1] of route time that is spent between routes
    :param resolution: (default=10) resolution of binning in minutes
    :return: array containing number of buses on a route as a function of time of day
    """

    time_slot_edges = np.arange(0, trip_summary['trip_end_time'].max()/60+resolution, resolution)  # time slot edges in minutes

    buffered_time_starts = trip_summary.trip_start_time / 60 - trip_summary.trip_duration / 60 * (deadhead / 2)
    c_starts = np.histogram(buffered_time_starts, bins=time_slot_edges)[0]

    buffered_time_ends = trip_summary.trip_start_time/60 + trip_summary.trip_duration/60*(1+deadhead/2)
    c_ends = np.histogram(buffered_time_ends, bins=time_slot_edges)[0]

    buses_in_traffic = np.cumsum(c_starts) - np.cumsum(c_ends)
    times = time_slot_edges[:-1]
    return times, buses_in_traffic


def _stop_elevations(stops):
    """
    adds elevation information to stops data frame
    :param stops: stops gtfs data frame
    :return: stops data frame with elevation info added
    """
    elevation_data = srtm.get_data()
    el = []
    for i, r in stops.iterrows():
        xy = r['geometry'].xy
        el.append(elevation_data.get_elevation(latitude=xy[1][0], longitude=xy[0][0]))
    stops['elevation'] = el
    return stops

def _elevation_from_shape(shapes):
    """
    calculates the elevation profile corresponding to the trip shapes geometry
    :param shapes: gtfs shapes data frame
    :return: dictionary of elevation profiles keyed by shape id
    """
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


if __name__=="__main__":
    import matplotlib.pyplot as plt

    inpath = '../data/full_greater_sydney_gtfs_static.zip'

    route_short_names = ["305", "320", '389', '406']
    route_desc = ['Sydney Buses Network']
    routes, trips, stops, stop_times, shapes = gtfs.read_busiest_week_data(inpath, route_short_names, route_desc)

    stops = _stop_elevations(stops)
    trip_summary = _summarise_trip_data(trips, stop_times, stops)


    times, buses_in_traffic = calc_buses_in_traffic(trip_summary)


    plt.plot(times/60,buses_in_traffic)
    plt.title('Buses in traffic graph for {} routes'.format(len(route_short_names)))
    plt.xlabel('Hour of week')
    plt.ylabel('# buses')
    plt.show()

    tmp = stop_times.groupby(by='unique_id')['elevation'].mean().reset_index(drop=True)




