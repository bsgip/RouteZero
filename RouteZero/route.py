import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import srtm
import geopy.distance
from tqdm import tqdm
from sklearn.metrics.pairwise import haversine_distances
from shapely.geometry import LineString

## RouteZero modules
import RouteZero.gtfs as gtfs
import RouteZero.weather as weather

"""
                Functions for processing route/trip data and appending information to this
"""

def process(routes, trips, stop_times, stops, patronage, shapes, get_temps=True):
    """
    Processes the gtfs data frames and summarises all relevant model input information in one data frame
    :param trips: gtfs trips data frame
    :param stop_times: gtfs stop_times data frame
    :param stops: gtfs stops data frame
    :param patronage: peak passenger on route information
    :param shapes: gtfs shapes dataframe
    :return: data frame summarising relevant model input information
    """
    # calculate shape lengths
    shapes = _calculate_shape_length(shapes)
    trips = trips.merge(shapes[['shape_id','length (m)']])

    # add passenger info to trips
    trips = _append_trip_patronage(routes, trips, patronage)
    # add elevatoin info to stops
    stops = _stop_elevations(stops)
    # summarise all model input data into one data frame
    trip_summary = _summarise_trip_data(trips, stop_times, stops)
    # add temperature data to trip_summary
    if get_temps:
        trip_summary = _trip_temperatures(trip_summary)

    return trip_summary


def update_patronage(trip_summary, patronage):
    if 'passengers' in trip_summary.columns:
        trip_summary.drop(columns='passengers')
    patronage_df = pd.DataFrame.from_dict(patronage)
    trip_summary = pd.merge(trip_summary, patronage_df[['route_short_name','passengers']], how='left')
    return trip_summary


def _append_trip_patronage(routes, trips, patronage):
    """
    appends passenger information to the trips data frame
    :param routes: gtfs routes dataframe
    :param trips: gtfs trips dataframe
    :param patronage: dict with keys "route_short_name", "passengers" where each have value in a list
    :return: trips dataframe with passenger information appended
    """
    patronage_df = pd.DataFrame.from_dict(patronage)
    routes['route_long_name'] = routes['route_long_name'].fillna(value='')
    patronage_df = patronage_df.astype({'route_short_name':routes['route_short_name'].dtype,'passengers':'int64'})
    tmp_df = pd.merge(routes[['route_short_name','route_id','agency_name','route_long_name']],patronage_df, how='left')
    return pd.merge(trips, tmp_df[['route_id','route_short_name','passengers','agency_name','route_long_name']])


def _calculate_shape_length(shapes):
    geometry = shapes['geometry']
    lengths = []
    for g in geometry:
        xy = g.coords.xy
        x = np.deg2rad(xy[0])
        y = np.deg2rad(xy[1])
        X = np.vstack([x,y]).T

        dist = haversine_distances(X) * 6371000
        length = np.sum(dist.diagonal(1))
        lengths.append(length)
    shapes['length (m)'] = lengths
    return shapes

def _summarise_trip_data(trips, stop_times, stops):
    """
    Summarises all pertinent information contained in trips and stop_times relevant to each unique trip
    :param trips: the trips data frame loaded from a gtfs feed and augmented with unique_id
    :param stop_times: the stop_times data frame loaded from the gtfs feed and augmented with unique_id
    :param stops: gtfs stop data frame with elevation information added
    :return: trip_summary data frame
    """
    stop_times = stop_times.merge(stops[['stop_id', 'elevation','geometry']], how='left')
    stop_times = pd.merge(stop_times, trips[['unique_id','route_id','direction_id','shape_id',
                                             'trip_id','route_short_name','agency_name', 'length (m)','route_long_name']], how='left')
    stop_times.sort_values(by=['unique_id', 'direction_id', 'stop_sequence'], ascending=True, inplace=True)

    trip_start_stops = stop_times.groupby(by='unique_id').head(1).reset_index(drop=True)
    trip_end_stops = stop_times.groupby(by='unique_id').tail(1).reset_index(drop=True)

    trip_summary = trip_start_stops[['unique_id','route_id', 'direction_id', 'shape_id',
                                     'trip_id','date','route_short_name','agency_name','route_long_name']].copy(deep=True)
    if ('shape_dist_traveled' in stop_times) and (trip_end_stops['shape_dist_traveled'].notna().sum()):
        trip_summary['trip_distance'] = (trip_end_stops['shape_dist_traveled'] - trip_start_stops['shape_dist_traveled']).to_numpy()

        if (trip_summary['trip_distance']/trip_end_stops['length (m)']).mean() < 0.002: # assume shape_dist was km
            trip_summary['trip_distance'] = trip_summary['trip_distance'] * 1000
        inds = trip_summary['trip_distance'].isna()
        trip_summary.loc[inds, 'trip_distance'] = trip_end_stops.loc[inds, 'length (m)']
    else:
        trip_summary['trip_distance'] = trip_end_stops['length (m)']

    trip_summary['trip_start_time'] = trip_start_stops['departure_time']
    trip_summary['trip_end_time'] = trip_end_stops['arrival_time']
    trip_summary['trip_duration'] =  trip_summary['trip_end_time'] - trip_summary['trip_start_time']
    trip_summary['average_speed_mps'] = trip_summary['trip_distance'] / trip_summary['trip_duration']
    trip_summary['average_gradient_%'] = (trip_end_stops['elevation']-trip_start_stops['elevation'])/trip_summary['trip_distance'] * 100

    trip_summary['start_loc_x'] = trip_start_stops['geometry'].values.x
    trip_summary['start_loc_y'] = trip_start_stops['geometry'].values.y
    trip_summary['start_el'] = trip_start_stops['elevation']
    trip_summary['end_loc_x'] = trip_end_stops['geometry'].values.x
    trip_summary['end_loc_y'] = trip_end_stops['geometry'].values.y
    trip_summary['end_el'] = trip_end_stops['elevation']

    # tmp = trip_summary[trip_summary['route_id'] == '74-320-sj2-1']
    # plt.plot(tmp['trip_start_time']/3600,tmp['average_speed_mps'],'x')
    # plt.title('Average speed on route (both directions)')
    # plt.ylabel('speed (mps)')
    # plt.xlabel('hour of week')
    # plt.show()

    # number of stop along route, count up stops and convert to stops/km
    df_tmp = stop_times.groupby(by=['route_id', 'direction_id', 'shape_id', 'unique_id'])[
        'stop_id'].count().reset_index().rename(columns={'stop_id': 'num_stops'})

    trip_summary = trip_summary.merge(df_tmp[['unique_id','num_stops']], how='left')
    trip_summary['stops_per_km'] = trip_summary['num_stops']/trip_summary['trip_distance']*1000

    # average elevation on trip
    trip_summary['av_elevation'] = stop_times.groupby(by='unique_id')['elevation'].mean().reset_index(drop=True)

    # average speed in km/h
    trip_summary['average_speed_kmh'] = trip_summary['average_speed_mps']*3.6

    return trip_summary

def calc_buses_in_traffic(trip_summary, deadhead=0.1, resolution=10, trip_ec=None):
    """
    calculated the number of buses in traffic as a function of the hour of the week
    :param trip_summary: trip summary data frame
    :param deadhead: (default=0.1) ratio [0->1] of route time that is spent between routes
    :param resolution: (default=10) resolution of binning in minutes
    :param trip_ec: (optional) energy consumption of each trip, if input then will return the departing trip energy
                    requirements and the return trip energy consumed arrays
    :return: array containing number of buses on a route as a function of time of day
    """

    time_slot_edges = np.arange(0, trip_summary['trip_end_time'].max()/60+resolution, resolution)  # time slot edges in minutes

    buffered_time_starts = trip_summary.trip_start_time / 60 - trip_summary.trip_duration / 60 * (deadhead / 2)
    c_starts = np.histogram(buffered_time_starts, bins=time_slot_edges)[0]

    buffered_time_ends = trip_summary.trip_start_time/60 + trip_summary.trip_duration/60*(1+deadhead/2)
    c_ends = np.histogram(buffered_time_ends, bins=time_slot_edges)[0]

    buses_in_traffic = np.cumsum(c_starts) - np.cumsum(c_ends)
    times = time_slot_edges[:-1]

    if trip_ec is not None:
        depart_trip_energy_reqs = np.histogram(buffered_time_starts, bins=time_slot_edges, weights=trip_ec * (1 + deadhead))[0]
        return_trip_enery_consumed = np.histogram(buffered_time_ends, bins=time_slot_edges, weights=trip_ec * (1 + deadhead))[0]
        return times, buses_in_traffic, depart_trip_energy_reqs, return_trip_enery_consumed

    else:
        return times, buses_in_traffic

def append_temperature_data(trip_summary, num_years=5, percentiles=[1, 99], disp=True):
    return _trip_temperatures(trip_summary, num_years=num_years, percentiles=percentiles, disp=disp)

def _trip_temperatures(trip_summary, num_years=5, percentiles=[1, 99], disp=True):
    """
    determines the 'design' temperature for each trip. The min temperature will be the percentile[0] temperature
    max temperature will be the percentile[1] temperature from the past num_years worth of recordings at the stop location
    and elevation
    :param stops: summarised trip data dataframe
    :param num_years: number of years worth of historical data to use
    :param percentiles: the percentiles used for extracting min and max temperature from the historical data
    :param disp: adds progress bar
    :return: trip_summary with temperature data added
    """
    trip_summary['max_temp'] = 0.
    trip_summary['min_temp'] = 0.

    df = trip_summary.drop_duplicates('route_short_name')
    start_loc_x = df['start_loc_x'].to_numpy()
    start_loc_y = df['start_loc_y'].to_numpy()
    start_el = df['start_el'].to_numpy()
    end_loc_x = df['end_loc_x'].to_numpy()
    end_loc_y = df['end_loc_y'].to_numpy()
    end_el = df['end_el'].to_numpy()
    route_short_name = df['route_short_name'].to_list()

    for i in tqdm(range(len(df)), desc='Getting min and max temperatures: ', disable=not disp):
        name = route_short_name[i]

        inds = trip_summary[trip_summary.route_short_name==name].index
        start_hour = np.mod(trip_summary[trip_summary.route_short_name==name].trip_start_time/3600, 24)
        end_hour = np.mod(trip_summary[trip_summary.route_short_name == name].trip_end_time/3600, 24)

        x = start_loc_x[i]
        y = start_loc_y[i]
        el = start_el[i]
        daily_low_min, daily_low_max, daily_high_min, daily_high_max = weather.location_design_temp([y, x],
                                                                                                    el,
                                                                                                    num_years=num_years,
                                                                                                    percentiles=percentiles)

        t1 = weather.daily_temp_profile(start_hour, daily_low_max, daily_high_max)
        t2 = weather.daily_temp_profile(end_hour, daily_low_max, daily_high_max)

        t5 = weather.daily_temp_profile(start_hour, daily_low_min, daily_high_min)
        t6 = weather.daily_temp_profile(end_hour, daily_low_min, daily_high_min)

        x = end_loc_x[i]
        y = end_loc_y[i]
        el = end_el[i]

        daily_low_min, daily_low_max, daily_high_min, daily_high_max = weather.location_design_temp([y, x], el)
        t3 = weather.daily_temp_profile(start_hour, daily_low_max, daily_high_max)
        t4 = weather.daily_temp_profile(end_hour, daily_low_max, daily_high_max)

        t7 = weather.daily_temp_profile(start_hour, daily_low_min, daily_high_min)
        t8 = weather.daily_temp_profile(end_hour, daily_low_min, daily_high_min)

        temp_max = np.vstack([t1, t2, t3, t4]).max(axis=0)
        temp_min = np.vstack([t5, t6, t7, t8]).min(axis=0)

        trip_summary.loc[inds, 'min_temp'] = temp_min
        trip_summary.loc[inds, 'max_temp'] = temp_max

    trip_summary.dropna(inplace=True)
    return trip_summary

def _stop_elevations(stops, disp=True):
    """
    adds elevation information to stops data frame
    :param stops: stops gtfs data frame
    :param disp: adds progress bar
    :return: stops data frame with elevation info added
    """
    elevation_data = srtm.get_data()
    el = []
    geoms = stops['geometry'].to_list()
    for g in tqdm(geoms, desc='Getting elevations', disable=not disp):
        xy = g.xy
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

def offset_shape_geometries(shapes):
    geometries = shapes['geometry'].values

    new_geom = []
    for g in tqdm(geometries, desc='offsetting shape geometries'):
        x,y = g.xy

        # df = pd.DataFrame.from_dict({'x':x.tolist(),'y':y.tolist()})
        # df.drop_duplicates(inplace=True, ignore_index=True)
        # x = df['x'].to_numpy()
        # y = df['y'].to_numpy()
        x = np.array(x.tolist())
        y = np.array(y.tolist())

        u = x[1:] - x[:-1]
        v = y[1:] - y[:-1]
        l = np.sqrt(u**2 + v**2)

        # getting rid of duplicates
        inds = np.argwhere(l <= 1e-12)
        x = np.delete(x, inds)
        y = np.delete(y, inds)
        u = x[1:] - x[:-1]
        v = y[1:] - y[:-1]
        l = np.sqrt(u**2 + v**2)

        uhat = u/l
        vhat = v/l

        u_perp = -vhat
        v_perp = uhat

        offset = 0.00005
        x_diff = np.zeros(len(x))
        y_diff = np.zeros(len(y))
        x_diff[0] = u_perp[0]*offset
        y_diff[0] = v_perp[0]*offset
        x_diff[1:-1] = (u_perp[1:] + u_perp[:-1])/2*offset
        y_diff[1:-1] = (v_perp[1:] + v_perp[:-1])/2*offset
        x_diff[-1] = u_perp[-1] * offset
        y_diff[-1] = v_perp[-1] * offset

        x_new = x + x_diff
        y_new = y + y_diff


        new_geom.append(LineString([(xx) for xx in zip(x_new,y_new)]))

    shapes['geometry'] = new_geom
    return shapes

if __name__=="__main__":
    import matplotlib.pyplot as plt

    # inpath = '../data/gtfs/act.zip'
    # inpath = '../data/gtfs/full_greater_sydney_gtfs_static.zip'
    # name = 'vic_regional_bus_gtfs'
    # name = 'act_gtfs'
    name = "brisbane_gtfs"
    inpath = '../data/gtfs/'+name+'.zip'


    route_short_names, route_desc = gtfs.read_route_desc_and_names(inpath)

    # route_short_names = ["305", "320", '389', '406']
    # route_names_df = pd.read_csv('../data/zenobe_routes.csv')
    # route_short_names = route_names_df['route_short_name'].to_list()
    # route_desc = ['Sydney Buses Network']

    routes, trips, stops, stop_times, shapes = gtfs.read_busiest_week_data(inpath, route_short_names, route_desc)

    patronage = {"route_short_name": route_short_names, "passengers":[38]*len(route_short_names)}

    trip_summary = process(routes, trips, stop_times, stops, patronage, shapes)
    times, buses_in_traffic = calc_buses_in_traffic(trip_summary)

    shapes = offset_shape_geometries(shapes)        #offset routes to left so they display next to each other



    plt.plot(times/60,buses_in_traffic)
    plt.title('Buses in traffic graph for {} routes'.format(len(route_short_names)))
    plt.xlabel('Hour of week')
    plt.ylabel('# buses')
    plt.show()
    #


    trip_summary.to_csv('../data/gtfs/'+name[:-5]+'/trip_data.csv')
    shapes.to_file('../data/gtfs/'+name[:-5]+'/shapes.shp')

    print("{} trips in week".format(len(trips)))
    print("{} trips in summary".format(len(trip_summary)))
    print("{:.2f}% success".format(len(trip_summary)/len(trips)*100))





