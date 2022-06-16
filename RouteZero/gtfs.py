import partridge as ptg
from zipfile import ZipFile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from tqdm import tqdm

"""
                        Functions for working with gtfs files
"""


# def to_csv(filename, routes, trips, stops, stop_times):       # probs not the right time to be saving data
#     routes.to_csv(filename+'_routes.csv')
#     trips.to_csv(filename+'_trips.csv')
#     stops.to_csv(filename+'_stops.csv')
#     stop_times(filename+'_stop_times.csv')

def read_route_desc_and_names(inpath):
    """
    read a data frame containing the route short names and route descriptions from which routes could be selected
    :param inpath: path to gtfs zip file
    :return: data frame
    """
    routes_all = _read_all_routes(inpath)
    if 'route_desc' in routes_all:
        return routes_all['route_short_name'].unique().tolist(), routes_all['route_desc'].unique().tolist()
    else:
        return routes_all['route_short_name'].unique().tolist(), None

def read_busiest_week_data(inpath, route_short_names, route_desc, disp=True):
    """
    reads in data for the selected routes and route descriptions for the busiest week in the gtfs file
    :param inpath: path to gtfs zip file
    :param route_short_names: selected route short names (list)
    :param route_desc: selected route descriptions (list)
    :param disp: (default = true) enables progress bar
    :return: data frames containing information for the routes, trips, stops, and stop_times
    """
    service_ids_by_date = _read_busiest_week_services(inpath)
    routes_list = []
    trips_list = []
    stops_list = []
    stop_times_list = []
    shapes_list = []


    for i, key in enumerate(tqdm(service_ids_by_date.keys(),desc='Reading trips for days of busiest week',disable=not disp)):
        service_ids = service_ids_by_date[key]
        routes, trips, stops, stop_times, shapes = _read_service_data(inpath, service_ids,
                                                                 route_short_names=route_short_names,
                                                                 route_desc=route_desc)

        stop_times['arrival_time'] += i*24*3600        # plus i days
        stop_times['departure_time'] += i*24*3600       # plus i days
        # trip id is only unique within a day
        trips['unique_id'] = trips['trip_id'] + '_' + str(i)
        trips['date'] = key
        stop_times['unique_id'] = stop_times['trip_id'] + '_' + str(i)
        stop_times['date'] = key

        # append data
        routes_list.append(routes)
        trips_list.append(trips)
        stops_list.append(stops)
        stop_times_list.append(stop_times)
        shapes_list.append(shapes)


    # merge data frames
    routes = pd.concat(routes_list, axis=0, ignore_index=True).drop_duplicates()
    trips = pd.concat(trips_list, axis=0, ignore_index=True).drop_duplicates()
    stops = pd.concat(stops_list, axis=0, ignore_index=True).drop_duplicates()
    stop_times = pd.concat(stop_times_list, axis=0, ignore_index=True).drop_duplicates()
    shapes = pd.concat(shapes_list, axis=0, ignore_index=True).drop_duplicates()

    try:
        agency = _read_agency(inpath)
        if len(agency)==1:
            routes['agency_name'] = agency['agency_name'].to_list()[0]
        else:
            routes = pd.merge(routes, agency[['agency_id', 'agency_name']], how='left')
    except:
        routes['agency'] = ""

    return routes, trips, stops, stop_times, shapes

def _read_all_routes(inpath):
    """
    Extracts all bus route data from gtfs feed zipfile and filters to only bus routes
    :param inpath: path to gtfs zip file
    :return: data frame containing all bus route information
    """
    with ZipFile(inpath) as z:
        routes = pd.read_csv(z.open('routes.txt'), delimiter=",")

    if ((routes.route_type >= 700) & (routes.route_type < 800)).sum():
        routes = routes.loc[(routes.route_type >= 700) & (routes.route_type < 800)].reset_index(drop=True)
    elif ((routes.route_type == 3)).sum():
        routes = routes.loc[(routes.route_type == 3)].reset_index(drop=True)

    return routes

def _read_agency(inpath):
    """
    Extracts agency information from gtfs feed zipfile
    :param inpath: path to gtfs zip file
    :return: data frame containing all agency information
    """
    with ZipFile(inpath) as z:
        agency = pd.read_csv(z.open('agency.txt'), delimiter=",")

    return agency

def _read_service_ids_by_date(inpath):
    service_ids_by_date = ptg.read_service_ids_by_date(inpath)
    return service_ids_by_date



def _read_busiest_week_services(inpath):
    """
    returns a dictionary that contains all dates and corresponding service ids
    :param inpath: path to gtfs zip file
    :return: dictionary
    """
    service_ids_by_date = ptg.read_busiest_week(inpath)
    return service_ids_by_date

def _read_service_data(inpath, service_ids, route_short_names=None, route_desc=None):
    """
    returns data frames corresponding to gtfs data for specified service_ids
    :param inpath: path to gtfs zip file
    :param service_ids: service ids that we want data for
    :param route_short_names: list of route short names to filter data down to
    :param route_desc: list of route descriptions that we wish to filter data down to
    :return:
    """
    if (route_short_names is not None) and (route_desc is not None):
        view = {'trips.txt': {'service_id': service_ids},
                'routes.txt':{'route_short_name':route_short_names,
                              'route_desc':route_desc}}
    elif route_desc is not None:
        view = {'trips.txt': {'service_id': service_ids},
                'routes.txt':{'route_desc':route_desc}}
    elif route_short_names is not None:
        view = {'trips.txt': {'service_id': service_ids},
                'routes.txt':{'route_short_name':route_short_names}}
    else:
        view = {'trips.txt': {'service_id': service_ids}}
    feed = ptg.load_geo_feed(inpath, view)
    routes = feed.routes
    trips = feed.trips

    stop_times = feed.stop_times
    stops = feed.stops
    shapes = feed.shapes
    return routes, trips, stops, stop_times, shapes

if __name__=="__main__":
    import matplotlib.pyplot as plt

    inpath = '../data/gtfs/full_greater_sydney_gtfs_static.zip'

    # all route short names and descriptions
    route_short_names, route_desc = read_route_desc_and_names(inpath)

    # from which the user selects the ones they are interested in
    route_short_names = ["305", "320", '389', '406']
    # route_short_names = ["526"]
    route_desc = ['Sydney Buses Network']

    routes, trips, stops, stop_times, shapes = read_busiest_week_data(inpath, route_short_names, route_desc)

    plt.hist(stop_times['arrival_time']/3600,bins=24*7)
    plt.xlabel('hours of week')
    plt.title('number of stops made by buses on selected routes')
    plt.show()





