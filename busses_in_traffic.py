import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from scipy.interpolate import interp1d
from weather import location_design_temp

# custom functions
from gtfs_routes import process_gtfs_routes
from ebus_energy_models import LinearRegressionAbdelatyModel

# for plotting the demo example
import matplotlib.pyplot as plt




if __name__=='__main__':
    gtfs_file = "./data/full_greater_sydney_gtfs_static.zip"        # location of the gtfs zip file
    route_short_names = ["305", "320"]      # the short names of the routes we want to get summaries of
    # route_short_names = ["305", "320", '389', '406',
    #                      '428', '430', '431', '433',
    #                      '437', '438N', '438X', '440',
    #                      '441', '442', '445', '469',
    #                      '470', '502', '503', '504']      # the short names of the routes we want to get summaries of
    route_desc = 'Sydney Buses Network'     # optional input if we also want to filter by particular types of routes
    cutoffs = [0, 6, 9, 15, 19, 22, 24]     # optional input for splitting up the route summary information into time windows
    passenger_loading = 38
    print('Processing routes '+", ".join(route_short_names))
    route_data, subset_shapes, elevation_profiles, trip_totals = process_gtfs_routes(gtfs_file, route_short_names, cutoffs=cutoffs, busiest_day=True, route_desc=route_desc)

    # get design day temperatures for route locations
    min_temps = []
    max_temps = []
    # avg_temps = []
    for i, r in route_data.iterrows():
        location_coords = (r['start_location'][1], r['start_location'][0] ) # geometry locations are (E, N) not (N, E)...
        elevation = elevation_profiles[r['shape_id']].mean()
        min_temp, max_temp, avg_temp = location_design_temp(location_coords, elevation, num_years=10, percentiles=[1, 99])
        min_temps.append(min_temp)
        max_temps.append(max_temp)
        # avg_temps.append(avg_temp)

    route_data['min_temp'] = min_temps
    route_data['max_temp'] = max_temps

    model = LinearRegressionAbdelatyModel()
    route_data = model.predict_routes(route_data, PL=passenger_loading)

    """ 
    Work out busses in traffic
    """
    buffer = 0.1        # add a 10% buffer to trip times
    resolution = 10      # resolution in 5 mins
    time_slot_edges = np.arange(0, 24*60, resolution)      # time slot edges in minutes
    c_starts = np.histogram(trip_totals.trip_start_time/60, bins=time_slot_edges)[0]

    buffered_time_ends = trip_totals.trip_start_time/60 + trip_totals.trip_duration/60*(1+buffer)
    c_ends = np.histogram(buffered_time_ends, bins=time_slot_edges)[0]

    busses_in_traffic = np.cumsum(c_starts) - np.cumsum(c_ends)

    plt.step(time_slot_edges[:-1]/60, busses_in_traffic)
    plt.title('Busses in traffic on {} sydney bus routes'.format(len(route_short_names)))
    plt.xlabel('Time of day (hour)')
    plt.ylabel('Number of busses')
    plt.show()

    #
    trip_totals['max_EC_total'] = np.nan
    trip_totals['min_EC_total'] = np.nan
    for i, r in route_data.iterrows():
        window = r.window
        shape_id = r.shape_id
        max_EC_total = r.max_EC_total
        min_EC_total = r.min_EC_total
        inds = (trip_totals.shape_id==shape_id) & (trip_totals.window==window)
        # trip_totals[inds]['max_EC_total'] == max_EC_total
        trip_totals.loc[inds, 'max_EC_total'] = max_EC_total
        trip_totals.loc[inds, 'min_EC_total'] = min_EC_total

    # work out energy requirements of busses current in traffic
    # worst case first
    e_starts = np.histogram(trip_totals.trip_start_time/60, bins=time_slot_edges, weights=trip_totals.max_EC_total*(1+buffer))[0]
    e_ends = np.histogram(buffered_time_ends, bins=time_slot_edges, weights=trip_totals.max_EC_total*(1+buffer))[0]

    energy_req_busses_max = np.cumsum(e_starts) - np.cumsum(e_ends)

    # bset case
    e_starts = np.histogram(trip_totals.trip_start_time/60, bins=time_slot_edges, weights=trip_totals.min_EC_total*(1+buffer))[0]
    e_ends = np.histogram(buffered_time_ends, bins=time_slot_edges, weights=trip_totals.min_EC_total*(1+buffer))[0]

    energy_req_busses_min = np.cumsum(e_starts) - np.cumsum(e_ends)

    plt.step(time_slot_edges[:-1],energy_req_busses_max,label='worst temperature')
    plt.step(time_slot_edges[:-1], energy_req_busses_min, label='best temperature')
    plt.ylabel('Energy requirement (kwh)')
    plt.title('Energy requirement of busses in traffic on {} sydney bus routes'.format(len(route_short_names)))
    plt.xlabel('Time of day (hour)')
    plt.legend()
    plt.show()