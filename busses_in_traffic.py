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
    # choose some things
    busiest = 'day'
    cutoffs = [0, 6, 9, 15, 19, 22, 24,
               30]  # optional input for splitting up the route summary information into time windows
    wrap_time = 24*60


    passenger_loading = 38

    gtfs_file = "./data/full_greater_sydney_gtfs_static.zip"        # location of the gtfs zip file
    # route_short_names = ["305", "320"]      # the short names of the routes we want to get summaries of
    route_short_names = ["305", "320", '389', '406',
                         '428', '430', '431', '433']
    # route_short_names = ["305", "320", '389', '406',
    #                      '428', '430', '431', '433',
    #                      '437', '438N', '438X', '440',
    #                      '441', '442', '445', '469',
    #                      '470', '502', '503', '504']      # the short names of the routes we want to get summaries of
    route_desc = 'Sydney Buses Network'     # optional input if we also want to filter by particular types of routes


    print('Processing routes '+", ".join(route_short_names))
    route_data, subset_shapes, elevation_profiles, trip_totals, _ = process_gtfs_routes(gtfs_file, route_short_names, cutoffs=cutoffs, busiest=busiest, route_desc=route_desc)

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
    deadhead_buffer = 0.1        # buffere for deadhead time adds half before and afteradd a 10% buffer to trip times
    resolution = 10      # resolution in 5 mins
    time_slot_edges = np.arange(0, 30*60, resolution)      # time slot edges in minutes

    buffered_time_starts = trip_totals.trip_start_time/60 - trip_totals.trip_duration/60*(deadhead_buffer/2)


    c_starts = np.histogram(buffered_time_starts, bins=time_slot_edges)[0]

    buffered_time_ends = trip_totals.trip_start_time/60 + trip_totals.trip_duration/60*(1+deadhead_buffer/2)
    c_ends = np.histogram(buffered_time_ends, bins=time_slot_edges)[0]

    busses_in_traffic = np.cumsum(c_starts) - np.cumsum(c_ends)

    t_starts = time_slot_edges[:-1]
    wrap_inds = t_starts > wrap_time    # todo: apply wrap to start as well in some way consistently
    busses_in_traffic[:sum(wrap_inds)] = busses_in_traffic[:sum(wrap_inds)]+busses_in_traffic[wrap_inds]

    t_wrap = t_starts[~wrap_inds]
    busses_in_traffic_wrap = busses_in_traffic[~wrap_inds]

    plt.step(t_wrap/60, busses_in_traffic_wrap)
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

    # energy usage in kw
    energy_usage = trip_totals.max_EC_total*(1+deadhead_buffer) / (trip_totals.trip_duration*(1+deadhead_buffer)/60/60)
    e_starts = np.histogram(buffered_time_starts, bins=time_slot_edges, weights=energy_usage)[0]
    e_ends = np.histogram(buffered_time_ends, bins=time_slot_edges, weights=energy_usage)[0]

    energy_req_busses_max = np.cumsum(e_starts) - np.cumsum(e_ends)

    energy_req_busses_max[:sum(wrap_inds)] = energy_req_busses_max[:sum(wrap_inds)]+energy_req_busses_max[wrap_inds]
    energy_req_busses_max_wrap = energy_req_busses_max[~wrap_inds]

    # best case
    energy_usage = trip_totals.min_EC_total * (1 + deadhead_buffer) / (trip_totals.trip_duration*(1+deadhead_buffer) / 60 / 60)
    e_starts = np.histogram(trip_totals.trip_start_time/60, bins=time_slot_edges, weights=energy_usage)[0]
    e_ends = np.histogram(buffered_time_ends, bins=time_slot_edges, weights=energy_usage)[0]

    energy_req_busses_min = np.cumsum(e_starts) - np.cumsum(e_ends)


    energy_req_busses_min[:sum(wrap_inds)] = energy_req_busses_min[:sum(wrap_inds)]+energy_req_busses_min[wrap_inds]
    energy_req_busses_min_wrap = energy_req_busses_min[~wrap_inds]

    # shift after 24 hours to morning


    plt.step(t_wrap/60,energy_req_busses_max_wrap,label='worst temperature')
    plt.step(t_wrap/60, energy_req_busses_min_wrap, label='best temperature')
    plt.ylabel('Energy usage (kw)')
    plt.title('Energy usage of busses in traffic on {} sydney bus routes'.format(len(route_short_names))+'\n Area under this graph is energy consumed (kwh)')
    plt.xlabel('Time of day (hour)')
    plt.legend()
    plt.show()

    # calculate cumulative energy consumption in kwh
    energy_consumption_max = np.cumsum(energy_req_busses_max) * (resolution / 60)
    energy_consumption_min = np.cumsum(energy_req_busses_min) * (resolution / 60)

    plt.step(time_slot_edges[:-1]/60, energy_consumption_max,label='worst temperature')
    plt.step(time_slot_edges[:-1]/60, energy_consumption_min, label='best temperature')
    plt.ylabel('Energy consumed (kwh)')
    plt.title('Cumulative energy consumed by busses in traffic on {} sydney bus routes'.format(len(route_short_names)))
    plt.xlabel('Time of day (hour)')
    plt.legend()
    plt.show()


    # calculate busses at depot graph
    min_required_busses = busses_in_traffic.max()
    max_battery_capacity = 368

    busses_at_depot = min_required_busses - busses_in_traffic_wrap
    max_cap_busses_at_depot = busses_at_depot * max_battery_capacity


    depart_trip_energy_reqs = np.histogram(buffered_time_starts, bins=time_slot_edges, weights=trip_totals.max_EC_total*(1+deadhead_buffer))[0]
    return_trip_enery_consumed = np.histogram(buffered_time_ends, bins=time_slot_edges, weights=trip_totals.max_EC_total*(1+deadhead_buffer))[0]
    depart_trip_energy_reqs[:sum(wrap_inds)] = depart_trip_energy_reqs[:sum(wrap_inds)]+depart_trip_energy_reqs[wrap_inds]
    depart_trip_energy_reqs = depart_trip_energy_reqs[~wrap_inds]
    return_trip_enery_consumed[:sum(wrap_inds)] = return_trip_enery_consumed[:sum(wrap_inds)]+return_trip_enery_consumed[wrap_inds]
    return_trip_enery_consumed = return_trip_enery_consumed[~wrap_inds]

    plt.subplot(2,2,1)
    plt.step(t_wrap/60, busses_at_depot)
    plt.xlabel('Time of day (hour)')
    plt.ylabel('Number of busses')
    plt.title('Busses at depot')

    plt.subplot(2,2,2)
    plt.step(t_wrap/60, max_cap_busses_at_depot)
    plt.xlabel('Time of day (hour)')
    plt.ylabel('Capacity (kwh)')
    plt.title('Battery capacity of busses at depot')

    plt.subplot(2,2,3)
    plt.step(t_wrap/60, depart_trip_energy_reqs,label='Departing trip requirements')
    plt.step(t_wrap/60, return_trip_enery_consumed, label='Returning trip consumed')
    plt.xlabel('Time of day (hour)')
    plt.ylabel('Energy (kwh)')
    plt.title('Trip energy')
    plt.legend()

    plt.subplot(2,2,4)
    plt.step(t_wrap/60, np.cumsum(depart_trip_energy_reqs))
    plt.xlabel('Time of day (hour)')
    plt.ylabel('Energy (kwh)')
    plt.title('Cumulative trip reuirements')
    plt.tight_layout()
    plt.show()

    ## Save some data for optimisation
    optim_data = pd.DataFrame(index=t_wrap,columns=['ED','ER','Nt'])
    optim_data['ED'] = depart_trip_energy_reqs
    optim_data['ER'] = return_trip_enery_consumed
    optim_data['Nt'] = busses_at_depot

# optim_data.to_csv('data/optim_data.csv')
