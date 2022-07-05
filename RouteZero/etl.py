import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import srtm

def etl_bus_data():
    in_folder = "../data/bus_data/raw_csv/"
    out_folder = "../data/bus_data/processed_csv/"
    files = [filename for filename in os.listdir(in_folder) if filename.endswith(".csv")]

    data_dict = {}
    for file in tqdm(files):
        df = pd.read_csv(os.path.join(in_folder, file), parse_dates=['Unnamed: 0']).rename(columns={"Unnamed: 0":"datetime"}).set_index("datetime")
        df.dropna(inplace=True)

        BUS_ID = df['BUS_ID'].unique().tolist()
        assert len(BUS_ID)==1, ' There should only be one bus id per raw file'

        if BUS_ID[0] in data_dict:
            data_dict[BUS_ID[0]] = pd.concat([data_dict[BUS_ID[0]], df])
        else:
            data_dict[BUS_ID[0]] = df

    for bus_id in data_dict:
        df = data_dict[bus_id].sort_index()
        df.to_csv(out_folder + str(bus_id) + '.csv')

def etl_transport():
    xls = pd.ExcelFile('../data/transport/transportNSWdata.xlsx')
    all = pd.read_excel(xls, 'DATA SHEET 1')
    count = 0
    trip_dict = {}
    REGOS = all['REGO'].unique().tolist()

    for REGO in tqdm(REGOS):
        # print(REGO)
        single_bus = all[all.REGO==REGO].copy()

        single_bus['arrive_datetime'] = pd.to_datetime(single_bus['DATE'] + ' ' + single_bus['ACTUAL_ARRIVE_TIME'])
        single_bus['depart_datetime'] = pd.to_datetime(single_bus['DATE'] + ' ' + single_bus['ACTUAL_DEPART_TIME'])

        ROUTES = single_bus['ROUTE'].unique().tolist()

        for ROUTE in ROUTES:
            # print(ROUTE)
            single_route = single_bus[single_bus.ROUTE==ROUTE].copy()
            single_route = single_route.dropna()

            # get trip index
            single_route.sort_values(by='arrive_datetime', inplace=True)
            single_route['trip_num'] = (single_route['SEQ'].diff() < 0).astype(int).cumsum()

            gb = single_route.groupby(by='trip_num')


            for (i,g) in gb:
                g.dropna(inplace=True)
                if not len(g):
                    continue
                if (len(g) == g['SEQ'].values[-1]) and (len(g) > 1):       # hopefully has all stops
                    depart_datetime = g['depart_datetime'].values
                    arrive_datetime = g['arrive_datetime'].values
                    time_delta = arrive_datetime[1:] - depart_datetime[:-1]

                    # replace <10 pass with 5

                    inds = g['IN_TRANSIT_DEPART']=='<10'
                    missing = np.sum(inds) / len(g)

                    g.loc[g['IN_TRANSIT_DEPART']=='<10','IN_TRANSIT_DEPART'] = np.floor(10*np.random.rand(sum(g['IN_TRANSIT_DEPART']=='<10')))
                    g['IN_TRANSIT_DEPART'] = pd.to_numeric(g['IN_TRANSIT_DEPART'])
                    av_pass = np.average(g['IN_TRANSIT_DEPART'].values[:-1], weights=pd.to_timedelta(time_delta).total_seconds().values)
                    sel = (~inds).values
                    sel[-1] = False
                    if sum(sel):
                        pass_true_part = np.average(g.loc[sel,'IN_TRANSIT_DEPART'].values, weights=pd.to_timedelta(time_delta[sel[:-1]]).total_seconds().values)
                        pass_true_weight = sum(pd.to_timedelta(time_delta[sel[:-1]]).total_seconds().values)/sum(pd.to_timedelta(time_delta).total_seconds().values)
                    else:
                        pass_true_part = 0
                        pass_true_weight = 0


                    trip = {'start_time':depart_datetime[0],
                            "end_time":arrive_datetime[-1],
                            "pass_filled":av_pass,
                            "duration":(pd.to_datetime(arrive_datetime[-1]) - pd.to_datetime(depart_datetime[0])).total_seconds(),
                            "num_stops":g['SEQ'].values[-1],
                            "direction":g['DIRECTION'].values[0],
                            "REGO":REGO,
                            "ROUTE":ROUTE,
                            "missing_pass":missing,
                            "pass_true_part":pass_true_part,
                            "pass_true_weight":pass_true_weight}

                    trip_dict[count] = trip

                    count += 1

    trip_df = pd.DataFrame.from_dict(trip_dict, orient='index')

    trip_df.to_csv('../data/transport/sheet1_trips.csv')
    return trip_df

def etl_merge_transport_bus():
    ## todo: also add elevation data once we get it

    trip_data_1 = pd.read_csv('../data/transport/sheet1_trips.csv', index_col='Unnamed: 0', parse_dates=['start_time', 'end_time'])
    trip_data_2 = pd.read_csv('../data/transport/sheet2_trips.csv', index_col='Unnamed: 0', parse_dates=['start_time', 'end_time'])
    trip_data = pd.concat([trip_data_1, trip_data_2], ignore_index=True)

    bus_data_folder = '../data/bus_data/processed_csv/'
    bus_files = [filename for filename in os.listdir(bus_data_folder) if filename.endswith(".csv")]
    bus_data_ids = [x[:-4] for x in bus_files]

    # REGOS = trip_data['REGO'].unique().tolist()
    REGOS = ["MO8380", "MO8387"]        # The two we actually have data for

    # REGO = 'MO8387'
    for REGO in REGOS:
        print(REGO)
        single_bus_trips = trip_data[trip_data.REGO==REGO]
        if REGO[2:] in bus_data_ids:
            bus_data = pd.read_csv('../data/bus_data/processed_csv/'+REGO[2:]+".csv", index_col='datetime', parse_dates=['datetime'])
            data_start_date = bus_data.index[0]
            data_end_date = bus_data.index[-1]

            for i in tqdm(range(len(single_bus_trips))):
                trip = single_bus_trips.iloc[i]
                start_dt = trip['start_time']
                end_dt = trip['end_time']

                # some checks
                if not ((start_dt > data_start_date) and (end_dt < data_end_date)):         # check teh data contains the dates of trip
                    checks_passed = False
                elif np.abs((bus_data.index - start_dt).total_seconds()).min() > 2*60:      # check we have data near the start of the trip
                     checks_passed = False
                    # continue
                elif np.abs((bus_data.index - end_dt).total_seconds()).min() > 2*60:        # check we have data near the end of the trip
                    checks_passed = False
                else:
                    checks_passed = True


                if checks_passed:

                    i_start = np.argmin(np.abs((bus_data.index - start_dt).total_seconds()))
                    i_end = np.argmin(np.abs((bus_data.index - end_dt).total_seconds()))

                    if i_start==i_end:
                        continue

                    start_data = bus_data.iloc[i_start]
                    end_data = bus_data.iloc[i_end]

                    distance = (end_data['ODOMETER'] - start_data['ODOMETER']) * 1000   # in

                    if (len(bus_data.iloc[i_start:i_end].drop_duplicates(subset=['STATE_OF_CHARGE',"ODOMETER"]))==1) or (distance==0):
                        continue

                    speed = distance / trip['duration']
                    start_SOC = start_data['STATE_OF_CHARGE']
                    end_SOC = end_data['STATE_OF_CHARGE']
                    delta_SOC = start_SOC - end_SOC
                    start_loc = [float(x) for x in start_data['GPS_TRACKING'][1:-1].split(",")]
                    end_loc = [float(x) for x in end_data['GPS_TRACKING'][1:-1].split(",")]

                    # elevations
                    elevation_data = srtm.get_data()
                    start_el = elevation_data.get_elevation(latitude=start_loc[0], longitude=start_loc[1])
                    end_el = elevation_data.get_elevation(latitude=end_loc[0], longitude=end_loc[1])
                    gradient = (end_el - start_el) / distance * 100

                    # put new data into
                    trip_data.loc[trip.name, "distance (m)"] = distance
                    trip_data.loc[trip.name, "speed (m/s)"] = speed
                    trip_data.loc[trip.name, "start SOC (%)"] = start_SOC
                    trip_data.loc[trip.name, "end SOC (%)"] = end_SOC
                    trip_data.loc[trip.name, "delta SOC (%)"] = delta_SOC
                    trip_data.loc[trip.name, "stops/km"] = trip["num_stops"]/distance*1000
                    trip_data.loc[trip.name, "start_loc_x"] = start_loc[1]
                    trip_data.loc[trip.name, "start_loc_y"] = start_loc[0]
                    trip_data.loc[trip.name, "end_loc_x"] = end_loc[1]
                    trip_data.loc[trip.name, "end_loc_y"] = end_loc[0]
                    trip_data.loc[trip.name, "start_el"] = start_el
                    trip_data.loc[trip.name, "end_el"] = end_el
                    trip_data.loc[trip.name, "gradient (%)"] = gradient


    trip_data[trip_data["delta SOC (%)"] <= 0] = np.nan      # should always be using energy
    trip_data.to_csv("../data/trip_data_merged.csv")

if __name__=="__main__":
    # etl_transport()
    # etl_bus_data()
    etl_merge_transport_bus()
    # #
    trip_data = pd.read_csv("../data/trip_data_merged.csv")
    trip_data.dropna()

    #
    plt.subplot(3,3,1)
    plt.scatter(trip_data['num_stops']/trip_data['distance (m)']*1000, 4.2*trip_data['delta SOC (%)']/trip_data['distance (m)']*1000)
    plt.xlabel("stops/km")

    plt.subplot(3,3,2)
    plt.scatter(trip_data['speed (m/s)'], 4.2*trip_data['delta SOC (%)']/trip_data['distance (m)']*1000)
    plt.xlabel("speed")

    plt.subplot(3,3,3)
    plt.scatter(trip_data['pass_true_part'], 4.2*trip_data['delta SOC (%)']/trip_data['distance (m)']*1000)
    plt.xlabel("av pass")

    plt.subplot(3,3,4)
    plt.scatter(trip_data['gradient (%)'], 4.2*trip_data['delta SOC (%)']/trip_data['distance (m)']*1000)
    plt.xlabel("gradient (%)")

    plt.subplot(3,3,5)
    plt.scatter(trip_data['start SOC (%)'], 4.2*trip_data['delta SOC (%)']/trip_data['distance (m)']*1000)
    plt.xlabel("start_soc")




    plt.tight_layout()
    plt.show()
    #
    #
    # #
    #
    #
