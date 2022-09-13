import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import srtm
import pytz
from sklearn.metrics.pairwise import haversine_distances
import geopandas as gpd

import sys
sys.path.insert(0, "/group/RouteZero/project_code/RouteZero")
from RouteZero import weather

# import RouteZero.weather as weather

def etl_bus_data():
    in_folder = "/routezero/data/bus_data/raw_csv/"
    out_folder = "/routezero/data/bus_data/processed_csv/"
    files = [filename for filename in os.listdir(in_folder) if filename.endswith(".csv")]

    data_dict = {}
    completeness = []
    for file in tqdm(files, desc="processing raw bus data files"):
        df = pd.read_csv(os.path.join(in_folder, file), parse_dates=['Unnamed: 0']).rename(columns={"Unnamed: 0":"datetime"}).set_index("datetime")
        df.dropna(inplace=True)

        bus_ids = df['BUS_ID'].unique().tolist()

        for bus_id in bus_ids:
            df_bus = df[df.BUS_ID==bus_id]
            if bus_id in data_dict:
                data_dict[bus_id] = pd.concat([data_dict[bus_id], df_bus])
            else:
                data_dict[bus_id] = df_bus

    for bus_id in data_dict:
        df = data_dict[bus_id].sort_index()
        td = df.index.max()-df.index.min()
        mins = int(td.total_seconds()/60)+1
        N = len(df)
        print("bus {} start {} end {} with {:.2f} percent complete data".format(bus_id, df.index.min(), df.index.max(),N/mins*100))
        completeness.append(N/mins*100)
        df.to_csv(out_folder + str(bus_id) + '.csv')
    print("average completeness {}".format(np.mean(completeness)))
    print("best completeness {}".format(np.max(completeness)))
    print("worst completeness {}".format(np.min(completeness)))

def etl_transport():
    sheets = ["Data 1", "Data 2", "Data", "Data"]
    xls = pd.ExcelFile('/routezero/data/transport/transportNSWdata_Jan2May.xlsx')
    xls2 = pd.ExcelFile("/routezero/data/transport/transportNSW_JuneJuly.xlsx")
    xls3 = pd.ExcelFile("/routezero/data/transport/transportNSW_August.xlsx")

    for k, sheet in enumerate(sheets):
        if k < 2:
            # continue
            all = pd.read_excel(xls, sheet)
        elif k==2:
            # continue
            all = pd.read_excel(xls2, sheet)
        elif k==3:
            all = pd.read_excel(xls3, sheet)
            all["DATE"] = all["DATE"].astype(str)

        REGOS = all['REGO'].unique().tolist()
        trip_dict = {}
        count = 0

        for REGO in tqdm(REGOS, desc="Extracting trip data by bus for data sheet {} of {}".format(k+1, len(sheets))):
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

                        g["IN_TRANSIT_DEPART"] = pd.to_numeric(g["IN_TRANSIT_DEPART"])
                        av_pass = np.average(g['IN_TRANSIT_DEPART'].values[:-1], weights=pd.to_timedelta(time_delta).total_seconds().values)

                        trip = {'start_time':depart_datetime[0],
                                "end_time":arrive_datetime[-1],
                                "duration":(pd.to_datetime(arrive_datetime[-1]) - pd.to_datetime(depart_datetime[0])).total_seconds(),
                                "num_stops":g['SEQ'].values[-1],
                                "direction":g['DIRECTION'].values[0],
                                "REGO":REGO,
                                "ROUTE":ROUTE,
                                "average_passengers":av_pass}

                        trip_dict[count] = trip

                        count += 1

        trip_df = pd.DataFrame.from_dict(trip_dict, orient='index')
        trip_df.to_csv('routezero/data/transport/sheet{}_trips.csv'.format(k+1))
    return trip_df

def etl_merge_transport_bus():

    trip_data_1 = pd.read_csv('/routezero/data/transport/sheet1_trips.csv', index_col='Unnamed: 0', parse_dates=[
        'start_time', 'end_time'])
    trip_data_2 = pd.read_csv('/routezero/data/transport/sheet2_trips.csv', index_col='Unnamed: 0', parse_dates=['start_time', 'end_time'])
    trip_data_3 = pd.read_csv('/routezero/data/transport/sheet3_trips.csv', index_col='Unnamed: 0', parse_dates=['start_time', 'end_time'])
    trip_data_4 = pd.read_csv('/routezero/data/transport/sheet4_trips.csv', index_col='Unnamed: 0', parse_dates=['start_time', 'end_time'])
    trip_data = pd.concat([trip_data_1, trip_data_2, trip_data_3, trip_data_4], ignore_index=True)

    bus_data_folder = '/routezero/data/bus_data/processed_csv/'
    bus_files = [filename for filename in os.listdir(bus_data_folder) if filename.endswith(".csv")]
    bus_data_ids = [x[:-4] for x in bus_files]

    REGOS = trip_data['REGO'].unique().tolist()

    for REGO in REGOS:
        print(REGO)
        single_bus_trips = trip_data[trip_data.REGO==REGO]
        if REGO[2:] in bus_data_ids:
            bus_data = pd.read_csv('/routezero/data/bus_data/processed_csv/'+REGO[2:]+".csv", index_col='datetime', parse_dates=['datetime'])
            data_start_date = bus_data.index[0]
            data_end_date = bus_data.index[-1]

            for i in tqdm(range(len(single_bus_trips)), desc="merging data for bus {}".format(REGO)):
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

                    ## GPS_track
                    gps_track = bus_data.iloc[i_start:i_end]["GPS_TRACKING"].values
                    path_x = []
                    path_y = []
                    for p in gps_track:
                        yx = p[1:-1].split(",")
                        path_x.append(yx[1])
                        path_y.append(yx[0])
                    path_x = ",".join(path_x)
                    path_y = ",".join(path_y)




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
                    trip_data.loc[trip.name, "path_x"] = path_x
                    trip_data.loc[trip.name, "path_y"] = path_y

    trip_data[trip_data["delta SOC (%)"] <= 0] = np.nan      # should always be using energy # todo: check impact of this
    trip_data.dropna(inplace=True)
    trip_data.reset_index(inplace=True)
    trip_data.to_csv("/routezero/data/trip_data.csv")
    print("Trip data merged for a total of {} trips".format(len(trip_data)))



def etl_add_historical_temps():
    aest = pytz.timezone("Australia/Sydney")

    trip_data = pd.read_csv("/routezero/data/trip_data.csv", parse_dates=["start_time", "end_time"], index_col="Unnamed: 0")
    trip_data['start_time'] = trip_data['start_time'].dt.tz_localize(aest)
    trip_data['end_time'] = trip_data['end_time'].dt.tz_localize(aest)
    trip_data.dropna(inplace=True)                                  # shouldn't need this

    weather_df = pd.read_csv("/routezero/data/routezero_weather.csv")

    weather_df['datetime'] = pd.to_datetime(weather_df['dt'], unit='s')
    weather_df.datetime = weather_df.datetime.dt.tz_localize('UTC').dt.tz_convert(aest)

    weather_df.drop(columns=["feels_like", "pressure", "humidity", "dew_point", "clouds", "visibility",
                             "wind_speed", "wind_deg", "wind_gust", "rain__1h", "weather__description",
                             "weather__main"], inplace=True)

    weather_df.sort_values(by="dt", inplace=True)
    weather_df.set_index('datetime', inplace=True)

    # separate by station
    weather_df['station_id'] = np.nan
    unique_stations = weather_df.drop_duplicates(subset=["lat", "lon"])
    station_lats = unique_stations['lat'].to_list()
    station_lons = unique_stations['lon'].to_list()
    weather_df.loc[(weather_df.lat == station_lats[0]) & (weather_df.lon == station_lons[0]), 'station_id'] = 0
    weather_df.loc[(weather_df.lat == station_lats[1]) & (weather_df.lon == station_lons[1]), 'station_id'] = 1
    weather_df.loc[(weather_df.lat == station_lats[2]) & (weather_df.lon == station_lons[2]), 'station_id'] = 2
    weather_df['station_id'] = weather_df['station_id'].astype(int)

    weather_data_start = weather_df.index.min()
    weather_data_end = weather_df.index.max()

    # closest station to start of each trip
    X = np.deg2rad(np.vstack([trip_data['start_loc_y'].to_numpy(), trip_data['start_loc_x'].to_numpy()]).T)
    Y = np.deg2rad(np.vstack([np.array(station_lats), np.array([station_lons])]).T)
    dists = haversine_distances(X, Y)
    start_stations = np.argmin(dists, axis=1)

    # closest station to end of each trip
    X = np.deg2rad(np.vstack([trip_data['end_loc_y'].to_numpy(), trip_data['end_loc_x'].to_numpy()]).T)
    Y = np.deg2rad(np.vstack([np.array(station_lats), np.array([station_lons])]).T)
    dists = haversine_distances(X, Y)
    end_stations = np.argmin(dists, axis=1)

    for i in tqdm(range(len(trip_data)), desc="Adding historical temperature record to trips"):
        trip = trip_data.iloc[i]

        start_dt = trip['start_time']
        end_dt = trip['end_time']

        start_station_data = weather_df[weather_df.station_id == start_stations[i]]
        end_station_data = weather_df[weather_df.station_id == end_stations[i]]

        if ((start_dt > weather_data_start) and (
                end_dt < weather_data_end)):  # check teh data contains the dates of trip
            start_temp = \
            start_station_data.loc[[abs(start_station_data['dt'] - start_dt.timestamp()).idxmin()]]["temp"].values[0]
            end_temp = end_station_data.loc[[abs(end_station_data['dt'] - end_dt.timestamp()).idxmin()]]["temp"].values[
                0]
            temp = (start_temp + end_temp) / 2

            ind = trip_data.index.values[i]
            trip_data.loc[ind, "temp"] = temp
            trip_data.loc[ind, "temp_data_exist"] = True
        else:
            ind = trip_data.index.values[i]
            trip_data.loc[ind, "temp"] = np.nan
            trip_data.loc[ind, "temp_data_exist"] = False

    trip_data["start_time"] = trip_data["start_time"].dt.tz_localize(None)
    trip_data["end_time"] = trip_data["end_time"].dt.tz_localize(None)
    trip_data.to_csv("/routezero/data/trip_data.csv")

def etl_average_temperature_profile():
    weather_df = pd.read_csv("/routezero/data/routezero_weather.csv")

    aest = pytz.timezone("Australia/Sydney")
    weather_df['datetime'] = pd.to_datetime(weather_df['dt'], unit='s')
    weather_df.datetime = weather_df.datetime.dt.tz_localize('UTC').dt.tz_convert(aest)

    weather_df.drop(columns=["feels_like","pressure","humidity","dew_point","clouds","visibility",
                             "wind_speed","wind_deg","wind_gust","rain__1h","weather__description",
                             "weather__main"],inplace=True)

    weather_df.sort_values(by="dt", inplace=True)
    weather_df.set_index('datetime',inplace=True)

    # separate by station
    weather_df['station_id'] = np.nan
    unique_stations = weather_df.drop_duplicates(subset=["lat","lon"])
    station_lats = unique_stations['lat'].to_list()
    station_lons = unique_stations['lon'].to_list()
    weather_df.loc[(weather_df.lat==station_lats[0]) & (weather_df.lon==station_lons[0]),'station_id'] = 0
    weather_df.loc[(weather_df.lat==station_lats[1]) & (weather_df.lon==station_lons[1]),'station_id'] = 1
    weather_df.loc[(weather_df.lat==station_lats[2]) & (weather_df.lon==station_lons[2]),'station_id'] = 2
    weather_df['station_id'] = weather_df['station_id'].astype(int)

    av = np.zeros(24,)
    count = 0

    for station in range(3):
        single_station = weather_df[weather_df.station_id == station].copy()
        days = [g[1] for g in single_station.groupby(single_station.index.date)]
        for day in days:
            plt.plot((day.dt-day.dt.values[0])/60/60, day.temp)
            if len(day.temp)==24:
                av += day.temp.values
                count += 1

    av = av / count

    plt.plot(np.arange(24), av, linewidth=2, color='k')
    plt.show()

    print(av)

def etl_add_estimated_temp():
    trip_data = pd.read_csv("/routezero/data/trip_data.csv", parse_dates=["start_time", "end_time"], index_col="Unnamed: 0")
    for i in tqdm(range(len(trip_data)), "Filling missing temp with estimated temperature based on daily min and max"):
        trip = trip_data.iloc[i]
        index = trip_data.index.values[i]
        if not trip["temp_data_exist"]:
            day = datetime.datetime(year=trip["start_time"].year, month=trip["start_time"].month, day=trip["start_time"].day)

            # temperature at start
            tmin, tmax = weather.get_temperature_by_date([trip["start_loc_y"], trip["start_loc_x"]], trip["start_el"], day)
            start_hour = trip["start_time"].hour + trip["start_time"].minute/60
            start_temp = weather.daily_temp_profile(start_hour, tmin, tmax)

            # temperature at end
            tmin, tmax = weather.get_temperature_by_date([trip["end_loc_y"], trip["end_loc_x"]], trip["end_el"], day)
            end_hour = trip["end_time"].hour + trip["end_time"].minute/60
            end_temp = weather.daily_temp_profile(end_hour, tmin, tmax)

            temp = (start_temp + end_temp)/2

            trip_data.loc[index, "temp"] = temp

    trip_data.to_csv("/routezero/data/trip_data.csv")


def analyse_gps_and_shape():
    trip_data = pd.read_csv("/routezero/data/trip_data.csv", parse_dates=["start_time", "end_time"], index_col="Unnamed: 0")

    gtfs_trip_data = pd.read_csv("../data/gtfs/greater_sydney/trip_data.csv")
    shapes = gpd.read_file("../data/gtfs/greater_sydney/shapes.shp")

    dist_away = []
    gtfs_grad = []
    gtfs_shape_ids = []
    gtfs_length = []
    for k in tqdm(range(len(trip_data)), desc="comparing gps tracking to gtfs shape files"):
        trip = trip_data.iloc[k]
        route = trip["ROUTE"]

        path_x = [float(x) for x in trip["path_x"].split(",")]
        path_y = [float(x) for x in trip["path_y"].split(",")]

        shape_ids = gtfs_trip_data[gtfs_trip_data["route_short_name"]==route]["shape_id"].unique().tolist()
        subset_shapes = shapes[shapes["shape_id"].isin(shape_ids)]

        X = np.vstack([path_y, path_x]).T

        geometry = subset_shapes["geometry"].values


        xy_list = []
        dist_list = []
        start_start = []
        start_end = []
        for i,g in enumerate(geometry):
            xy = g.xy
            Y = np.vstack([xy[1], xy[0]]).T
            dists = haversine_distances(X, Y) * 6371    # dists in kms

            start_start.append(dists[0,0])
            start_end.append(dists[0,-1])
            xy_list.append(xy)
            dist_list.append(dists.min(axis=1).mean())

        inds_sort = np.argsort(dist_list)
        trip_dist = trip['distance (m)']
        shape_lengths = subset_shapes['length (m)'].to_numpy()
        ii_fallback = None
        ii_best = None
        for ii in inds_sort:
            if (start_start[ii] < start_end[ii]):
                ii_fallback = ii
                if np.abs(trip_dist - shape_lengths[ii]) < 2000:
                    ii_best = ii
                    break

        if ii_best:
            i_closest = ii_best
        elif ii_fallback:
            i_closest = ii_fallback
        else:
            i_closest = inds_sort[0]
        min_dist = dist_list[i_closest]

        ind = subset_shapes.index.values[i_closest]
        s_id = subset_shapes.loc[ind, "shape_id"]
        gtfs_grad.append(gtfs_trip_data[gtfs_trip_data["shape_id"]==s_id]["average_gradient_%"].values[0])
        gtfs_shape_ids.append(s_id)
        dist_away.append(min_dist)
        gtfs_length.append(subset_shapes.loc[ind, "length (m)"])

    plt.hist(dist_away, bins=30, range=[0,15])
    plt.show()

    trip_data['dist from shape id'] = dist_away
    trip_data['gradient of shape id'] = gtfs_grad
    trip_data['gtfs shape id'] = gtfs_shape_ids
    trip_data['gtfs length (m)'] = gtfs_length


    return trip_data

def remove_gps_outliers(trip_data):
    print("removing {} trips as too far from shape id".format((trip_data["dist from shape id"] > 5).sum()))
    trip_data = trip_data[trip_data["dist from shape id"] < 5]
    length_diff = trip_data['distance (m)'] - trip_data['gtfs length (m)']
    print("removing {} trips as having too great a difference in length from shape id".format(np.sum(np.abs(length_diff)>3000)))
    trip_data = trip_data[np.abs(length_diff)<=3000]
    return trip_data

def add_bus_type_and_energy_use(trip_data):
    trip_data = trip_data.copy()
    trip_data["bus type"] = ""
    trip_data["ec (kWh)"] = 0
    #
    byd_bus_df = pd.read_csv("/routezero/data/BYD_BUS_IDS.csv")
    byd_bus_regos =["MO"+ str(s) for s in byd_bus_df['BYD_BUS_IDS'].to_list()]
    trip_data.loc[trip_data["REGO"].isin(byd_bus_regos), "bus type"] = "BYD"
    trip_data.loc[~trip_data["REGO"].isin(byd_bus_regos), "bus type"] = "Yutong"

    byd_cap = 368       #kWh
    yutong_cap = 422    #kWh

    trip_data.loc[trip_data["bus type"]=="BYD",'ec (kWh)'] = trip_data[trip_data["bus type"]=="BYD"]["delta SOC (%)"]/100 * byd_cap
    trip_data.loc[trip_data["bus type"]=="Yutong",'ec (kWh)'] = trip_data[trip_data["bus type"]=="Yutong"]["delta SOC (%)"]/100 * yutong_cap
    #
    trip_data["ec/km (kWh/km)"] = trip_data["ec (kWh)"]/trip_data['distance (m)']*1000

    return trip_data

def calculate_measurement_variance(trip_data):
    trip_data["1SOC"] = 4.22
    trip_data.loc[trip_data["bus type"] == "BYD", "1SOC"] = 3.68
    tmp = (trip_data["1SOC"] / (trip_data["distance (m)"] / 1000))
    trip_data["meas_variance"] = ((tmp*2) ** 2 / 12)
    # v = np.sqrt(((2 * tmp) ** 2 / 12)).mean()
    return trip_data

if __name__=="__main__":
    transport_df = etl_transport()                   # 1
    etl_bus_data()                    # 2
    etl_merge_transport_bus()         # 3
    etl_add_historical_temps()        # 4
    etl_add_estimated_temp()          # 5
    # #
    trip_data = analyse_gps_and_shape()
    trip_data = remove_gps_outliers(trip_data)
    # # #
    trip_data = add_bus_type_and_energy_use(trip_data)
    trip_data = calculate_measurement_variance(trip_data)
    # #
    trip_data.to_csv("/routezero/data/trip_data.csv")
    # #
    # trip_data = pd.read_csv("../data/trip_data.csv", parse_dates=["start_time", "end_time"], index_col="Unnamed: 0")

