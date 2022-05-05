import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import srtm
import geocoder
import geopy
from geopy import distance as geo_dist
from geopy import Point

depot_GPS_point = Point(longitude=151.15932034790586, latitude=-33.876127015878886)

df_BYD = pd.read_csv('data/BYD_BUS_IDS.csv')

df = pd.read_csv('data/bus_data/8387_ANU_minute_data_2022-01-01_2022-01-31.csv', parse_dates=['Unnamed: 0'])
df.rename(columns={'Unnamed: 0':'datetime'}, inplace=True)


# todo: 1) Expand GPS data to make processing later easier
df[['GPS_N','GPS_E']] = df['GPS_TRACKING'].str.strip(')').str.strip('(').str.split(',',1,expand=True).apply(pd.to_numeric)

# todo: 2) assign the correct battery capacity based on which buy type it is
if len(df['BUS_ID'].unique()) > 1:
    sys.exit("file contains multiple bus IDs - this needs accounting for")
bus_id = df['BUS_ID'].to_numpy()[0]
if bus_id in df_BYD['BYD_BUS_IDS'].to_numpy():  # BUS is a BYD bus
    max_battery_capacity = 368
else:                                           # BUS is a Yutong bus
    max_battery_capacity = 422

# todo: add battery capacity at every time instant
df['Battery_capacity'] = max_battery_capacity * df['STATE_OF_CHARGE'] / 100

# todo: 3) break into chunks based on segments of valid data
# remove leading or trailing nans
df = df[df.first_valid_index():df.last_valid_index()+1]

# break into chunks based on location of nan
valid = df.notna().all(axis=1)
df['segment'] = (valid.astype(int).diff()==1).astype(int).cumsum()
df.dropna(inplace=True)

grouped = df.groupby(df.segment)
df_list = [g.dropna() for i,g in grouped]

for i, df in enumerate(df_list):
    print('group {} has length {}'.format(i, len(df)))

# todo: 4) add elevation data
for df in df_list:
    GPS_N = df['GPS_N'].to_numpy()
    GPS_E = df['GPS_E'].to_numpy()
    elevation_data = srtm.get_data()
    els = np.array([elevation_data.get_elevation(n, e) for n, e in zip(GPS_N, GPS_E)])
    df['elevation'] = els



# todo: 5) determine if charging

for df in df_list:
    SOC_change = df['STATE_OF_CHARGE'].diff().to_numpy()
    SOC = df['STATE_OF_CHARGE'].to_numpy()
    odo = df['ODOMETER'].to_numpy()
    odo_diff = df['ODOMETER'].diff().to_numpy()
    charging = np.zeros((len(SOC),), dtype=bool)

    is_moving = False
    is_charging = False


    for i in range(0, len(SOC)):
        is_moving = odo_diff[i] > 0       # moved from last odo measurement
        next_increase = np.where(SOC_change[i:] > 0)[0]     # time steps till increases
        next_decrease = np.where(SOC_change[i:] < 0)[0]     # time steps till decreases
        # pad the above
        next_increase = np.hstack([next_increase, len(SOC)])
        next_decrease = np.hstack([next_decrease, len(SOC)])

        if is_charging:                     # its state was charging
            if (is_moving) or (next_decrease[0]==0):                   # if it moved or SOC decreased
                is_charging = False         #
            # while charging estimate the rate at which charging occurs
            if (len(next_increase) >= 2) and (next_increase[1] < next_decrease[0]):
                inv_rate = inv_rate = next_increase[1]-next_increase[0]
            if next_decrease[0] <= next_increase[0]:         # next change is a decrease
                if i - last_increase > inv_rate:            # no longer charging
                    is_charging=False
            elif (next_increase[0] > i+inv_rate) and (i-last_increase > inv_rate):
                is_charging=False

            if next_increase[0]==0:
                last_increase=i


        else:
            if not is_moving:               # can only start charging if its not moving

                if (len(next_increase) >= 2) and (next_increase[1] < next_decrease[0]):     # we want at least two consecutive increases before a decrease
                    # estimate charging rate
                    inv_rate = next_increase[1]-next_increase[0]
                    if next_increase[0] <= inv_rate:
                        is_charging = True


        charging[i] = is_charging

    df['is_charging'] = charging

    # plt.subplot(3,1,1)
    # plt.plot(df['STATE_OF_CHARGE'])
    # # plt.plot(df_list[10]['SOC_smooth'])
    #
    # plt.subplot(3,1,2)
    # plt.plot(charging)
    #
    # plt.subplot(3,1,3)
    # plt.plot(odo_diff > 0)
    # plt.tight_layout()
    # plt.show()


# todo: 6) Calculate speed and accelerations
for df in df_list:
    df['speed_mps'] = df['ODOMETER'].diff().fillna(method='ffill')*1000/60      # assumes 1 minute time intervals (which should be correct)
    df['acceleration'] = df['speed_mps'].diff().fillna(method='ffill')/60
    # print('max acceleration is {} and max deceleration is {}'.format(df.acceleration.max(),df.acceleration.min()))

# plt.subplot(3,1,1)
# plt.plot(df['ODOMETER'][:100])
#
# plt.subplot(3,1,2)
# plt.plot(df['speed_mps'][:100])
#
# plt.subplot(3,1,3)
# plt.plot(df['acceleration'])


plt.tight_layout()
plt.show()

# todo: 6) add temperature data


# todo: 7) see if we can match to trips / routes
subset_shapes = pd.read_csv('data/pre_processed_gtfs_data/subset_shapes.csv')
import geopandas as gpd
subset_shapes['geometry'] = gpd.GeoSeries.from_wkt(subset_shapes['geometry'])
gdf = gpd.GeoDataFrame(subset_shapes, geometry=subset_shapes.geometry)

#todo: this equation is broke for some reason
def haversine(N1, E1, N2, E2):
    """
    :param N1: Array of latitudes for the first points
    :param E1: Array of longitudes for the first points
    :param N2: Array of latitudes for the first points
    :param E2: Array of longitudes for the first points
    :return dists: array of distances between all points
    """
    r = 6371.0
    N1 = np.atleast_2d(N1).T
    E1 = np.atleast_2d(E1).T
    N2 = np.atleast_2d(N2)
    E2 = np.atleast_2d(E2)

    dists = 2*r*np.arcsin(np.sqrt(np.sin((N1-N2)/2)**2 + np.cos(N1)*np.cos(N2)*np.sin((E1-E2)/2)**2))
    return dists

route_N_lists = []
route_E_lists = []
max_dist = []
for i, r in gdf.iterrows():
    N = []
    E = []
    for p in r.geometry.coords:
        N.append(p[1])
        E.append(p[0])
    N = np.array(N)
    E = np.array(E)
    route_N_lists.append(N)
    route_E_lists.append(E)

    plt.plot(E, N, color='k')

    dists = np.sqrt((np.atleast_2d(E).T-np.atleast_2d(df_list[49]['GPS_E'].to_numpy()))**2+(np.atleast_2d(N).T-np.atleast_2d(df_list[49]['GPS_N'].to_numpy()))**2)


    # dists = haversine(N, E, df_list[49]['GPS_N'].to_numpy(), df_list[49]['GPS_E'].to_numpy())
    max_dist.append(dists.min(axis=1).max())

max_dist = np.array(max_dist)

plt.plot(df_list[49]['GPS_E'], df_list[49]['GPS_N'],marker='.', linestyle='--')
plt.plot(route_E_lists[41],route_N_lists[41], color='k')

# plt.xlim((df_list[49]['GPS_E'].min(), df_list[49]['GPS_E'].max()))
# plt.ylim((df_list[49]['GPS_N'].min(), df_list[49]['GPS_N'].max()))
plt.xlim((151.05, 151.26))
plt.ylim((-33.95, -33.8))
plt.show()
# for i,s in subset_shapes.iterrows():
#     for p in s.geometry:
#         print(p)




## Try writing a Particle Filter for the state of charge problem

