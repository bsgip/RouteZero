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


plt.plot(df_list[49]['GPS_E'], df_list[49]['GPS_N'])
plt.show()

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

    plt.subplot(3,1,1)
    plt.plot(df['STATE_OF_CHARGE'])
    # plt.plot(df_list[10]['SOC_smooth'])

    plt.subplot(3,1,2)
    plt.plot(charging)

    plt.subplot(3,1,3)
    plt.plot(odo_diff > 0)
    plt.tight_layout()
    plt.show()


# todo: 6) add temperature data

# todo: 7) see if we can match to trips / routes


## Try writing a Particle Filter for the state of charge problem

