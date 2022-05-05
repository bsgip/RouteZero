import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from geopy import distance as geo_dist
from geopy import Point

depot_GPS_point = Point(longitude=151.15932034790586, latitude=-33.876127015878886)
depot_radius = 0.5      # distance to depot loc that we will consider a bus at the dept (km)
state_of_charge_quant = 1    # the state of charge information has a quantisation of 1%

df_BYD = pd.read_csv('data/BYD_BUS_IDS.csv')

df = pd.read_csv('data/bus_data/8387_ANU_minute_data_2022-01-01_2022-01-31.csv', parse_dates=['Unnamed: 0'])
df.rename(columns={'Unnamed: 0':'datetime'}, inplace=True)
df.fillna(method='bfill', inplace=True)

# Expand GPS data to make processing later easier
df[['GPS_N','GPS_E']] = df['GPS_TRACKING'].str.strip(')').str.strip('(').str.split(',',1,expand=True).apply(pd.to_numeric)



# bus_id =

if len(df['BUS_ID'].unique()) > 1:
    sys.exit("file contains multiple bus IDs - this needs accounting for")

bus_id = df['BUS_ID'].to_numpy()[0]

if bus_id in df_BYD['BYD_BUS_IDS'].to_numpy():  # BUS is a BYD bus
    max_battery_capacity = 368
else:                                           # BUS is a Yutong bus
    max_battery_capacity = 422


# todo: add a max_battery_capacity column to the data frame
df['Battery_capacity'] = max_battery_capacity * df['STATE_OF_CHARGE'] / 100

# todo: determine if bus at depot and parked
ODO = df['ODOMETER'].to_numpy()
df['at_depot'] = False
test= []
for i, r in df.iterrows():
    point1 = Point(latitude=r['GPS_N'], longitude=r['GPS_E'])
    if (geo_dist.distance(point1, depot_GPS_point).km < depot_radius) and not(ODO[i] < ODO[i+1] ):
        df.at[i,'at_depot'] = True
        test.append(geo_dist.distance(point1, depot_GPS_point).km)


# todo: Add a charging flag column to data frame
# need some window over which to look to see if its charging because of the resolution on SOC

window = 5      # in minutes
# Try applying moving average filter
df['SOC_MA'] = df['STATE_OF_CHARGE'].rolling(window).mean().shift(1-window, fill_value=df['STATE_OF_CHARGE'].tail().values[0])

df['is_charging'] = ((df['SOC_MA'].diff() >0) & (df['at_depot'])).shift(-1, fill_value=False)


test = df[df['is_charging'] * (~df['at_depot'])]

# todo: Work out depot location
charging_df = df[df.is_charging]
not_charging_df = df[~df.is_charging]
tmp = charging_df[(charging_df['GPS_E']<151.17)&(charging_df['GPS_E']>151.15)].reset_index()


plt.scatter(not_charging_df.GPS_E,not_charging_df.GPS_N,marker='.')
plt.scatter(charging_df.GPS_E,charging_df.GPS_N,marker='.')
plt.scatter(df[df['at_depot']].GPS_E,df[df['at_depot']].GPS_N,marker='.')
plt.show()



plt.subplot(3,1,1)
plt.scatter(charging_df['datetime'][:1000],charging_df['GPS_E'][:1000])
plt.subplot(3,1,2)
plt.plot(df['datetime'],df['STATE_OF_CHARGE'])
plt.scatter(charging_df['datetime'][:1000],charging_df['STATE_OF_CHARGE'][:1000],color='r',marker='.')
import datetime
plt.xlim([datetime.date(2022, 1, 3), datetime.date(2022, 1, 4)])
plt.subplot(3,1,3)
plt.plot(df['datetime'],df['ODOMETER'])
plt.xlim([datetime.date(2022, 1, 3), datetime.date(2022, 1, 4)])

plt.show()

# plt.plot(df['datetime'],df['SOC_MA'])
# plt.scatter(charging_df['datetime'][:1000],charging_df['STATE_OF_CHARGE'][:1000],color='r',marker='.')
# import datetime
# plt.xlim([datetime.datetime(2022, 1, 3,5), datetime.datetime(2022, 1, 3,10)])
# plt.show()

# todo:
# todo: add a at_depot flag column to data frame
# todo: Append elevation data to dataframe
# todo:

plt.subplot(3,1,1)
plt.plot(df.datetime[2300:2500], df.ODOMETER[2300:2500],label='Odometer')

plt.subplot(3,1,2)
plt.plot(df.datetime[2300:2500], df.STATE_OF_CHARGE[2300:2500], label='SOC')
plt.plot(df.datetime[2300:2500], df.SOC_MA[2300:2500], label='SOC_MA')
plt.plot(df.datetime[2300:2500], df.is_charging[2300:2500]*15+70, label='is_charging')
plt.show()


on_trip = False
trip_km = []
trip_charge = []
for i, r in df.iterrows():
    if i > 0:
        if on_trip:
            if r['STATE_OF_CHARGE'] > last_state_of_charge:
                on_trip = False
                trip_km.append(r['ODOMETER']-odo_start)
                trip_charge.append(r['STATE_OF_CHARGE']-charge_start)
        else:
            if r['STATE_OF_CHARGE'] < last_state_of_charge:
                charge_start = last_state_of_charge
                odo_start = last_odo
                on_trip=True
    last_state_of_charge = r['STATE_OF_CHARGE']
    last_odo = r['ODOMETER']

plt.scatter(trip_km,trip_charge)
plt.xlabel('travelling (km)')
plt.ylabel('Change in battery capacity (%)')
plt.show()

trip_km = np.array(trip_km)
trip_charge = np.array(trip_charge)

ind = trip_km > 0
trip_km = trip_km[ind]
trip_charge = trip_charge[ind]


energy_usage = -trip_charge / 100 * max_battery_capacity / trip_km

plt.hist(energy_usage)
plt.show()

