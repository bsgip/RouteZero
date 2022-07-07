from datetime import datetime, time, date, timedelta
from meteostat import Point, Daily, Monthly, Stations
from calendar import monthrange
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd

# class TemperatureData():
#     """
#     A class for working with the historical hourly temperature data that we have scraped
#     """
#     def __init__(self, datafile):
#         """
#         init function
#         :param datafile: path to csv
#         """
#         df = pd.read_csv(datafile)
#         df['datetime'] = pd.to_datetime(df['dt'], unit='s')
#         gb = df.groupby('lat')
#         self.station_dfs = [x[1].reset_index() for x in gb]
#         self.station_locs = np.array([[x['lat'][0], x['lon'][0]] for x in self.station_dfs])
#
#     def __call__(self, latitude, longitude, datetime):
#         """
#         Returns the temperature recorded at the station nearest to (latitude, longitude) at the closest time to datetime
#         :param latitude: latitude in degrees
#         :param longitude: longitude in degrees
#         :param datetime: datetime to get temperature of
#         :return:
#         """
#         p2 = np.array([[latitude, longitude]])
#         d = haversine_distances(np.deg2rad(self.station_locs), np.deg2rad(p2))
#         station_ind = np.argmin(d)      # index of closest station
#         df = self.station_dfs[station_ind]
#         dt_ind = np.argmin((datetime-df['datetime']).abs())       # closest recording in time
#         temp = df.loc[dt_ind, 'temp']
#
#         return temp
#

def get_temperature_by_date(location_coords, elevation, date):
    """
    Gets the min and max temperature at the location and elevation for a given date
    :param location_coords: (N,E) gps coords
    :param elevation: float value at location (m)
    :param date: datetime
    :return:
    """

    point = Point(location_coords[0], location_coords[1], elevation)
    daily = Daily(point, date, date)
    temp_data = daily.fetch()
    tmin = temp_data["tmin"].values[0]
    tmax = temp_data["tmax"].values[0]
    return tmin, tmax



def location_design_temp(location_coords, elevation, num_years=5, percentiles=(1, 99)):
    """
    Determines the design day max and min temperatures for a location by looking at num_years
    historical daily temperature data. For the min temperature we use the percentiles[0] value
    of the daily mins, i.e. by default this excludes 1% of the lowest temperatures. For the max we
    use percentiles[1] value, which by default excludes 1% of the hottest days.
    :param location_coords: (N, E) gps coords
    :param elevation: float value for elevation at location (m)
    :param num_years: (default=10) number of years of daily temperature data to consider
    :param percentiles: (default=[1,99]) the upper and lower percentiles for temperatures to consider
    :return: min_temp, max_temp, avg_temp
    """
    temp_data = historical_daily_temperatures(num_years, location_coords, elevation)

    daily_low_min = np.percentile(temp_data.tmin, percentiles[0])
    daily_low_max = np.percentile(temp_data.tmin, percentiles[1])
    daily_high_min = np.percentile(temp_data.tmax, percentiles[0])
    daily_high_max = np.percentile(temp_data.tmax, percentiles[1])

    return daily_low_min, daily_low_max, daily_high_min, daily_high_max

def historical_daily_temperatures(num_years, location_coords, elevation):
    stations = Stations()
    station = stations.nearby(location_coords[0], location_coords[1])
    station_df = station.fetch()
    station_df = station_df[station_df.daily_end.notnull()]
    station_df = station_df[station_df.daily_start.notnull()]

    station_df = station_df[station_df.daily_end.notnull()]
    station_df = station_df[station_df.daily_start.notnull()]
    station_df = station_df.sort_values(by='distance')

    temp_data = None
    for i in range(len(station_df)):
        nearest = station_df.iloc[i]

        date_end = nearest['daily_end']
        date_start = date_end - timedelta(days=365.25*num_years)

        daily = Daily(nearest.name,date_start, date_end)
        data = daily.fetch()

        temp_data = data[['tmin','tmax']].copy(deep=True)
        temp_data.dropna(inplace=True)

        if len(temp_data) > 100:
             break


    return temp_data

def historical_monthly_temperatures(start, end, location_coords, elevation):
    # start as list [year, month, day]
    # end as list [year, month, day]
    start = datetime(start[0], start[1], start[2])
    end = datetime(end[0], end[1], end[2])
    location = Point(location_coords[0], location_coords[1], alt=elevation)
    # location.method = 'weighted'  # method can also be nearest (default) or weighted

    # data = Daily(location, start, end)
    data = Monthly(location, start, end)
    df = data.fetch()
    return df

def typical_months_temperatures(month, location_coords, elevation):
    tavg_list = []
    tmin_list = []
    tmax_list = []
    for year in range(2000,2021):
        days = monthrange(year, month)
        start = datetime(year, month, 1)
        end = datetime(year, month, days[1])
        location = Point(location_coords[0], location_coords[1], alt=elevation)
        data = Daily(location, start, end)
        data = data.fetch()
        tavg_list.append(data['tavg'].values)
        tmin_list.append(data['tmin'].values)
        tmax_list.append(data['tmax'].values)

    tavg = np.vstack(tavg_list)
    tmin = np.vstack(tmin_list)
    tmax = np.vstack(tmax_list)
    return tavg.mean(axis=0), tmin.mean(axis=0), tmax.mean(axis=0)

def daily_temp_profile(hour, low, high):
    standard_temps = np.array([19.54894928, 19.26742754, 18.92731884, 18.67521739, 18.44326087,
       18.22728261, 18.13927536, 18.09126812, 18.69884058, 20.03550725,
       21.30811594, 22.26322464, 22.90880435, 23.31202899, 23.42528986,
       23.46021739, 23.20536232, 22.6626087 , 22.0709058 , 21.50652174,
       20.90311594, 20.49789855, 20.14619565, 19.77431159])

    norm_standard_temp = (standard_temps-standard_temps.min())/(standard_temps.max()-standard_temps.min())

    profile = norm_standard_temp * (high-low) + low

    hours = np.arange(24)

    return np.interp(np.mod(hour,24), hours, profile)

if __name__=='__main__':
    from sklearn.metrics.pairwise import haversine_distances

    weather_df = pd.read_csv("../data/routezero_weather.csv")
    weather_df['datetime'] = pd.to_datetime(weather_df['dt'], unit='s')

    trips_data = pd.read_csv('../data/trip_data_leichhardt.csv')

    station_lats = weather_df['lat'].unique()
    station_lons = weather_df['lon'].unique()

    latitude = trips_data['start_loc_y'].to_numpy()[0]
    longitude = trips_data['start_loc_x'].to_numpy()[0]

    temps = TemperatureData("../data/routezero_weather.csv")
    d = temps(latitude=latitude, longitude=longitude, datetime=weather_df['datetime'][0])






