from datetime import datetime, time, date
from meteostat import Point, Daily, Monthly, Stations
from calendar import monthrange
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd

class TemperatureData():
    """
    A class for working with the historical hourly temperature data that we have scraped
    """
    def __init__(self, datafile):
        """
        init function
        :param datafile: path to csv
        """
        df = pd.read_csv(datafile)
        df['datetime'] = pd.to_datetime(df['dt'], unit='s')
        gb = df.groupby('lat')
        self.station_dfs = [x[1].reset_index() for x in gb]
        self.station_locs = np.array([[x['lat'][0], x['lon'][0]] for x in self.station_dfs])

    def __call__(self, latitude, longitude, datetime):
        """
        Returns the temperature recorded at the station nearest to (latitude, longitude) at the closest time to datetime
        :param latitude: latitude in degrees
        :param longitude: longitude in degrees
        :param datetime: datetime to get temperature of
        :return:
        """
        p2 = np.array([[latitude, longitude]])
        d = haversine_distances(np.deg2rad(self.station_locs), np.deg2rad(p2))
        station_ind = np.argmin(d)      # index of closest station
        df = self.station_dfs[station_ind]
        dt_ind = np.argmin((datetime-df['datetime']).abs())       # closest recording in time
        temp = df.loc[dt_ind, 'temp']

        return temp





def location_design_temp(location_coords, elevation, num_years=5, percentiles=[1, 99]):
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
    start = [2021-num_years, 1, 1]
    end = [2021, 1, 1]
    data = historical_daily_temperatures(start, end, location_coords, elevation)
    temp_data = data[['tavg','tmin','tmax']].copy(deep=True)
    temp_data.dropna(inplace=True)

    daily_low_min = np.percentile(temp_data.tmin, percentiles[0])
    daily_low_max = np.percentile(temp_data.tmin, percentiles[1])
    daily_high_min = np.percentile(temp_data.tmax, percentiles[0])
    daily_high_max = np.percentile(temp_data.tmax, percentiles[1])

    return daily_low_min, daily_low_max, daily_high_min, daily_high_max

def historical_daily_temperatures(start, end, location_coords, elevation):
    # start as list [year, month, day]
    # end as list [year, month, day]
    start = datetime(start[0], start[1], start[2])
    end = datetime(end[0], end[1], end[2])
    location = Point(location_coords[0], location_coords[1], alt=elevation)
    # location.method = 'weighted'  # method can also be nearest (default) or weighted

    data = Daily(location, start, end)
    # data = Monthly(location, start, end)
    df = data.fetch()
    return df

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

def daily_temp_profile(hour, low, high, low_hour=6, high_hour=15):

    hours = [(high_hour-24), low_hour, high_hour, 24+low_hour, 24+high_hour]
    temps = [high, low, high, low, high]
    cs = CubicSpline(hours, temps, bc_type='periodic')

    return cs(hour)

if __name__=='__main__':
    import time
    import matplotlib.pyplot as plt
    # import pandas as pd
    import geopandas as gpd
    from shapely import wkt
    from sklearn.metrics.pairwise import haversine_distances

    weather_df = pd.read_csv("../data/routezero_weather.csv")
    weather_df['datetime'] = pd.to_datetime(weather_df['dt'], unit='s')

    trips_data = pd.read_csv('../data/trip_data_leichhardt.csv')

    station_lats = weather_df['lat'].unique()
    station_lons = weather_df['lon'].unique()

    latitude = trips_data['start_loc_y'].to_numpy()[0]
    longitude = trips_data['start_loc_x'].to_numpy()[0]

    # p1 = np.deg2rad(np.array([[latitude, longitude]]))
    # p2 = np.deg2rad(np.array([[weather_df['lat'].to_numpy()[0], weather_df['lon'].to_numpy()[0]]]))

    temps = TemperatureData("../data/routezero_weather.csv")
    d = temps(latitude=latitude, longitude=longitude, datetime=weather_df['datetime'][0])

    # lat1 = latitude
    # lon1 = longitude
    #
    # lat2 = weather_df['lat'].to_numpy()[0]
    # lon2 = weather_df['lon'].to_numpy()[0]
    #
    # # haversine formula
    # R = 7371
    # dLat = np.deg2rad(lat1 - lat2)
    # dLon = np.deg2rad(lon1 - lon2)
    # a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dLon / 2) * np.sin(dLon / 2)
    # c =




    # trips_data['passengers'] = 38
    #
    # num_years = 5
    #
    # start = [2021-num_years, 1, 1] #datetime.datetime(2021 - num_years, 1, 1)
    # end = [2021, 1, 1]
    #
    # df = trips_data.drop_duplicates('route_short_name')
    #
    # for i, r in df.iterrows():
    #     route_short_name = r['route_short_name']
    #
    #     inds = trips_data[trips_data.route_short_name==route_short_name].index
    #     start_hour = np.mod(trips_data[trips_data.route_short_name==route_short_name].trip_start_time/3600, 24)
    #     end_hour = np.mod(trips_data[trips_data.route_short_name == route_short_name].trip_end_time/3600, 24)
    #
    #     x = r['start_loc_x']
    #     y = r['start_loc_y']
    #     el = r['start_el']
    #     daily_low_min, daily_low_max, daily_high_min, daily_high_max = location_design_temp([y, x], el)
    #
    #     t1 = daily_temp_profile(start_hour, daily_low_max, daily_high_max)
    #     t2 = daily_temp_profile(end_hour, daily_low_max, daily_high_max)
    #
    #     t5 = daily_temp_profile(start_hour, daily_low_min, daily_high_min)
    #     t6 = daily_temp_profile(end_hour, daily_low_min, daily_high_min)
    #
    #     x = r['end_loc_x']
    #     y = r['end_loc_y']
    #     el = r['end_el']
    #
    #     daily_low_min, daily_low_max, daily_high_min, daily_high_max = location_design_temp([y, x], el)
    #     t3 = daily_temp_profile(start_hour, daily_low_max, daily_high_max)
    #     t4 = daily_temp_profile(end_hour, daily_low_max, daily_high_max)
    #
    #     t7 = daily_temp_profile(start_hour, daily_low_min, daily_high_min)
    #     t8 = daily_temp_profile(end_hour, daily_low_min, daily_high_min)
    #
    #     temp_max = np.vstack([t1, t2, t3, t4]).max(axis=0)
    #     temp_min = np.vstack([t5, t6, t7, t8]).min(axis=0)
    #
    #     trips_data.loc[inds, 'min_temp'] = temp_min
    #     trips_data.loc[inds, 'max_temp'] = temp_max





