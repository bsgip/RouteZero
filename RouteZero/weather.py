# Import Meteostat library and dependencies
from datetime import datetime, time, date
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Monthly
from calendar import monthrange
import numpy as np


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
    data = historical_monthly_temperatures(start, end, location_coords, elevation)
    temp_data = data[['tavg','tmin','tmax']].copy(deep=True)
    temp_data.dropna(inplace=True)
    # std = np.std(data.tmin)

    min_tmp = np.percentile(temp_data.tmin, percentiles[0])
    max_tmp = np.percentile(temp_data.tmax, percentiles[1])
    avg_tmp = np.mean(temp_data.tavg)
    return min_tmp, max_tmp, avg_tmp

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


if __name__=='__main__':
    import time
    num_years = 5
    location_coords = [-32.8960021, 151.734908] # for mayfield east
    elevation = 100

    start = [2021 - num_years, 1, 1]
    end = [2021, 1, 1]

    t1 = time.time()
    data = historical_daily_temperatures(start, end, location_coords, elevation)
    t2 = time.time()
    print(t2-t1)

    plt.hist(data.tmin, bins=30)
    plt.hist(data.tavg, bins=30)
    plt.hist(data.tmax, bins=30)
    plt.show()

    min_temp, max_temp, avg_temp = location_design_temp(location_coords, elevation, num_years=10, percentiles=[1, 99])

    print('Design day min temp for mayfield = ', min_temp)
    print('Design day max temp for mayfield = ', max_temp)
    print('Design day average temp for mayfield = ', avg_temp)

