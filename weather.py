# Import Meteostat library and dependencies
from datetime import datetime, time, date
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Monthly
from calendar import monthrange
import numpy as np
import pandas as pd
# from astral import LocationInfo
from suntime import Sun

def get_sun_times(location_coords, ymd):
    # s = sun(city.observer, date=datetime.date(ymd[0], ymd[1], ymd[2]))
    # city = LocationInfo(51.5, -0.116)
    sun = Sun(location_coords[0], location_coords[1])
    day = date(ymd[0], ymd[1], ymd[2])
    sunrise = sun.get_local_sunrise_time(day).time()
    sunset = sun.get_local_sunset_time(day).time()
    sunrise = datetime(ymd[0], ymd[1], ymd[2],sunrise.hour, sunrise.minute)
    sunset = datetime(ymd[0], ymd[1], ymd[2], sunset.hour, sunset.minute)
    noon = sunrise + (sunset-sunrise)/2
    return sunrise, sunset, noon

def historical_daily_temperatures(start, end, location_coords, elevation):
    # start as list [year, month, day]
    # end as list [year, month, day]
    start = datetime(start[0], start[1], start[2])
    end = datetime(end[0], end[1], end[2])
    location = Point(location_coords[0], location_coords[1], alt=elevation)
    # location.method = 'weighted'  # method can also be nearest (default) or weighted

    data = Daily(location, start, end)
    data = data.fetch()
    return data

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
    start = [2020,1,1]
    end = [2020,1,31]
    location_coords = [-32.8960021, 151.734908] # for mayfield east
    elevation = 10

    data = historical_daily_temperatures(start, end, location_coords, elevation)
    data.plot(y=['tavg', 'tmin', 'tmax'])
    plt.show()

    tavg, tmin, tmax = typical_months_temperatures(1, location_coords, elevation)
    plt.plot(tavg)
    plt.plot(tmin)
    plt.plot(tmax)
    plt.show()

    ymd = [2022, 1, 20]
    sunrise, sunset, noon = get_sun_times(location_coords, ymd)


