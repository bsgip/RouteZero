"""
        Script for case study 1 in the paper
        Transport and energy sector coupling

"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from RouteZero.models import PredictionPipe
import RouteZero.bus as ebus
from RouteZero.route import calc_buses_in_traffic
from sklearn.metrics.pairwise import haversine_distances
from RouteZero.map import _create_gdf_of_value, _folium_open, _create_gdf_map
from RouteZero.models import summarise_results

NUM_PASSENGERS = 15
DEADHEAD = 0.1
RESOLUTION = 10


# routes within 30km of fairfield (roughly middle of the sydney regions?)
# agencies = ["Transit Systems"]
trips_data = pd.read_csv('../data/gtfs/greater_sydney/trip_data.csv')
X = np.vstack([trips_data["start_loc_y"].to_numpy(), trips_data["start_loc_x"].to_numpy()]).T
# sydney_gps = np.deg2rad([-33.8688, 151.2093]).reshape((1,2))
sydney_gps = np.deg2rad([-33.8666632, 150.916663]).reshape((1,2))  # fairfield coordinates as its more in the middle
X = np.deg2rad(X)
d = haversine_distances(X, Y=sydney_gps)* 6371000/1000
trips_data = trips_data[d <= 30]
shapes = gpd.read_file('../data/gtfs/greater_sydney/shapes.shp')
# trips_data = trips_data[trips_data["agency_name"].isin(agencies)]


trips_data['passengers'] = NUM_PASSENGERS
bus = ebus.BYD()
prediction_pipe = PredictionPipe()

ec_km, ec_total = prediction_pipe.predict_worst_case(trips_data, bus)

times, buses_in_traffic, depart_trip_energy_reqs, return_trip_enery_consumed = calc_buses_in_traffic(trips_data,
                                                                                                     deadhead=DEADHEAD,
                                                                                                     resolution=RESOLUTION,
                                                                                                     trip_ec=ec_total)

print("min requried buses = ", buses_in_traffic.max())

# summarise daily used power
used_daily = np.histogram(times / 60, bins=np.arange(0, 24 * 8, 24), weights=depart_trip_energy_reqs)[0]

plt.subplot(3,1,1)
plt.plot(times/60, buses_in_traffic)
plt.xlabel("hour of week")
plt.title("buses on trips")
plt.ylabel("# buses")

plt.subplot(3,1,2)
plt.plot(times/60, depart_trip_energy_reqs/1000)
plt.xlabel("hour of week")
plt.title("energy required by starting trips")
plt.ylabel("energy (MWh)")

plt.subplot(3,1,3)
plt.bar(np.arange(1,8), used_daily/1000)
plt.xlabel("day of week")
plt.title("Energy used on trips (MWh)")
plt.ylabel("energy (MWh)")

plt.tight_layout()
plt.show()

print("Busiest week from GTFS data is from {} to {}".format(trips_data.date.min(), trips_data.date.max()))
print("Busiest week in the GTFS data has {} trips".format(len(trips_data)))
print("Total week energy usage is {:.2f}MWh".format(used_daily.sum()/1000))
print("Max daily energy usage is {:.2f}MWh".format(used_daily.max()/1000))

# plot on map
route_summaries = summarise_results(trips_data, ec_km, ec_total)

gdf = _create_gdf_of_value(route_summaries, shapes, window='6:00 - 9:30')

map_title = "Route Energy Consumption"
colorbar_str = 'energy per km'

m = _create_gdf_map(gdf, map_title, colorbar_str)
_folium_open(m, map_title + '.html')