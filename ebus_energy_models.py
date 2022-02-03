import numpy as np
import pandas as pd
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from gtfs_routes import process_gtfs_routes
from weather import location_design_temp



class LinearRegressionAbdelatyModel:
    """
    A linear regression model for eBus energy consumption taken from

    Abdelaty, H.; Mohamed, M. APrediction Model for Battery Electric Bus Energy Consumption in Transit.
    Energies 2021, 14, 2824. https://doi.org/10.3390/en14102824

    model:

    EC = B0 + B1*GR + B2*SoCi + B3*RC + B4*HVAC + B5*PL + B6*Dagg + B7*SD + B8*Va + e

    Model is then altered slightly so that hvac is factored off time not distance

    where:
    EC: energy consumption (KWh/km)
    GR: road grade average (%) [-100, 100]
    SoCi: initial battery state of charge (%) [0, 100]
    RC: road condition in three levels (three levels)
    HVAC: auxilliary systems (heating, cooling etc) (KW)
    PL: passenger loading (passengers)
    Dagg: driver aggressiveness (three levels)
    SD: stop density (stops/km)
    Va: average velocity (km/h)

    Parameter values are given as
    B0 = -0.782
    B1 = 0.38
    B2 = 0.0124
    B3 = 0.26
    B4 = 0.036
    B5 = 0.005
    B6 = 0.065
    B7 = 0.128
    B8 = 0.007
    """
    def __init__(self):
        # self.B = np.array([-0.782, 0.38, 0.0124, 0.26, 0.036, 0.005, 0.065, 0.128, 0.007])
        self.B = np.array([-0.782, 0.38, 0.0124, 0.26, 0.005, 0.065, 0.128, 0.007])
        self.hvac_ref_vals = [13.75, 6.7, 3.0, 1.25, 2.0, 10.75]
        self.temp_ref_vals = [-20, -10, 0, 10, 20, 30]
        self.hvac_func = interp1d(self.temp_ref_vals, self.hvac_ref_vals, kind='cubic', fill_value='extrapolate', bounds_error=False)

    def _predict(self, X):
        return np.dot(X, self.B)

    # def predict(self, GR, SoCi, RC, HVAC, PL, Dagg, SD, Va):
    #     X = np.array([1., GR, SoCi, RC, HVAC, PL, Dagg, SD, Va])
    #     self._predict(X)

    def predict_routes(self, route_data, SoCi=1., Dagg=2., PL=38, RC=1.):
        route_data['min_EC_km'] = 0.
        route_data['max_EC_km'] = 0.
        route_data['min_EC_total'] = 0.
        route_data['max_EC_total'] =0.
        for i, r in route_data.iterrows():
            hvac = []
            hvac.append(model.hvac_from_temp(r['max_temp'], 1/(r['average_speed_mps']*3.6)))
            hvac.append(model.hvac_from_temp(r['min_temp'], 1/(r['average_speed_mps']*3.6)))
            if (10. < r['max_temp']) and (10. > r['min_temp']):
                hvac.append(model.hvac_from_temp(10., 1/(r['average_speed_mps']*3.6)))
            hvac_min = np.min(hvac)
            hvac_max = np.max(hvac)
            X = [1., r['av_grade_%'], SoCi, RC, PL, Dagg, r['stops_km'], r['average_speed_mps'] * 3.6]
            EC_min = self._predict(X) + hvac_min
            EC_max = self._predict(X) + hvac_max
            route_data.loc[i, 'min_EC_km'] = EC_min
            route_data.loc[i, 'max_EC_km'] = EC_max
            route_data.loc[i, 'min_EC_total'] = EC_min * r['trip_distance'] / 1000  # convert metres to km
            route_data.loc[i, 'max_EC_total'] = EC_max * r['trip_distance'] / 1000
        return route_data

    def hvac_from_temp(self, temps, hours_km):
        hvacs = self.hvac_func(temps) * hours_km
        return hvacs

if __name__ == "__main__":

    gtfs_file = "./data/full_greater_sydney_gtfs_static.zip"        # location of the gtfs zip file
    route_short_names = ["305", "320"]      # the short names of the routes we want to get summaries of
    route_desc = 'Sydney Buses Network'     # optional input if we also want to filter by particular types of routes
    # cutoffs = [0, 6, 9, 15, 19, 22, 24]     # optional input for splitting up the route summary information into time windows

    print('Processing routes '+", ".join(route_short_names))
    route_data, subset_shapes, elevation_profiles = process_gtfs_routes(gtfs_file, route_short_names, cutoffs=None, busiest_day=True, route_desc=route_desc)

    # get design day temperatures for route locations
    min_temps = []
    max_temps = []
    # avg_temps = []
    for i, r in route_data.iterrows():
        location_coords = (r['start_location'][1], r['start_location'][0] ) # geometry locations are (E, N) not (N, E)...
        elevation = elevation_profiles[r['shape_id']].mean()
        min_temp, max_temp, avg_temp = location_design_temp(location_coords, elevation, num_years=10, percentiles=[1, 99])
        min_temps.append(min_temp)
        max_temps.append(max_temp)
        # avg_temps.append(avg_temp)

    route_data['min_temp'] = min_temps
    route_data['max_temp'] = max_temps

    model = LinearRegressionAbdelatyModel()
    route_data = model.predict_routes(route_data)