import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

"""

                module for electric bus energy consumption models

"""

class Model:
    """
        parent class for energy models

        inputs:
            const: A constant value
            GR: road grade average (%) [-100, 100]
            PL: passenger loading (passengers)
            SD: stop density (stops/km)
            Va: average velocity (km/h)
            SoCi: initial battery state of charge (%) [0, 100]
            RC: road condition in three levels (three levels)
            Dagg: driver aggressiveness (three levels)
            t: temperature (degrees)
            todo: work out how to include buss mass in some meaningful way

        Outputs:
            EC: energy consumption (KWh/km)

    """
    def __init__(self):
        """
            initialises default parameters
        """
        self.rc = [1, 2, 3]                  # road condition has three possible levels
        self.driver_agg = [1, 2, 3]          # driver aggressiveness has three possible levels


    def _build_regressor_variations(self):
        return None

    def _build_hottest_day_regressor(self, trips_data, bus):
        rc = self.rc[2]
        dagg = self.driver_agg[2]
        soc = bus.soc
        temp = np.atleast_2d(trips_data['max_temp'].to_numpy()).T
        X = Model._build_regressor_matrix(trips_data, soc, rc, dagg, temp)
        return X

    def _build_coldest_day_regressor(self, trips_data, bus):
        rc = self.rc[2]
        dagg = self.driver_agg[2]
        soc = bus.soc
        temp = np.atleast_2d(trips_data['min_temp'].to_numpy()).T
        X = Model._build_regressor_matrix(trips_data, soc, rc, dagg, temp)
        return X

    @staticmethod
    def _build_regressor_matrix(trips_data, soc, rc, dagg, temp):
        n = len(trips_data)
        ones = np.ones((n, 1))
        x = trips_data[['average_gradient_%','passengers','stops_per_km','average_speed_kmh']].to_numpy()
        X = np.hstack([ones, x, soc * ones, rc * ones, dagg * ones, temp])
        return X

    def _predict(self, X):
        """
            Place holder method to be overriden by child classes
        """
        (r, c) = X.shape
        return X @ np.ones((c,1))

    def predict(self, trips_data, bus):
        X = self._build_hottest_day_regressor(trips_data, bus)
        y = self._predict(X)
        return y


class LinearRegressionAbdelatyModel(Model):
    """
    A linear regression model for eBus energy consumption taken from

    Abdelaty, H.; Mohamed, M. APrediction Model for Battery Electric Bus Energy Consumption in Transit.
    Energies 2021, 14, 2824. https://doi.org/10.3390/en14102824

    model:

    EC = B0 + B1*GR + B2*SoCi + B3*RC + B4*HVAC + B5*PL + B6*Dagg + B7*SD + B8*Va + e

    Model is then altered slightly so that hvac is factored off time and temperature not distance

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
        super().__init__()
        self.B = np.array([-0.782, 0.38, 0.005, 0.128, 0.007, 0.0124, 0.26, 0.065]) # reordered and without hvac to match parent class
        self.hvac_ref_vals = [13.75, 6.7, 3.0, 1.25, 2.0, 10.75]
        self.temp_ref_vals = [-20, -10, 0, 10, 20, 30]
        self.hvac_func = interp1d(self.temp_ref_vals, self.hvac_ref_vals, kind='cubic', fill_value='extrapolate', bounds_error=False)

    def _predict(self, X):
        temps = X[:, -1]
        hours_km = 1/X[:,4]
        hvac = self._hvac_from_temp(temps, hours_km)
        return np.dot(X[:, :-1], self.B) + hvac

    def _hvac_from_temp(self, temps, hours_km):
        hvacs = self.hvac_func(temps) * hours_km
        return hvacs

if __name__=="__main__":
    import RouteZero.bus as ebus
    trips_data = pd.read_csv('../data/test_trip_summary.csv')
    bus = ebus.BYD()
    model = LinearRegressionAbdelatyModel()
    EC_km = model.predict(trips_data, bus)

