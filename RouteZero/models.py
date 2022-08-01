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
        rc = self.rc[1]
        dagg = self.driver_agg[1]
        soc = bus.get_soc_percent()
        temp = np.atleast_2d(trips_data['max_temp'].to_numpy()).T
        X = Model._build_regressor_matrix(trips_data, soc, rc, dagg, temp)
        return X

    def _build_coldest_day_regressor(self, trips_data, bus):
        rc = self.rc[1]
        dagg = self.driver_agg[1]
        soc = bus.get_soc_percent()    # state or charge as percent
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
        return None

    def predict_hottest(self, trips_data, bus):
        X = self._build_hottest_day_regressor(trips_data, bus)
        y = self._predict(X)
        ec_km = y
        ec_total = y * trips_data['trip_distance']/1000
        return ec_km, ec_total

    def predict_worst_temp(self, trips_data, bus):
        X = self._build_hottest_day_regressor(trips_data, bus)
        y1 = self._predict(X)

        X = self._build_coldest_day_regressor(trips_data, bus)
        y2 = self._predict(X)

        y = np.vstack([y1, y2]).max(axis=0)

        ec_km = y
        ec_total = y * trips_data['trip_distance']/1000
        return ec_km, ec_total


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


class LinearRegMod(Model):
    """
    Modified from

    Abdelaty, H.; Mohamed, M. APrediction Model for Battery Electric Bus Energy Consumption in Transit.
    Energies 2021, 14, 2824. https://doi.org/10.3390/en14102824

    model:

    EC = B0 + B1*GR + B2*SoCi + B3*RC + B4*HVAC + B5*PL + B6*Dagg + B7*SD + B8*Va + B9 * SoCiFull + B10* grad**2+ e

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
    B0 = -0.2438821
    B1 = 0.1225219
    B9 = 0.7
    B10 = 0.01469553


    """
    def __init__(self):
        super().__init__()
        self.B = np.array([-0.2438821, 0.1225219, 0.005, 0.128, 0.007, -0.005676, 0.26, 0.065, 0.63, 0.01469553]) # reordered and without hvac to match parent class
        # self.hvac_ref_vals = [13.75, 6.7, 3.0, 1.25, 2.0, 10.75]
        self.hvac_ref_vals = [9.37057329+0.00841158+0.12998606,  7.48830395+0.0431898+0.00377488, 15.12797907+0.22957565-0.34177559]
        # self.hvac_ref_vals = [0, 0, 0, 0, 0, 0]
        # self.temp_ref_vals = [-20, -10, 0, 10, 20, 30]
        self.temp_ref_vals = [10, 20, 30]
        # self.hvac_func = interp1d(self.temp_ref_vals, self.hvac_ref_vals, kind='cubic', fill_value='extrapolate', bounds_error=False)
        self.hvac_func = interp1d(self.temp_ref_vals, self.hvac_ref_vals, kind='quadratic', fill_value='extrapolate', bounds_error=False)

    def _predict(self, X):
        temps = X[:, -1]
        soci_full = X[:, [-4]] > 97
        grad_square = X[:, [1]]**2
        X = np.hstack([X[:, :-1], soci_full, grad_square])
        hours_km = 1/X[:,4]
        hvac = self._hvac_from_temp(temps, hours_km)
        return np.dot(X, self.B) + hvac

    def _hvac_from_temp(self, temps, hours_km):
        hvacs = self.hvac_func(temps) * hours_km
        return hvacs


def summarise_results(trips_data, ec_km, ec_total):
    """
    creates a dataframe that summarises the routes key parameters and energy usages binned within time windows
    :param trips_data: summarised trips dataframe
    :param ec_km: the energy usage per km for each trip in trips_data
    :param ec_total: the total energy usage for each trip in trips_data
    :return:
    """

    windows = [0, 6, 9.5, 12, 15, 18, 22, 24]

    tmp = trips_data.copy()
    tmp['start_hour'] = np.mod(tmp['trip_start_time']/3600,24)
    tmp['ec/km (kwh/km)'] = ec_km
    tmp['ec (kwh)'] = ec_total
    tmp.drop(columns=['agency_name','trip_id','unique_id','date','start_loc_x','Unnamed: 0',
                      'start_loc_y','start_el','end_loc_x','end_loc_y','end_el','av_elevation',
                      'trip_start_time','trip_end_time','average_speed_mps'], inplace=True)

    tmp.reset_index(inplace=True, drop=True)

    tmp['hour window'] = pd.cut(tmp['start_hour'], windows)

    # tmp.groupby(by=['route_id','direction_id','shape_id','window'])['ec/km'].max()
    tmp.sort_values(by='ec/km (kwh/km)', inplace=True, ascending=False)
    df = tmp.groupby(by=['route_id','direction_id','shape_id','hour window']).head(1).reset_index(drop=True)

    df['trip_duration'] = df['trip_duration']/60
    df.drop(columns='start_hour',inplace=True)
    df.rename(columns={'average_gradient_%':'average gradient (%)', "stops_per_km":"stops/km",
                        "average_speed_kmh":"average speed (km/h)","max_temp":"possible max temp",
                       "min_temp":"possible min temp", "trip_distance":"trip distance (m)",
                       "trip_duration":"trip duration (mins)", "num_stops":"number stops"}, inplace=True)
    df['hour window'] = [(x.left, x.right) for x in df['hour window'].values]

    return df


if __name__=="__main__":
    import RouteZero.bus as ebus
    trips_data = pd.read_csv('../data/gtfs/leichhardt/trip_data.csv')
    trips_data['passengers'] = 38
    bus = ebus.BYD()
    model = LinearRegressionAbdelatyModel()
    ec_km, ec_total = model.predict_worst_temp(trips_data, bus)

    df = summarise_results(trips_data, ec_km, ec_total)


