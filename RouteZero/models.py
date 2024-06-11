"""

                RouteZero module containing the machine learning predictive models

                The model used in the web app is run using the PredictionPipe class.
                A description of this model is found in the Documetnation
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import json
from RouteZero.route import calc_buses_in_traffic



class PredictionPipe():
    "pipeline to transform and make predictions on the gtfs and user supplied data"
    def __init__(self, saved_params_file="../data/bayes_lin_reg_model_2.json"):
        self.model = BayesianLinearRegression()
        self.model.load(filename=saved_params_file)
        self.add_constant = Add_constant()
        self.add_soc_full = Add_soc_full()
        self.feature_square = Feature_square(["max_temp", "min_temp"])
        self.select_features_max_temp = Select_features(["constant", "stops_per_km", "average_gradient_%",
                                                        "max_temp", "average_speed_kmh", "passengers",
                                                        "start SOC (%)", "soc > 97", "max_temp_square",
                                                        ])
        self.select_features_min_temp = Select_features(["constant", "stops_per_km", "average_gradient_%",
                                                        "min_temp", "average_speed_kmh", "passengers",
                                                        "start SOC (%)", "soc > 97", "min_temp_square",
                                                        ])


    def add_soc(self, data, bus):
        soc = bus.get_soc_percent()
        data["start SOC (%)"] = soc
        return data

    def _transform_data(self, data, bus):
        data = self.add_constant.transform(data)
        data = self.feature_square.transform(data)
        data = self.add_soc(data, bus)
        data = self.add_soc_full.transform(data)
        return data

    def predict_worst_case(self, trip_data, bus):
        trip_data = self._transform_data(trip_data.copy(), bus)
        distance = trip_data['trip_distance'].to_numpy() / 1000
        X_max_temp = self.select_features_max_temp.transform(trip_data)
        X_min_temp = self.select_features_min_temp.transform(trip_data)
        y_max_temp, y_max_temp_std = self.model.predict(X_max_temp)
        y_min_temp, y_min_temp_std = self.model.predict(X_min_temp)
        # 95% confidence conservative upper value
        y_max_temp_conf = y_max_temp.flatten() + 2 * y_max_temp_std
        y_min_temp_conf = y_min_temp.flatten() + 2 * y_min_temp_std

        ec_km = np.maximum(y_max_temp_conf, y_min_temp_conf)
        ec_total = ec_km * distance
        return ec_km, ec_total




# new model class
class BayesianLinearRegression():
    """ Fits a linear model using bayesian regression
        Expects the following features in order
        ["constant","stops/km", "gradient (%)", "temp", "speed (km/h)",
                        "average_passengers", "start SOC (%)", "soc > 97", "temp_square"]
    """

    def __init__(self, prior_std=10, regressor_vars=None):
        self.prior_std = prior_std
        self.regressor_vars = regressor_vars
        self.params = None
        self.post_var = None

    def fit(self, X, y=None, meas_variance=0.01):
        X = X.to_numpy().astype(float)
        # sigma = self.meas_std
        prior_var = self.prior_std ** 2
        params = np.linalg.solve((X.T @ (X /np.atleast_2d(meas_variance).T) + np.eye(X.shape[1]) / prior_var),
                                 X.T @ (y / np.atleast_2d(meas_variance).T))
        post_var = np.linalg.inv(X.T @ (X /np.atleast_2d(meas_variance).T) + np.eye(X.shape[1]) / prior_var)
        self.params = params
        self.post_var = post_var

        return self

    def predict(self, X):
        X = X.to_numpy().astype(float)
        y = X @ self.params

        # pred_std = np.sqrt(np.diag(X @ self.post_var @ X.T))
        # fast way of getting the diagonal standard deviations
        pred_std = np.sqrt((X @ self.post_var * X).sum(axis=1))

        return y, pred_std

    def save(self, filename):
        tmp = self.__dict__
        for key in tmp:
            if isinstance(tmp[key], np.ndarray):
                tmp[key] = tmp[key].tolist()
        with open(filename, 'w') as f:
            json.dump(tmp, f)

    def load(self, filename):
        with open(filename) as f:
            tmp = json.load(f)
            for key in tmp:
                if (key == "params") or (key == "post_var"):
                    setattr(self, key, np.array(tmp[key]))
                else:
                    setattr(self, key, tmp[key])
        return self


# Feature transformers
class Add_CrossProducts():
    """ adds the product of two features """

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        assert len(variables) > 1, "must have more than one variable to make combinations with"
        self.variables = variables
        self.combinations = [(a, b) for idx, a in enumerate(variables) for b in variables[idx + 1:]]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for p in self.combinations:
            X[p[0]+"_"+p[1]] = X[p[0]]*X[p[1]]

        # X["speed (km/h)"] = X["speed (m/s)"] * 3.6
        return X

class Add_constant():
    """ Adds a constant (1) value for regression"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["constant"] = 1.
        return X


class Add_kmph():
    """ add kmph speed"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["speed (km/h)"] = X["speed (m/s)"] * 3.6
        return X


class Add_soc_full():
    """ add state of charge full indicator variable"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["soc > 97"] = X["start SOC (%)"] > 97
        return X


class Feature_square():
    """ adds squared value of a feature """

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            X[var + "_square"] = X[var] ** 2
        return X


class Select_features():
    """ keep only specified subset of features"""

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[self.variables].copy()
        return X



# old model class
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

        Outputs:
            EC: energy consumption (KWh/km)

    """

    def __init__(self):
        """
            initialises default parameters
        """
        self.rc = [1, 2, 3]  # road condition has three possible levels
        self.driver_agg = [1, 2, 3]  # driver aggressiveness has three possible levels

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
        soc = bus.get_soc_percent()  # state or charge as percent
        temp = np.atleast_2d(trips_data['min_temp'].to_numpy()).T
        X = Model._build_regressor_matrix(trips_data, soc, rc, dagg, temp)
        return X

    @staticmethod
    def _build_regressor_matrix(trips_data, soc, rc, dagg, temp):
        n = len(trips_data)
        ones = np.ones((n, 1))
        x = trips_data[['average_gradient_%', 'passengers', 'stops_per_km', 'average_speed_kmh']].to_numpy()
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
        ec_total = y * trips_data['trip_distance'] / 1000
        return ec_km, ec_total

    def predict_worst_temp(self, trips_data, bus):
        X = self._build_hottest_day_regressor(trips_data, bus)
        y1 = self._predict(X)

        X = self._build_coldest_day_regressor(trips_data, bus)
        y2 = self._predict(X)

        y = np.vstack([y1, y2]).max(axis=0)

        ec_km = y
        ec_total = y * trips_data['trip_distance'] / 1000
        return ec_km, ec_total


class LinearRegressionAbdelatyModel(Model):
    """
    A linear regression model for eBus energy consumption taken from

    Abdelaty, H.; Mohamed, M. A Prediction Model for Battery Electric Bus Energy Consumption in Transit.
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
        self.B = np.array([-0.782, 0.38, 0.005, 0.128, 0.007, 0.0124, 0.26,
                           0.065])  # reordered and without hvac to match parent class
        self.hvac_ref_vals = [13.75, 6.7, 3.0, 1.25, 2.0, 10.75]
        self.temp_ref_vals = [-20, -10, 0, 10, 20, 30]
        self.hvac_func = interp1d(self.temp_ref_vals, self.hvac_ref_vals, kind='cubic', fill_value='extrapolate',
                                  bounds_error=False)

    def _predict(self, X):
        temps = X[:, -1]
        hours_km = 1 / X[:, 4]
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
        self.B = np.array([-0.2438821, 0.1225219, 0.005, 0.128, 0.007, -0.005676, 0.26, 0.065, 0.63,
                           0.01469553])  # reordered and without hvac to match parent class
        # self.hvac_ref_vals = [13.75, 6.7, 3.0, 1.25, 2.0, 10.75]
        self.hvac_ref_vals = [9.37057329 + 0.00841158 + 0.12998606, 7.48830395 + 0.0431898 + 0.00377488,
                              15.12797907 + 0.22957565 - 0.34177559]
        # self.hvac_ref_vals = [0, 0, 0, 0, 0, 0]
        # self.temp_ref_vals = [-20, -10, 0, 10, 20, 30]
        self.temp_ref_vals = [10, 20, 30]
        # self.hvac_func = interp1d(self.temp_ref_vals, self.hvac_ref_vals, kind='cubic', fill_value='extrapolate', bounds_error=False)
        self.hvac_func = interp1d(self.temp_ref_vals, self.hvac_ref_vals, kind='quadratic', fill_value='extrapolate',
                                  bounds_error=False)

    def _predict(self, X):
        temps = X[:, -1]
        soci_full = X[:, [-4]] > 97
        grad_square = X[:, [1]] ** 2
        X = np.hstack([X[:, :-1], soci_full, grad_square])
        hours_km = 1 / X[:, 4]
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
    tmp['start_hour'] = np.mod(tmp['trip_start_time'] / 3600, 24)
    tmp['ec/km (kwh/km)'] = ec_km
    tmp['ec (kwh)'] = ec_total
    tmp.drop(columns=['agency_name', 'trip_id', 'unique_id', 'date', 'start_loc_x', 'Unnamed: 0',
                      'start_loc_y', 'start_el', 'end_loc_x', 'end_loc_y', 'end_el', 'av_elevation',
                      'trip_start_time', 'trip_end_time', 'average_speed_mps'], inplace=True)

    tmp.reset_index(inplace=True, drop=True)

    tmp['hour window'] = pd.cut(tmp['start_hour'], windows)

    # tmp.groupby(by=['route_id','direction_id','shape_id','window'])['ec/km'].max()
    tmp.sort_values(by='ec/km (kwh/km)', inplace=True, ascending=False)
    df = tmp.groupby(by=['route_id', 'direction_id', 'shape_id', 'hour window']).head(1).reset_index(drop=True)

    df['trip_duration'] = df['trip_duration'] / 60
    df.drop(columns='start_hour', inplace=True)
    df.rename(columns={'average_gradient_%': 'average gradient (%)', "stops_per_km": "stops/km",
                       "average_speed_kmh": "average speed (km/h)", "max_temp": "possible max temp",
                       "min_temp": "possible min temp", "trip_distance": "trip distance (m)",
                       "trip_duration": "trip duration (mins)", "num_stops": "number stops"}, inplace=True)
    df['hour window'] = [(x.left, x.right) for x in df['hour window'].values]

    return df


if __name__ == "__main__":
    import RouteZero.bus as ebus
    import matplotlib.pyplot as plt

    # trips_data = pd.read_csv('../data/gtfs/greater_sydney/trip_data.csv')
    trips_data = pd.read_csv('../data/gtfs/act/trip_data.csv')
    # trips_data = trips_data[trips_data['agency_name']=='Newcastle Transport']
    trips_data['passengers'] = 38
    bus = ebus.BYD()
    prediction_pipe = PredictionPipe()

    ec_km, ec_total = prediction_pipe.predict_worst_case(trips_data, bus)
    # model = LinearRegressionAbdelatyModel()
    # ec_km, ec_total = model.predict_worst_temp(trips_data, bus)

    df = summarise_results(trips_data, ec_km, ec_total)

    times, buses_in_traffic, depart_trip_energy_reqs, return_trip_enery_consumed = calc_buses_in_traffic(trips_data,
                                                                                                       deadhead=0.1,
                                                                                                         resolution=10,
                                                                                                         trip_ec=ec_total)
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(times/60,buses_in_traffic)
    plt.xlabel('Hour of week')
    plt.ylabel('# buses')

    plt.subplot(1,2,2)
    plt.plot(times/60, depart_trip_energy_reqs)
    # plt.plot(times/60, return_trip_enery_consumed, label="returning trips")
    plt.xlabel("Hour of week")
    plt.ylabel("Energy required by departing trips (kWh)")

    plt.tight_layout()
    plt.show()
