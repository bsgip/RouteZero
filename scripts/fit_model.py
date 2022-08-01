import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from feature_engine.selection import (
    DropFeatures
)
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class Passenger_imputer():
    """ imputes missing passenger values """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        true_pass = X["pass_true_part"].to_numpy()
        true_weight = X["pass_true_weight"].to_numpy()
        passengers_imputed = true_pass * true_weight + 4 * (1 - true_weight)
        X["passengers imputed"] = passengers_imputed
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
        X["speed (km/h)"] = X["speed (m/s)"]*3.6
        return X


class Add_soc_full():
    """ add state of charge full indicator variable"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["start full"] = X["start SOC (%)"] > 97
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
            X[var+"_square"] = X[var] ** 2
        return X

class Select_features():
    """ keep only specified subset of features"""
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def Transform(self, X):
        X = X[self.variables].copy()
        return X

class BayesianLinearRegression():
    """ Fits a linear model using bayesian regression"""
    def __init__(self, prior_std=10, meas_std=0.1):
        self.prior_std = prior_std
        self.meas_std = meas_std

    def fit(self, X, y=None):
        X = X.to_numpy().astype(float)
        sigma = self.meas_std
        prior_var = self.prior_std**2
        params = np.linalg.solve((X.T @ X / sigma ** 2 + np.eye(X.shape[1]) / prior_var), X.T @ y / sigma ** 2)
        post_var = np.linalg.inv(X.T @ X / sigma ** 2 + np.eye(X.shape[1])/prior_var)
        # self.A = X.T @ X / sigma ** 2 + np.diag(1 / prior_var)
        self.params = params
        self.post_var = post_var
        return self

    def predict(self, X):
        X = X.to_numpy().astype(float)
        y = X @ self.params
        pred_std = np.sqrt(np.diag(X @ self.post_var @ X.T))
        return y, pred_std




if __name__=="__main__":
    import matplotlib.pyplot as plt
    trip_data = pd.read_csv("../data/trip_data_outliers_removed.csv", parse_dates=["start_time", "end_time"], index_col="Unnamed: 0")

    INDEPENDENT_VARS = ["constant","stops/km", "gradient (%)", "temp", "speed (km/h)", "passengers imputed", "start SOC (%)", "start full", "temp_square",]
    TARGET = ["ec/km (kWh/km)"]

    X_train, X_test, y_train, y_test = train_test_split(
        trip_data.drop(TARGET, axis=1),  # predictors
        trip_data[TARGET],  # target
        test_size=0.1,  # percentage of obs in test set
        random_state=0)  # seed to ensure reproducibility

    # X_train.shape, X_test.shape

    FEATURES_TO_DROP = list(set(trip_data.columns.tolist()) - set(INDEPENDENT_VARS+TARGET))

    ebus_pipe = Pipeline([
        ("passenger imputation", Passenger_imputer()),
        ("add constant", Add_constant()),
        ("kmph speed", Add_kmph()),
        ("soc full indicator", Add_soc_full()),
        ("squared features", Feature_square(variables=["temp","gradient (%)"])),
        ("drop features", DropFeatures(FEATURES_TO_DROP)),
        # ("linear regression model", LinearRegression(fit_intercept=False))
        ##
    ])




    ebus_pipe.fit(X_train, y_train)

    X_train = ebus_pipe.transform(X_train)
    X_test = ebus_pipe.transform(X_test)

    model = BayesianLinearRegression(meas_std=0.27)
    model.fit(X_train, y_train)

    y_train_pred, y_train_std = model.predict(X_train)

    error = (y_train - y_train_pred)
    plt.hist(error, bins=30)
    plt.title("training set errors")
    plt.show()
    print("train prediction error std was {}".format(error.std()))

    PLOT_VARS = ["start SOC (%)", "gradient (%)", "temp", "passengers imputed", "stops/km", "speed (km/h)"]

    for i, var in enumerate(PLOT_VARS):
        plt.subplot(3,2,i+1)
        plt.scatter(X_train[var], error, marker='.', s=0.1)
        plt.xlabel(var)

    plt.tight_layout()
    plt.show()


    y_test_pred, y_test_std = model.predict(X_test)
    error = (y_test.to_numpy() - y_test_pred)
    plt.hist(error, bins=30)
    plt.title("test set errors")
    plt.show()
    print("test prediction error std was {}".format(error.std()))


    tmp_range = np.linspace(10,30)
    out = tmp_range * -0.07 + tmp_range**2*0.002
    plt.plot(tmp_range, out)
    plt.title("offset energy consumption by temperature plot")
    plt.show()
