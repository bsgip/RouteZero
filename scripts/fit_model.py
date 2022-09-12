import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from RouteZero.models import BayesianLinearRegression, Add_constant, Add_kmph, Add_soc_full, Select_features, \
    Feature_square, Add_CrossProducts
from sklearn.ensemble import RandomForestRegressor

if __name__=="__main__":
    import matplotlib.pyplot as plt
    trip_data = pd.read_csv("../data/trip_data_outliers_removed.csv", parse_dates=["start_time", "end_time"], index_col="Unnamed: 0")


    INDEPENDENT_VARS = ["constant","stops/km", "gradient (%)", "temp", "speed (km/h)",
                        "average_passengers", "start SOC (%)", "soc > 97", "temp_square"]
    TARGET = ["ec/km (kWh/km)"]

    X_train, X_test, y_train, y_test = train_test_split(
        trip_data.drop(TARGET, axis=1),  # predictors
        trip_data[TARGET],  # target
        test_size=0.1,  # percentage of obs in test set
        random_state=0)  # seed to ensure reproducibility

    # X_train.shape, X_test.shape

    ebus_pipe = Pipeline([
        ("add constant", Add_constant()),
        ("kmph speed", Add_kmph()),
        ("soc full indicator", Add_soc_full()),
        ("squared features", Feature_square(variables=["temp"])),
        ("select features in order", Select_features(variables=INDEPENDENT_VARS)),
    ])


    meas_variances = X_train["meas_variance"].to_numpy()
    meas_variance_test = X_test["meas_variance"].to_numpy()
    weights_test = (1/meas_variance_test) / (1/meas_variance_test).sum()
    weights_test2 = (1/np.sqrt(meas_variance_test)) / (1/np.sqrt(meas_variance_test)).sum()

    ebus_pipe.fit(X_train, y_train)

    X_train = ebus_pipe.transform(X_train)
    X_test = ebus_pipe.transform(X_test)

    model = BayesianLinearRegression(regressor_vars=X_train.columns.tolist())


    model.load("../data/bayes_lin_reg_model_2.json")
    # model.fit(X_train, y_train, meas_variance=0.27**2)
    # model.fit(X_train, y_train, meas_variance=meas_variances)

    # model = RandomForestRegressor(max_depth=10, random_state=0)
    # model.fit(X_train, y_train.to_numpy().flatten())
    # y_train_pred = model.predict(X_train)

    # save model
    # model.save("../data/bayes_lin_reg_model.json")        # before adding measurement weights
    # model.save("../data/bayes_lin_reg_model_2.json")        # with measurement weights



    y_train_pred, y_train_std = model.predict(X_train)

    error = (y_train - y_train_pred)
    # error = (y_train.to_numpy().flatten() - y_train_pred)
    plt.hist(error, bins=30)
    plt.title("training set errors")
    plt.show()
    print("train prediction error std was {}".format(error.std()))
    print("train prediction error mean was {}".format(error.mean()))

    C_std = np.sqrt(np.diag(model.post_var))
    for i in range(9):
        print("c_{} for {} has mean {:.3g} and 95% CI of [{:.3g}, {:.3g}]".format(i, X_train.columns[i],
                                                                                  model.params[i][0],
                                                                      model.params[i][0]-2*C_std[i],
                                                                      model.params[i][0]+2*C_std[i]))


    PLOT_VARS = ["start SOC (%)", "gradient (%)", "temp", "average_passengers", "stops/km", "speed (km/h)"]

    for i, var in enumerate(PLOT_VARS):
        plt.subplot(3,2,i+1)
        plt.scatter(X_train[var], error, marker='.', s=0.1)
        plt.xlabel(var)

    plt.tight_layout()
    plt.show()

    y_test_pred, y_test_std = model.predict(X_test)
    # y_test_pred = model.predict(X_test)
    error = (y_test.to_numpy() - y_test_pred)
    # error = (y_test.to_numpy().flatten() - y_test_pred)

    plt.hist(error, bins=30)
    plt.title("test set errors")
    plt.show()
    print("test prediction error std was {}".format(error.std()))
    print("weighted test prediction error mean was {}".format((error.flatten()*weights_test).sum()))


    tmp_range = np.linspace(10,30)
    out = tmp_range * -0.07 + tmp_range**2*0.002
    plt.plot(tmp_range, out)
    plt.title("offset energy consumption by temperature plot")
    plt.show()

    tmp = X_train.drop(columns=["temp_square", "soc > 97", "constant"])
    input_mean = tmp.mean()
    input_max = tmp.max()
    input_min = tmp.min()
    vars = tmp.columns.tolist()


    plt.figure(figsize=(10,8))
    for i, var in enumerate(vars):
        trial = pd.DataFrame(columns=vars)
        trial[var] = np.linspace(input_min[var],input_max[var])
        for var2 in vars:
            if var2 != var:
                trial[var2] = input_mean[var2]

        trial = Add_constant().transform(trial)
        trial = Add_soc_full().transform(trial)
        trial = Feature_square(variables=["temp", "gradient (%)"]).transform(trial)
        trial = trial[INDEPENDENT_VARS]

        y_trial, y_trial_std = model.predict(trial)

        plt.subplot(3,2,i+1)
        plt.fill_between(trial[var], (y_trial.flatten() - 2*y_trial_std), y_trial.flatten()+2*y_trial_std, alpha=0.3, label="95% CI")
        plt.plot(trial[var], y_trial, label='mean sensitivity')
        plt.legend()
        if var=="temp":
            plt.xlabel("Temperature ($^\circ$C)")
        elif var=="average_passengers":
            plt.xlabel("average passengers")
        else:
            plt.xlabel(var)
        plt.ylabel('ec/km (kwh/km)')

    plt.tight_layout()
    plt.show()

