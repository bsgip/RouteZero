import pandas as pd
import numpy as np
import RouteZero.models as Models

def impute_missing_passengers(trip_data):
    true_pass = trip_data["pass_true_part"].to_numpy()
    true_weight = trip_data["pass_true_weight"].to_numpy()
    passengers_imputed = true_pass * true_weight + 4 * (1-true_weight)

    trip_data["passengers_imputed"] = passengers_imputed
    return trip_data

def add_constant(trip_data):
    trip_data['constant'] = 1
    return trip_data

def add_kmph(trip_data):
    trip_data["speed (km/h)"] = trip_data["speed (m/s)"]*3.6
    return trip_data

def add_driver_agg(trip_data, driver_aggressiveness):
    trip_data["driver aggressiveness"] = driver_aggressiveness
    return trip_data

def add_road_condition(trip_data, road_condition):
    trip_data["road condition"] = road_condition
    return trip_data


def extract_model_Y_X(trip_data):
    X = trip_data[["constant","gradient (%)", "average_passengers", "stops/km",
                   "speed (km/h)", "start SOC (%)", "road condition", "driver aggressiveness", "temp"]].to_numpy()
    Y = trip_data[['ec/km (kWh/km)']].to_numpy().flatten()
    return X, Y


if __name__=="__main__":

    import matplotlib.pyplot as plt
    trip_data = pd.read_csv("/routezero/data/trip_data.csv", parse_dates=["start_time", "end_time"], index_col="Unnamed: 0")
    road_condition = 2                  # levels 1, 2, 3
    driver_aggressiveness = 2           # levels 1, 2, 3

    # bias_shift = -0.555
    # bias_shift = 0.9439
    bias_shift = 0

    # model = Models.LinearRegressionAbdelatyModel()
    model = Models.LinearRegMod()

    # trip_data = impute_missing_passengers(trip_data)
    trip_data = add_constant(trip_data)
    trip_data = add_kmph(trip_data)
    trip_data = add_road_condition(trip_data, road_condition=road_condition)
    trip_data = add_driver_agg(trip_data, driver_aggressiveness=driver_aggressiveness)

    distance = trip_data["distance (m)"]
    temperature = trip_data["temp"]
    start_soc = trip_data["start SOC (%)"]
    is_byd = trip_data['bus type'] == 'BYD'
    conv = np.zeros(len(is_byd))
    conv[is_byd] = 3.68
    conv[~is_byd] = 4.22

    X, Y = extract_model_Y_X(trip_data)
    Ypred = model._predict(X) + bias_shift
    errors = Y-Ypred
    print("all data")
    print("mean error: {}".format(errors.mean()))
    print("std errors: {}".format(errors.std()))

    plt.hist(errors, bins=30, range=[-1.5,1.5])
    plt.xlabel("error (kWh/km)")
    plt.title("all data")
    plt.show()

    ec_error = distance*errors/1000
    perc_SOC_error = ec_error/conv
    print('percent SOC error std {:.2f}'.format(perc_SOC_error.std()))
    plt.hist(perc_SOC_error, bins=30)
    plt.axvline(-1., linestyle='--',color='k')
    plt.axvline(1., linestyle='--',color='k')
    plt.xlabel('errors (%) SOC')
    plt.title('all data')
    plt.show()

    temp_range = np.linspace(temperature.min(), temperature.max(), 100)
    p = np.polyfit(temperature, errors * trip_data['speed (m/s)'].to_numpy() * 3.6, deg=2)
    temp_ref_vals = [10, 20, 30]
    tt = np.polyval(p, temp_ref_vals)
    p = np.polyfit(temperature, errors, deg=2)

    plt.scatter(temperature, errors, marker='.')
    plt.plot(temp_range, np.polyval(p, temp_range), linestyle='--', color='r')
    plt.title("error by temperature")
    plt.xlabel("temperature (degrees)")
    plt.ylabel("Y-Yhat (kWh/km)")
    plt.show()

    np.mean(errors[start_soc>97])
    soc_range = np.linspace(start_soc.min(), 100, 100)
    p = np.polyfit(start_soc[start_soc<90], errors[start_soc<90], deg=1)
    plt.scatter(start_soc, errors, marker='.')
    plt.plot(soc_range, np.polyval(p, soc_range), linestyle='--', color='r')
    plt.title("error by start soc")
    plt.xlabel("start soc (%)")
    plt.ylabel("Y-Yhat (kWh/km)")
    plt.show()

    plt.scatter(trip_data["average_passengers"], errors, marker='.')
    plt.title("error by imputed passengers")
    plt.xlabel("passengers")
    plt.ylabel("Y-Yhat (kWh/km)")
    plt.show()

    grad_range = np.linspace(trip_data["gradient (%)"].min(), trip_data["gradient (%)"].max(), 100)
    p = np.polyfit(trip_data["gradient (%)"], errors, deg=2)
    plt.scatter(trip_data["gradient (%)"], errors, marker='.')
    plt.plot(grad_range, np.polyval(p, grad_range), linestyle='--', color='r')
    plt.title("error by gradient")
    plt.xlabel("gradient (%)")
    plt.ylabel("Y-Yhat (kWh/km)")
    plt.show()

    plt.scatter(trip_data["stops/km"], errors, marker='.')
    plt.title("error by stops/km")
    plt.xlabel("stops/km")
    plt.ylabel("Y-Yhat (kWh/km)")
    plt.show()

    plt.scatter(trip_data["speed (km/h)"], errors, marker='.')
    plt.title("error by speed")
    plt.xlabel("speed (km/h)")
    plt.ylabel("Y-Yhat (kWh/km)")
    plt.show()

    # remove some outliers and save dataset
    err_std = errors.std()
    inds_outlier = np.abs(errors) > (3 * err_std)
    print("{} outliers_removed".format(np.sum(inds_outlier)))
    trip_data_outliers_removed = trip_data[~inds_outlier].reset_index(drop=True)

    trip_data_outliers_removed.drop(columns=["index", "road condition", "driver aggressiveness"])
    trip_data_outliers_removed.to_csv("/routezero/data/trip_data_outliers_removed.csv")

