import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



trip_data = pd.read_csv("../data/trip_data_outliers_removed.csv", parse_dates=["start_time", "end_time"],
                        index_col="Unnamed: 0")

PLOT_VARS = ["start SOC (%)", "gradient (%)", "temp", "average_passengers", "stops/km", "speed (km/h)"]


for i, var in enumerate(PLOT_VARS):
    plt.subplot(3, 2, i+1)
    plt.hist(trip_data[var], bins=30)
    plt.xlabel(var)
    plt.ylabel("# trips")
plt.tight_layout()
plt.show()

plt.hist(trip_data["ec/km (kWh/km)"], bins=30)
plt.xlabel("ec/km (kWh/km)")
plt.ylabel("# trips")
plt.show()


for i, var in enumerate(PLOT_VARS):
    plt.subplot(3, 2, i+1)
    plt.scatter(trip_data[var], trip_data["ec/km (kWh/km)"], marker='.')
    plt.xlabel(var)
    plt.ylabel("ec/km (kWh/km)")
plt.tight_layout()
plt.show()