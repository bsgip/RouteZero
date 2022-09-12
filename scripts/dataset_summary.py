import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



trip_data = pd.read_csv("../data/trip_data_outliers_removed.csv", parse_dates=["start_time", "end_time"],
                        index_col="Unnamed: 0")

PLOT_VARS = ["start SOC (%)", "gradient (%)", "temp", "average_passengers", "stops/km", "speed (km/h)"]


plt.figure(figsize=(10, 8))
for i, var in enumerate(PLOT_VARS):
    plt.subplot(3, 3, i+1)
    plt.hist(trip_data[var], bins=30)
    if var == "temp":
        plt.xlabel("Temperature ($^\circ$C)")
    elif var == "average_passengers":
        plt.xlabel("average passengers")
    else:
        plt.xlabel(var)
    plt.ylabel("# trips")

plt.subplot(3,3,9)
plt.hist(trip_data["ec/km (kWh/km)"], bins=30)
plt.xlabel("ec/km (kWh/km)")
plt.ylabel("# trips")

plt.subplot(3,3,8)
# plt.hist(trip_data["delta SOC (%)"], bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5])
plt.hist(trip_data["distance (m)"]/1000, bins=20)
plt.xlabel("distance (km)")
plt.ylabel("# trips")

plt.subplot(3,3,7)
# plt.hist(trip_data["delta SOC (%)"], bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5])
plt.hist(trip_data["delta SOC (%)"], bins=20)
plt.xlabel("SOC used (%)")
plt.ylabel("# trips")

plt.tight_layout()
plt.show()

plt.subplot(1,2,2)
plt.hist(trip_data["ec/km (kWh/km)"], bins=30)
plt.xlabel("ec/km (kWh/km)")
plt.ylabel("# trips")

plt.subplot(1,2,1)
# plt.hist(trip_data["delta SOC (%)"], bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5])
plt.hist(trip_data["delta SOC (%)"], bins=20)
plt.xlabel("SOC used (%)")
plt.ylabel("# trips")

plt.tight_layout()
plt.show()


for i, var in enumerate(PLOT_VARS):
    plt.subplot(2, 3, i+1)
    plt.scatter(trip_data[var], trip_data["ec/km (kWh/km)"], marker='.')
    plt.xlabel(var)
    plt.ylabel("ec/km (kWh/km)")


plt.tight_layout()
plt.show()

plt.scatter(trip_data["start SOC (%)"], trip_data["ec/km (kWh/km)"], marker='.', alpha=0.3)
plt.xlabel('start state of charge (%)')
plt.ylabel("ec/km (kWh/km)")
plt.show()

for var in PLOT_VARS:
    print("{} has mean {}, max {}, min {}".format(var, trip_data[var].mean(), trip_data[var].max(), trip_data[
        var].min()))

print("ec/km has mean {}, max {}, min {}".format(trip_data["ec/km (kWh/km)"].mean(),trip_data["ec/km (kWh/km)"].max(
),trip_data["ec/km (kWh/km)"].min()))



print("measurement std computed:",np.sqrt(trip_data["meas_variance"]).mean())

