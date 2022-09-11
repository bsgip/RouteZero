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


trip_data["1SOC"] = 4.22
trip_data.loc[trip_data["bus type"]=="BYD", "1SOC"] = 3.68



tmp = (trip_data["1SOC"]/(trip_data["distance (m)"] / 1000))
v = np.sqrt((tmp**2/12)).mean()
print("measurement error computed:",v)

d1 = trip_data[trip_data["bus type"]=="BYD"]["distance (m)"].to_numpy()/1000
d2 = trip_data[trip_data["bus type"]!="BYD"]["distance (m)"].to_numpy()/1000

v1 = (3.68/d1.mean())**2/12
v2 = (4.22/d2.mean())**2/12