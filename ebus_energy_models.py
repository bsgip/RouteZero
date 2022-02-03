import numpy as np
import gtfs_functions as gtfs
import pandas as pd
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
import srtm
import geopy.distance

# todo: temperature minimum and maximum for period



class linearRegressionAbdelatyModel:
    """
    A linear regression model for eBus energy consumption taken from

    Abdelaty, H.; Mohamed, M. APrediction Model for Battery Electric Bus Energy Consumption in Transit.
    Energies 2021, 14, 2824. https://doi.org/10.3390/en14102824

    model:

    EC = B0 + B1*GR + B2*SoCi + B3*RC + B4*HVAC + B5*PL + B6*Dagg + B7*SD + B8*Va + e

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
        self.B = np.array([-0.782, 0.38, 0.0124, 0.26, 0.036, 0.005, 0.065, 0.128, 0.007])

    def _predict(self, X):
        return np.dot(X, self.B)

    def predict(self, GR, SoCi, RC, HVAC, PL, Dagg, SD, Va):
        X = np.array([GR, SoCi, RC, HVAC, PL, Dagg, SD, Va])
        self._predict(X)

if __name__ == "__main__":
    print('Testing ebus model')









