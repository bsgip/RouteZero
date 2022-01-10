# RouteZero
Electric bus modelling project. The goal is to create a model
that can be used to predict the battery capacity used by an eBus 
when completing a given route and help to specify the number of busses 
required, the charging depots and their batteries required to service
that route.

## Available data
The data can be separated into two sets:

- **Modelling data**. This is historical data from which model parameters and suitability can be determined
- **User input data**. This is information an end user enters from which the predictions need to be made

### Modelling data
The base data set will be provided by Zenobe and may need to be augmented

**Data provided by Zenobe:**

- Bus Specifications (Not sure what this will have yet)
- Grid connection information 
- Route details (length, stops)   (Zenobe to get info from Transit Systems)
- Timetables  (Zenobe to get info from Transit Systems)
- Passenger load (Zenobe to ask Transit Systems if they can/are willing to share this as Zenobe has no capability to monitor this)
- Bus data from January onwards, after all buses have been (hopefully) fitted with telematics and Zenobe is collecting data (see format below):

| Reading DateTime | Bus ID | Odometer | State of charge | GPS tracking |
|:------:|:------:|:------:|:------:|:------:|
| 1/7/2022 10:00 | 8031 | Numeric | Numeric | GPS point |
|1/7/2022 10:01 | 8031 | Numeric | Numeric | GPS point |

**Need to confirm the frequency at which the bus data will be collected**

**Data augmentation:**

- Weather information (primarily temperature) --- to be added based on location and time of day
- Elevation profile --- to be calculated from GPS tracking information
- Speed and acceleration profile --- to be calculated from GPS tracking information

### User input Data
This is information an end user enters from which the model will make predictions 
and recommendations.

This information will include:

- Bus specifications
- Some grid connection options (location of depot + is end of route charging allowed)
- Route details (length, stops)
- Required route timetable (frequency, days and times operating)

Data augmentation will be needed to get the following:
- Likely weather (temperature) --- based on location and time of year/day
- Elevation profile --- based on route information
- Speed and acceleration profiles for the route

### Information that might be lacking

- **battery state of health** ---
Typical to claim that battery capacity will linearly degrade over its lifetime
to end at a certain amount (i.e. over 5 years the battery capacity will degrade to be 80% of its
initial) capacity. It is also typical for ebus operaters to avoid using the bus below say 20% capacity
in order to avoid the worst of the nonlinear region of battery operation.
- **Battery temperature** --- Battery management systems may have battery cooling. This impacts performance in two ways: 1) keeps battery at optimal temperature reducing impact on health and improving performance. 2) Uses energy

## Electric Bus models
Initial models to consider:

1. Hjelkrem, O. A., Lervåg, K. Y., Babri, S., Lu, C., & Södersten, C. J. (2021). A battery electric bus energy consumption model for strategic purposes: Validation of a proposed model structure with data from bus fleets in China and Norway. Transportation Research Part D: Transport and Environment, 94, 102804. --- **A white box model**
2. Abdelaty, H., & Mohamed, M. (2021). A Prediction Model for Battery Electric Bus Energy Consumption in Transit. Energies, 14(10), 2824 --- **a Multivariate Multiple Regression model**
3. Li, P., Zhang, Y., Zhang, Y., & Zhang, K. (2021). Prediction of electric bus energy consumption with stochastic speed profile generation modelling and data driven method based on real-world big data. Applied Energy, 298, 117204. --- **Random forest and KNN**
4. Abdelaty, H., Al-Obaidi, A., Mohamed, M., & Farag, H. E. (2021). Machine learning prediction models for battery-electric bus energy consumption in transit. Transportation Research Part D: Transport and Environment, 96, 102868. --- **LSTM AND NN**
5. Chen, Y., Zhang, Y., & Sun, R. (2021). Data-driven estimation of energy consumption for electric bus under real-world driving conditions. Transportation Research Part D: Transport and Environment, 98, 102969. --- ** still to read **

Challenge with many of the models will be creating synthetic speed and 
acceleration profiles based off the end user input data. The following may
have some insights

- Kivekäs, K., VepsäläInen, J., & Tammi, K. (2018). Stochastic driving cycle synthesis for analyzing the energy consumption of a battery electric bus. IEEE Access, 6, 55586-55598.

A possible option is to generate rudimentary synthetic speed and acceleration profiles using
the python interface to OpenMaps

### Model 1 - A white box model 
This is a white box model.
Requires as an input a timestamped trajectory containing:
- speed
- distance travelled
- elevation (slope alpha)
- door status
- External temperature (T_exterior)

Model parameters and coefficients:

|   Parameter   |                   Description                   |    Value used in paper    | Range in other literature |
|:-------------:|:-----------------------------------------------:|:-------------------------:|:-------------------------:|
|      Cr       |        coefficient of rolling resistance        |           0.01            |       0.006 - 0.02        |
|      Cd       |               coefficient of drag               |            0.7            |         0.6 - 0.8         |
|    Pother     |          Additional power used by HVAC          |             2             |           2 - 7           |
|   eta_ebus    |         Efficiency of battery to motion         |           0.82            |        0.63 - 0.9         |
|  eta_battery  |       Efficiency of battery to aux power        |            0.9            |        0.63 - 0.9         |
|   eta_recup   |       efficiency of regenerative breaking       |           0.82            |        0.64 - 0.82        |
|      A_0      |                    door area                    |             3             |                           |
|       H       |                   door height                   |             2             |                           |
|       m       |        gross mass (passenger plus buss )        |       8800 - 12040        |                           |
|      rho      |                 density of air                  |         1.2kg/m^3         |                           |
|    A_front    |             frontal area of vehicle             |           7.99            |                           |
|     T_bus     |             temperature inside bus              |   winter 18 / summer 22   |                           |
| P_ventilation | power required to drive air through ventialtion |           0.5KW           |                           |
|     V_bus     |                  volume of bus                  |           67m^3           |                           |
|      c_p      |          specific heat capacity of air          |         1.005kg K         |                           |
| Q_ventilation |      volume flow rate from bus to exterior      |       V_bus / 450s        |                           |
|    Q_doors    |       volume flor rate out of open doors        | A_0/5 sqrt(g' H) + 0.0625 |                           |
|      g'       |        effective acceleration of gravity        | g delta T / average temp  |                           |
|    N_doors    |                 number of doors                 |             2             |                           |

** Model exclusion **

- Dependence on state of charge
- Dependence on health of battery

### Model 2 - A Multivariate Multiple Regression Model
This is a multivariate linear regression model. Model given by

![equation](https://latex.codecogs.com/svg.image?E_c&space;=&space;\beta_0&space;&plus;&space;\beta_1&space;GR&space;&plus;&space;\beta_2&space;D_{agg}&space;&plus;&space;\beta_3&space;R_c&space;&plus;&space;\beta_4&space;HVAC&space;&plus;&space;\beta_5&space;P_L&space;&plus;&space;\beta_6&space;S_D&space;&plus;&space;\beta_7&space;V_a&space;&plus;&space;\beta_8&space;SoCi&space;&plus;&space;\beta_9&space;L)

Output:
- E_c in Kw/h

The following inputs are considered:
- Average Road Grade (GR) as a percent
- Driver agressiveness (D_agg) three levels that correspond to max accel and decells?
- HVAC is the energy consumption due to heating, ventillation and air conditioning (KW), this could be related to external temperature (they give some relations)
- passenger loading (PL) number of passengers
- stop intensity of the route (SD) in stops/km
- Average speed (Va) in Km/h
- Initial state of battery charge (SoCi) as a percent
- Length of route (L) in metres
- Road condition (Rc) three levels which is related to rolling resistance

| Driver aggressiveness (D_agg) | Max acceleration | max Deceleration |    description     |
|:-----------------------------:|:----------------:|:----------------:|:------------------:|
|            level 1            |     0.5m/s^2     |     1.5m/s^2     |    Slow driving    |
|            level 2            |     1.5m/s^2     |     2.5m/s^2     |   Normal driving   |
|            level 3            |     2.5m/s^2     |      4m/s^2      | Aggressive driving |

| Road condition (Rc) | Coefficient of rolling resistance |      Description       |
|:-------------------:|:---------------------------------:|:----------------------:|
|       level 1       |              < 0.006              |     Good dry road      |
|       level 2       |              < 0.02               |     fair wet road      |
|       level 3       |              > 0.02               | poor icy road or slush |

| HVAC energy consumption (KW/h) | approx external temperature |
|:------------------------------:|:---------------------------:|
|             13.75              |             -20             |
|              6.7               |             -10             |
|              3.0               |              0              |
|              1.25              |             10              |
|              2.0               |             20              |
|             10.75              |             30              | 

**Exclusions / assumptions**
- No model for HVAC, HVAC energy consumption is assumed known as an input
- Model considers averages of road grade, accelerations and speed over route