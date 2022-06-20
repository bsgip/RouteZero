# RouteZero
Electric bus energy usage prediction given high level route information and depot charging feasibility and optimisation.

## Installation

clone the repo and then
```angular2html
pip install .
```

Install the CBC optimisation solver by following the instructions at https://zoomadmin.com/HowToInstall/UbuntuPackage/coinor-cbc

then checkout scripts/UI_process_notebook.ipynb

this will also require you to install jupyter packages and ipywidgets

### For the UI

```angular2html
pip install dash
pip install -e git+ssh://git@github.com/bsgip/dash-blueprint.git@master#egg=dash_blueprint
```


## gtfs data
Greater sydney:
    - https://opendata.transport.nsw.gov.au/dataset/timetables-complete-gtfs
    - last updated 2022-03-21

Public transport victoria:
    - https://discover.data.vic.gov.au/dataset/ptv-timetable-and-geographic-information-2015-gtfs
    - last updated 06/01/2022 

## Design


### Inputs

*   Route information (GTFS)
*   Timetable information (GTFS)
*   Passenger loading (normal and peak)
*   Select bus from list which defines
    *   battery capacity
    *   mass
    *   passenger capacity
    *   charging rate limit
    *   End of life battery capacity (default 80%)
*   Type of charger (max charging rate or charger)
*   (optional) A specific sequence of routes to check for feasibility

### Parameters for first pass calculations

*   Between route deadhead time and energy (nominally 10% of route time and energy)
*   depot deadhead time i.e. additional time for returning and starting/finishing charge (nominally 15 mins)

### First pass calculation outputs

Route feasibility information:

*   Energy required for each route as total kWh and kWh/km
*   From the above, is each route feasible
*   if sequence was provided, is the sequence feasible

Depot and timetable minimum feasibility requirements:
- Minimum number of buses based on max buses on route at a given time
- Minimum grid connection limit at the depot
- Minimum number of chargers
- the charging per time graph

### Second pass calculation inputs:

- preferred charging times (select time slots that are preferred)
- Number of additional busses
- Depot battery capacity
- Additional charger sets (number/power for each set)

### Second pass calculation outputs:
- Feasibility of timetable for a given combination of # buses, # chargers, battery capacity, grid connection limit
Any values that were optimised
- The charging per time graph

### Export options
Results can be exported to CSV

  

### Nice to haves

*   Repeated worst case condition days and how many can be handled
*   Automatic comparisons
*   Other plots
*   Emissions saved on routes (would require a diesel bus consumption model)




## deployment
Need to have these dash settings
```angular2html
server = Flask(__name__)
app = dash.Dash(
__name__,
server=server,
suppress_callback_exceptions=True,
compress=True,
title="RouteZero",
update_title=None,
)
```

and then run

```angular2html
gunicorn --workers=5 --threads=1 -b 0.0.0.0:8050 app:server
```

