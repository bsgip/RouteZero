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
pip install dash_extensions
pip install -e git+ssh://git@github.com/bsgip/dash-blueprint.git@master#egg=dash_blueprint
```


## gtfs data
Greater sydney:
    - https://opendata.transport.nsw.gov.au/dataset/timetables-complete-gtfs
    - last updated 2022-06-04

Public transport victoria:
    - https://discover.data.vic.gov.au/dataset/ptv-timetable-and-geographic-information-2015-gtfs
    - last updated 06/01/2022 

ACT:
    - https://www.transport.act.gov.au/contact-us/information-for-developers
    - last updated 08/04/2022

Tas:
    - https://www.metrotas.com.au/community/gtfs/
    - last updated 18/05/2022

Northern Territory
    - https://dipl.nt.gov.au/data/bus-timetable-data-and-geographic-information
    - last updated 23/04/2022

Western Australia
    - https://www.transperth.wa.gov.au/About/Spatial-Data-Access
    - last updated 23/06/2022

South Australia
    - https://data.sa.gov.au/data/dataset/https-gtfs-adelaidemetro-com-au
    - last updated 30/10/2022

## Design





## deployment
run
```angular2html
gunicorn --workers=5 --threads=1 -b 0.0.0.0:8050 --timeout 600 app:server
```

nginx server time out https://ubiq.co/tech-blog/increase-request-timeout-nginx/
