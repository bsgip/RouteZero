# RouteZero
Python package for the RouteZero project. RouteZero provides predictions of electric bus energy usage when 
undertaking a trip on a specific route and depot charging optimisation. For a detailed description of the project 
goals, design, and theory see documentation/repord.pdf. 

For the purpose of the describing the code the following definitions are used:
- A *bus route* is defined as a sequence of bus stops and the path taken between them. A bus might do the same
route several times a day or week and multiple buses might be on the same route at the same time. A bus
route is what we think of if we were to say that “Newcastle West to University via Carrington” route is not
a very direct route between Newcastle West and Carrington.
- A *bus trip* is a single occurrence of a bus undertaking a given route. It has a specific start and end time.
This is what saying “the 9:11am Monday bus from Newcastle West to University” would be referring to.
Each trip on a route is considered to have different energy requirements as the bus may encounter different
traffic and weather conditions as well as having a different number of passengers.
- A *trip timetable* is defined as the schedule of trips that occur on a route or a collection of routes. It is what
a member of the public would use when they check what times they can catch a bus from stop A to stop
B. Importantly, a trip timetable provides no information about which bus is operating which trips/routes.
Likewise, it does not provide information about the sequence of trips a bus is undertaking or about when a
bus would return or depart the depot.

Information about the routes, trips, and timetable is extracted from Google Transit Feed Specification files. These 
files need to undergo preprocessing.

## Repository contents
The repository consists of
- A package of functions for data processing, prediction, and optimisation `routezero/routezero`
- A dash web application `RouteZero/app.py` and the pages in `RouteZero/pages`
- Some additional scripts used when generating the machine learning model and results for the report `RouteZero/scripts`
- The web application and data management functions expect data to be located in a folder `RouteZero/data`. This 
  folder is gitignored due to size.

## Installation
To install the RouteZero package complete the following steps:
1. Clone the repository
```
git clone git@github.com:bsgip/RouteZero.git
```

2. Activate virtual environment
```
python3 -m venv venv
source ./venv/bin/activate
```

3. Install the package
```
cd RouteZero
pip install -e .
```

4. Install the CBC optimisation solver by following the instructions at https://zoomadmin.com/HowToInstall/UbuntuPackage/coinor-cbc
```
sudo apt-get update -y
sudo apt-get install -y coinor-cbc
```

5. (For the webb application) Install dash-blueprint
```
pip install -e git+ssh://git@github.com/bsgip/dash-blueprint.git@master#egg=dash_blueprint

```

6. (Optional) Download preprocessed data files.  Data can be found in the BSGIP onedrive `Documents/12_Major Projects/RouteZero/data`, place the data in `RouteZero/data/`


## Processing GTFS files
To extract data from a GTFS file in preparation for use with the web application follow these steps:
1. Locate the gtfs zip file according to `data/gtfs/<name>_gtfs.zip`
2. Edit line 337 of `routezero/route.py` to be `name=<name>_gtfs`
3. Run `python routezero/route.py`
4. Check that the csv `data/gtfs/<name>/trip_data.csv` is created


### Current data sources
These are the current data sources and when they were last updated

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
    
Queensland
    - https://www.data.qld.gov.au/dataset/general-transit-feed-specification-gtfs-seq
    - last updated 09/07/2019

## Running the web application
Before running the web application make sure the required data files are inside `RouteZero/data/`.
Data can be found in the BSGIP onedrive `Documents/12_Major Projects/RouteZero/data`.


### Locally
Run the command
```angular2html
python app.py
```

### Deploying to the RouteZero server

The RouteZero web application is publicly deployed at routezero.cecs.anu.edu.au.
To update the deployment use the following instructions. If updating code make sure to merge into the 
deploy branch.

First, you will need to connect to the server using ssh. To do this, edit the ssh config file `.ssh/config'
to include
```
host routezero.cecs.anu.edu.au
  HostName Localhost
  Port 23455
  User u1118557
  IdentityFile ~/.ssh/id_rsa
  ProxyCommand ssh u1118557@der-lab.cecs.anu.edu.au nc %h %p 2> /dev/null
```

Then connect by running
```
ssh routezero.cecs.anu.edu.au
```

If new data needs to be transfered do so using rsync
```
rsync -av --progress -e ssh /path_to_data/data routezero.cecs.anu.edu.au:~/app/RouteZero/
```

Pull any updates from the deploy branch
```
cd app/RouteZero/
git pull
```

The web application is run in the background using screen https://linuxize.com/post/how-to-use-linux-screen/

Check if there is already a running screen
```
screen -ls
```
If so attach to the screen using and stop the server running by pressing ctrl-c
```
screen -r
```
else create a new screen
```
screen
```
Activate the virtual environment (if it is not activated)
```
source app/venv/bin/activate
```

Install package updates
```
pip install .
```

Start the server running again
```angular2html
gunicorn --workers=12 --threads=6 -b 0.0.0.0:8050 --timeout 600 app:server
```

Disconnect from screen by pressing ctrl-a d


#### Server time out issue

If the server needs to be reconfigured for some reason then there is a 
nginx server time out issue that needs to be fixed. See https://ubiq.co/tech-blog/increase-request-timeout-nginx/
