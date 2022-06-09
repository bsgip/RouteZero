import os
import inflection
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import random
import dash_blueprint as dbp
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State

from RouteZero import route
import RouteZero.bus as ebus
from RouteZero.models import LinearRegressionAbdelatyModel
from RouteZero.optim import Extended_feas_problem

app = Dash(__name__, suppress_callback_exceptions=True)

app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

GTFS_FOLDER = "./data/gtfs"
TRIP_FILENAME = "trip_data.csv"
SHP_FILENAME = "shapes.shp"

RESOLUTION = 10

class AppData:
    def __init__(self):
        """ Nothing to do here """
        # self.trips_data = read_gtfs_file(gtfs_name)
        pass

    def read_gtfs_file(self, gtfs_name):
        path = os.path.join(GTFS_FOLDER, gtfs_name, TRIP_FILENAME)
        self.trips_data = pd.read_csv(path)

    def set_deadhead(self, deadhead_percent):
        self.deadhead = deadhead_percent/100

    def set_optim_options(self, min_charge_time, start_charge, final_charge, reserve_capacity):
        self.min_charge_time = min_charge_time
        self.start_charge = start_charge/100
        self.final_charge = final_charge/100
        self.reserve_capacity =  reserve_capacity/100

    def set_charger_power(self, charger_power):
        chargers = {"power":charger_power, "number":"optim", "cost":10}
        self.chargers = chargers

    def add_battery_parameters(self, depot_bat_cap, depot_bat_power, depot_bat_eff):
        battery = {"power":depot_bat_power,
                   "capacity":depot_bat_cap,
                   "efficiency":depot_bat_eff}
        self.battery=battery

    def set_passenger_loading(self, passengers):
        trips_data_sel = self.trips_data_sel.copy()
        trips_data_sel['passengers'] = passengers
        self.trips_data_sel = trips_data_sel

    def get_routes(self):
        return self.trips_data["route_short_name"].unique().tolist()

    def subset_data(self, selected_routes):
        trips_data_sel = self.trips_data[self.trips_data["route_short_name"].isin(selected_routes)]
        self.trips_data_sel = trips_data_sel

    def get_subset_data(self):
        return self.trips_data_sel

    def add_bus_parameters(self,  max_passengers,bat_capacity,charging_power,
                   gross_mass,charging_eff,eol_capacity):
        self.bus = ebus.Bus(max_passengers=max_passengers, battery_capacity=bat_capacity,
                       charging_rate=charging_power, gross_mass=gross_mass,
                       charging_efficiency=charging_eff, end_of_life_cap=eol_capacity/100)

    def predict_energy_consumption(self):
        model = LinearRegressionAbdelatyModel()
        subset_trip_data = self.get_subset_data()
        ec_km, ec_total = model.predict_hottest(subset_trip_data, self.bus)
        self.ec_km = ec_km
        self.ec_total = ec_total
        return ec_km

    def store_init_feas_results(self, results):
        self.init_feas_results = results

# initialise global object to hold app data
appdata = AppData()


def run_init_feasibility():
    battery = appdata.battery
    chargers = appdata.chargers
    grid_limit="optim"
    bus = appdata.bus
    trips_data = appdata.get_subset_data()
    ec_total = appdata.ec_total
    deadhead = appdata.deadhead
    min_charge_time = appdata.min_charge_time
    start_charge = appdata.start_charge
    final_charge = appdata.final_charge
    reserve = appdata.reserve_capacity
    problem = Extended_feas_problem(trips_data, ec_total, bus, chargers, grid_limit, start_charge=start_charge, final_charge=final_charge,
                                  deadhead=deadhead,resolution=RESOLUTION, min_charge_time=min_charge_time, reserve=reserve,
                                  battery=battery)
    results = problem.solve()
    appdata.store_init_feas_results(results)


def create_optim_results_plot(results):

    times = results["times"]/60
    grid_limit = results["grid_limit"]
    battery_power = results['battery_action']
    charging_power = results["charging_power"]
    aggregate_power = results["aggregate_power"]

    data = {"hour of week": times, "charging power": charging_power}
    df = pd.DataFrame(data)

    fig = px.line(df, x="hour of week", y="charging power",title="Nice title")

    fig.update_layout(
        xaxis = dict(
            tickformat="digit",
            tickmode='linear',
            tick0=0,
            dtick=6
        )
    )

    return dcc.Graph(id="charging-graph", figure=fig)


def get_gtfs_options():
    folders = [
        folder
        for folder in os.listdir(GTFS_FOLDER)
        if os.path.isdir(os.path.join(GTFS_FOLDER, folder))
    ]
    return [{"value": item, "label": inflection.titleize(item)} for item in folders]


def read_shp_file(gtfs_name):
    path = os.path.join(GTFS_FOLDER, gtfs_name, SHP_FILENAME)
    return gpd.read_file(path)


def create_routes_map_figure(gtfs_name, map_title):
    gdf = read_shp_file(gtfs_name)

    from RouteZero.map import create_map
    ec_km = appdata.predict_energy_consumption()

    colorbar_str = 'energy per km'
    m = create_map(trips_data=appdata.get_subset_data(), shapes=gdf,value=ec_km, map_title=map_title, colorbar_str=colorbar_str)
    # save html
    path = map_title + '.html'
    html_page = f'{path}'
    m.save(html_page)

    return m



def calculate_buses_in_traffic():
    trips_data_sel = appdata.get_subset_data()
    times, buses_in_traffic = route.calc_buses_in_traffic(
        trips_data_sel, deadhead=appdata.deadhead, resolution=RESOLUTION
    )
    return (times / 60).astype(int), buses_in_traffic

def create_route_options():
    return [
        html.H4("Route options:"),
        dbp.FormGroup(
            label='deadhead (%)',
            inline=True,
            children=dbp.Slider(
                id="deadhead",
                value=10.,
                min=0.0,
                max=100,
                stepSize=1.
            )
        ),
        dbp.FormGroup(
            label='Peak passengers',
            inline=True,
            children=dbp.NumericInput(
                id="peak-passengers", value=38, stepSize=1
            )
        ),
        dbp.Button(id="confirm-route-options", children="Next"),
    ]


def create_feas_optim_options():
    return [
        html.H4("Optimisation options:"),
        dbp.FormGroup(
            label='Min plugin time (mins)',
            inline=True,
            children=dbp.NumericInput(
                id='min-charge-time', value=60, stepSize=1
            )
        ),
        dbp.FormGroup(
            label='start charge (%)',
            inline=True,
            children=dbp.Slider(
                id="start-charge",
                value=90,
                min=0.0,
                max=100.,
                stepSize=1.
            )
        ),
        dbp.FormGroup(
            label='Final charge (%)',
            inline=True,
            children=dbp.Slider(
                id="final-charge",
                value=80,
                min=0.0,
                max=100.,
                stepSize=1.
            )
        ),
        dbp.FormGroup(
            label='Bus reserve capacity (%)',
            inline=True,
            children=dbp.Slider(
                id="reserve-capacity",
                value=20,
                min=0.0,
                max=100.,
                stepSize=1.
            )
        ),
        dbp.Button(id="confirm-optim-options", children="Next", n_clicks=0),
    ]

def create_depot_options():
    return [
        html.H4("Depot options:"),
        dbp.FormGroup(
            label='Max charger power (kW)',
            inline=True,
            children=dbp.NumericInput(
                id="charger-power", value=150, stepSize=1
            )
        ),
        dbp.FormGroup(
            label='Battery capacity (kWh)',
            inline=True,
            children=dbp.NumericInput(
                id="depot-battery-capacity", value=0, stepSize=1
            )
        ),
        dbp.FormGroup(
            label='Battery power (kW)',
            inline=True,
            children=dbp.NumericInput(
                id="depot-battery-power", value=0, stepSize=1
            )
        ),
        dbp.FormGroup(
            label='Battery efficiency',
            inline=True,
            children=dbp.Slider(
                id="depot-battery-eff",
                value=0.95,
                min=0.0,
                max=1.,
                stepSize=0.01
            )
        ),
    ]

def create_bus_options():
    return [
        html.H4("Bus options:"),
        dbp.FormGroup(
            label="Max Passengers",
            inline=True,
            children=dbp.NumericInput(
                id="max-passenger-count", value=70, stepSize=1
            ),
        ),
        dbp.FormGroup(
            label="Battery capacity (kWh)",
            inline=True,
            children=dbp.NumericInput(
                id="battery-capacity-kwh", value=400, stepSize=1
            ),
        ),
        dbp.FormGroup(
            label="Charging power (kW)",
            inline=True,
            children=dbp.NumericInput(
                id="charging-capacity-kw", value=300, stepSize=1
            ),
        ),
        dbp.FormGroup(
            label="Gross mass (kg)",
            inline=True,
            children=dbp.NumericInput(
                id="gross-mass-kg", value=18000, stepSize=1
            ),
        ),
        dbp.FormGroup(
            label="Charging efficiency",
            inline=True,
            children=dbp.Slider(
                id="charging-efficiency",
                value=0.9,
                min=0.0,
                max=1.0,
                stepSize=0.01,
            ),
        ),
        dbp.FormGroup(
            label="End of life capacity (%)",
            inline=True,
            children=dbp.Slider(
                id="eol-capacity", value=80, min=0.0, max=100, stepSize=1.
            ),
        ),
        dbp.Button(id="confirm-bus-options", children="Next"),
    ]


app.layout = html.Div(
    className="grid-container",
    children=[
        # html.Div(className="header", children="RouteZero User Interface"),
        html.Div(
            className="sidenav",
            children=[
                html.H4("Select data source:"),
                dbp.FormGroup(
                    id="formgroup",
                    required=True,
                    children=[dbp.Select(id="gtfs-selector", items=get_gtfs_options())],
                ),
                html.Div(id="route-selection-form"),
                html.Div(id="route-options-form"),
                html.Div(id="bus-information-form"),
                html.Div(id="depot-options-form"),
                html.Div(id="feas-optim-options-form")
            ],
        ),
        html.Div(
            className="main",
            children=[
                html.Div(id="results-bus-number", children=None),
                html.Div(id="results-route-map", children=None),
                html.Div(id="results-init-feas", children=None)
            ],
        ),
    ],
)


@app.callback(
    Output("route-selection-form", "children"),
    Input("gtfs-selector", "value"),
    prevent_initial_callbacks=True,
)
def get_route_selection_form(gtfs_name):

    if gtfs_name is not None:
        appdata.read_gtfs_file(gtfs_name)
        routes = appdata.get_routes()
        return [
            html.Div(
                children=[
                    html.H4("Select Routes:"),
                    dcc.Dropdown(
                        id="route-selector",
                        options=[{"value": item, "label": item} for item in routes],
                        value=["MTL", "NYC"],
                        multi=True,
                    ),
                    # dbp.MultiSelect(
                    #     id="route-selector",
                    #     required=True,
                    #     items=[{"value": item, "label": item} for item in routes],
                    # ),
                    dbp.Button(id="route-selector-all", children="All", n_clicks=0),
                    dbp.Button(id="route-selector-confirm", children="Next"),
                ]
            ),
        ]

@app.callback(
    Output("route-options-form", "children"),
    [Input("route-selector-confirm", "n_clicks")],
    # [State("route-selector", "value")],
    # prevent_initial_callbacks=True,
)
def get_route_options_form(n_clicks):
    if n_clicks:
        return html.Div(
            id="route-options-form",children=create_route_options()
        )

@app.callback(
    Output("route-selector", "value"),
    [Input("route-selector-all", "n_clicks")],
    prevent_initial_callback=True,
)
def select_all_routes(n_clicks):
    if n_clicks:
        routes = appdata.get_routes()
        options = [{"value": item, "label": item} for item in routes]
        return [option["value"] for option in options]

@app.callback(
    Output("results-bus-number", "children"),
    [Input("confirm-route-options", "n_clicks")],
    [State("route-selector", "value"), State("deadhead","value"),
     State("peak-passengers", "value")],
    prevent_initial_callbacks=True,
)
def calc_bus_number_output(n_clicks, routes, deadhead_percent, peak_passengers):
    if n_clicks is None or routes is None:
        return "Select Bus Routes"
    appdata.set_deadhead(deadhead_percent)
    appdata.subset_data(routes)
    appdata.set_passenger_loading(peak_passengers)
    times, buses_in_traffic = calculate_buses_in_traffic()

    data = {"hour of week": times, "# buses": buses_in_traffic}
    df = pd.DataFrame(data)

    fig = px.line(df, x="hour of week", y="# buses", title="Buses on routes throughout the week")

    fig.update_layout(
        xaxis = dict(
            tickformat="digit",
            tickmode='linear',
            tick0=0,
            dtick=6
        )
    )

    return dcc.Graph(id="bus-count-graph", figure=fig)


@app.callback(
    Output("bus-information-form", "children"),
    [Input("confirm-route-options", "n_clicks")],
    # prevent_initial_callbacks=True
)
def show_bus_options_form(n_clicks):
    if n_clicks:
        return html.Div(
            id="bus-information-form", children=create_bus_options()
        )


@app.callback(
    Output("results-route-map", "children"),
    [Input("confirm-bus-options", "n_clicks")],
    [State("gtfs-selector", "value"), State("max-passenger-count","value"),
     State("battery-capacity-kwh","value"), State("charging-capacity-kw", "value"),
     State("gross-mass-kg","value"), State("charging-efficiency","value"),
     State("eol-capacity","value")],
    # prevent_initial_callbacks=True
)
def show_route_map(n_clicks, gtfs_file, max_passengers,bat_capacity,charging_power,
                   gross_mass,charging_eff,eol_capacity):
    appdata.add_bus_parameters(max_passengers,bat_capacity,charging_power,
                   gross_mass,charging_eff,eol_capacity)

    if n_clicks:
        map_title = "Route Energy Consumption"
        create_routes_map_figure(gtfs_file, map_title)
        return html.Div(
            children=html.Iframe(id='map', srcDoc=open(map_title+'.html').read(),width="80%",height=500)
        )

@app.callback(
    Output("feas-optim-options-form", "children"),
    [Input("confirm-bus-options", "n_clicks")],
    prevent_initial_callbacks=True
)
def show_init_optim_options_form(n_clicks):
    if n_clicks:
        return [
            html.Div(
                id="depot-information-form", children=create_depot_options()
            ),
            html.Div(
                id="feas-optim-form", children=create_feas_optim_options()
            )
        ]


@app.callback(
    Output("results-init-feas", "children"),
    [Input("confirm-optim-options", "n_clicks")],
    [State("charger-power","value"), State("depot-battery-capacity","value"),
     State("depot-battery-power", "value"), State("depot-battery-eff","value"),
     State("min-charge-time","value"), State("start-charge","value"),
     State("final-charge","value"),State("reserve-capacity","value")],
    prevent_initial_callbacks=True
)
def run_initial_feasibility(n_clicks, charger_power, depot_bat_cap, depot_bat_power,
                            depot_bat_eff, min_charge_time, start_charge, final_charge,
                            reserve_capacity):
    if n_clicks:
        appdata.add_battery_parameters(depot_bat_cap, depot_bat_power, depot_bat_eff)
        appdata.set_charger_power(charger_power)
        appdata.set_optim_options(min_charge_time, start_charge, final_charge, reserve_capacity)
        run_init_feasibility()
        results = appdata.init_feas_results
        return create_optim_results_plot(results)



if __name__ == "__main__":
    app.run_server(debug=True)
