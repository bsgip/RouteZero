import os
import inflection
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
DEFAULT_DEADHEAD = 10.
DEFAULT_PEAK_PASSENGER = 38
DEFAULT_MIN_CHARGE_TIME = 60
DEFAULT_START_CHARGE = 90
DEFAULT_FINAL_CHARGE = 90
DEFAULT_RESERVE_CAPACITY = 20
DEFAULT_CHARGER_POWER = 150
DEFAULT_DEPOT_BATTERY_EFF = 0.95


class AppData:
    def __init__(self):
        """ Nothing to do here """
        pass

    @staticmethod
    def read_gtfs_file(gtfs_name):
        path = os.path.join(GTFS_FOLDER, gtfs_name, TRIP_FILENAME)
        return pd.read_csv(path)

    @staticmethod
    def battery_dict(depot_bat_cap, depot_bat_power, depot_bat_eff):
        battery = {"power": depot_bat_power,
                   "capacity": depot_bat_cap,
                   "efficiency": depot_bat_eff}
        return battery

    @staticmethod
    def set_passenger_loading(subset_trips, passengers):
        trips_data_sel = subset_trips.copy()
        trips_data_sel['passengers'] = passengers
        return trips_data_sel

    @staticmethod
    def get_routes(trips_data):
        return trips_data["route_short_name"].unique().tolist()

    @staticmethod
    def subset_data(selected_routes, trips_data):
        trips_data_sel = trips_data[trips_data["route_short_name"].isin(selected_routes)]
        return trips_data_sel

    @staticmethod
    def get_bus(bus_dict):
        max_passengers = bus_dict["max_passengers"]
        bat_capacity = bus_dict["bat_capacity"]
        charging_power = bus_dict["charging_power"]
        gross_mass = bus_dict["gross_mass"]
        charging_eff = bus_dict['charging_eff']
        eol_capacity = bus_dict["eol_capacity"]
        return ebus.Bus(max_passengers=max_passengers, battery_capacity=bat_capacity,
                            charging_rate=charging_power, gross_mass=gross_mass,
                            charging_efficiency=charging_eff, end_of_life_cap=eol_capacity / 100)

    @staticmethod
    def predict_energy_consumption(bus, subset_trip_data):
        model = LinearRegressionAbdelatyModel()
        ec_km, ec_total = model.predict_hottest(subset_trip_data, bus)
        return ec_km, ec_total


def create_options_dict():
    return dict(advanced_options=False,
                deadhead=DEFAULT_DEADHEAD / 100,
                min_charge_time=DEFAULT_MIN_CHARGE_TIME,
                start_charge=DEFAULT_START_CHARGE,
                final_charge=DEFAULT_FINAL_CHARGE,
                reserve_capacity=DEFAULT_RESERVE_CAPACITY,
                )


def run_init_feasibility(options_dict, ec_dict):
    grid_limit = "optim"
    ec_total = ec_dict["ec_total"]
    deadhead = options_dict["deadhead"]
    min_charge_time = options_dict["min_charge_time"]
    start_charge = options_dict["start_charge"]
    final_charge = options_dict["final_charge"]
    reserve = options_dict["reserve_capacity"]
    battery = options_dict["battery"]
    chargers = options_dict["chargers"]
    bus = AppData.get_bus(options_dict["bus_dict"])
    problem = Extended_feas_problem(None, ec_total, bus, chargers, grid_limit, start_charge=start_charge,
                                    final_charge=final_charge,
                                    deadhead=deadhead, resolution=RESOLUTION, min_charge_time=min_charge_time,
                                    reserve=reserve,
                                    battery=battery, ec_dict=ec_dict)
    results = problem.solve()
    return results


def create_optim_results_plot(results):
    times = results["times"] / 60
    grid_limit = results["grid_limit"]
    battery_power = results['battery_action']
    charging_power = results["charging_power"]
    aggregate_power = results["aggregate_power"]

    data = {"hour of week": times, "total charging power": charging_power, "depot battery power": battery_power,
            "aggregate power": aggregate_power}
    df = pd.DataFrame(data)

    fig = px.line(df, x="hour of week", y=df.columns[1:-1], title="Nice title")
    fig.add_trace(go.Scatter(x=df["hour of week"], y=df["aggregate power"], name='aggregate power',
                             line=dict(dash='dash')))
    fig.add_hline(y=grid_limit, line_dash="dash", annotation_text="Required grid limit",
                  annotation_position="top right")

    fig.update_layout(
        xaxis=dict(
            tickformat="digit",
            tickmode='linear',
            tick0=0,
            dtick=6
        ),
        legend_title=None,
        yaxis_title='Power (kW)'
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


def create_routes_map_figure(gtfs_name, map_title, ec_km, subset_trip_data):
    gdf = read_shp_file(gtfs_name)

    from RouteZero.map import create_map

    colorbar_str = 'energy per km'
    m = create_map(trips_data=subset_trip_data, shapes=gdf, value=ec_km, map_title=map_title,
                   colorbar_str=colorbar_str)
    # save html
    path = map_title + '.html'
    html_page = f'{path}'
    m.save(html_page)

    return m


def create_route_options():
    return [
        html.H4("Route options:"),
        dbp.FormGroup(
            label='deadhead (%)',
            inline=True,
            children=dbp.Slider(
                id="deadhead",
                value=DEFAULT_DEADHEAD,
                min=0.0,
                max=100,
                stepSize=1.,
                labelStepSize=50,
            )
        ),
        dbp.FormGroup(
            label='Peak passengers',
            inline=True,
            children=dbp.NumericInput(
                id="peak-passengers", value=DEFAULT_PEAK_PASSENGER, stepSize=1
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
                id='min-charge-time', value=DEFAULT_MIN_CHARGE_TIME, stepSize=1
            )
        ),
        dbp.FormGroup(
            label='start charge (%)',
            inline=True,
            children=dbp.Slider(
                id="start-charge",
                value=DEFAULT_START_CHARGE,
                min=0.0,
                max=100.,
                stepSize=1.,
                labelStepSize=50,
            )
        ),
        dbp.FormGroup(
            label='Final charge (%)',
            inline=True,
            children=dbp.Slider(
                id="final-charge",
                value=DEFAULT_FINAL_CHARGE,
                min=0.0,
                max=100.,
                stepSize=1.,
                labelStepSize=50,
            )
        ),
        dbp.FormGroup(
            label='Bus reserve capacity (%)',
            inline=True,
            children=dbp.Slider(
                id="reserve-capacity",
                value=DEFAULT_RESERVE_CAPACITY,
                min=0.0,
                max=100.,
                stepSize=1.,
                labelStepSize=50,
            )
        ),
        dbp.Button(id="confirm-optim-options", children="Next", n_clicks=0),
    ]


def create_depot_options(advanced_options):
    return [
        html.H4("Depot options:"),
        dbp.FormGroup(
            label='Max charger power (kW)',
            inline=True,
            children=dbp.NumericInput(
                id="charger-power", value=DEFAULT_CHARGER_POWER, stepSize=1
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
        html.Div(children=[
            dbp.FormGroup(
                label='Battery efficiency',
                inline=True,
                children=dbp.Slider(
                    id="depot-battery-eff",
                    value=DEFAULT_DEPOT_BATTERY_EFF,
                    min=0.0,
                    max=1.,
                    stepSize=0.01
                )
            )], hidden=not advanced_options),
    ]


def create_bus_options(advanced_options):
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
        html.Div(children=[
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
            )], hidden=not advanced_options),
        html.Div(children=[
            dbp.FormGroup(
                label="End of life capacity (%)",
                inline=True,
                children=dbp.Slider(
                    id="eol-capacity", value=80, min=0.0, max=100, stepSize=1., labelStepSize=50,
                ),
            )], hidden=not advanced_options),
        dbp.Button(id="confirm-bus-options", children="Next"),
    ]


app.layout = html.Div(
    className="grid-container",
    children=[
        # html.Div(className="header", children=html.H1("RouteZero EBus energy consumption and depot charging model")),
        html.Div(
            className="sidenav",
            children=[
                html.H4("Select data source:"),
                dbp.FormGroup(
                    id="formgroup",
                    required=True,
                    children=[dbp.Select(id="gtfs-selector", items=get_gtfs_options())],
                ),
                dbp.Checkbox("advanced options", id='advanced-options-checkbox', checked=False),
                html.Div(id="route-selection-form"),
                html.Div(id="route-options-form"),
                html.Div(id="bus-information-form"),
                html.Div(id="depot-options-form"),
                html.Div(id="feas-optim-options-form"),
                html.Div(id="hidden-div", style={"display": "none"}, children=None),
                # html.Div(id="options-store",style={"display":"none"})
            ], style={'padding': 10, 'flex': 1}
        ),
        html.Div(
            className="main",
            children=[
                html.Div(id="results-bus-number", children=None),
                html.Div(id="results-route-map", children=None),
                dcc.Loading(html.Div(id="results-init-feas", children=None))
            ],
        ),
        dcc.Store(id="route-names-store", storage_type="session"),
        dcc.Store(id="bus-store", data=dict(), storage_type="session"),
        dcc.Store(id="ec-store", data=dict(), storage_type="session"),
        dcc.Store(id="init-results-store", data=dict(), storage_type="session"),
    ],

)


@app.callback(
    [Output("route-selection-form", "children"),
     Output("route-names-store", "data")],
    Input("gtfs-selector", "value"),
    prevent_initial_callbacks=True,
)
def get_route_selection_form(gtfs_name):
    if gtfs_name is not None:
        trips_data = AppData.read_gtfs_file(gtfs_name)
        route_names = AppData.get_routes(trips_data)
        return [
            html.Div(
                children=[
                    html.H4("Select routes serviced by depot:"),
                    dcc.Dropdown(
                        id="route-selector",
                        options=[{"value": item, "label": item} for item in route_names],
                        value=["MTL", "NYC"],
                        multi=True,
                    ),
                    dbp.Button(id="route-selector-all", children="All", n_clicks=0),
                    dbp.Button(id="route-selector-confirm", children="Next"),
                ]
            ),
        ], route_names
    else:
        return (None, None)


@app.callback(
    Output("route-options-form", "children"),
    [Input("route-selector-confirm", "n_clicks")],
    [State("route-selector", "value")],
    # prevent_initial_callbacks=True,
)
def get_route_options_form(n_clicks, routes_sel):
    if n_clicks and (routes_sel is not None):
        return html.Div(
            id="route-options-form", children=create_route_options()
        )


@app.callback(
    Output("route-selector", "value"),
    [Input("route-selector-all", "n_clicks")],
    [State("route-names-store", "data")],
    prevent_initial_callback=True,
)
def select_all_routes(n_clicks, route_names):
    if n_clicks:
        options = [{"value": item, "label": item} for item in route_names]
        return [option["value"] for option in options]


def create_buses_in_traffic_plots(times, buses_in_traffic):
        times = times/60
        data = {"hour of week": times, "# buses": buses_in_traffic}
        df = pd.DataFrame(data)

        fig = px.line(df, x="hour of week", y="# buses", title="Buses on routes throughout the week")

        fig.update_layout(
            xaxis=dict(
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
    [State("advanced-options-checkbox", "checked")]
    # prevent_initial_callbacks=True
)
def show_bus_options_form(n_clicks, advanced_options):
    if n_clicks:
        return html.Div(
            id="bus-information-form", children=create_bus_options(advanced_options)
        )


@app.callback(
    [Output("results-route-map", "children"),
     Output("results-bus-number", "children"),
     Output("bus-store", "data"),
     Output("ec-store", "data")],
    [Input("confirm-bus-options", "n_clicks")],
    [State("gtfs-selector", "value"), State("max-passenger-count", "value"),
     State("battery-capacity-kwh", "value"), State("charging-capacity-kw", "value"),
     State("gross-mass-kg", "value"), State("charging-efficiency", "value"),
     State("eol-capacity", "value"), State("gtfs-selector", "value"),
     State("route-selector", "value"), State("peak-passengers", "value"),
     State("deadhead", "value")],
    prevent_initial_callbacks=True
)
def show_route_results(n_clicks, gtfs_file, max_passengers, bat_capacity, charging_power,
                   gross_mass, charging_eff, eol_capacity, gtfs_name, routes_sel,
                   peak_passengers, deadhead_percent):
    if n_clicks:
        bus_dict = {"max_passengers":max_passengers, "bat_capacity":bat_capacity,
                    "charging_power":charging_power, "gross_mass":gross_mass,
                    "charging_eff":charging_eff, "eol_capacity":eol_capacity}
        bus = AppData.get_bus(bus_dict)
        trips_data = AppData.read_gtfs_file(gtfs_name)
        subset_trip_data = AppData.subset_data(routes_sel, trips_data)
        subset_trip_data = AppData.set_passenger_loading(subset_trip_data, peak_passengers)
        ec_km, ec_total = AppData.predict_energy_consumption(bus, subset_trip_data)
        times, buses_in_traffic, depart_trip_energy_req, return_trip_energy_cons = route.calc_buses_in_traffic(subset_trip_data,
                                                                                                         deadhead_percent/100,
                                                                                                         RESOLUTION,
                                                                                                         ec_total)
        ec_dict = {"ec_km": ec_km,
                   "ec_total": ec_total,
                   "times": times,
                   "buses_in_traffic":buses_in_traffic,
                   "depart_trip_energy_req":depart_trip_energy_req,
                   "return_trip_energy_cons":return_trip_energy_cons}

        map_title = "Route Energy Consumption"
        create_routes_map_figure(gtfs_file, map_title, ec_km, subset_trip_data)
        return (html.Div(
                    children=html.Iframe(id='map', srcDoc=open(map_title + '.html').read(), width="80%", height="600vh")
                ),
                create_buses_in_traffic_plots(times, buses_in_traffic),
                bus_dict, ec_dict)
    else:
        return (None, None, None, None)


@app.callback(
    Output("feas-optim-options-form", "children"),
    [Input("confirm-bus-options", "n_clicks")],
    [State("advanced-options-checkbox", "checked")],
    prevent_initial_callbacks=True
)
def show_init_optim_options_form(n_clicks, advanced_options):
    if n_clicks:
        return [
            html.Div(
                id="depot-information-form", children=create_depot_options(advanced_options)
            ),
            html.Div(
                id="feas-optim-form", children=create_feas_optim_options()
            )
        ]


@app.callback(
    Output("results-init-feas", "children"),
    [Input("confirm-optim-options", "n_clicks")],
    [State("charger-power", "value"), State("depot-battery-capacity", "value"),
     State("depot-battery-power", "value"), State("depot-battery-eff", "value"),
     State("min-charge-time", "value"), State("start-charge", "value"),
     State("final-charge", "value"), State("reserve-capacity", "value"),
     State("deadhead", "value"), State("bus-store", "data"),
     State("ec-store", "data")],
    prevent_initial_callbacks=True
)
def run_initial_feasibility(n_clicks, charger_power, depot_bat_cap, depot_bat_power,
                            depot_bat_eff, min_charge_time, start_charge, final_charge,
                            reserve_capacity, deadhead_percent, bus_dict, ec_dict):
    if n_clicks:
        battery = AppData.battery_dict(depot_bat_cap, depot_bat_power, depot_bat_eff)
        chargers = {"power":charger_power, "number":"optim","cost":10}
        options_dict = {"deadhead":deadhead_percent/100,
                        "min_charge_time":min_charge_time,
                        "start_charge":start_charge/100,
                        "final_charge":final_charge/100,
                        "reserve_capacity":reserve_capacity/100,
                        "chargers":chargers,
                        "battery":battery,
                        "bus_dict":bus_dict}
        results = run_init_feasibility(options_dict, ec_dict)
        return create_optim_results_plot(results)
    else:
        return None


if __name__ == "__main__":
    app.run_server(debug=True)