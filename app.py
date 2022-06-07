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

app = Dash(__name__, suppress_callback_exceptions=True)

app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

GTFS_FOLDER = "./data/gtfs"
TRIP_FILENAME = "trip_data.csv"
SHP_FILENAME = "shapes.shp"


class AppData:
    def __init__(self):
        """ Nothing to do here """
        # self.trips_data = read_gtfs_file(gtfs_name)
        pass

    def read_gtfs_file(self, gtfs_name):
        path = os.path.join(GTFS_FOLDER, gtfs_name, TRIP_FILENAME)
        self.trips_data = pd.read_csv(path)

    def get_routes(self):
        return self.trips_data["route_short_name"].unique().tolist()

    def subset_data(self, selected_routes):
        trips_data_sel = self.trips_data[self.trips_data["route_short_name"].isin(selected_routes)]
        self.trips_data_sel = trips_data_sel

    def get_subset_data(self):
        return self.trips_data_sel

    def predict_energy_consumption(self):
        bus = ebus.BYD()
        model = LinearRegressionAbdelatyModel()
        subset_trip_data = self.get_subset_data().copy()
        subset_trip_data['passengers'] = 38     # todo: allow this to be user settable
        ec_km, ec_total = model.predict_hottest(subset_trip_data, bus)
        self.ec_km = ec_km
        self.ec_total = ec_total
        return ec_km

# initialise global object to hold app data
appdata = AppData()


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



def calculate_buses_in_traffic(selected_routes):
    appdata.subset_data(selected_routes)
    trips_data_sel = appdata.get_subset_data()
    times, buses_in_traffic = route.calc_buses_in_traffic(
        trips_data_sel, deadhead=0.1, resolution=10
    )
    return times, buses_in_traffic


def create_additional_options():
    return [
        dbp.FormGroup(
            id="route-additional-options",
            label="Bus Options",
            children=[
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
                    label="Charging capacity (kW)",
                    inline=True,
                    children=dbp.NumericInput(
                        id="charging-capacity-kw", value=150, stepSize=1
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
                    label="End of life capacity",
                    inline=True,
                    children=dbp.Slider(
                        id="eol-capacity", value=0.8, min=0.0, max=1.0, stepSize=0.01
                    ),
                ),
                dbp.Button(id="confirm-additional-options", children="Next"),
            ],
        )
    ]


app.layout = html.Div(
    className="grid-container",
    children=[
        # html.Div(className="header", children="RouteZero User Interface"),
        html.Div(
            className="sidenav",
            children=[
                dbp.FormGroup(
                    id="formgroup",
                    required=True,
                    children=[dbp.Select(id="gtfs-selector", items=get_gtfs_options())],
                ),
                html.Div(id="route-selection-form"),
            ],
        ),
        html.Div(
            className="main",
            children=[
                html.Div(id="results-bus-number", children=None),
                html.Div(id="results-route-map", children=None)
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
                    dbp.MultiSelect(
                        id="route-selector",
                        required=True,
                        items=[{"value": item, "label": item} for item in routes],
                    ),
                    dbp.Button(id="route-selector-confirm", children="Next"),
                ]
            ),
            html.Div(id="additional-information-form"),
        ]


@app.callback(
    Output("results-bus-number", "children"),
    [Input("route-selector-confirm", "n_clicks")],
    [State("route-selector", "value")],
    prevent_initial_callbacks=True,
)
def calc_bus_number_output(n_clicks, routes):
    if n_clicks is None or routes is None:
        return "Select Bus Routes"
    times, buses_in_traffic = calculate_buses_in_traffic(routes)

    data = {"hour of week": times, "# buses": buses_in_traffic}
    df = pd.DataFrame(data)

    fig = px.line(df, x="hour of week", y="# buses", title="Buses in traffic")

    fig.update_layout(
        xaxis = dict(
            tickformat = "digit",
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1000
        )
    )

    return dcc.Graph(id="bus-count-graph", figure=fig)


@app.callback(
    Output("additional-information-form", "children"),
    [Input("route-selector-confirm", "n_clicks")],
    [State("route-selector", "value"), State("gtfs-selector", "value")],
    # prevent_initial_callbacks=True
)
def show_additional_options_form(n_clicks, routes, gtfs_file):
    if n_clicks:
        return html.Div(
            id="additional-information-form", children=create_additional_options()
        )


@app.callback(
    Output("results-route-map", "children"),
    [Input("confirm-additional-options", "n_clicks")],
    [State("gtfs-selector", "value")],
    # prevent_initial_callbacks=True
)
def show_route_map(n_clicks, gtfs_file):
    if n_clicks:
        map_title = "Route Energy Consumption"
        create_routes_map_figure(gtfs_file, map_title)
        return html.Div(
            children=html.Iframe(id='map', srcDoc=open(map_title+'.html').read(),width="80%",height=500)
        )



if __name__ == "__main__":
    app.run_server(debug=True)
