import os
import inflection
import pandas as pd
import geopandas as gpd
import plotly.colors as px_colors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# dash stuff
import dash_blueprint as dbp
from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, \
    ServersideOutputTransform, callback
import dash
import dash_bootstrap_components as dbc

from RouteZero import route
import RouteZero.bus as ebus
from RouteZero.models import LinearRegressionAbdelatyModel, summarise_results
from RouteZero.optim import Extended_feas_problem
from RouteZero.optim import determine_charger_use

# app = DashProxy(__name__, suppress_callback_exceptions=True, transforms=[ServersideOutputTransform()])
#
# app.scripts.config.serve_locally = True
# app.css.config.serve_locally = True

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


dash.register_page(__name__, path='/')

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
    def get_agencies(trips_data):
        return trips_data["agency_name"].unique().tolist()

    @staticmethod
    def subset_data(selected_routes, trips_data, agency_name):
        tmp = trips_data[trips_data["agency_name"] == agency_name]
        trips_data_sel = tmp[tmp["route_short_name"].isin(selected_routes)]
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


def run_init_feasibility(options_dict, ec_dict, num_buses):
    grid_limit = "optim"
    deadhead = options_dict["deadhead"]
    min_charge_time = options_dict["min_charge_time"]
    start_charge = options_dict["start_charge"]
    final_charge = options_dict["final_charge"]
    reserve = options_dict["reserve_capacity"]
    battery = options_dict["battery"]
    chargers = options_dict["chargers"]
    bus = AppData.get_bus(options_dict["bus_dict"])
    problem = Extended_feas_problem(None, None, bus, chargers, grid_limit, start_charge=start_charge,
                                    final_charge=final_charge,
                                    deadhead=deadhead, resolution=RESOLUTION, min_charge_time=min_charge_time,
                                    reserve=reserve, num_buses=num_buses,
                                    battery=battery, ec_dict=ec_dict)

    results = problem.solve()
    used_daily, charged_daily = problem.summarise_daily()
    results['used_daily'] = used_daily
    results['charged_daily'] = charged_daily
    chargers_in_use = determine_charger_use(chargers, problem.Nt_avail, results["charging_power"], problem.windows)
    results['chargers_in_use'] = chargers_in_use
    return results

def display_init_summary(init_results):
    text = """
    ###### Route energy usage analysis results
    - Data source is the publicly available {gtfs_name} GTFS data. From this the busiest week has been extracted for analysis.
    - {num_routes} routes have been selected.
    - The 'Buses on route' graph shows how many buses are active on these routes throughout the week.
    - From this we can see a minimum of {num_buses} buses are required for the subsequent analysis.
    - The energy requirement for these routes has been predicted using a data-driven model and considering:
        - worst case temperatures at the location throughout the year,
        - worst case loading (all trips have peak loading),
        - energy requirements change throughout the day due to different traffic conditions and temperature.
    - The max energy required on a single route is {max_energy:.1f}kWh and the average energy required is {av_energy:.1f}kWh{bus_cap}.
    - The predicted total energy required on active routes is shown in the right hand graph.
    - The map shows the energy requirements of specific routes during the selected time window. 
    """.format(gtfs_name=' '.join(elem.capitalize() for elem in init_results['gtfs_name'].replace("_"," ").split()),
               num_routes=init_results["num_routes"],
               num_buses=init_results['num_buses'],
               max_energy=init_results['max_energy'],
               av_energy=init_results['av_energy'],
               bus_cap='' if init_results['bus_eol_capacity']>init_results['max_energy'] else ". **This is greater than the end of life bus battery capacity. Increase the bus battery capacity before proceeding**")

    return dcc.Markdown(text)


def display_optim_summary(results):
    setup_summary = """ 
    ###### Setup summary:
    - Trip deadhead (additional time and energy between trips): {deadhead}%
    - Peak number of passengers considered on routes: {peak_passengers}
    - {bus_num} buses were used with:
        - battery capacity: {bus_bat_cap}kWh
        - max charging rate: {bus_charge}kW
        - max passengers: {max_pass}
        - gross mass: {gross_mass}kg
        - efficiency: {bus_eta}%
        - end of life capacity: {bus_eol}%
    - Depot charger power: {charger_power:.1f}kw 
    - Depot onsite battery with:
        - capacity: {depot_bat_cap:.1f}kWh
        - power rating: {depot_bat_power:.1f}kW
        - effiency: {depot_bat_eta}%
    - Optimisation options:
        - sum bus battery start of week charge: {start_charge}%
        - required bus battery end of week charge: {final_charge}%
        - Minimum time allowed to plug in a bus: {min_charge_time}mins
        - Desired reserve sum bus battery capacity: {reserve}%
    """.format(bus_num=results['num_buses'],
               bus_bat_cap=results['bus']['capacity'],
               bus_charge=results['bus']['max_charging'],
               max_pass=results['bus']['max_passengers'],
               gross_mass=results['bus']['gross_mass'],
               bus_eol=results['bus']['end_of_life_cap']*100,
               bus_eta=results['bus']['efficiency']*100,
               charger_power=results['chargers']['power'][0],
               depot_bat_cap=results["battery_spec"]['capacity'],
               depot_bat_power=results["battery_spec"]["power"],
               depot_bat_eta=results["battery_spec"]["efficiency"]*100,
               start_charge=results["start_charge"]*100,
               final_charge=results['final_charge']*100,
               min_charge_time=results["min_charge_time"],
               reserve=results['reserve']*100,
               deadhead=results["deadhead"]*100,
               peak_passengers=results["peak_passengers"]
               )

    try_text = "Try increasing the 'max charging power'/'bus battery capacity'/'number of buses'"

    results_summary = """
    ###### Depot charging analysis summary
    - The depot could {sol_text}sufficiently charge the buses {failed_sol}
    - Desired reserve capacity of {reserve}% was {reserve_text} achieved{failed_reserve} 
    - Desired end of week charge of {final_charge}% was {final_text}achieved{failed_final}
    - Required grid connection: {grid_con:.1f}kW 
    - {num_chargers} chargers of {charger_power}kW required to be shared by the {num_buses} buses
    """.format(grid_con=results['grid_limit'],
               num_chargers=int(results['chargers']['number'][0]),
               charger_power=int(results['chargers']['power'][0]),
               num_buses=results['num_buses'],
               sol_text='' if results['infeasibility_%'] < 0.01 else 'not ',
               reserve_text='not' if results['reserve_infease_%'] > 0.01 else '',
               reserve=results["reserve"]*100,
               final_charge=results['final_charge']*100,
               failed_sol='' if results['infeasibility_%'] < 0.01 else '. Failed by {:.1f}%. '.format(results['infeasibility_%']) +try_text,
               failed_reserve='' if results['reserve_infease_%'] < 0.01 else '. Failed by {:.1f}%. '.format(results['reserve_infease_%']) +try_text,
               final_text='' if results['final_soc_infeas_%'] < 0.01 else 'not ',
               failed_final='' if results['final_soc_infeas_%'] < 0.01 else '. Failed by {:.1f}%. '.format(results['final_soc_infeas_%']) +try_text,
               )

    return [dbc.Row("",style={"height":"5rem"}),
            dcc.Markdown(results_summary),
            dcc.Markdown(setup_summary),
            ]
    # return [dbc.Row("",style={"height":"5rem"}),
    #         header_text,
    #         sol_text,
    #         bus_text,
    #         charger_text,
    #         ]


def create_optim_results_plot(results):
    times = results["times"] / 60
    grid_limit = results["grid_limit"]
    battery_power = results['battery_action']
    charging_power = results["charging_power"]
    aggregate_power = results["aggregate_power"]
    energy_available = results["total_energy_available"]
    battery_soc = results['battery_soc']
    reserve_energy = results['reserve_energy']
    used_daily = results['used_daily']
    charged_daily = results['charged_daily']

    data = {"hour of week": times, "total charging power": charging_power, "depot battery power": battery_power,
            "aggregate power": aggregate_power}
    df = pd.DataFrame(data)

    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=['Charging power', 'Sum state of charge',
                                        'Daily energy use and charging', 'Chargers in use'])

    fig.add_trace(
        go.Scatter(x=times, y=charging_power, name="sum bus charging", legendgroup='1'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=times, y=battery_power, name="battery power", legendgroup='1'),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df["hour of week"], y=df["aggregate power"], name='aggregate power',
                             line=dict(dash='dash'), legendgroup='1'), row=1, col=1)
    fig.add_hline(y=grid_limit, line_dash="dash", annotation_text="Required grid limit",
                  annotation_position="top right", row=1, col=1)

    fig.add_trace(go.Scatter(x=times, y=energy_available, name='sum bus battery', legendgroup='2'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=battery_soc, name='depot battery', legendgroup='2'),
                  row=2, col=1)
    fig.add_hline(y=reserve_energy, line_dash="dash", annotation_text="Reserve bus capacity",
                  annotation_position="top right", row=2, col=1)

    fig.add_trace(go.Bar(x=['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'], y=used_daily / 1000, name='used daily',
                         legendgroup='3'),
                  row=3, col=1)
    fig.add_trace(
        go.Bar(x=['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'], y=charged_daily / 1000, name='charged daily',
               legendgroup='3'),
        row=3, col=1)

    chargers_in_use = results['chargers_in_use']
    chargers = results['chargers']
    r, c = chargers_in_use.shape
    for i in range(c):
        fig.add_trace(go.Scatter(x=times, y=chargers_in_use[:, i],
                                 name="{}kW chargers".format(chargers['power'][i]),
                                 legendgroup=4),
                      row=4, col=1)

    fig.update_layout(
        xaxis=dict(
            tickformat="digit",
            tickmode='linear',
            tick0=0,
            dtick=6
        ),
        legend_title=None,
        yaxis_title='Power (kW)',
        height=1066,  # 800/3*4,
        width=800,
        legend_tracegroupgap=200,
        xaxis2=dict(
            tickformat="digit",
            tickmode='linear',
            tick0=0,
            dtick=6
        ),
        yaxis2_title='State of charge (kWh)',
        yaxis3_title='Energy (MWh)',
        xaxis4=dict(
            tickformat="digit",
            tickmode='linear',
            tick0=0,
            dtick=6
        ),
        yaxis4_title='Number',
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


def create_routes_map_figure(gtfs_name, map_title, route_summaries, window):
    gdf = read_shp_file(gtfs_name)

    from RouteZero.map import create_map

    colorbar_str = 'energy per km'
    m = create_map(route_summaries=route_summaries, shapes=gdf, map_title=map_title,
                   colorbar_str=colorbar_str, window=window)

    return m._repr_html_()


def create_route_options():
    return [
        html.H5("Step 2) Predicting energy usage on routes"),
        html.H6("Route options:"),
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
                id="peak-passengers", value=DEFAULT_PEAK_PASSENGER, stepSize=1, min=0
            )
        ),
    ]


def create_feas_optim_options():
    return [
        html.H6("Optimisation options:"),
        dbp.FormGroup(
            label='Min plugin time (mins)',
            inline=True,
            children=dbp.NumericInput(
                id='min-charge-time', value=DEFAULT_MIN_CHARGE_TIME, stepSize=1,min=0
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
        dbp.Button(id="confirm-optim-options", children="Optimise charging", n_clicks=0),
    ]


def create_depot_options(advanced_options, ec_dict):
    min_buses = int(ec_dict['buses_in_traffic'].max())
    max_buses = int(max(1.2 * min_buses, min_buses+5))
    return [
        html.H5("Step 3) Optimise charging at depot"),
        html.P("Optimises the aggregate charging profile to find the minimum power rating"
               " for the depot grid connection and the minimum number of bus chargers required."),
        html.H6("Depot options:"),
        dbp.FormGroup(
            label='Max charger power (kW)',
            inline=True,
            children=dbp.NumericInput(
                id="charger-power", value=DEFAULT_CHARGER_POWER, stepSize=1, min=0
            )
        ),
        dbp.FormGroup(
            label='On-site battery \n capacity (kWh)',
            inline=True,
            children=[dbp.NumericInput(
                id="depot-battery-capacity", value=0, stepSize=1, min=0
            )]
        ),
        dbp.FormGroup(
            label='On-site battery power (kW)',
            inline=True,
            children=dbp.NumericInput(
                id="depot-battery-power", value=0, stepSize=1, min=0
            )
        ),
        dbp.FormGroup(
            label='Number of buses',
            inline=True,
            children=dbp.Slider(
                id="num-buses-slider",
                value=min_buses,
                min=min_buses,
                max=max_buses,
                stepSize=1,
                labelStepSize=int(np.floor((max_buses-min_buses)/2))
            )
        ),
        html.Div(children=[
            dbp.FormGroup(
                label='On-site battery efficiency',
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
        html.H6("Bus options:"),
        dbp.FormGroup(
            label="Max Passengers",
            inline=True,
            children=dbp.NumericInput(
                id="max-passenger-count", value=70, stepSize=1, min=0
            ),
        ),
        dbp.FormGroup(
            label="Battery capacity (kWh)",
            inline=True,
            children=dbp.NumericInput(
                id="battery-capacity-kwh", value=400, stepSize=1, min=0
            ),
        ),
        dbp.FormGroup(
            label="Charging power (kW)",
            inline=True,
            children=dbp.NumericInput(
                id="charging-capacity-kw", value=300, stepSize=1, min=0
            ),
        ),
        dbp.FormGroup(
            label="Gross mass (kg)",
            inline=True,
            children=dbp.NumericInput(
                id="gross-mass-kg", value=18000, stepSize=1, min=0
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
        dbp.Button(id="confirm-bus-options", children="Predict route energy usage"),
    ]


layout = html.Div(
    className="grid-container",
    children=[
        html.Div(
            className="sidenav",
            children=[
                html.H5("Step 1) Select gtfs source and routes"),
                html.H6("Select data source:"),
                dbp.FormGroup(
                    id="formgroup",
                    required=True,
                    children=[dbp.Select(id="gtfs-selector", items=get_gtfs_options())],
                ),
                dbp.Checkbox("advanced options", id='advanced-options-checkbox', checked=False),
                dcc.Loading(html.Div(id="agency-selection-form")),
                html.Div(id="route-selection-form"),
                html.Div(id="route-options-form"),
                html.Div(id="bus-information-form"),
                html.Div(id="depot-options-form"),
                html.Div(id="feas-optim-options-form"),
            ], style={'padding': 10, 'flex': 1}
        ),
        html.Div(
            className="main",
            children=[
                dcc.Loading(html.Div(id="results-bus-number", children=None)),
                dcc.Loading(html.Div(id="results-route-map", children=None)),
                dcc.Loading(html.Div(id="results-init-feas", children=None))
            ]
        ),
        dcc.Store(id="agency-store", storage_type="memory"),
        dcc.Store(id="bus-store", data=dict(), storage_type="memory"),
        dcc.Store(id="ec-store", data=dict(), storage_type="memory"),
        # dcc.Store(id="init-results-store", data=dict(), storage_type="memory"),
        dcc.Store(id="route-summary-store", data=dict(), storage_type="memory"),
        dcc.Download(id="download-dataframe-csv")
    ],

)


@callback(
    [Output("agency-selection-form", "children"),
     ServersideOutput("agency-store", "data")],
    Input("gtfs-selector", "value"),
    prevent_initial_callback=True
)
def get_agency_selection_form(gtfs_name):
    if gtfs_name is not None:
        trips_data = AppData.read_gtfs_file(gtfs_name)
        agency_names = AppData.get_agencies(trips_data)
        route_agency_df = trips_data[["route_short_name", "agency_name"]].drop_duplicates()
        route_agency_dict = route_agency_df.to_dict()
        return [
                   html.Div(
                       children=[
                           html.H6("Select agency:"),
                           dbp.Select(id="agency-selector",
                                      items=[{"value": item, "label": item} for item in agency_names],
                                      ),

                       ]
                   ),
               ], route_agency_dict
    else:
        return (None, None)


@callback(
    Output("route-selection-form", "children"),
    Input("agency-selector", "value"),
    [State("agency-store", "data")],
    prevent_initial_callbacks=True,
)
def get_route_selection_form(agency_name, route_agency_dict):
    if agency_name is not None:
        route_agency_df = pd.DataFrame.from_dict(route_agency_dict)
        tmp = route_agency_df[route_agency_df["agency_name"] == agency_name]
        route_names = tmp['route_short_name'].unique().tolist()
        # trips_data = AppData.read_gtfs_file(gtfs_name)
        # route_names = AppData.get_routes(trips_data)
        return [
                   html.Div(
                       children=[
                           html.H6("Select routes serviced by depot:"),
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
               ]



@callback(
    [Output("route-options-form", "children"),
     Output("bus-information-form", "children")],
    [Input("route-selector-confirm", "n_clicks")],
    [State("route-selector", "value"),
     State("advanced-options-checkbox", "checked")],
    # prevent_initial_callbacks=True,
)
def get_route_options_form(n_clicks, routes_sel, advanced_options):
    if n_clicks and (routes_sel is not None):
        return html.Div(
            id="route-options-form", children=create_route_options()
        ), show_bus_options_form(advanced_options)

    else:
        return (None, None)


@callback(
    Output("route-selector", "value"),
    [Input("route-selector-all", "n_clicks")],
    [State("agency-store", "data"), State("agency-selector", "value")],
    prevent_initial_callback=True,
)
def select_all_routes(n_clicks, route_agency_dict, agency_name):
    if n_clicks:
        route_agency_df = pd.DataFrame.from_dict(route_agency_dict)
        tmp = route_agency_df[route_agency_df["agency_name"] == agency_name]
        route_names = tmp['route_short_name'].unique().tolist()
        options = [{"value": item, "label": item} for item in route_names]
        return [option["value"] for option in options]


def create_buses_in_traffic_plots(times, buses_in_traffic, energy_req):
    times = times / 60
    if times[-1] < 168:
        times = np.hstack([times, np.array([times[-1]+0.01, 168])])
        buses_in_traffic = np.hstack([buses_in_traffic, np.array([0.0, 0.0])])
        energy_req = np.hstack([energy_req, np.array([0.0, 0.0])])
    if times[0] > 0:
        times = np.hstack([np.array([0, times[0]-0.01], times)])
        buses_in_traffic = np.hstack([np.array([0.0, 0.0]), buses_in_traffic])
        energy_req = np.hstack([np.array([0.0, 0.0]), energy_req])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['buses on route',
                                        'Total energy required on active routes'])

    cols = px_colors.DEFAULT_PLOTLY_COLORS
    fig.add_trace(
        go.Scatter(x=times, y=buses_in_traffic, name='', line=dict(color=cols[0])),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=times, y=energy_req, name='', line=dict(color=cols[0])),
        row=1, col=2
    )

    fig.update_layout(
        xaxis=dict(
            tickformat="digit",
            tickmode='linear',
            tick0=0,
            dtick=6,
            title='Hour of week',
            range=[0,168]
        ),
        yaxis=dict(title='# buses'),
        xaxis2=dict(
            tickformat="digit",
            tickmode='linear',
            tick0=0,
            dtick=6,
            title='Hour of week',
            range=[0, 168]
        ),
        yaxis2=dict(title='Energy (kWh)'),
        showlegend=False
    )

    return dcc.Graph(id="bus-count-graph", figure=fig)


def show_bus_options_form(advanced_options):
    return html.Div(
        id="bus-information-form", children=create_bus_options(advanced_options)
    )


@callback(
    [
     Output("results-bus-number", "children"),
     ServersideOutput("bus-store", "data"),
     ServersideOutput("ec-store", "data"),
     ServersideOutput("route-summary-store", "data")],
    [Input("confirm-bus-options", "n_clicks")],
    [State("max-passenger-count", "value"),
     State("battery-capacity-kwh", "value"), State("charging-capacity-kw", "value"),
     State("gross-mass-kg", "value"), State("charging-efficiency", "value"),
     State("eol-capacity", "value"), State("gtfs-selector", "value"),
     State("route-selector", "value"), State("peak-passengers", "value"),
     State("deadhead", "value"), State("agency-selector", "value")],
    prevent_initial_callbacks=True
)
def predict_energy_usage(n_clicks, max_passengers, bat_capacity, charging_power,
                       gross_mass, charging_eff, eol_capacity, gtfs_name, routes_sel,
                       peak_passengers, deadhead_percent, agency_name):
    if n_clicks:
        bus_dict = {"max_passengers": max_passengers, "bat_capacity": bat_capacity,
                    "charging_power": charging_power, "gross_mass": gross_mass,
                    "charging_eff": charging_eff, "eol_capacity": eol_capacity}
        bus = AppData.get_bus(bus_dict)
        trips_data = AppData.read_gtfs_file(gtfs_name)
        subset_trip_data = AppData.subset_data(routes_sel, trips_data, agency_name)
        subset_trip_data = AppData.set_passenger_loading(subset_trip_data, peak_passengers)
        ec_km, ec_total = AppData.predict_energy_consumption(bus, subset_trip_data)

        route_summaries = summarise_results(subset_trip_data, ec_km, ec_total)

        times, buses_in_traffic, depart_trip_energy_req, return_trip_energy_cons = route.calc_buses_in_traffic(
            subset_trip_data,
            deadhead_percent / 100,
            RESOLUTION,
            ec_total)
        ec_dict = {"depart_trip_energy_req": depart_trip_energy_req,
                   "return_trip_energy_cons": return_trip_energy_cons,
                   "buses_in_traffic": buses_in_traffic,
                   "times": times}

        ## calculate energy requirement over duration of trips
        route_energy_usage = np.cumsum(depart_trip_energy_req) - np.cumsum(return_trip_energy_cons)

        window_options = create_window_options(route_summaries['hour window'].unique().tolist())

        init_results = {"gtfs_name":gtfs_name,
                        "num_routes":len(route_summaries['route_short_name'].unique()),
                        "num_buses":int(buses_in_traffic.max()),
                        "max_energy":ec_total.max(),
                        "av_energy":ec_total.mean(),
                        "bus_eol_capacity":bus.usable_capacity}

        return (
                [dbc.Container([dbc.Row(display_init_summary(init_results)),
                                dbc.Row(create_buses_in_traffic_plots(times, buses_in_traffic, route_energy_usage))]),
                 html.Center(
                     html.Div([
                         html.P("Time window for map results:    ",
                                style={"display": "inline-block", "padding": 10}),
                         # dcc.Dropdown(window_options,
                         #              id='window-selector',
                         #              placeholder='Select time window',
                         #              style={'display':'inline-block','width':200}),
                         dbp.Select(items=window_options,
                                    label=window_options[0]["label"],
                                    value=window_options[0]["value"],
                                    id="window-selector"),
                         html.Button('Download CSV', id="btn-ec-results")]
                     ), style={"height": 100}),
                 ],
                bus_dict, ec_dict, route_summaries.to_dict(orient='index'))
    else:
        return (None, None, None, None)

def create_window_options(hour_windows):
    sorted_windows =sorted(hour_windows, key=lambda x: x[0])
    window_strings = ["{one:.1f} - {two:.1f}".format(one=x[0], two=x[1]) for x in sorted_windows]
    return [{"value": item, "label": item} for item in window_strings]

@callback(
    Output("results-route-map", "children"),
    Input("window-selector", "value"),
    [State("route-summary-store", "data"), State("gtfs-selector", "value")],
    prevent_initial_callback=True,
)
def show_energy_usage_map(window, route_summary_dict, gtfs_name):
    if window is not None:
        df = pd.DataFrame.from_dict(route_summary_dict, orient='index')
        map_title = "Energy consumption of routes between " + window
        map_html = create_routes_map_figure(gtfs_name, map_title, df, window=window)

        return html.Center(html.Div(
                                        children=html.Iframe(id='map', srcDoc=map_html, width="90%", height="850vh")
                                    ))

@callback(
   Output("download-dataframe-csv", "data"),
    Input("btn-ec-results", "n_clicks"),
    [State("route-summary-store", "data")],
    prevent_initial_callback=True
)
def download_ec_results(n_clicks, route_summary_dict):
    if n_clicks:
        df = pd.DataFrame.from_dict(route_summary_dict, orient='index')
        df = df[['route_short_name','hour window', 'route_id', 'direction_id']+list(df.columns)[4:-1]]
        return dcc.send_data_frame(df.to_csv, "route_energy_usage_summary.csv")

@callback(
    Output("feas-optim-options-form", "children"),
    # [Input("confirm-bus-options", "n_clicks")],
    Input("window-selector", "value"),
    [State("advanced-options-checkbox", "checked"), State("ec-store", "data")],
    prevent_initial_callbacks=True
)
def show_init_optim_options_form(n_clicks, advanced_options, ec_dict):
    if n_clicks:
        return [
            html.Div(
                id="depot-information-form", children=create_depot_options(advanced_options, ec_dict)
            ),
            html.Div(
                id="feas-optim-form", children=create_feas_optim_options()
            )
        ]


@callback(
    Output("results-init-feas", "children"),
    [Input("confirm-optim-options", "n_clicks")],
    [State("charger-power", "value"), State("depot-battery-capacity", "value"),
     State("depot-battery-power", "value"), State("depot-battery-eff", "value"),
     State("min-charge-time", "value"), State("start-charge", "value"),
     State("final-charge", "value"), State("reserve-capacity", "value"),
     State("deadhead", "value"), State("bus-store", "data"),
     State("ec-store", "data"), State("peak-passengers", "value"),
     State("num-buses-slider", "value")],
    prevent_initial_callbacks=True
)
def run_initial_feasibility(n_clicks, charger_power, depot_bat_cap, depot_bat_power,
                            depot_bat_eff, min_charge_time, start_charge, final_charge,
                            reserve_capacity, deadhead_percent, bus_dict, ec_dict,
                            peak_passengers, num_buses):
    if n_clicks:
        battery = AppData.battery_dict(depot_bat_cap, depot_bat_power, depot_bat_eff)
        chargers = {"power": charger_power, "number": "optim", "cost": 10}
        options_dict = {"deadhead": deadhead_percent / 100,
                        "min_charge_time": min_charge_time,
                        "start_charge": start_charge / 100,
                        "final_charge": final_charge / 100,
                        "reserve_capacity": reserve_capacity / 100,
                        "chargers": chargers,
                        "battery": battery,
                        "bus_dict": bus_dict}
        results = run_init_feasibility(options_dict, ec_dict, num_buses)
        results['peak_passengers'] = peak_passengers
        out = dbc.Container(dbc.Row([
            dbc.Col(display_optim_summary(results), width={"size":3, "offset":1}),
            dbc.Col(create_optim_results_plot(results), width=6),
        ]),fluid=True)
        return out
        # return html.Center(create_optim_results_plot(results))
    else:
        return None



