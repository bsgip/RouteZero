import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__)

def create_about_field():
    text = """
    RouteZero was developed by the Battery Storage and Grid Integration Program at the ANU.  

    The tool’s development was funded by the Australian Renewable Energy Agency as part of the Next Generation Electric Bus 
    Depot project https://arena.gov.au/projects/next-generation-electric-bus-depot/ led by Zenobe and Transgrid. 
    This project is the first large scale deployment of electric buses in Australia, with 40 electric buses being rolled 
    out into the Transit Systems fleet. These buses operate out of the Leichardt depot in Sydney, NSW. 

    RouteZero incorporates two models. The first is a data-driven model of electricity usage for a given electric buses 
    on a particular route under particular traffic and weather conditions. This model is refined for Australian conditions 
    based on performance data of the electric buses in Leichardt. 

    The second model optimises the charging of the bus fleet to meet the operational demands of the timetabled routes 
    while minimising the peak demand that the depot places on the electricity network. This model functions with the 
    aggregated pool of buses, as we do not have scheduling information of which bus is driving on which route at which time. 
    """

    return dbc.Container(dbc.Row([
        dbc.Col(dcc.Markdown(text), width={"size":8, "offset":2})
    ]), style={"margin-top":"7rem"})

layout = html.Div(children=[
    create_about_field()
])
