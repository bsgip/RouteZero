import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__)

layout = html.Div(children=[
    dcc.Markdown("""
    ## About
    Documentation to come
    """)
], style={'margin-left': "18rem"})
