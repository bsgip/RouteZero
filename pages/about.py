import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__)

layout = html.Div(children=[
    dcc.Markdown("""
    ## About
    Documentation to come
    """)
    # dbc.Col(html.H1(children='About RouteZero'), align='center'),
    # html.H1(children='About RouteZero'),
    #
    # html.Div(children='''
    #     About contents.
    # '''),

], style={'margin-left': "18rem"})
