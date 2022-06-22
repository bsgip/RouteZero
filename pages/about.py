import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(children=[
    html.H1(children='About RouteZero'),

    html.Div(children='''
        About contents.
    '''),

])