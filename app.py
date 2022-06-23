from flask import Flask
from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, \
    ServersideOutputTransform
import dash
import dash_bootstrap_components as dbc

# backend = RedisStore()
# backend = FileSystemStore(threshold=100)

app = DashProxy(__name__, use_pages=True, suppress_callback_exceptions=True, transforms=[ServersideOutputTransform()],
                external_stylesheets=[dbc.themes.BOOTSTRAP])

app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


navbar = dbc.Navbar(dbc.Container([
    dbc.Row(
      [
        dbc.Col(html.Img(src='./assets/Zenobe-logo-Glow.png', height="35px"), style={"padding":5}),
        dbc.Col(html.Img(src='./assets/BSGIP-logo.png', height="45px"), style={"padding":5}),
      ],
      align="center",
      className="g-0",
    ),
    dbc.Row(dbc.Col([html.H1('RouteZero (alpha)'),"ebus energy usage model / depot charging optimiser"])),
    dbc.Row(
        [dbc.Col(dbc.NavItem(dbc.NavLink(page['name'], href=page['relative_path']))) for page in dash.page_registry.values()],
        align="center",
        className="g-0",
    )
]
))


app.layout = html.Div(children=[
    html.Div(
        children=[
            navbar
        ],
    ),
    html.Div(
        dash.page_container
    )
])


if __name__ == '__main__':
	app.run_server(debug=True)