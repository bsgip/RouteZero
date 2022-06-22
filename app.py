from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, \
    ServersideOutputTransform
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Container import Container

app = DashProxy(__name__, use_pages=True, suppress_callback_exceptions=True, transforms=[ServersideOutputTransform()])

app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


navbar = dbc.NavbarSimple(children=[
    html.Img(src='./assets/Zenobe-logo-Glow.png', height='30px'),
    html.H1('RouteZero')
] + [dbc.NavItem(dbc.NavLink(page['name'], href=page['relative_path'])) for page in dash.page_registry.values()])


app.layout = html.Div(children=[
    html.Div(
        children=[
            navbar
        ],
        className='header'
    ),
    html.Div(
        dash.page_container
    )
])



if __name__ == '__main__':
	app.run_server(debug=True)