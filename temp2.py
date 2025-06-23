import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_extensions.enrich as de  # For better performance (optional)

# Dark theme (Morph Dark)
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.MORPH],
    suppress_callback_exceptions=True
)

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        # Sidebar
        dbc.Col(width=2, style={"backgroundColor": "#1f2c3d", "minHeight": "100vh", "color": "white", "padding": "20px"}, children=[
            html.H5("GENERAL", style={"fontWeight": "bold", "marginBottom": "30px"}),
            dcc.Link("🏠 Home", href="/", style={"display": "block", "padding": "10px", "color": "white", "textDecoration": "none", "marginBottom": "5px", "backgroundColor": "#1abc9c", "borderRadius": "5px"}),
            dcc.Link("• Dashboard", href="/dashboard", style={"display": "block", "padding": "10px", "color": "white", "textDecoration": "none"}),
            dcc.Link("• Dashboard2", href="/dashboard2", style={"display": "block", "padding": "10px", "color": "white", "textDecoration": "none"}),
            dcc.Link("• Dashboard3", href="/dashboard3", style={"display": "block", "padding": "10px", "color": "white", "textDecoration": "none"}),
        ]),

        # Main content
        dbc.Col(width=10, children=[
            html.Div(dash.page_container, style={"padding": "40px"})
        ])
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=1234)