import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

dash.register_page(__name__, path="/dashboard", name="Dashboard")

layout = html.Div([
    dbc.Row([
        dbc.Col(html.Div([
            html.H2("2500", style={"color": "white"}),
            html.P("▲ 4% From last Week", style={"color": "lightgreen"})
        ], className="text-center"), width=3),

        dbc.Col(html.Div([
            html.H2("123.50", style={"color": "white"}),
            html.P("▲ 3% From last Week", style={"color": "lightgreen"})
        ], className="text-center"), width=3),

        dbc.Col(html.Div([
            html.H2("2,500", style={"color": "white"}),
            html.P("▲ 34% From last Week", style={"color": "lightgreen"})
        ], className="text-center"), width=3),

        dbc.Col(html.Div([
            html.H2("4,567", style={"color": "white"}),
            html.P("▼ 12% From last Week", style={"color": "salmon"})
        ], className="text-center"), width=3),
    ], className="mb-4"),

    html.H4("Network Activities", style={"marginTop": "30px", "color": "white"}),
    html.P("Graph title sub-title", style={"color": "lightgray"}),

    dcc.Graph(
        figure=go.Figure([
            go.Scatter(x=["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 06"], y=[20, 60, 30, 50, 80], fill='tozeroy', name='Series A'),
            go.Scatter(x=["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 06"], y=[80, 30, 70, 120, 60], fill='tozeroy', name='Series B'),
        ]).update_layout(
            paper_bgcolor="#1f2c3d",
            plot_bgcolor="#1f2c3d",
            font_color="white"
        )
    )
])