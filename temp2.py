import dash
from dash import html, dcc
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div(style={"margin": "0 10%", "fontFamily": "Arial"}, children=[
    html.Div([
        html.Div([
            html.H2("2500"),
            html.P("▲ 4% From last Week", style={"color": "green"})
        ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),

        html.Div([
            html.H2("123.50"),
            html.P("▲ 3% From last Week", style={"color": "green"})
        ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),

        html.Div([
            html.H2("2,500"),
            html.P("▲ 34% From last Week", style={"color": "green"})
        ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),

        html.Div([
            html.H2("4,567"),
            html.P("▼ 12% From last Week", style={"color": "red"})
        ], style={"display": "inline-block", "width": "24%", "textAlign": "center"})
    ], style={"padding": "20px 0"}),

    html.H3("Network Activities"),
    html.P("Graph title sub-title"),

    dcc.Graph(
        figure=go.Figure([
            go.Scatter(x=["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 06"], 
                       y=[20, 60, 30, 50, 80], fill='tozeroy', name='Series 1'),
            go.Scatter(x=["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 06"], 
                       y=[80, 30, 70, 120, 60], fill='tozeroy', name='Series 2'),
        ])
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=1234)