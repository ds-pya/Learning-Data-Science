import dash
from dash import dcc, html
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard"

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        # Sidebar
        dbc.Col(width=2, style={"backgroundColor": "#2c3e50", "minHeight": "100vh", "color": "white", "padding": "20px"}, children=[
            html.H5("GENERAL", style={"fontWeight": "bold", "marginBottom": "30px"}),
            html.Div([
                html.Div("🏠 Home", style={"padding": "10px", "backgroundColor": "#1abc9c", "borderRadius": "5px", "marginBottom": "5px"}),
                html.Div("• Dashboard", style={"padding": "10px", "fontWeight": "bold", "marginBottom": "5px"}),
                html.Div("• Dashboard2", style={"padding": "10px", "marginBottom": "5px"}),
                html.Div("• Dashboard3", style={"padding": "10px", "marginBottom": "5px"}),
            ])
        ]),

        # Main Content
        dbc.Col(width=10, children=[
            html.Div(style={"padding": "40px"}, children=[

                # KPI Cards
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H2("2500"),
                        html.P("▲ 4% From last Week", style={"color": "green"})
                    ], className="text-center"), width=3),

                    dbc.Col(html.Div([
                        html.H2("123.50"),
                        html.P("▲ 3% From last Week", style={"color": "green"})
                    ], className="text-center"), width=3),

                    dbc.Col(html.Div([
                        html.H2("2,500"),
                        html.P("▲ 34% From last Week", style={"color": "green"})
                    ], className="text-center"), width=3),

                    dbc.Col(html.Div([
                        html.H2("4,567"),
                        html.P("▼ 12% From last Week", style={"color": "red"})
                    ], className="text-center"), width=3),
                ], className="mb-4"),

                # Graph Title
                html.H4("Network Activities", style={"marginTop": "30px"}),
                html.P("Graph title sub-title", style={"marginBottom": "20px"}),

                # Graph
                dcc.Graph(
                    figure=go.Figure([
                        go.Scatter(
                            x=["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 06"],
                            y=[20, 60, 30, 50, 80],
                            fill='tozeroy', mode='lines', name='Series A'
                        ),
                        go.Scatter(
                            x=["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 06"],
                            y=[80, 30, 70, 120, 60],
                            fill='tozeroy', mode='lines', name='Series B'
                        ),
                    ]).update_layout(
                        margin={"l": 10, "r": 10, "t": 10, "b": 10},
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        xaxis_title="",
                        yaxis_title="",
                        height=400
                    )
                )
            ])
        ])
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=1234)