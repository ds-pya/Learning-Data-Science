import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pages.home as home
import pages.keyword as keyword
import pages.score as score

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for deployment

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(width=3, children=[
            html.H4("📁 데이터 업로드", className="mt-3"),
            dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                       style={'height': '60px', 'border': '1px dashed gray', 'textAlign': 'center'}),
            html.Div(id='file-name', className='mt-3'),
            html.Hr(),
            dcc.Tabs(
                id='tabs', value='tab-home',
                children=[
                    dcc.Tab(label='🏠 Home', value='tab-home'),
                    dcc.Tab(label='🔑 Keyword', value='tab-keyword'),
                    dcc.Tab(label='📊 Score', value='tab-score'),
                ]
            ),
        ], style={'borderRight': '1px solid lightgray', 'height': '100vh', 'padding': '1rem'}),
        dbc.Col(width=9, children=html.Div(id='tab-content', className='p-4'))
    ])
])

# 탭별 페이지 불러오기
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_page(tab):
    if tab == 'tab-home':
        return home.layout
    elif tab == 'tab-keyword':
        return keyword.layout
    elif tab == 'tab-score':
        return score.layout
    return html.Div("알 수 없는 탭입니다.")

if __name__ == '__main__':
    app.run_server(debug=True)