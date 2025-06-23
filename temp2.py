import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# 기본 레이아웃
app.layout = dbc.Container(fluid=True, children=[
    # 중앙 탭 영역
    dbc.Row([
        dbc.Col([
            dcc.Tabs(
                id='tabs', value='tab-keyword',
                children=[
                    dcc.Tab(label='🔑 Keyword', value='tab-keyword', className="custom-tab", selected_className="custom-tab-selected"),
                    dcc.Tab(label='📊 Score', value='tab-score', className="custom-tab", selected_className="custom-tab-selected"),
                ],
                className="justify-content-center mt-3"
            )
        ], width=12, className="d-flex justify-content-center")
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Div(id='tab-content', className="p-4")
        ], width=10, className="mx-auto")  # 가운데 정렬 + 여백
    ])
])

# 탭별 콘텐츠 처리
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    return render_keyword_tab() if tab == 'tab-keyword' else render_score_tab()


# Keyword 탭
def render_keyword_tab():
    return html.Div([
        html.H4("🔑 Keyword 분석", className="mb-4"),
        dcc.Upload(
            id='upload-keyword',
            children=html.Div(['👉 ', html.A('여기에 CSV 파일을 업로드 하세요')]),
            style={
                'width': '100%', 'height': '100px', 'lineHeight': '100px',
                'borderWidth': '2px', 'borderStyle': 'dashed',
                'borderRadius': '10px', 'textAlign': 'center',
                'backgroundColor': '#f9f9f9'
            },
            multiple=False
        ),
        html.Div(id='keyword-output', className="mt-4")
    ])


# Score 탭
def render_score_tab():
    return html.Div([
        html.H4("📊 Score 분석", className="mb-4"),
        dcc.Upload(
            id='upload-score',
            children=html.Div(['👉 ', html.A('여기에 CSV 파일을 업로드 하세요')]),
            style={
                'width': '100%', 'height': '100px', 'lineHeight': '100px',
                'borderWidth': '2px', 'borderStyle': 'dashed',
                'borderRadius': '10px', 'textAlign': 'center',
                'backgroundColor': '#f9f9f9'
            },
            multiple=False
        ),
        html.Div(id='score-output', className="mt-4")
    ])


# 업로드 처리: Keyword
@app.callback(
    Output('keyword-output', 'children'),
    Input('upload-keyword', 'contents'),
    State('upload-keyword', 'filename')
)
def handle_keyword_upload(contents, filename):
    if contents is None:
        return ""
    df = parse_contents(contents)
    return html.Div([
        html.H6(f"파일명: {filename}"),
        html.P(f"행 수: {len(df):,} / 열 수: {len(df.columns)}"),
        html.P("컬럼명: " + ", ".join(df.columns))
    ])


# 업로드 처리: Score
@app.callback(
    Output('score-output', 'children'),
    Input('upload-score', 'contents'),
    State('upload-score', 'filename')
)
def handle_score_upload(contents, filename):
    if contents is None:
        return ""
    df = parse_contents(contents)
    return html.Div([
        html.H6(f"파일명: {filename}"),
        html.P(f"행 수: {len(df):,} / 열 수: {len(df.columns)}"),
        html.P("컬럼명: " + ", ".join(df.columns))
    ])


# 파일 파싱 함수
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))


# CSS 스타일 추가
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>대시보드</title>
        {%favicon%}
        {%css%}
        <style>
            .custom-tab {
                font-size: 18px;
                padding: 10px 20px;
            }
            .custom-tab-selected {
                border-bottom: 3px solid #0d6efd !important;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=1234)