import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "데이터 분석 대시보드"

# 기본 레이아웃
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        # 좌측 패널
        dbc.Col(width=3, children=[
            html.H4("📁 데이터 업로드", className="mt-3"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                },
                multiple=False
            ),
            html.Div(id='file-name', className='mt-3', style={'fontWeight': 'bold'}),
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

        # 우측 콘텐츠
        dbc.Col(width=9, children=[
            html.Div(id='tab-content', className='p-4')
        ]),
    ])
])

# 파일 저장용 스토리지
app.file_data = {
    'df': None,
    'filename': ''
}

# 콜백: 파일 업로드 처리
@app.callback(
    Output('file-name', 'children'),
    Output('tab-content', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('tabs', 'value'),
)
def update_output(contents, filename, current_tab):
    if contents is None:
        return "", html.Div("파일을 업로드해주세요.")

    # CSV 파일 로딩
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except:
        return f"{filename} (❌ 로드 실패)", html.Div("CSV 형식이 아닙니다.")

    app.file_data['df'] = df
    app.file_data['filename'] = filename

    return f"📄 현재 파일: {filename}", render_tab_content(current_tab, df)


# 콜백: 탭 변경 시 콘텐츠 변경
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def update_tab_content(tab_value):
    df = app.file_data.get('df')
    return render_tab_content(tab_value, df)


# 탭별 콘텐츠 생성 함수
def render_tab_content(tab, df):
    if df is None:
        return html.Div("먼저 파일을 업로드해주세요.", className="text-muted")

    if tab == 'tab-home':
        return render_home_tab(df)
    elif tab == 'tab-keyword':
        return html.Div("🔑 Keyword 분석 그래프 영역 (추후 구현)")
    elif tab == 'tab-score':
        return html.Div("📊 Score 분석 그래프 영역 (추후 구현)")
    return html.Div("알 수 없는 탭입니다.")


# Home 탭: 메타데이터 출력
def render_home_tab(df):
    summary = []
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            start = df['date'].min().strftime("%Y-%m-%d")
            end = df['date'].max().strftime("%Y-%m-%d")
            summary.append(f"- **기간**: {start} ~ {end}")
        except:
            summary.append("- **기간**: 날짜 형식 파싱 실패")

    summary.append(f"- **총 행 수**: {len(df):,} rows")
    summary.append(f"- **총 열 수**: {len(df.columns):,} columns")
    summary.append("- **열 목록**: " + ", ".join(df.columns))

    return dcc.Markdown("\n".join(summary))


if __name__ == '__main__':
    import base64
    app.run_server(debug=True)