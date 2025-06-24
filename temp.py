# utils/plot_generator.py
import pandas as pd
import plotly.graph_objs as go

def make_bar_plot(records):
    df = pd.DataFrame(records)

    # 예시: 계층적 인덱스 2레벨을 "index_1", "index_2" 컬럼으로 가정
    required_cols = {"index_1", "index_2", "value"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Required columns: {required_cols}")

    # 고정된 인덱스 셋 (순서를 유지하고 누락된 항목은 0으로)
    all_index = pd.MultiIndex.from_tuples([
        ('A', 'a1'), ('A', 'a2'),
        ('B', 'b1'), ('B', 'b2'),
        ('C', 'c1')
    ], names=["index_1", "index_2"])

    df = df.set_index(["index_1", "index_2"]).reindex(all_index, fill_value=0)
    df = df.reset_index()

    # Plotly sunburst로 folding 계층적 바 그래프 흉내 가능 (가로 바 지원 X)
    fig = go.Figure()

    for level1 in df['index_1'].unique():
        sub_df = df[df['index_1'] == level1]
        fig.add_trace(go.Bar(
            y=[f"{level1} / {lvl2}" for lvl2 in sub_df["index_2"]],
            x=sub_df["value"],
            name=level1,
            orientation="h",
            showlegend=False
        ))

    fig.update_layout(
        title="📊 Hierarchical Horizontal Bar Plot",
        barmode="stack",
        paper_bgcolor="#1f2c3d",
        plot_bgcolor="#1f2c3d",
        font_color="white",
        height=400 + 20 * len(df)
    )
    return fig