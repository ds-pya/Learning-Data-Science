import plotly.graph_objs as go

def make_stacked_bar(keyword_list, score_list, source_score_list, marker):
    import plotly.express as px

    # 정의된 source → 색상 매핑
    source_to_color = {
        "naver": "#d62728",
        "google": "#1f77b4",
        "bing": "#2ca02c",
        "youtube": "#9467bd",
        "kakao": "#ff7f0e"
    }

    all_sources = list(source_to_color.keys())

    traces = []

    for source in all_sources:
        x, y, hovertexts = [], [], []

        for i, keyword in enumerate(keyword_list):
            score = source_score_list[i].get(source, 0.0)
            if score > 0:
                x.append(score)
                y.append(keyword)
                hovertexts.append(
                    f"<b>{keyword}</b><br>Source: {source}<br>Score: {round(score, 3)}"
                )

        traces.append(go.Bar(
            x=x,
            y=y,
            orientation="h",
            name=source,
            marker=dict(
                color=[source_to_color[source]] * len(x),
                line=dict(width=0)
            ),
            hoverinfo="text",
            hovertext=hovertexts,
            showlegend=False
        ))

    fig = go.Figure(traces)

    # 총합 점수 텍스트 추가
    for keyword, total in zip(keyword_list, score_list):
        if total > 0:
            fig.add_trace(go.Scatter(
                x=[total],
                y=[keyword],
                mode="text",
                text=[f"{round(total, 3)}"],
                textposition="middle right",
                showlegend=False,
                hoverinfo="skip",
                textfont=dict(color="black")
            ))

    # y축 텍스트 색상 설정
    tick_colors = [marker.get(k, "black") for k in keyword_list]

    fig.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(
            tickfont=dict(color=tick_colors)
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        margin=dict(l=10, r=80, t=40, b=10),
        height=40 * len(keyword_list) + 200
    )

    return fig