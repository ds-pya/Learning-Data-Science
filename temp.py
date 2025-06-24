import plotly.graph_objs as go
import plotly.express as px

def make_stacked_bar_custom(keyword_list, score_list, source_score_list, marker):
    # 모든 source 추출
    all_sources = set()
    for d in source_score_list:
        all_sources.update(d.keys())
    all_sources = sorted(all_sources)

    # 선 색상 매핑
    border_palette = px.colors.qualitative.Set1
    source_to_border = {
        source: border_palette[i % len(border_palette)]
        for i, source in enumerate(all_sources)
    }

    traces = []

    for source in all_sources:
        x, y, colors, line_colors, hovertexts = [], [], [], [], []

        for i, keyword in enumerate(keyword_list):
            source_score = source_score_list[i].get(source, 0.0)
            if source_score > 0:
                x.append(source_score)
                y.append(keyword)
                colors.append(marker.get(keyword, "#888888"))
                line_colors.append(source_to_border[source])
                hovertexts.append(f"<b>{keyword}</b><br>Source: {source}<br>Score: {source_score:.2f}")

        traces.append(go.Bar(
            x=x,
            y=y,
            orientation='h',
            name=source,
            marker=dict(
                color=colors,
                line=dict(color=line_colors, width=2)
            ),
            hoverinfo="text",
            hovertext=hovertexts,
            showlegend=True
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        barmode="stack",
        title="Stacked Horizontal Bar Chart",
        paper_bgcolor="#1f2c3d",
        plot_bgcolor="#1f2c3d",
        font_color="white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(title="Source", traceorder="normal", orientation="v", x=1.02, y=1)
    )
    return fig