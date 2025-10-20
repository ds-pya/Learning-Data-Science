import io
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def fig_to_png_bytes():
    # 원하는 라이브러리(Plotly/Matplotlib)로 그린 뒤 PNG 바이트로 변환
    fig, ax = plt.subplots(figsize=(2.2, 1.2), dpi=200)  # 미니 크기
    ax.plot([1,2,3,2,5])
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def compose_chart_with_stats(chart_png: bytes, stats: dict) -> bytes:
    chart = Image.open(io.BytesIO(chart_png)).convert("RGBA")
    # 텍스트 영역 폭/높이 대략 잡기
    Wc, Hc = chart.size
    text_lines = [f"{k}: {v}" for k, v in stats.items()]
    # 텍스트 높이 대충 계산(고정 폭 폰트 가정)
    line_h, pad = 14, 6
    Ht = pad + len(text_lines)*line_h + pad
    canvas = Image.new("RGBA", (max(Wc, 320), Hc+Ht), "WHITE")

    # 붙이기
    canvas.paste(chart, (0,0))
    draw = ImageDraw.Draw(canvas)
    y = Hc + pad
    for line in text_lines:
        draw.text((6, y), line, fill=(0,0,0))
        y += line_h

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return out.read()

rows = []
for name, series, stats in [
    ("A", [1,2,3,2,5], {"mean": 2.6, "p90": 5}),
    ("B", [3,1,4,1,5], {"mean": 2.8, "p90": 5}),
]:
    chart_png = fig_to_png_bytes()
    combo_png = compose_chart_with_stats(chart_png, stats)
    rows.append({"name": name, "card": combo_png})

df = pd.DataFrame(rows)

st.dataframe(
    df,
    column_config={
        "card": st.column_config.ImageColumn("Chart + Stats"),
    },
    hide_index=True
)