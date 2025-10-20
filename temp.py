import io, base64
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def fig_to_png_bytes(series):
    fig, ax = plt.subplots(figsize=(2.2, 1.2), dpi=200)
    ax.plot(series)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def compose_chart_with_stats(chart_png: bytes, stats: dict) -> bytes:
    chart = Image.open(io.BytesIO(chart_png)).convert("RGBA")
    Wc, Hc = chart.size
    line_h, pad = 14, 6
    lines = [f"{k}: {v}" for k, v in stats.items()]
    Ht = pad + len(lines)*line_h + pad
    W = max(Wc, 320)
    canvas = Image.new("RGBA", (W, Hc+Ht), "WHITE")
    canvas.paste(chart, (0,0))
    d = ImageDraw.Draw(canvas)
    y = Hc + pad
    for t in lines:
        d.text((6, y), t, fill=(0,0,0))
        y += line_h
    out = io.BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return out.read()

def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

rows = []
for name, series, stats in [
    ("A", [1,2,3,2,5], {"mean": 2.6, "p90": 5}),
    ("B", [3,1,4,1,5], {"mean": 2.8, "p90": 5}),
]:
    chart_png = fig_to_png_bytes(series)
    combo_png = compose_chart_with_stats(chart_png, stats)
    rows.append({"name": name, "card": to_data_url(combo_png)})

df = pd.DataFrame(rows)

st.dataframe(
    df,
    column_config={
        "card": st.column_config.ImageColumn("Chart + Stats", width="medium")
    },
    hide_index=True,
    use_container_width=True,
)