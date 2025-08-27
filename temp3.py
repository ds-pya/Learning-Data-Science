import streamlit as st
import pandas as pd
import numpy as np
import ast

st.subheader("분포 설정 테이블 (콤팩트 뷰)")
csv = st.file_uploader("index=topic, columns=source, cell=dict 형태 CSV 업로드", type=["csv"])

def parse_cell(x):
    if pd.isna(x): 
        return {"Cover": False, "Distribution": None, "Parameter": {}, "FallbackWeights": 0.0}
    if isinstance(x, dict): 
        return x
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return {"Cover": False, "Distribution": None, "Parameter": {}, "FallbackWeights": 0.0}

def format_cell(d):
    cover = d.get("Cover", False)
    dist  = d.get("Distribution", "")
    params = d.get("Parameter", {}) or {}
    # 아주 콤팩트한 표시 (토픽명이 길어 한 줄 유지용)
    parts = []
    if dist: parts.append(dist)
    if "median" in params: parts.append(f"m={params['median']}")
    if "p95" in params:   parts.append(f"p95={params['p95']}")
    if "p90" in params:   parts.append(f"p90={params['p90']}")
    if "lambda" in params:parts.append(f"λ={params['lambda']}")
    if "mean" in params:  parts.append(f"μ={params['mean']}")
    if "var" in params:   parts.append(f"var={params['var']}")
    if "pi0" in params:   parts.append(f"π0={params['pi0']}")
    text = " ".join(parts) if parts else "-"
    # 커버 여부 아이콘
    icon = "✓" if cover else "✗"
    return f"{icon} {text}"

if csv is not None:
    raw = pd.read_csv(csv, index_col=0)
    parsed = raw.applymap(parse_cell)
    display = parsed.applymap(format_cell)

    # 폰트 크기 조절 (px)
    font_px = st.slider("폰트 크기(px)", min_value=9, max_value=14, value=11, step=1)

    # 스타일링: Cover=True 초록톤 / False 회색톤, 한 줄 표시(줄바꿈 방지)
    def bg_style(df_parsed):
        # same shape DataFrame of css strings
        css = pd.DataFrame("", index=df_parsed.index, columns=df_parsed.columns)
        for r in df_parsed.index:
            for c in df_parsed.columns:
                d = df_parsed.loc[r, c]
                cov = bool(d.get("Cover", False))
                color = "#e8f7ee" if cov else "#f4f4f5"
                css.loc[r, c] = f"background-color: {color};"
        return css

    styler = (display.style
              .apply(lambda _: bg_style(parsed), axis=None)
              .set_table_styles([
                  {"selector":"table", "props":[("border-collapse","separate"),("border-spacing","0 6px")]},
                  {"selector":"th, td", "props":[("font-size", f"{font_px}px"),("white-space","nowrap"),("padding","6px 8px")]},
              ])
              .set_properties(**{
                  "border":"1px solid #e5e7eb",
                  "border-radius":"6px",
              }))

    st.write(styler)
else:
    st.info("CSV를 업로드하면 테이블이 표시됩니다.")