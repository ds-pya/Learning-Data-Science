import streamlit as st
import pandas as pd
import numpy as np
import ast
from copy import deepcopy

st.subheader("분포 설정 테이블 (편집 가능, 콤팩트 뷰)")
csv = st.file_uploader("index=topic, columns=source, cell=dict 형태 CSV 업로드", type=["csv"])

# ---------- helpers ----------
DIST_OPTIONS = ["LN","ZILN","P","ZIP","NB","ZINB"]

EMPTY_CELL = {"Cover": False, "Distribution": "LN", "Parameter": {"median":30, "p95":150}, "FallbackWeights": 0.0}

def parse_cell(x):
    if pd.isna(x): return deepcopy(EMPTY_CELL)
    if isinstance(x, dict): return deepcopy(x)
    try:
        d = ast.literal_eval(str(x))
        # 최소 키 보정
        if "Cover" not in d: d["Cover"] = False
        if "Distribution" not in d: d["Distribution"] = "LN"
        if "Parameter" not in d: d["Parameter"] = {}
        if "FallbackWeights" not in d: d["FallbackWeights"] = 0.0
        if not isinstance(d["Parameter"], dict): d["Parameter"] = {}
        return d
    except Exception:
        return deepcopy(EMPTY_CELL)

def cell_bg_hex(cover: bool) -> str:
    return "#e8f7ee" if cover else "#f4f4f5"

def get_param_defaults(dist: str, current: dict) -> dict:
    cur = dict(current or {})
    if dist == "LN":
        cur.setdefault("median", 30.0)
        # p90 또는 p95 중 하나만 있으면 유지
        if "p90" in cur and "p95" not in cur:
            cur.setdefault("p90", cur["p90"])
        else:
            cur.setdefault("p95", 150.0)
    elif dist == "ZILN":
        cur.setdefault("pi0", 0.6)
        cur.setdefault("median", 30.0)
        cur.setdefault("p95", 150.0)
    elif dist == "P":
        cur.setdefault("lambda", 3.0)
    elif dist == "ZIP":
        cur.setdefault("lambda", 1.0)
        cur.setdefault("pi0", 0.6)
    elif dist == "NB":
        cur.setdefault("mean", 1.0)
        cur.setdefault("var", 1.8)
    elif dist == "ZINB":
        cur.setdefault("mean", 0.6)
        cur.setdefault("var", 1.6)
        cur.setdefault("pi0", 0.6)
    return cur

# ---------- UI ----------
if csv is None:
    st.info("CSV를 업로드하면 편집 가능한 테이블이 표시됩니다.")
    st.stop()

raw = pd.read_csv(csv, index_col=0)
parsed = raw.applymap(parse_cell)

topics = list(parsed.index)
sources = list(parsed.columns)

# 콤팩트 스타일 + 폰트 크기
font_px = st.slider("폰트 크기(px)", 9, 14, 11, 1)
st.markdown(f"""
<style>
.small * {{font-size:{font_px}px !important;}}
.gridcell {{
  border:1px solid #e5e7eb; border-radius:8px; padding:6px 8px;
  background:#fbfbfc; 
}}
.gridwrap {{ display:block; }}
.gridtitle {{ font-weight:600; white-space:nowrap; }}
.compact .stSlider, .compact .stNumberInput, .compact .stSelectbox, .compact .stCheckbox {{
  padding:0 !important; margin:0 !important;
}}
th, td {{ padding: 0 !important; }}
</style>
""", unsafe_allow_html=True)

with st.expander("🔧 분포/파라미터 설정 (펼치면 표시)", expanded=True):
    # 헤더
    header = st.columns([1] + [3]*len(sources), gap="small")
    header[0].markdown("<div class='small gridtitle'>Topic \\ Source</div>", unsafe_allow_html=True)
    for j, src in enumerate(sources, start=1):
        header[j].markdown(f"<div class='small gridtitle'>{src}</div>", unsafe_allow_html=True)

    # ROWS
    updated = parsed.copy()
    for t in topics:
        row = st.columns([1] + [3]*len(sources), gap="small")
        row[0].markdown(f"<div class='small gridtitle'>{t}</div>", unsafe_allow_html=True)

        for j, src in enumerate(sources, start=1):
            cell_dict = deepcopy(parsed.loc[t, src])
            base_key = f"{t}|{src}"
            with row[j]:
                # 셀 배경색: Cover
                bg = cell_bg_hex(bool(cell_dict.get("Cover", False)))
                st.markdown(f"<div class='gridcell small compact' style='background:{bg}'>", unsafe_allow_html=True)

                # Cover 토글
                cov = st.checkbox("Cover", value=bool(cell_dict.get("Cover", False)),
                                  key=f"{base_key}::cover", help="이 소스가 해당 토픽을 직접 커버하는지")

                # 분포 선택
                dist = st.selectbox("분포", DIST_OPTIONS, index=DIST_OPTIONS.index(cell_dict.get("Distribution","LN")),
                                    key=f"{base_key}::dist", label_visibility="collapsed")

                # 파라미터 입력 (콤팩트)
                params = get_param_defaults(dist, cell_dict.get("Parameter", {}))
                if dist == "LN":
                    c1,c2 = st.columns(2, gap="small")
                    with c1: params["median"] = st.number_input("median", 0.0, value=float(params["median"]), step=5.0, key=f"{base_key}::ln_m")
                    # p90/p95 중 CSV에 존재하는 키 유지
                    if "p90" in cell_dict.get("Parameter", {}):
                        with c2: params["p90"] = st.number_input("p90", 0.0, value=float(params.get("p90", 150.0)), step=10.0, key=f"{base_key}::ln_p90")
                        params.pop("p95", None)
                    else:
                        with c2: params["p95"] = st.number_input("p95", 0.0, value=float(params.get("p95", 150.0)), step=10.0, key=f"{base_key}::ln_p95")
                        params.pop("p90", None)

                elif dist == "ZILN":
                    c1,c2,c3 = st.columns(3, gap="small")
                    with c1: params["pi0"] = st.slider("pi0", 0.0,1.0, float(params["pi0"]), 0.05, key=f"{base_key}::ziln_pi0")
                    with c2: params["median"] = st.number_input("median", 0.0, value=float(params["median"]), step=5.0, key=f"{base_key}::ziln_m")
                    with c3: params["p95"] = st.number_input("p95", 0.0, value=float(params["p95"]), step=10.0, key=f"{base_key}::ziln_p95")

                elif dist == "P":
                    params["lambda"] = st.number_input("lambda", min_value=1e-6, value=float(params["lambda"]), step=0.5, key=f"{base_key}::p_lam")

                elif dist == "ZIP":
                    c1,c2 = st.columns(2, gap="small")
                    with c1: params["lambda"] = st.number_input("lambda", min_value=1e-6, value=float(params["lambda"]), step=0.2, key=f"{base_key}::zip_lam")
                    with c2: params["pi0"] = st.slider("pi0", 0.0,1.0, float(params["pi0"]), 0.05, key=f"{base_key}::zip_pi0")

                elif dist == "NB":
                    c1,c2 = st.columns(2, gap="small")
                    with c1: params["mean"] = st.number_input("mean", 0.0, value=float(params["mean"]), step=0.1, key=f"{base_key}::nb_mean")
                    with c2: params["var"]  = st.number_input("var",  min_value=1e-6, value=float(params["var"]), step=0.1, key=f"{base_key}::nb_var")

                elif dist == "ZINB":
                    c1,c2,c3 = st.columns(3, gap="small")
                    with c1: params["mean"] = st.number_input("mean", 0.0, value=float(params["mean"]), step=0.1, key=f"{base_key}::zinb_mean")
                    with c2: params["var"]  = st.number_input("var",  min_value=1e-6, value=float(params["var"]), step=0.1, key=f"{base_key}::zinb_var")
                    with c3: params["pi0"]  = st.slider("pi0", 0.0,1.0, float(params["pi0"]), 0.05, key=f"{base_key}::zinb_pi0")

                st.markdown("</div>", unsafe_allow_html=True)

                # 업데이트 반영
                new_cell = {
                    "Cover": bool(cov),
                    "Distribution": dist,
                    "Parameter": params,
                    "FallbackWeights": cell_dict.get("FallbackWeights", 0.0)
                }
                updated.loc[t, src] = new_cell

# ---------- 저장/다운로드 ----------
# dict를 문자열로 직렬화하여 CSV 저장
save_df = updated.applymap(lambda d: str(d))
st.download_button(
    "📥 편집 결과 CSV 다운로드",
    data=save_df.to_csv(index=True).encode("utf-8"),
    file_name="distribution_editable_table.csv",
    mime="text/csv"
)

# 미리보기 (콤팩트 텍스트)
st.caption("미리보기 (일부):")
st.dataframe(save_df.head(10))