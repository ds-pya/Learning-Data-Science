import streamlit as st
import pandas as pd

# --- 상단: CSV 업로드 & 펼침 영역 ---
st.subheader("모델 설정")
csv_file = st.file_uploader("source, topic, fallback(boolean) CSV 업로드", type=["csv"])

if csv_file is not None:
    df = pd.read_csv(csv_file)
    # 안전 변환
    df["source"] = df["source"].astype(str)
    df["topic"] = df["topic"].astype(str)
    df["fallback"] = df["fallback"].astype(str).str.lower().isin(["true","1","yes","y","t"])
    topics = sorted(df["topic"].unique().tolist())
    sources = sorted(df["source"].unique().tolist())

    with st.expander("🔧 분포/파라미터 설정 (펼치면 표시)", expanded=False):
        # 헤더 행
        cols = st.columns([1] + [max(2, 7 // max(1, len(sources))) for _ in sources])
        cols[0].markdown("**Topic \\ Source**")
        for j, src in enumerate(sources, start=1):
            cols[j].markdown(f"**{src}**")

        # 각 토픽 행
        for t in topics:
            row_cols = st.columns([1] + [max(2, 7 // max(1, len(sources))) for _ in sources])
            row_cols[0].markdown(f"**{t}**")

            for j, src in enumerate(sources, start=1):
                cell = row_cols[j]
                # fallback 표시 (CSV 기준)
                fb = bool(df[(df["topic"] == t) & (df["source"] == src)]["fallback"].any())
                fb_badge = "✅ fallback" if fb else "❌ fallback 없음"
                cell.caption(fb_badge)

                # 고유 키
                base_key = f"{t}|{src}"
                # 드롭다운(6개 분포)
                dist = cell.selectbox(
                    "분포",
                    options=["LN", "ZILN", "P", "ZIP", "NB", "ZINB"],
                    index=0,
                    key=f"{base_key}::dist",
                    label_visibility="collapsed"
                )

                # 분포별 파라미터 에디터 (간단 기본값)
                if dist == "LN":
                    m = cell.number_input("median", min_value=0.0, value=30.0, step=5.0, key=f"{base_key}::ln_m")
                    p95 = cell.number_input("p95", min_value=0.0, value=150.0, step=10.0, key=f"{base_key}::ln_p95")
                elif dist == "ZILN":
                    pi0 = cell.slider("pi0", 0.0, 1.0, 0.6, 0.05, key=f"{base_key}::ziln_pi0")
                    m = cell.number_input("median", min_value=0.0, value=30.0, step=5.0, key=f"{base_key}::ziln_m")
                    p95 = cell.number_input("p95", min_value=0.0, value=150.0, step=10.0, key=f"{base_key}::ziln_p95")
                elif dist == "P":
                    lam = cell.number_input("lambda", min_value=1e-6, value=3.0, step=0.5, key=f"{base_key}::p_lam")
                elif dist == "ZIP":
                    lam = cell.number_input("lambda", min_value=1e-6, value=1.0, step=0.2, key=f"{base_key}::zip_lam")
                    pi0 = cell.slider("pi0", 0.0, 1.0, 0.6, 0.05, key=f"{base_key}::zip_pi0")
                elif dist == "NB":
                    mean = cell.number_input("mean", min_value=0.0, value=1.0, step=0.1, key=f"{base_key}::nb_mean")
                    var = cell.number_input("var", min_value=1e-6, value=1.8, step=0.1, key=f"{base_key}::nb_var")
                elif dist == "ZINB":
                    mean = cell.number_input("mean", min_value=0.0, value=0.6, step=0.1, key=f"{base_key}::zinb_mean")
                    var = cell.number_input("var", min_value=1e-6, value=1.6, step=0.1, key=f"{base_key}::zinb_var")
                    pi0 = cell.slider("pi0", 0.0, 1.0, 0.6, 0.05, key=f"{base_key}::zinb_pi0")
else:
    st.info("CSV를 업로드하면 설정 테이블이 표시됩니다.")