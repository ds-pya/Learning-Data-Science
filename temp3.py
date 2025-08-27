# 1) CSS로 셀 경계/여백 강화 + 컴팩트 위젯
st.markdown("""
<style>
.grid-cell{border:1px solid #e5e7eb;border-radius:10px;padding:6px 8px;background:#fbfbfc}
.badge{display:inline-block;padding:2px 6px;border-radius:6px;font-size:11px;background:#eef2ff;color:#3730a3}
.badge.off{background:#f1f5f9;color:#64748b}
.small label{font-size:12px !important;margin-bottom:2px !important}
.small .stSlider, .small .stNumberInput{padding-top:0 !important;margin-top:0 !important}
</style>
""", unsafe_allow_html=True)

with st.expander("🔧 분포/파라미터 설정 (펼치면 표시)", expanded=False):
    header = st.columns([1] + [3]*len(sources), gap="small")
    header[0].markdown("**Topic \\ Source**")
    for j, src in enumerate(sources, start=1):
        header[j].markdown(f"**{src}**")

    for t in topics:
        row = st.columns([1] + [3]*len(sources), gap="small")
        row[0].markdown(f"**{t}**")

        for j, src in enumerate(sources, start=1):
            with row[j]:
                fb = bool(df[(df["topic"]==t)&(df["source"]==src)]["fallback"].any())
                st.markdown(f"<div class='grid-cell'>", unsafe_allow_html=True)
                st.markdown(f"<span class='badge{' off' if not fb else ''}'>" + ("fallback" if fb else "no fb") + "</span>", unsafe_allow_html=True)

                base_key = f"{t}|{src}"
                dist = st.selectbox(
                    "분포", ["LN","ZILN","P","ZIP","NB","ZINB"],
                    index=0, key=f"{base_key}::dist", label_visibility="collapsed"
                )

                # 파라미터: 더 컴팩트하게 2~3열 그리드로 배치
                if dist == "LN":
                    c1,c2 = st.columns(2, gap="small")
                    with c1: st.number_input("median", min_value=0.0, value=30.0, step=5.0, key=f"{base_key}::ln_m", help="분(중앙값)")
                    with c2: st.number_input("p95", min_value=0.0, value=150.0, step=10.0, key=f"{base_key}::ln_p95", help="95퍼센타일")
                elif dist == "ZILN":
                    c1,c2,c3 = st.columns(3, gap="small")
                    with c1: st.slider("pi0", 0.0,1.0,0.6,0.05, key=f"{base_key}::ziln_pi0")
                    with c2: st.number_input("median", min_value=0.0, value=30.0, step=5.0, key=f"{base_key}::ziln_m")
                    with c3: st.number_input("p95", min_value=0.0, value=150.0, step=10.0, key=f"{base_key}::ziln_p95")
                elif dist == "P":
                    st.number_input("lambda", min_value=1e-6, value=3.0, step=0.5, key=f"{base_key}::p_lam")
                elif dist == "ZIP":
                    c1,c2 = st.columns(2, gap="small")
                    with c1: st.number_input("lambda", min_value=1e-6, value=1.0, step=0.2, key=f"{base_key}::zip_lam")
                    with c2: st.slider("pi0", 0.0,1.0,0.6,0.05, key=f"{base_key}::zip_pi0")
                elif dist == "NB":
                    c1,c2 = st.columns(2, gap="small")
                    with c1: st.number_input("mean", min_value=0.0, value=1.0, step=0.1, key=f"{base_key}::nb_mean")
                    with c2: st.number_input("var",  min_value=1e-6, value=1.8, step=0.1, key=f"{base_key}::nb_var")
                elif dist == "ZINB":
                    c1,c2,c3 = st.columns(3, gap="small")
                    with c1: st.number_input("mean", min_value=0.0, value=0.6, step=0.1, key=f"{base_key}::zinb_mean")
                    with c2: st.number_input("var",  min_value=1e-6, value=1.6, step=0.1, key=f"{base_key}::zinb_var")
                    with c3: st.slider("pi0", 0.0,1.0,0.6,0.05, key=f"{base_key}::zinb_pi0")

                st.markdown("</div>", unsafe_allow_html=True)