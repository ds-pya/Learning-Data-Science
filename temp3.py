# streamlit_app.py
# Streamlit dashboard for exploring 5 candidate distributions:
# 1) Lognormal (duration, continuous)
# 2) Poisson (count, discrete)
# 3) Negative Binomial (count, discrete)
# 4) Zero-Inflated Poisson (count, discrete)
# 5) Zero-Inflated Negative Binomial (count, discrete)
#
# Features:
# - Parameterization pickers (e.g., Lognormal: median&p95 or mu&sigma under log1p)
# - Live plots for CDF and PMF/PDF
# - Adjustable x-range and resolution
# - Download sampled curves as CSV
#
# Run:
#   pip install streamlit
#   streamlit run streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import math
from math import erf, sqrt, lgamma
import matplotlib.pyplot as plt

st.set_page_config(page_title="Distribution Explorer", layout="wide")

# -----------------------------
# Math helpers
# -----------------------------

Z95 = 1.6448536269514722
Z99 = 2.3263478740408408

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

# -------- Lognormal (log1p parameterization) --------
def lognorm_params_from_median_p95(median, p95):
    median = max(median, 0.0)
    p95 = max(p95, median + 1e-9)
    mu = np.log1p(median)
    q95 = np.log1p(p95)
    sigma = max((q95 - mu) / Z95, 1e-8)
    return mu, sigma

def lognorm_pdf(x, mu, sigma):
    # PDF for y = log1p(x) ~ Normal(mu, sigma^2), x >= 0
    x = np.maximum(x, 0.0)
    z = (np.log1p(x) - mu) / sigma
    return (1.0 / (sigma * np.sqrt(2*np.pi) * (1.0 + x))) * np.exp(-0.5 * z * z)

def lognorm_cdf(x, mu, sigma):
    z = (np.log1p(np.maximum(x, 0.0)) - mu) / sigma
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

# -------- Poisson --------
def poisson_pmf(k, lam):
    if k < 0:
        return 0.0
    return np.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf(k, lam):
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, int(k) + 1):
        s += poisson_pmf(i, lam)
    return min(1.0, s)

# -------- Negative Binomial (r, p) with mean/var conversions --------
def nb_rp_from_mean_var(mean, var):
    # r>0, p in (0,1). Using parameterization:
    # mean = r * p / (1 - p), var = r * p / (1 - p)^2  => var > mean
    # Solve: r = mean^2 / (var - mean), p = mean / (mean + r)
    var = max(var, mean + 1e-6)
    r = (mean * mean) / (var - mean)
    p = mean / (mean + r)
    r = max(r, 1e-8)
    p = min(max(p, 1e-9), 1 - 1e-9)
    return r, p

def nb_pmf(k, r, p):
    # pmf(k) = C(k+r-1, k) * (1-p)^r * p^k
    if k < 0:
        return 0.0
    return np.exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1) + r * np.log(1 - p) + k * np.log(p))

def nb_cdf(k, r, p):
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, int(k) + 1):
        s += nb_pmf(i, r, p)
    return min(1.0, s)

# -------- Zero-Inflated Poisson --------
def zip_pmf(k, lam, pi0):
    pi0 = clamp01(pi0)
    if k == 0:
        return pi0 + (1 - pi0) * poisson_pmf(0, lam)
    return (1 - pi0) * poisson_pmf(k, lam)

def zip_cdf(k, lam, pi0):
    pi0 = clamp01(pi0)
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, int(k) + 1):
        s += zip_pmf(i, lam, pi0)
    return min(1.0, s)

# -------- Zero-Inflated Negative Binomial --------
def zinb_pmf(k, r, p, pi0):
    pi0 = clamp01(pi0)
    if k == 0:
        return pi0 + (1 - pi0) * nb_pmf(0, r, p)
    return (1 - pi0) * nb_pmf(k, r, p)

def zinb_cdf(k, r, p, pi0):
    pi0 = clamp01(pi0)
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, int(k) + 1):
        s += zinb_pmf(i, r, p, pi0)
    return min(1.0, s)

# -----------------------------
# UI
# -----------------------------
st.title("Distribution Explorer (CDF / PMF/PDF)")

dist = st.sidebar.selectbox(
    "분포 선택",
    ["Lognormal (duration)", "Poisson", "Negative Binomial", "Zero-Inflated Poisson", "Zero-Inflated Negative Binomial"]
)

show_pdf = st.sidebar.checkbox("PDF/PMF 보기", value=True)
show_cdf = st.sidebar.checkbox("CDF 보기", value=True)

if dist == "Lognormal (duration)":
    st.sidebar.markdown("### Lognormal 파라미터")
    mode = st.sidebar.radio("파라미터화 방식", ["median & p95 (권장)", "mu & sigma (log1p 스케일)"], index=0)
    if mode == "median & p95 (권장)":
        median = st.sidebar.number_input("median (minutes)", min_value=0.0, value=30.0, step=5.0)
        p95    = st.sidebar.number_input("p95 (minutes)", min_value=0.0, value=150.0, step=10.0)
        mu, sigma = lognorm_params_from_median_p95(median, p95)
    else:
        mu    = st.sidebar.number_input("mu (on log1p scale)", value=np.log1p(30.0), step=0.1, format="%.3f")
        sigma = st.sidebar.number_input("sigma (>0)", min_value=1e-6, value=0.6, step=0.05, format="%.3f")

    x_max = st.sidebar.number_input("X max (minutes)", min_value=10.0, value=float(np.expm1(mu + Z99*sigma)), step=10.0)
    n_pts = st.sidebar.slider("연속축 샘플 포인트 수", min_value=100, max_value=1000, value=400, step=50)

    xs = np.linspace(0, x_max, n_pts)
    data = {"x": xs}
    if show_pdf:
        data["pdf"] = lognorm_pdf(xs, mu, sigma)
    if show_cdf:
        data["cdf"] = lognorm_cdf(xs, mu, sigma)

    df = pd.DataFrame(data)
    st.subheader("Lognormal (duration)")
    fig, ax = plt.subplots(figsize=(8,4))
    if show_pdf:
        ax.plot(df["x"], df["pdf"], label="PDF")
    if show_cdf:
        ax.plot(df["x"], df["cdf"], label="CDF")
    ax.set_xlabel("minutes")
    ax.set_ylim(bottom=0)
    ax.legend()
    st.pyplot(fig)
    st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8"), file_name="lognormal_curves.csv", mime="text/csv")

elif dist == "Poisson":
    st.sidebar.markdown("### Poisson 파라미터")
    lam = st.sidebar.number_input("lambda (평균)", min_value=1e-6, value=3.0, step=0.5)
    # practical range
    k_max = int(np.ceil(lam + 6 * np.sqrt(max(lam, 1e-6)) + 10))
    k_max = st.sidebar.slider("K max (표시 최대 카운트)", min_value=10, max_value=max(20, k_max), value=min(40, k_max))

    ks = np.arange(0, k_max + 1)
    data = {"k": ks}
    if show_pdf:
        data["pmf"] = [poisson_pmf(int(k), lam) for k in ks]
    if show_cdf:
        data["cdf"] = [poisson_cdf(int(k), lam) for k in ks]
    df = pd.DataFrame(data)

    st.subheader("Poisson (count)")
    fig, ax = plt.subplots(figsize=(8,4))
    if show_pdf:
        ax.step(df["k"], df["pmf"], where="post", label="PMF")
    if show_cdf:
        ax.step(df["k"], df["cdf"], where="post", label="CDF")
    ax.set_xlabel("count")
    ax.set_ylim(bottom=0, top=1 if show_cdf and not show_pdf else None)
    ax.legend()
    st.pyplot(fig)
    st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8"), file_name="poisson_curves.csv", mime="text/csv")

elif dist == "Negative Binomial":
    st.sidebar.markdown("### Negative Binomial 파라미터")
    nb_mode = st.sidebar.radio("파라미터화 방식", ["mean & var (권장)", "r & p"], index=0)
    if nb_mode == "mean & var (권장)":
        mean = st.sidebar.number_input("mean", min_value=0.0, value=1.0, step=0.1)
        var  = st.sidebar.number_input("var (mean보다 커야 함)", min_value=1e-6, value=1.8, step=0.1)
        r, p = nb_rp_from_mean_var(mean, var)
    else:
        r = st.sidebar.number_input("r (>0)", min_value=1e-6, value=1.0, step=0.1)
        p = st.sidebar.number_input("p (0~1)", min_value=1e-6, max_value=1-1e-6, value=0.5, step=0.05)

    # practical range: mean and std from r,p
    mean_nb = r * p / (1 - p)
    var_nb  = r * p / ((1 - p) ** 2)
    std_nb  = np.sqrt(var_nb)
    k_max_suggest = int(np.ceil(mean_nb + 6 * std_nb))
    k_max = st.sidebar.slider("K max (표시 최대 카운트)", min_value=10, max_value=max(20, k_max_suggest), value=min(40, k_max_suggest))

    ks = np.arange(0, k_max + 1)
    data = {"k": ks}
    if show_pdf:
        data["pmf"] = [nb_pmf(int(k), r, p) for k in ks]
    if show_cdf:
        data["cdf"] = [nb_cdf(int(k), r, p) for k in ks]
    df = pd.DataFrame(data)

    st.subheader("Negative Binomial (count)")
    fig, ax = plt.subplots(figsize=(8,4))
    if show_pdf:
        ax.step(df["k"], df["pmf"], where="post", label="PMF")
    if show_cdf:
        ax.step(df["k"], df["cdf"], where="post", label="CDF")
    ax.set_xlabel("count")
    ax.set_ylim(bottom=0, top=1 if show_cdf and not show_pdf else None)
    ax.legend()
    st.pyplot(fig)
    st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8"), file_name="neg_binomial_curves.csv", mime="text/csv")

elif dist == "Zero-Inflated Poisson":
    st.sidebar.markdown("### ZIP 파라미터")
    lam = st.sidebar.number_input("lambda (포아송 평균)", min_value=1e-6, value=1.0, step=0.2)
    pi0 = st.sidebar.slider("pi0 (제로 팽창 확률)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    k_max = int(np.ceil(lam + 6 * np.sqrt(max(lam, 1e-6)) + 10))
    k_max = st.sidebar.slider("K max (표시 최대 카운트)", min_value=10, max_value=max(20, k_max), value=min(40, k_max))

    ks = np.arange(0, k_max + 1)
    data = {"k": ks}
    if show_pdf:
        data["pmf"] = [zip_pmf(int(k), lam, pi0) for k in ks]
    if show_cdf:
        data["cdf"] = [zip_cdf(int(k), lam, pi0) for k in ks]
    df = pd.DataFrame(data)

    st.subheader("Zero-Inflated Poisson (count)")
    fig, ax = plt.subplots(figsize=(8,4))
    if show_pdf:
        ax.step(df["k"], df["pmf"], where="post", label="PMF")
    if show_cdf:
        ax.step(df["k"], df["cdf"], where="post", label="CDF")
    ax.set_xlabel("count")
    ax.set_ylim(bottom=0, top=1 if show_cdf and not show_pdf else None)
    ax.legend()
    st.pyplot(fig)
    st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8"), file_name="zip_curves.csv", mime="text/csv")

elif dist == "Zero-Inflated Negative Binomial":
    st.sidebar.markdown("### ZINB 파라미터")
    zinb_mode = st.sidebar.radio("파라미터화 방식", ["mean & var & pi0 (권장)", "r & p & pi0"], index=0)
    if zinb_mode == "mean & var & pi0 (권장)":
        mean = st.sidebar.number_input("mean", min_value=0.0, value=0.6, step=0.1)
        var  = st.sidebar.number_input("var (mean보다 커야 함)", min_value=1e-6, value=1.6, step=0.1)
        r, p = nb_rp_from_mean_var(mean, var)
        pi0  = st.sidebar.slider("pi0 (제로 팽창 확률)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    else:
        r   = st.sidebar.number_input("r (>0)", min_value=1e-6, value=1.0, step=0.1)
        p   = st.sidebar.number_input("p (0~1)", min_value=1e-6, max_value=1-1e-6, value=0.5, step=0.05)
        pi0 = st.sidebar.slider("pi0 (제로 팽창 확률)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    mean_nb = r * p / (1 - p)
    var_nb  = r * p / ((1 - p) ** 2)
    std_nb  = np.sqrt(var_nb)
    k_max_suggest = int(np.ceil(mean_nb + 6 * std_nb))
    k_max = st.sidebar.slider("K max (표시 최대 카운트)", min_value=10, max_value=max(20, k_max_suggest), value=min(40, k_max_suggest))

    ks = np.arange(0, k_max + 1)
    data = {"k": ks}
    if show_pdf:
        data["pmf"] = [zinb_pmf(int(k), r, p, pi0) for k in ks]
    if show_cdf:
        data["cdf"] = [zinb_cdf(int(k), r, p, pi0) for k in ks]
    df = pd.DataFrame(data)

    st.subheader("Zero-Inflated Negative Binomial (count)")
    fig, ax = plt.subplots(figsize=(8,4))
    if show_pdf:
        ax.step(df["k"], df["pmf"], where="post", label="PMF")
    if show_cdf:
        ax.step(df["k"], df["cdf"], where="post", label="CDF")
    ax.set_xlabel("count")
    ax.set_ylim(bottom=0, top=1 if show_cdf and not show_pdf else None)
    ax.legend()
    st.pyplot(fig)
    st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8"), file_name="zinb_curves.csv", mime="text/csv")

# -----------------------------
# Sidebar tips
# -----------------------------
with st.sidebar.expander("Tips / 가이드", expanded=False):
    st.markdown("""
- **로그정규**: median & p95로 직관적으로 보정하는 걸 권장합니다. (duration heavy-tail)
- **포아송 vs 음이항**: 분산이 평균에 비해 큰지 확인하세요. var ≫ mean이면 음이항.
- **제로팽창(π0)**: 0의 과잉 정도입니다. π0→0이면 ZIP→포아송, ZINB→음이항에 수렴합니다.
- **표시 범위**: K max / X max를 늘리면 꼬리까지 확인 가능합니다.
""")