import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from math import erf, sqrt, lgamma

# -----------------------------
# Math helpers
# -----------------------------
Z95 = 1.6448536269514722
Z99 = 2.3263478740408408

def lognorm_params_from_median_p95(median, p95):
    mu = np.log1p(max(median, 0.0))
    q95 = np.log1p(max(p95, median + 1e-9))
    sigma = max((q95 - mu) / Z95, 1e-8)
    return mu, sigma

def lognorm_cdf(xs, mu, sigma):
    # vectorized log-normal CDF under log1p parameterization
    z = (np.log1p(np.maximum(xs, 0.0)) - mu) / sigma
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))

def poisson_cdf(k, lam):
    k = int(np.floor(k))
    s = 0.0
    for i in range(0, k + 1):
        s += np.exp(-lam) * (lam ** i) / np.math.factorial(i)
    return min(1.0, s)

def negbin_params_from_mean_var(mean, var):
    var = max(var, mean + 1e-6)
    r = (mean * mean) / (var - mean)
    p = mean / (mean + r)
    r = max(r, 1e-6)
    p = min(max(p, 1e-9), 1 - 1e-9)
    return r, p

def negbin_pmf(k, r, p):
    # pmf(k) = C(k+r-1, k) * (1-p)^r * p^k
    return np.exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1) + r * np.log(1 - p) + k * np.log(p))

def negbin_cdf(k, mean, var):
    r, p = negbin_params_from_mean_var(mean, var)
    k = int(np.floor(k))
    s = 0.0
    for i in range(0, k + 1):
        s += negbin_pmf(i, r, p)
    return min(1.0, s)

def zip_cdf(k, lam, pi0):
    if k < 0: return 0.0
    return pi0 + (1.0 - pi0) * poisson_cdf(k, lam)

def zinb_cdf(k, mean, var, pi0):
    if k < 0: return 0.0
    return pi0 + (1.0 - pi0) * negbin_cdf(k, mean, var)

# -----------------------------
# Plotter
# -----------------------------
DURATION_SOURCES = {"app","you","web"}
COUNT_SOURCES    = {"ex","poi","cal","noti"}

def plot_cdf_grid_from_csv(
    csv_path: str,
    topics: list,
    sources: list,
    *,
    fallback_color: str = "tab:orange",
    fallback_linestyle: str = "--",
    show_legend: bool = True
):
    """
    CSV(컬럼: source,topic,dist_type,parameter,comment,fallback)를 읽어
    행=topics, 열=sources 격자로 CDF를 그립니다.
      - prior 있으면 파란 실선
      - 해당 소스 타입의 fallback 있으면 오렌지 점선으로 overlay
    """
    df = pd.read_csv(csv_path)
    # normalize / parse
    df["source"]    = df["source"].astype(str)
    df["topic"]     = df["topic"].astype(str)
    df["dist_type"] = df["dist_type"].astype(str)
    df["fallback"]  = df["fallback"].astype(str).str.lower().isin(["true","1","yes"])
    df["parameter"] = df["parameter"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else {})

    # fallback rows
    fb_dur = df[(df["source"]=="fallback_duration") & (df["fallback"])]
    fb_cnt = df[(df["source"]=="fallback_count")   & (df["fallback"])]
    fb_dur_spec = fb_dur.iloc[0] if len(fb_dur)>0 else None
    fb_cnt_spec = fb_cnt.iloc[0] if len(fb_cnt)>0 else None

    def _plot_cdf(ax, dist_type, params, is_fallback=False):
        if dist_type == "lognorm":
            mu, sigma = lognorm_params_from_median_p95(params["median"], params["p95"])
            x99 = np.expm1(mu + Z99*sigma)
            xs = np.linspace(0, max(1.0, x99), 200)
            ys = lognorm_cdf(xs, mu, sigma)
            kw = {"linestyle": fallback_linestyle, "color": fallback_color} if is_fallback else {}
            ax.plot(xs, ys, **kw)
            ax.set_xlabel("minutes")

        elif dist_type == "poisson":
            lam = float(params["lambda"])
            xmax = int(np.ceil(lam + 4*np.sqrt(max(lam, 1e-6))))
            xs = np.arange(0, max(1, xmax+1))
            ys = [poisson_cdf(x, lam) for x in xs]
            kw = {"linestyle": fallback_linestyle, "color": fallback_color} if is_fallback else {}
            ax.step(xs, ys, where="post", **kw)
            ax.set_xlabel("count")

        elif dist_type == "neg_binom":
            mean, var = float(params["mean"]), float(params["var"])
            std  = np.sqrt(var)
            xmax = int(np.ceil(mean + 4*std))
            xs = np.arange(0, max(1, xmax+1))
            ys = [negbin_cdf(x, mean, var) for x in xs]
            kw = {"linestyle": fallback_linestyle, "color": fallback_color} if is_fallback else {}
            ax.step(xs, ys, where="post", **kw)
            ax.set_xlabel("count")

        elif dist_type == "zip":
            lam, pi0 = float(params["lambda"]), float(params["pi0"])
            xmax = int(np.ceil(lam + 4*np.sqrt(max(lam, 1e-6))))
            xs = np.arange(0, max(1, xmax+1))
            ys = [zip_cdf(x, lam, pi0) for x in xs]
            kw = {"linestyle": fallback_linestyle, "color": fallback_color} if is_fallback else {}
            ax.step(xs, ys, where="post", **kw)
            ax.set_xlabel("count")

        elif dist_type == "zinb":
            mean, var, pi0 = float(params["mean"]), float(params["var"]), float(params["pi0"])
            std  = np.sqrt(var)
            xmax = int(np.ceil(mean + 4*std))
            xs = np.arange(0, max(1, xmax+1))
            ys = [zinb_cdf(x, mean, var, pi0) for x in xs]
            kw = {"linestyle": fallback_linestyle, "color": fallback_color} if is_fallback else {}
            ax.step(xs, ys, where="post", **kw)
            ax.set_xlabel("count")

        else:
            ax.text(0.5, 0.5, f"(unknown {dist_type})", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            return

        ax.set_ylim(0, 1)
        ax.set_ylabel("CDF")

    # grid
    n_rows, n_cols = len(topics), len(sources)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6*n_cols, 3.0*n_rows), squeeze=False)

    for i, topic in enumerate(topics):
        for j, source in enumerate(sources):
            ax = axes[i][j]
            # plot prior (if any)
            mask = (df["source"]==source) & (df["topic"]==topic) & (~df["fallback"])
            if mask.any():
                r = df[mask].iloc[0]
                _plot_cdf(ax, r["dist_type"], r["parameter"], is_fallback=False)
                ax.plot([], [], label="prior")  # legend handle
            else:
                ax.text(0.5, 0.6, "(no prior)", ha="center", va="center")

            # overlay fallback by source type
            if source in DURATION_SOURCES and fb_dur_spec is not None:
                _plot_cdf(ax, fb_dur_spec["dist_type"], fb_dur_spec["parameter"], is_fallback=True)
                ax.plot([], [], linestyle=fallback_linestyle, color=fallback_color, label="fallback")
            elif source in COUNT_SOURCES and fb_cnt_spec is not None:
                _plot_cdf(ax, fb_cnt_spec["dist_type"], fb_cnt_spec["parameter"], is_fallback=True)
                ax.plot([], [], linestyle=fallback_linestyle, color=fallback_color, label="fallback")

            ax.set_title(f"{source} — {topic}")
            if show_legend:
                ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    return fig, axes