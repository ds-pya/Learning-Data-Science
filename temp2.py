import matplotlib.pyplot as plt
import numpy as np
from math import erf, sqrt, log1p

def lognorm_params(sample_series):
    """샘플 기반 log1p 변환 후 mu, sigma 추정"""
    if len(sample_series)==0:
        return None, None
    x = np.log1p(sample_series.clip(lower=0))
    mu, sigma = x.mean(), x.std(ddof=0) + 1e-8
    return mu, sigma

def lognorm_cdf(x, mu, sigma):
    z = (np.log1p(x)-mu)/sigma
    return 0.5*(1+erf(z/sqrt(2)))

def plot_initial_cdf_grid(samples_by_st, sources, topics, max_cols=4):
    """
    samples_by_st: dict[(source,topic)] -> pd.Series of exposures
    sources: list of source names (열)
    topics : list of topic names  (행)
    """
    n_rows = len(topics)
    n_cols = min(len(sources), max_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)

    for i, topic in enumerate(topics):
        for j, source in enumerate(sources[:n_cols]):
            ax = axes[i][j]
            samp = samples_by_st.get((source, topic))
            if samp is None or len(samp)==0:
                ax.text(0.5,0.5,"(no data)",ha="center",va="center")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{source}-{topic}")
                continue

            mu, sigma = lognorm_params(samp)
            if mu is None:
                ax.text(0.5,0.5,"(not enough data)",ha="center",va="center")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{source}-{topic}")
                continue

            xmax = samp.quantile(0.995)
            xs = np.linspace(0, xmax, 200)
            ys = [lognorm_cdf(x, mu, sigma) for x in xs]
            ax.plot(xs, ys)
            ax.set_ylim(0,1)
            ax.set_title(f"{source}-{topic}")

    plt.tight_layout()
    plt.show()