import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def plot_initial_distribution(samples_by_st, source, topic):
    samp = samples_by_st.get((source, topic))
    if samp is None or len(samp)==0:
        print(f"No data for ({source}, {topic})")
        return
    
    mu, sigma = lognorm_params(samp)
    if mu is None: 
        print("Not enough data")
        return

    xmax = samp.quantile(0.995)  # 상위 99.5%까지 범위
    xs = np.linspace(0, xmax, 200)
    ys = [lognorm_cdf(x, mu, sigma) for x in xs]

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, label=f"lognorm CDF ({source}-{topic})")
    plt.title(f"Initial Distribution CDF for {topic} from {source}")
    plt.xlabel("Exposure")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()