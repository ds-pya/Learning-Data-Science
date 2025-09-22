import numpy as np
from scipy.stats import norm, poisson, nbinom

def zero_inflated_lognormal(value: float, p0: float, median: float, p90: float) -> float:
    if not np.isfinite(value) or not np.isfinite(p0) or not np.isfinite(median) or not np.isfinite(p90):
        return np.nan
    if value < 0:
        return 0.0

    pp0 = np.clip(p0, 0.0, 1.0 - 1e-12)
    if value == 0:
        return np.clip(pp0 * 0.5, 0.0, 1.0)

    z90 = norm.ppf(0.9)
    mu = np.log(max(median, 1e-12))
    safe_p90 = max(p90, max(median, 1e-12) * (1.0 + 1e-12))
    sigma = max((np.log(safe_p90) - mu) / z90, 1e-9)

    cdf_ln = norm.cdf((np.log(value) - mu) / sigma)
    pct = pp0 + (1.0 - pp0) * cdf_ln
    return np.clip(pct, 0.0, 1.0)


def zero_inflated_poisson(value: float, p0: float, mean: float) -> float:
    if not np.isfinite(value) or not np.isfinite(p0) or not np.isfinite(mean):
        return np.nan
    if value < 0:
        return 0.0

    pp0 = np.clip(p0, 0.0, 1.0 - 1e-12)
    lam = max(mean, 1e-12)
    pois = poisson(mu=lam)

    if value == 0:
        pmf0 = pois.pmf(0)
        return np.clip((pp0 + (1.0 - pp0) * pmf0) * 0.5, 0.0, 1.0)

    k = int(np.floor(value))
    cdf_below = pois.cdf(k - 1) if k > 0 else 0.0
    pmf_k = pois.pmf(k)

    pct = pp0 + (1.0 - pp0) * (cdf_below + 0.5 * pmf_k)
    return np.clip(pct, 0.0, 1.0)


def zero_inflated_negative_binomial(value: float, p0: float, mean: float, variance: float) -> float:
    if not np.isfinite(value) or not np.isfinite(p0) or not np.isfinite(mean) or not np.isfinite(variance):
        return np.nan
    if value < 0:
        return 0.0

    pp0 = np.clip(p0, 0.0, 1.0 - 1e-12)
    m = max(mean, 1e-12)
    v = max(variance, m * (1.0 + 1e-9))  # var > mean

    # NB(r, p): mean = r*(1-p)/p, var = r*(1-p)/p^2
    r = max(m * m / (v - m), 1e-9)
    p = np.clip(r / (r + m), 1e-12, 1.0 - 1e-12)

    # scipy nbinom: number of successes = r, prob of success = p
    nb = nbinom(n=r, p=p)

    if value == 0:
        pmf0 = nb.pmf(0)
        return np.clip((pp0 + (1.0 - pp0) * pmf0) * 0.5, 0.0, 1.0)

    k = int(np.floor(value))
    cdf_below = nb.cdf(k - 1) if k > 0 else 0.0
    pmf_k = nb.pmf(k)

    pct = pp0 + (1.0 - pp0) * (cdf_below + 0.5 * pmf_k)
    return np.clip(pct, 0.0, 1.0)