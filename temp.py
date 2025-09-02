import numpy as np, math

# ==================== 공통 헬퍼 ====================
def _to_arr(x):
    if isinstance(x, np.ndarray): return x.astype(float, copy=False), ("nd", None)
    if isinstance(x, (list, tuple)): return np.asarray(x, dtype=float), ("list", type(x))
    if isinstance(x, (np.floating, float, int, np.integer)): return np.asarray([float(x)]), ("scalar", type(x))
    a = np.asarray(x, dtype=float); return a, ("nd", None)

def _restore(y, keep):
    k, tp = keep
    if k == "scalar": return float(y[0])
    if k == "list": return y.tolist()
    return y

def _is_int(a, tol=1e-12):
    return (np.abs(a - np.round(a)) <= tol)

# ==================== ZILN(p0, median, p90) ====================
def _ziln_mu_sigma(median, p90):
    z90 = 1.2815515655446004
    mu = np.log(median)
    sigma = (np.log(p90) - np.log(median)) / z90
    if sigma <= 0: raise ValueError("ZILN: p90 > median 이어야 합니다.")
    return mu, sigma

def ziln_pdf(x, p0, median, p90):
    a, keep = _to_arr(x)
    mu, sg = _ziln_mu_sigma(median, p90)
    out = np.zeros_like(a, dtype=float)
    pos = a > 0
    if np.any(pos):
        z = (np.log(a[pos]) - mu) / sg
        logpdf = -np.log(a[pos]) - np.log(sg) - 0.5*np.log(2*np.pi) - 0.5*z*z
        out[pos] = (1.0 - p0) * np.exp(logpdf)
    return _restore(out, keep)

def ziln_cdf_mid(x, p0, median, p90):
    a, keep = _to_arr(x)
    mu, sg = _ziln_mu_sigma(median, p90)
    out = np.zeros_like(a, dtype=float)
    out[a < 0] = 0.0
    at0 = (a == 0); out[at0] = 0.5 * p0
    gt0 = a > 0
    if np.any(gt0):
        z = (np.log(a[gt0]) - mu) / sg
        out[gt0] = p0 + (1.0 - p0) * (0.5*(1.0 + np.erf(z/np.sqrt(2.0))))
    return _restore(out, keep)

# ==================== ZIP(p0, mean) ====================
def _poisson_logpmf(k, lam):
    logfac = np.vectorize(math.lgamma)(k + 1.0)
    return -lam + k*np.log(lam) - logfac

def _zip_lambda_omega(p0, m, tol=1e-10, max_iter=200):
    if m <= 0: return 0.0, 1.0
    if p0 < np.exp(-m) - 1e-12:
        raise ValueError("ZIP: (p0, mean) 불가 (p0 < exp(-mean)).")
    if abs(p0 - np.exp(-m)) <= 1e-12:
        return m, 0.0
    f = lambda lam: 1.0 - (m/lam)*(1.0 - np.exp(-lam)) - p0
    lo, hi = 1e-12, max(10.0, m*20.0)
    flo, fhi = f(lo), f(hi)
    grow = 0
    while flo*fhi > 0 and grow < 20:
        hi *= 2.0; fhi = f(hi); grow += 1
    if flo*fhi > 0:
        lam = max(m, 1.0)
        for _ in range(60):
            e = np.exp(-lam)
            g = 1.0 - (m/lam)*(1.0 - e) - p0
            dg = m*((1.0 - e)/lam**2 - e/lam)
            step = g/dg
            lam = max(lam - step, 1e-12)
            if abs(step) < tol: break
    else:
        for _ in range(max_iter):
            mid = 0.5*(lo+hi)
            fm = f(mid)
            if abs(fm) < tol or (hi-lo) < 1e-10: lam = float(mid); break
            if flo*fm <= 0: hi, fhi = mid, fm
            else: lo, flo = mid, fm
        else:
            lam = float(0.5*(lo+hi))
    omega = 1.0 - m/lam
    omega = min(max(omega, 0.0), 1.0)
    return lam, omega

def zip_pdf(x, p0, mean):
    a, keep = _to_arr(x)
    out = np.zeros_like(a, dtype=float)
    lam, omega = _zip_lambda_omega(p0, mean)
    ok = _is_int(a) & (a >= 0)
    if np.any(ok):
        k = np.round(a[ok]).astype(int)
        logpmf = _poisson_logpmf(k.astype(float), lam)
        pmf = (1.0 - omega) * np.exp(logpmf)
        pmf[k == 0] += omega
        out[ok] = pmf
    return _restore(out, keep)

def zip_cdf_mid(x, p0, mean):
    a, keep = _to_arr(x)
    out = np.zeros_like(a, dtype=float)
    lam, omega = _zip_lambda_omega(p0, mean)
    out[a < 0] = 0.0
    xf = np.floor(a).astype(int)
    for i, xi in enumerate(a):
        if xi < 0: continue
        kf = int(xf[i])
        ks = np.arange(0, kf+1, dtype=float)
        logpmf = _poisson_logpmf(ks, lam)
        pmf = (1.0 - omega) * np.exp(logpmf)
        if pmf.size > 0: pmf[0] += omega
        cdf_floor = pmf.sum()
        if _is_int(np.asarray([xi]))[0]:
            pmf_x = (1.0 - omega) * np.exp(_poisson_logpmf(float(kf), lam))
            if kf == 0: pmf_x += omega
            out[i] = (cdf_floor - pmf_x) + 0.5*pmf_x
        else:
            out[i] = cdf_floor
    return _restore(out, keep)

# ==================== ZINB(p0, mean, r) ====================
# NB 파라미터화: pmf(k) = C(k+r-1, k) * (1-p)^r * p^k,  k=0,1,...
# base-NB 평균 = r*p/(1-p), 분산 = r*p/(1-p)^2 = mean + mean^2/r
# ZINB: omega = 구조적 0 비율, 총평균 = (1-omega)*base_mean,  P(X=0)=omega + (1-omega)*(1-p)^r
def _nb_logpmf(k, r, p):
    k = np.asarray(k, dtype=float)
    logC = np.vectorize(math.lgamma)(k + r) - math.lgamma(r) - np.vectorize(math.lgamma)(k + 1.0)
    return logC + r*np.log(1.0 - p) + k*np.log(p)

def _zinb_params_from_p0_mean_r(p0, mean, r, tol=1e-10, max_iter=200):
    if r <= 0: raise ValueError("ZINB: r>0 필요.")
    if mean < 0: raise ValueError("ZINB: mean>=0 필요.")
    if mean == 0: 
        # 전체 평균 0이면 전부 0
        return 0.5, 0.0, 1.0  # (q dummy, p=0, omega=1)
    # q = 1-p ∈ (0,1).  z=q^r
    # mean = ((1-p0)/(1 - z)) * r * (1-q)/q
    def g(q):
        z = q**r
        return ((1.0 - p0)/(1.0 - z)) * r * (1.0 - q)/q - mean
    lo, hi = 1e-12, 1.0 - 1e-12
    glo, ghi = g(lo), g(hi)
    # 보장: g(lo) > 0 (거의 무한대), g(hi) < 0 (0에 근접)
    if not (glo > 0 and ghi < 0):
        # 수치/불가 조합
        raise ValueError("ZINB: (p0, mean, r) 조합이 불가하거나 수치적으로 불안정합니다.")
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        gm = g(mid)
        if abs(gm) < tol or (hi-lo) < 1e-10:
            q = float(mid); break
        if gm > 0: lo = mid
        else: hi = mid
    else:
        q = float(0.5*(lo+hi))
    p = 1.0 - q
    z = q**r
    omega = (p0 - z) / (1.0 - z)
    omega = min(max(omega, 0.0), 1.0)
    return q, p, omega

def zinb_pdf(x, p0, mean, r):
    a, keep = _to_arr(x)
    out = np.zeros_like(a, dtype=float)
    q, p, omega = _zinb_params_from_p0_mean_r(p0, mean, r)
    ok = _is_int(a) & (a >= 0)
    if np.any(ok):
        k = np.round(a[ok]).astype(int)
        logpmf = _nb_logpmf(k.astype(float), r, p)
        pmf = (1.0 - omega) * np.exp(logpmf)
        pmf[k == 0] += omega
        out[ok] = pmf
    return _restore(out, keep)

def zinb_cdf_mid(x, p0, mean, r):
    a, keep = _to_arr(x)
    out = np.zeros_like(a, dtype=float)
    q, p, omega = _zinb_params_from_p0_mean_r(p0, mean, r)
    out[a < 0] = 0.0
    xf = np.floor(a).astype(int)
    for i, xi in enumerate(a):
        if xi < 0: continue
        kf = int(xf[i])
        ks = np.arange(0, kf+1, dtype=float)
        logpmf = _nb_logpmf(ks, r, p)
        pmf = (1.0 - omega) * np.exp(logpmf)
        if pmf.size > 0: pmf[0] += omega
        cdf_floor = pmf.sum()
        if _is_int(np.asarray([xi]))[0]:
            pmf_x = (1.0 - omega) * np.exp(_nb_logpmf(float(kf), r, p))
            if kf == 0: pmf_x += omega
            out[i] = (cdf_floor - pmf_x) + 0.5*pmf_x
        else:
            out[i] = cdf_floor
    return _restore(out, keep)