import numpy as np, math

# ---------- 공통 헬퍼 ----------
def _to_arr(x):
    if isinstance(x, np.ndarray): return x.astype(float, copy=False), ("nd", None)
    if isinstance(x, (list, tuple)): return np.asarray(x, dtype=float), ("list", type(x))
    if isinstance(x, (np.floating, float, int, np.integer)): return np.asarray([float(x)]), ("scalar", type(x))
    # fallback
    a = np.asarray(x, dtype=float); return a, ("nd", None)

def _restore(y, keep):
    k, tp = keep
    if k == "scalar": return float(y[0])
    if k == "list": return y.tolist()
    return y

def _is_int(a, tol=1e-12):
    return (np.abs(a - np.round(a)) <= tol)

# ---------- ZILN(p0, median, p90) ----------
# median = exp(mu), p90 = exp(mu + z90*sigma), z90≈1.2815515655446004
def _ziln_mu_sigma(median, p90):
    z90 = 1.2815515655446004
    mu = np.log(median)
    sigma = (np.log(p90) - np.log(median)) / z90
    if sigma <= 0: raise ValueError("p90>median 이어야 합니다.")
    return mu, sigma

def ziln_pdf(x, p0, median, p90):
    a, keep = _to_arr(x)
    mu, sg = _ziln_mu_sigma(median, p90)
    out = np.zeros_like(a, dtype=float)
    pos = a > 0
    if np.any(pos):
        z = (np.log(a[pos]) - mu) / sg
        # log-pdf(안정화)
        logpdf = -np.log(a[pos]) - np.log(sg) - 0.5*np.log(2*np.pi) - 0.5*z*z
        out[pos] = (1.0 - p0) * np.exp(logpdf)
    # a==0은 점질량(p0)이고 연속 pdf는 0
    return _restore(out, keep)

def ziln_cdf_mid(x, p0, median, p90):
    a, keep = _to_arr(x)
    mu, sg = _ziln_mu_sigma(median, p90)
    out = np.zeros_like(a, dtype=float)
    out[a < 0] = 0.0
    at0 = (a == 0)
    out[at0] = 0.5 * p0
    gt0 = a > 0
    if np.any(gt0):
        z = (np.log(a[gt0]) - mu) / sg
        # Φ(z) = 0.5*(1+erf(z/sqrt(2)))
        cdf_ln = 0.5*(1.0 + np.erf(z/np.sqrt(2.0)))
        out[gt0] = p0 + (1.0 - p0) * cdf_ln
    return _restore(out, keep)

# ---------- ZIP(p0, mean) ----------
# 주어진 mean=m과 P(X=0)=p0에서 λ, ω(구조적 zero 비율) 해결:
# m = (1-ω)λ, p0 = ω + (1-ω) e^{-λ} -> λ는 f(λ)=1 - (m/λ)(1-e^{-λ}) - p0 = 0의 해
def _zip_lambda_omega(p0, m, tol=1e-10, max_iter=200):
    if m <= 0: return 0.0, 1.0  # 평균 0이면 모두 0
    # 실현 가능성 체크: p0 >= e^{-m} (ω=0일 때 최소 zero)
    if p0 < np.exp(-m) - 1e-12:
        raise ValueError("주어진 (p0, mean) 조합은 ZIP로 불가(p0<exp(-mean)).")
    # 특수해: ω=0 가능
    if abs(p0 - np.exp(-m)) <= 1e-12:
        return m, 0.0
    # 이분법
    f = lambda lam: 1.0 - (m/lam)*(1.0 - np.exp(-lam)) - p0
    lo, hi = 1e-12, max(10.0, m*20.0)
    flo, fhi = f(lo), f(hi)
    # 단조는 아니지만 보통 부호차 존재. 없으면 hi를 키움
    grow = 0
    while flo*fhi > 0 and grow < 20:
        hi *= 2.0
        fhi = f(hi); grow += 1
    if flo*fhi > 0:
        # 수치적으로 곤란하면 뉴턴 보조
        lam = max(m, 1.0)
        for _ in range(50):
            e = np.exp(-lam)
            g = 1.0 - (m/lam)*(1.0 - e) - p0
            dg = m*((1.0 - e)/lam**2 - e/lam)
            step = g/dg
            lam -= step
            if lam <= 1e-9: lam = 1e-9
            if abs(step) < 1e-10: break
        lam = float(lam)
    else:
        # 표준 이분법
        for _ in range(max_iter):
            mid = 0.5*(lo+hi)
            fm = f(mid)
            if abs(fm) < tol or (hi-lo) < 1e-10:
                lam = float(mid); break
            if flo*fm <= 0: hi, fhi = mid, fm
            else: lo, flo = mid, fm
        else:
            lam = float(0.5*(lo+hi))
    omega = 1.0 - m/lam
    omega = min(max(omega, 0.0), 1.0)
    return lam, omega

def _poisson_logpmf(k, lam):
    # log pmf = -lam + k*log(lam) - lgamma(k+1)
    logfac = np.vectorize(math.lgamma)(k + 1.0)
    return -lam + k*np.log(lam) - logfac

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
    # 정수/비정수 분기
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
            # mid-CDF
            # pmf(xi)
            pmf_x = (1.0 - omega) * np.exp(_poisson_logpmf(float(kf), lam))
            if kf == 0: pmf_x += omega
            out[i] = (cdf_floor - pmf_x) + 0.5*pmf_x
        else:
            out[i] = cdf_floor
    return _restore(out, keep)