import numpy as np
import math

# ---------- helpers ----------
def _to_array_and_keeper(x):
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False), ("ndarray", None)
    if isinstance(x, (np.floating, float, int)):
        return np.asarray([x], dtype=float), ("scalar", type(x))
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float), ("list", type(x))
    # np.float32/64 등
    if isinstance(x, (np.float32, np.float64, np.int32, np.int64)):
        return np.asarray([float(x)]), ("scalar", type(x))
    # fallback
    arr = np.asarray(x, dtype=float)
    return arr, ("ndarray", None)

def _restore_type(arr_out, keeper):
    kind, tp = keeper
    if kind == "scalar":
        v = float(arr_out[0])
        # 입력이 int였다면 int로 강제하지는 않음(확률/누적이라 실수 유지)
        return v
    if kind == "list":
        return arr_out.tolist()
    return arr_out

def _is_integer_array(x, tol=1e-12):
    return np.abs(x - np.round(x)) <= tol

def _safe_log1p(x):
    # 큰 값 안정화용: log(1+x)
    return np.log1p(x)

def _normal_cdf(z):
    # Φ(z) = 0.5 * (1 + erf(z/sqrt(2)))
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))

# ---------- ZILN: Zero-Inflated LogNormal ----------
# params: pi in [0,1] (zero inflation), mu, sigma>0 (lognormal)
def ziln_pdf(x, pi, mu, sigma):
    arr, keep = _to_array_and_keeper(x)
    out = np.zeros_like(arr, dtype=float)

    pos = arr > 0
    if np.any(pos):
        z = (np.log(arr[pos]) - mu) / sigma
        # log-pdf (안정화)
        log_pdf = -np.log(arr[pos]) - np.log(sigma) - 0.5*np.log(2*np.pi) - 0.5*(z**2)
        out[pos] = (1.0 - pi) * np.exp(log_pdf)

    at0 = arr == 0
    # 연속분포의 밀도는 0에서 0이나, ZI의 점질량은 확률질량이므로 pdf 개념상 0(연속)과 구분됩니다.
    # 필요시 "point-mass" 확인용으로 별도 반환이 필요하나, 여기서는 pdf 배열만 반환(0 유지).
    # pmf 형태가 필요한 경우는 ZIP/ZINB를 사용하세요.

    return _restore_type(out, keep)

def ziln_cdf_mid(x, pi, mu, sigma):
    arr, keep = _to_array_and_keeper(x)
    out = np.zeros_like(arr, dtype=float)

    # x < 0 : 0
    lt0 = arr < 0
    out[lt0] = 0.0

    # x == 0 : mid 적용 -> pi/2
    at0 = arr == 0
    out[at0] = 0.5 * pi

    # x > 0 : F = pi + (1-pi)*Φ((ln x - mu)/sigma)
    gt0 = arr > 0
    if np.any(gt0):
        z = (np.log(arr[gt0]) - mu) / sigma
        out[gt0] = pi + (1.0 - pi) * _normal_cdf(z)

    return _restore_type(out, keep)

# ---------- ZIP: Zero-Inflated Poisson ----------
# params: omega in [0,1], lam>0
def _poisson_logpmf(k, lam):
    # log P = -lam + k*log(lam) - lgamma(k+1)
    # k, lam는 브로드캐스트 가능해야 함
    k = np.asarray(k)
    lam = np.asarray(lam)
    logfac = np.vectorize(math.lgamma)(k + 1.0)
    return -lam + k * np.log(lam) - logfac

def zip_pdf(x, omega, lam):
    arr, keep = _to_array_and_keeper(x)
    out = np.zeros_like(arr, dtype=float)

    # 비정수는 0
    is_int = _is_integer_array(arr) & (arr >= 0)
    k = np.round(arr[is_int]).astype(int)

    if np.any(is_int):
        logpmf = _poisson_logpmf(k, lam)
        base_pmf = np.exp(logpmf)
        pmf = (1.0 - omega) * base_pmf
        # k==0에 zero-inflation 질량 더하기
        pmf[k == 0] += omega
        out[is_int] = pmf

    return _restore_type(out, keep)

def zip_cdf_mid(x, omega, lam, kmax=None):
    arr, keep = _to_array_and_keeper(x)
    out = np.zeros_like(arr, dtype=float)

    # 음수 -> 0
    out[arr < 0] = 0.0

    # 누적 합산을 위해 floor(x)까지 pmf 합
    # 성능 이슈 방지를 위해 상한 kmax 제공 가능(없으면 floor(x) 사용)
    xf = np.floor(arr).astype(int)
    n = arr.size

    for i in range(n):
        xi = arr[i]
        if xi < 0:
            continue
        k_floor = int(xf[i]) if kmax is None else min(int(xf[i]), int(kmax))
        ks = np.arange(0, k_floor + 1, dtype=int)
        logpmf = _poisson_logpmf(ks, lam)
        pmf = (1.0 - omega) * np.exp(logpmf)
        if ks.size > 0 and ks[0] == 0:
            pmf[0] += omega
        cdf_at_floor = pmf.sum()

        # mid: 정수면 0.5*pmf(k), 아니면 그대로
        if _is_integer_array(np.asarray([xi]))[0]:
            pmf_x = zip_pdf(xi, omega, lam)
            out[i] = (cdf_at_floor - pmf_x) + 0.5 * pmf_x
        else:
            out[i] = cdf_at_floor

    return _restore_type(out, keep)

# ---------- ZINB: Zero-Inflated Negative Binomial ----------
# 파라미터화: r>0, p in (0,1)
# pmf(k) = C(k+r-1, k) * (1-p)^r * p^k  (k=0,1,2,...)
def _nb_logpmf(k, r, p):
    # log C = lgamma(k+r) - lgamma(r) - lgamma(k+1)
    k = np.asarray(k, dtype=float)
    r = float(r)
    p = float(p)
    logC = np.vectorize(math.lgamma)(k + r) - math.lgamma(r) - np.vectorize(math.lgamma)(k + 1.0)
    return logC + r * np.log(1.0 - p) + k * np.log(p)

def zinb_pdf(x, omega, r, p):
    arr, keep = _to_array_and_keeper(x)
    out = np.zeros_like(arr, dtype=float)

    is_int = _is_integer_array(arr) & (arr >= 0)
    k = np.round(arr[is_int]).astype(int)

    if np.any(is_int):
        logpmf = _nb_logpmf(k, r, p)
        base_pmf = np.exp(logpmf)
        pmf = (1.0 - omega) * base_pmf
        pmf[k == 0] += omega
        out[is_int] = pmf

    return _restore_type(out, keep)

def zinb_cdf_mid(x, omega, r, p, kmax=None):
    arr, keep = _to_array_and_keeper(x)
    out = np.zeros_like(arr, dtype=float)

    out[arr < 0] = 0.0
    xf = np.floor(arr).astype(int)
    n = arr.size

    for i in range(n):
        xi = arr[i]
        if xi < 0:
            continue
        k_floor = int(xf[i]) if kmax is None else min(int(xf[i]), int(kmax))
        ks = np.arange(0, k_floor + 1, dtype=int)
        logpmf = _nb_logpmf(ks, r, p)
        pmf = (1.0 - omega) * np.exp(logpmf)
        if ks.size > 0 and ks[0] == 0:
            pmf[0] += omega
        cdf_at_floor = pmf.sum()

        if _is_integer_array(np.asarray([xi]))[0]:
            pmf_x = zinb_pdf(xi, omega, r, p)
            out[i] = (cdf_at_floor - pmf_x) + 0.5 * pmf_x
        else:
            out[i] = cdf_at_floor

    return _restore_type(out, keep)