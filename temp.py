# pip install distfit scipy numpy matplotlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from distfit import distfit
from scipy import stats
import matplotlib.pyplot as plt


# --- 비음수 지지(>=0) 분포만: Commons Math와 1:1 대응되는 것들 ---
NONNEG_SUPPORTED = [
    'lognorm',       # LogNormalDistribution
    'gamma',         # GammaDistribution
    'weibull_min',   # WeibullDistribution
    'expon',         # ExponentialDistribution
    'chi2',          # ChiSquaredDistribution
    'f',             # FDistribution
    'pareto',        # ParetoDistribution
]
# (선택) 스케일링이 필요한 분포들
BETA_UNIFORM = ['beta', 'uniform']  # BetaDistribution([0,1]), UniformRealDistribution([a,b])


@dataclass
class FitResult:
    name: str
    mode: str                 # "ZI" or "HURDLE"
    p0: float
    params: Dict[str, Any]    # {'arg': tuple, 'loc': float, 'scale': float}
    ll: float
    aic: float
    bic: float
    n: int
    n_zero: int
    n_pos: int
    ks_stat: Optional[float] = None
    ks_pvalue: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class ZIMixtureSelector:
    """
    distfit 기반 자동화 셀렉터 (입력 데이터 ≥ 0 전제)
    - mode="ZI": Zero-Inflated (x=0 점질량 + 양수부 분포)
    - mode="HURDLE": 허들 (0은 0, 양수영역 CDF 재정규화)

    추가:
      - cdf(x), pdf(x) 구현
      - plot_fit_pdf(): 양수부 히스토그램 + 혼합 pdf + x=0 점질량 표시
      - plot_fit_cdf(): 누적 히스토그램 + 혼합 cdf

    분포 후보:
      기본은 NONNEG_SUPPORTED (Commons Math와 매핑되는 비음수 지지 분포).
      beta/uniform은 데이터 스케일링이 필요한 케이스이므로 기본 제외(옵션으로 포함 가능).
    """
    def __init__(self,
                 candidates: Optional[List[str]] = None,
                 mode: str = "ZI",
                 use_bic: bool = False,
                 ks_on_positive: bool = True,
                 allow_beta_uniform: bool = False,
                 beta_uniform_strategy: str = "auto"):  # "auto"|"skip"
        if candidates is None:
            self.candidates = NONNEG_SUPPORTED.copy()
            if allow_beta_uniform:
                self.candidates += BETA_UNIFORM
        else:
            # 사용자가 준 후보 중 비음수 지지 위주 사용 권장
            self.candidates = candidates

        self.mode = mode.upper()
        assert self.mode in {"ZI", "HURDLE"}
        self.use_bic = use_bic
        self.ks_on_positive = ks_on_positive

        # beta/uniform 사용 시 데이터 스케일링 전략
        self.allow_beta_uniform = allow_beta_uniform
        self.beta_uniform_strategy = beta_uniform_strategy

        self._positive_fit = None
        self._positive_df = None
        self.best_: Optional[FitResult] = None
        self.details_: List[FitResult] = []
        self._x_minmax: Optional[Tuple[float, float]] = None  # (min_pos, max_pos) for scaling if needed

    @staticmethod
    def _safe_log(x, eps=1e-12):
        return np.log(np.maximum(x, eps))

    def _assert_nonnegative(self, x: np.ndarray):
        if np.any(x < 0):
            neg_max = float(np.min(x))
            raise ValueError(f"All inputs must be >= 0. Found min={neg_max}")

    @staticmethod
    def _num_params(arg_tuple) -> int:
        # k = 1 (p0) + len(arg) + 2(loc, scale)
        return 1 + len(arg_tuple) + 2

    def _loglik(self, x, p0, dist_obj, arg, loc, scale) -> float:
        x = np.asarray(x)
        n0 = np.sum(x == 0)
        xp = x[x > 0]

        # pdf on positives (분포 loc/scale로 음수 쪽 support가 열리지 않도록 distfit가 알아서 맞춤)
        pdf_pos = dist_obj.pdf(xp, *arg, loc=loc, scale=scale)

        if self.mode == "ZI":
            # f(x) = p0 * 1{x=0} + (1-p0) * f_plus(x) for x>0
            ll = n0 * self._safe_log(p0).sum() + np.sum(self._safe_log((1 - p0) * pdf_pos))
        else:
            # HURDLE: f(x) = p0 for x=0 ; (1-p0) * f_plus(x) / (1-F_plus(0)) for x>0
            cdf0 = dist_obj.cdf(0.0, *arg, loc=loc, scale=scale)
            denom = 1 - cdf0
            pdf_pos_norm = pdf_pos / np.maximum(denom, 1e-12)
            ll = n0 * self._safe_log(p0).sum() + np.sum(self._safe_log((1 - p0) * pdf_pos_norm))
        return ll

    def _maybe_scale_for_beta_uniform(self, xp: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
        """
        beta/uniform은 기본적으로 [0,1] 또는 [a,b] 범위를 가정.
        allow_beta_uniform=True이고 strategy="auto"인 경우, 양수 데이터 xp를 [0,1]로 선형 스케일링하여
        distfit에 던지고, 모수 추정 후 실제 스케일로 역변환은 생략(선택)합니다.
        """
        if not self.allow_beta_uniform or self.beta_uniform_strategy != "auto":
            return xp, None
        x_min = float(np.min(xp))
        x_max = float(np.max(xp))
        if x_max <= 0 or x_max == x_min:
            # 상수/퇴화 상태면 스케일링 의미 없음
            return xp, None
        # [0,1] 스케일로 매핑
        xp_scaled = (xp - x_min) / (x_max - x_min)
        self._x_minmax = (x_min, x_max)
        return xp_scaled, (x_min, x_max)

    def fit(self, x: np.ndarray) -> "ZIMixtureSelector":
        x = np.asarray(x, dtype=float)
        self._assert_nonnegative(x)
        n = len(x)
        if n == 0:
            raise ValueError("Empty data.")

        # 1) p0 추정
        p0 = float(np.mean(x == 0.0))

        # 2) 양수 부분
        xp = x[x > 0]
        if len(xp) == 0:
            # 전부 0인 경우
            self.best_ = FitResult(
                name="pointmass_at_zero",
                mode=self.mode,
                p0=1.0,
                params={},
                ll=0.0,
                aic=np.inf,
                bic=np.inf,
                n=n, n_zero=n, n_pos=0,
                ks_stat=None, ks_pvalue=None
            )
            self.details_ = [self.best_]
            return self

        # beta/uniform 고려: 필요시 [0,1] 스케일
        xp_for_fit, minmax = self._maybe_scale_for_beta_uniform(xp)

        # distfit: 비음수 지지 분포들만 후보로
        d = distfit(todf=True, distr=self.candidates, method='parametric')
        d.fit_transform(xp_for_fit)  # RSS 기반 피팅/랭킹
        df = d.summary.sort_values('score')  # 모든 후보

        results: List[FitResult] = []
        for _, row in df.iterrows():
            # distfit 버전에 따라 컬럼명이 다를 수 있어 방어적 접근
            name = row.get('name', row.get('distr', row.get('distribution')))
            arg  = tuple(row.get('arg', ()) or ())
            loc  = float(row.get('loc', 0.0))
            scale= float(row.get('scale', 1.0))

            # 베타/유니폼 스케일링을 썼다면, 여기서 실제 좌표계로 역해석이 필요할 수 있음
            # 단, 혼합 로그우도 계산에서 dist.pdf/ cdf는 xp(원좌표) 기준이어야 하므로
            # 베타/유니폼을 쓰는 경우엔 현재 템플릿에서는 "평가 대상에서 제외"하거나,
            # 별도의 스케일-변환 cdf/pdf를 정의해주는 편이 안전.
            if (name in BETA_UNIFORM) and (minmax is not None):
                # 안전하게 이번 랭킹에선 스킵 (필요하면 스케일 역변환 로직을 추가하세요)
                continue

            dist_obj = getattr(stats, name)

            # 3) 혼합 로그우도 계산 (원좌표 xp 기준)
            ll = self._loglik(x, p0=p0, dist_obj=dist_obj, arg=arg, loc=loc, scale=scale)

            # 4) AIC/BIC
            k = self._num_params(arg)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll

            # 5) (옵션) 양수구간 KS 검정
            if self.ks_on_positive:
                ks_stat, ks_p = stats.kstest(xp, name, args=(*arg, loc, scale))
            else:
                ks_stat, ks_p = None, None

            results.append(FitResult(
                name=f'{self.mode}-{name}',
                mode=self.mode,
                p0=p0,
                params={'arg': arg, 'loc': loc, 'scale': scale},
                ll=float(ll), aic=float(aic), bic=float(bic),
                n=n, n_zero=int(np.sum(x == 0)), n_pos=len(xp),
                ks_stat=ks_stat, ks_pvalue=ks_p
            ))

        if not results:
            raise RuntimeError(
                "No candidate produced a valid result. "
                "If you enabled beta/uniform with scaling, add inverse-scaling logic for pdf/cdf."
            )

        # 6) 최종 선택
        key = (lambda r: r.bic) if self.use_bic else (lambda r: r.aic)
        best = min(results, key=key)
        self.best_ = best
        self.details_ = sorted(results, key=key)
        self._positive_fit = d
        self._positive_df = df
        return self

    def summary(self) -> Dict[str, Any]:
        if self.best_ is None:
            raise RuntimeError("Call .fit(x) first.")
        return {
            'best': self.best_.to_dict(),
            'top5': [r.to_dict() for r in self.details_[:5]]
        }

    # ---------- PDF & CDF ----------
    def pdf(self, x_grid: np.ndarray) -> np.ndarray:
        """선택된 혼합모형의 pdf (x>0에서만 연속; x=0은 점질량이므로 pdf로 표시되지 않음)"""
        if self.best_ is None or self.best_.name == "pointmass_at_zero":
            raise RuntimeError("Model not fitted or degenerate.")
        p0 = self.best_.p0
        arg, loc, scale = self.best_.params['arg'], self.best_.params['loc'], self.best_.params['scale']
        base = getattr(stats, self.best_.name.split('-', 1)[1])

        x_grid = np.asarray(x_grid, dtype=float)
        if np.any(x_grid < 0):
            raise ValueError("pdf() expects x_grid >= 0 for this nonnegative-support setup.")

        pdf = np.zeros_like(x_grid, dtype=float)
        pos_mask = x_grid > 0
        base_pdf = base.pdf(x_grid[pos_mask], *arg, loc=loc, scale=scale)
        if self.mode == "ZI":
            pdf[pos_mask] = (1 - p0) * base_pdf
        else:
            cdf0 = base.cdf(0.0, *arg, loc=loc, scale=scale)
            pdf[pos_mask] = (1 - p0) * base_pdf / max(1 - cdf0, 1e-12)
        return pdf

    def cdf(self, x_grid: np.ndarray) -> np.ndarray:
        """선택된 혼합모형의 CDF (x<=0에서는 p0 점프 포함). 입력은 0 이상을 권장."""
        if self.best_ is None or self.best_.name == "pointmass_at_zero":
            raise RuntimeError("Model not fitted or degenerate.")
        p0 = self.best_.p0
        arg, loc, scale = self.best_.params['arg'], self.best_.params['loc'], self.best_.params['scale']
        base = getattr(stats, self.best_.name.split('-', 1)[1])

        x_grid = np.asarray(x_grid, dtype=float)
        F = np.zeros_like(x_grid, dtype=float)

        nonpos = x_grid <= 0
        F[nonpos] = np.where(x_grid[nonpos] < 0, 0.0, p0)  # x<0이면 0, x=0이면 p0

        pos = x_grid > 0
        base_cdf = base.cdf(x_grid[pos], *arg, loc=loc, scale=scale)
        if self.mode == "ZI":
            F[pos] = p0 + (1 - p0) * base_cdf
        else:
            cdf0 = base.cdf(0.0, *arg, loc=loc, scale=scale)
            denom = max(1 - cdf0, 1e-12)
            F[pos] = p0 + (1 - p0) * (base_cdf - cdf0) / denom
        return F

    # ---------- Plotting ----------
    def plot_fit_pdf(self, x: np.ndarray, bins: int = 50, show_hist: bool = True):
        """
        pdf 플롯:
          - (옵션) 양수 데이터 히스토그램 density=True
          - 혼합 pdf 오버레이
          - x=0 점질량을 zero-bin 폭으로 스케일해 높이 p0/width로 스템 표시(면적≈p0)
        """
        if self.best_ is None:
            raise RuntimeError("Call .fit(x) first.")
        x = np.asarray(x, dtype=float)
        self._assert_nonnegative(x)

        p0 = self.best_.p0
        xp = x[x > 0]
        if len(xp) == 0:
            plt.figure(figsize=(8, 4.5))
            plt.title("All mass at zero (p0≈1.0)")
            plt.axvline(0, linestyle='--')
            plt.show()
            return

        plt.figure(figsize=(8, 4.5))

        if show_hist:
            counts, edges, _ = plt.hist(xp, bins=bins, density=True, alpha=0.25, label='data (x>0) density')
        else:
            counts, edges = np.histogram(xp, bins=bins, density=True)

        # 혼합 pdf 오버레이
        grid = np.linspace(max(1e-12, xp.min()), np.percentile(xp, 99.5), 600)
        pdf_vals = self.pdf(grid)
        plt.plot(grid, pdf_vals, lw=2, label=f'{self.best_.name} pdf (x>0)')

        # x=0 점질량
        zero_bin_width = edges[0] if edges[0] > 0 else (edges[1] - edges[0])
        spike_height = p0 / max(zero_bin_width, 1e-12)
        plt.vlines(0.0, 0.0, spike_height, colors='r', linestyles='-', linewidth=2,
                   label=f'point mass at 0 (area≈p0={p0:.3f})')

        plt.xlim(left=0)
        plt.title(f"Best: {self.best_.name} | AIC={self.best_.aic:.1f} | BIC={self.best_.bic:.1f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_fit_cdf(self, x: np.ndarray, bins: int = 100):
        """
        cdf 플롯:
          - 전체 데이터 누적 히스토그램 (density=True, cumulative=True, alpha=0.5)
          - 적합 CDF 곡선
        """
        if self.best_ is None:
            raise RuntimeError("Call .fit(x) first.")
        x = np.asarray(x, dtype=float)
        self._assert_nonnegative(x)

        plt.figure(figsize=(8, 4.5))
        upper = np.percentile(x[x > 0], 99.5) if np.any(x > 0) else 1.0
        bins_edges = np.linspace(0.0, upper, bins)
        plt.hist(x, bins=bins_edges, density=True, cumulative=True, alpha=0.5, label='empirical CDF')

        grid = np.linspace(bins_edges[0], bins_edges[-1], 800)
        cdf_vals = self.cdf(grid)
        plt.plot(grid, cdf_vals, lw=2, label=f'{self.best_.name} CDF')

        plt.axvline(0.0, linestyle='--', linewidth=1)
        plt.xlim(left=0)
        plt.ylim(0, 1.0)
        plt.title(f"Empirical vs Fitted CDF | p0≈{self.best_.p0:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()


# -----------------------------
# 사용 예시
# -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 4000
    true_p0 = 0.3
    zeros = rng.random(n) < true_p0
    pos = rng.gamma(shape=2.0, scale=1.0, size=n)  # (≥0)
    data = np.where(zeros, 0.0, pos)

    selector = ZIMixtureSelector(
        mode="ZI",
        candidates=None,              # 기본 NONNEG_SUPPORTED 사용
        use_bic=False,
        ks_on_positive=True,
        allow_beta_uniform=False      # 필요시 True로 두고 스케일링 로직 보강
    )
    selector.fit(data)
    print(selector.summary())

    # selector.plot_fit_pdf(data, bins=60, show_hist=True)
    # selector.plot_fit_cdf(data, bins=80)