# pip install distfit scipy numpy matplotlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
from distfit import distfit
from scipy import stats
import matplotlib.pyplot as plt


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
    distfit 기반 자동화 셀렉터
    - mode="ZI": Zero-Inflated (0에 점질량 + 원분포)
    - mode="HURDLE": 허들 (0은 0, 양수영역 확률을 1-p0로 재정규화)
    """
    def __init__(self,
                 candidates: Optional[List[str]] = None,
                 mode: str = "ZI",
                 use_bic: bool = False,
                 ks_on_positive: bool = True):
        self.candidates = candidates or ['lognorm', 'gamma', 'weibull_min', 'fisk']  # fisk == loglogistic
        self.mode = mode.upper()
        assert self.mode in {"ZI", "HURDLE"}
        self.use_bic = use_bic
        self.ks_on_positive = ks_on_positive
        self._positive_fit = None
        self.best_: Optional[FitResult] = None
        self.details_: List[FitResult] = []

    @staticmethod
    def _safe_log(x, eps=1e-12):
        return np.log(np.maximum(x, eps))

    def _loglik(self, x, p0, dist_obj, arg, loc, scale) -> float:
        x = np.asarray(x)
        n0 = np.sum(x == 0)
        xp = x[x > 0]
        # pdf/cdf on positives
        pdf_pos = dist_obj.pdf(xp, *arg, loc=loc, scale=scale)
        if self.mode == "ZI":
            # f(x) = p0 * 1{x=0} + (1-p0) * f_plus(x) for x>0
            ll = n0 * self._safe_log(p0).sum() + np.sum(self._safe_log((1 - p0) * pdf_pos))
        else:
            # HURDLE: f(x) = p0 for x=0 ; (1-p0) * f_plus(x) / (1-F_plus(0)) for x>0
            cdf0 = dist_obj.cdf(0, *arg, loc=loc, scale=scale)
            denom = 1 - cdf0
            pdf_pos_norm = pdf_pos / np.maximum(denom, 1e-12)
            ll = n0 * self._safe_log(p0).sum() + np.sum(self._safe_log((1 - p0) * pdf_pos_norm))
        return ll

    @staticmethod
    def _num_params(arg_tuple) -> int:
        # k = 1 (p0) + len(arg) + 2(loc, scale)
        return 1 + len(arg_tuple) + 2

    def fit(self, x: np.ndarray) -> "ZIMixtureSelector":
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n == 0:
            raise ValueError("Empty data.")

        # 1) p0 추정 (필요하면 EM/MLE로 공동최적화 확장 가능)
        p0 = float(np.mean(x == 0.0))

        # 2) 양수 부분 distfit
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

        d = distfit(distr=self.candidates, method='parametric')
        d.fit_transform(xp)  # RSS 기반 피팅/랭킹

        results: List[FitResult] = []
        for rec in d.model:  # 각 후보 분포 결과
            name = rec['name']
            pars = rec['params']  # {'name':..., 'arg': tuple, 'loc': float, 'scale': float}
            arg, loc, scale = pars['arg'], pars['loc'], pars['scale']
            dist_obj = getattr(stats, name)

            # 3) 혼합 로그우도 계산
            ll = self._loglik(x, p0=p0, dist_obj=dist_obj, arg=arg, loc=loc, scale=scale)

            # 4) AIC/BIC
            k = self._num_params(arg)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll

            # 5) (옵션) 양수구간 KS 검정
            if self.ks_on_positive:
                # distfit로 추정된 분포 모수 기반 CDF 사용
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

        # 6) 최종 선택
        key = (lambda r: r.bic) if self.use_bic else (lambda r: r.aic)
        best = min(results, key=key)
        self.best_ = best
        self.details_ = sorted(results, key=key)
        self._positive_fit = d  # 저장(원하면 시각화에 사용)
        return self

    def summary(self) -> Dict[str, Any]:
        if self.best_ is None:
            raise RuntimeError("Call .fit(x) first.")
        return {
            'best': self.best_.to_dict(),
            'top5': [r.to_dict() for r in self.details_[:5]]
        }

    def pdf(self, x_grid: np.ndarray) -> np.ndarray:
        """선택된 혼합모형의 전체 pdf (x>0에서만 연속, x=0은 점질량)"""
        if self.best_ is None or self.best_.name == "pointmass_at_zero":
            raise RuntimeError("Model not fitted or degenerate.")
        p0 = self.best_.p0
        arg, loc, scale = self.best_.params['arg'], self.best_.params['loc'], self.best_.params['scale']
        base = getattr(stats, self.best_.name.split('-', 1)[1])
        pdf = np.zeros_like(x_grid, dtype=float)
        pos_mask = x_grid > 0
        base_pdf = base.pdf(x_grid[pos_mask], *arg, loc=loc, scale=scale)
        if self.mode == "ZI":
            pdf[pos_mask] = (1 - p0) * base_pdf
        else:
            cdf0 = base.cdf(0, *arg, loc=loc, scale=scale)
            pdf[pos_mask] = (1 - p0) * base_pdf / max(1 - cdf0, 1e-12)
        # (참고) x=0의 점질량은 pdf 그래프에선 표시하지 않음
        return pdf

    def plot_fit(self, x: np.ndarray, bins: int = 50):
        """히스토그램(전체 데이터) + 양수구간 혼합 pdf 오버레이 + 0점질량 주석"""
        if self.best_ is None:
            raise RuntimeError("Call .fit(x) first.")
        x = np.asarray(x, dtype=float)
        p0 = self.best_.p0

        # 히스토그램
        plt.figure(figsize=(8, 4.5))
        counts, edges, _ = plt.hist(x[x > 0], bins=bins, density=True, alpha=0.3, label='data (x>0)')

        # pdf 오버레이 (양수영역)
        grid = np.linspace(max(1e-9, np.min(x[x > 0])), np.percentile(x[x > 0], 99.5), 400)
        pdf = self.pdf(grid)
        plt.plot(grid, pdf, lw=2, label=f'{self.best_.name} pdf (x>0)')

        # 0점질량(주석)
        plt.axvline(0, ymin=0, ymax=0.15, linestyle='--', linewidth=2)
        plt.text(edges[0] if len(edges) else 0, max(counts) * 0.15 if len(counts) else 0.1,
                 f"Point mass at 0: p0≈{p0:.3f}", va='bottom', ha='left')

        plt.title(f"Best: {self.best_.name} | AIC={self.best_.aic:.1f} | BIC={self.best_.bic:.1f}")
        plt.legend()
        plt.tight_layout()
        plt.show()


# -----------------------------
# 사용 예시
# -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 예시 데이터: ZI-Lognormal (p0=0.35, lognormal(mu=0.5, sigma=0.7))
    n = 5000
    true_p0 = 0.35
    zeros = rng.random(n) < true_p0
    pos = rng.lognormal(mean=0.5, sigma=0.7, size=n)
    data = np.where(zeros, 0.0, pos)

    # 1) 제로팽창(ZI) 모드
    zi_selector = ZIMixtureSelector(mode="ZI", use_bic=False,
                                    candidates=['lognorm', 'gamma', 'weibull_min', 'fisk'])
    zi_selector.fit(data)
    print("ZI summary:", zi_selector.summary())

    # 2) 허들(HURDLE) 모드
    hurdle_selector = ZIMixtureSelector(mode="HURDLE", use_bic=False,
                                        candidates=['lognorm', 'gamma', 'weibull_min', 'fisk'])
    hurdle_selector.fit(data)
    print("HURDLE summary:", hurdle_selector.summary())

    # 필요시 플롯(양수영역 pdf + 0 점질량 주석)
    # zi_selector.plot_fit(data)
    # hurdle_selector.plot_fit(data)