"""Otimização de pesos com scipy SLSQP: long-only, soma = 1, bounds por ativo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

import portfolio as P
import universe as U

_CONS_SUM = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
_OPTIONS = {"maxiter": U.OPT_MAX_ITER, "ftol": U.OPT_FTOL}


@dataclass
class OptResult:
    weights: np.ndarray
    success: bool
    message: str


def make_bounds(n: int, w_min: float, w_max: float) -> list[tuple[float, float]]:
    return [(w_min, w_max)] * n


def check_feasible(bounds) -> str | None:
    lo = sum(b[0] for b in bounds)
    hi = sum(b[1] for b in bounds)
    if lo > 1 + 1e-9 or hi < 1 - 1e-9:
        return (f"Bounds inviáveis para {len(bounds)} ativos: soma dos mínimos = {lo:.0%}, "
                f"soma dos máximos = {hi:.0%} (a soma dos pesos precisa ser 100%)")
    return None


def _start(bounds) -> np.ndarray:
    """Pesos iguais, projetados nos bounds quando necessário."""
    n = len(bounds)
    w = np.full(n, 1 / n)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    for _ in range(100):
        w = np.clip(w, lo, hi)
        gap = 1 - w.sum()
        if abs(gap) < 1e-12:
            break
        free = (w < hi - 1e-12) if gap > 0 else (w > lo + 1e-12)
        if not free.any():
            break
        w[free] += gap / free.sum()
    return w


def _run(fun, bounds, extra_cons=()) -> OptResult:
    err = check_feasible(bounds)
    if err:
        return OptResult(np.full(len(bounds), np.nan), False, err)
    res = minimize(fun, _start(bounds), method="SLSQP", bounds=bounds,
                   constraints=(_CONS_SUM, *extra_cons), options=_OPTIONS)
    if not res.success:
        return OptResult(np.full(len(bounds), np.nan), False, str(res.message))
    w = np.clip(res.x, 0, None)
    return OptResult(w / w.sum(), True, str(res.message))


def max_sharpe(mu, cov, rf: float, bounds) -> OptResult:
    mu, cov = np.asarray(mu), np.asarray(cov)

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 0.0))
        return -(w @ mu - rf) / vol if vol > 0 else 0.0

    return _run(neg_sharpe, bounds)


def min_vol(cov, bounds) -> OptResult:
    cov = np.asarray(cov)
    return _run(lambda w: w @ cov @ w, bounds)  # minimizar variância ≡ minimizar σ_p (mais estável)


def max_sortino(returns: pd.DataFrame, periods: int, rf: float, bounds) -> OptResult:
    R = returns.to_numpy()
    mu = R.mean(axis=0) * periods

    def neg_sortino(w):
        rp = R @ w
        down = np.sqrt(np.mean(np.minimum(rp, 0.0) ** 2)) * np.sqrt(periods)
        return -(w @ mu - rf) / down if down > 0 else 0.0

    return _run(neg_sortino, bounds)


def risk_parity(cov, bounds) -> OptResult:
    cov = np.asarray(cov)
    n = len(cov)

    def objective(w):
        rc = P.risk_contribution(w, cov)
        return float(np.nansum((rc - 1 / n) ** 2)) * 1e3

    return _run(objective, bounds)


def equal_weights(bounds) -> OptResult:
    err = check_feasible(bounds)
    if err:
        return OptResult(np.full(len(bounds), np.nan), False, err)
    w = _start(bounds)
    return OptResult(w, True, "Pesos iguais" + ("" if np.allclose(w, w[0]) else " (ajustados aos bounds)"))


def optimize(objective: str, mu, cov, returns: pd.DataFrame, periods: int, rf: float, bounds) -> OptResult:
    if objective == "Max Sharpe":
        return max_sharpe(mu, cov, rf, bounds)
    if objective == "Min Vol":
        return min_vol(cov, bounds)
    if objective == "Max Sortino":
        return max_sortino(returns, periods, rf, bounds)
    if objective == "Risk Parity":
        return risk_parity(cov, bounds)
    if objective == "Igual":
        return equal_weights(bounds)
    raise ValueError(f"Objetivo desconhecido: {objective}")


def _return_range(mu, bounds) -> tuple[float, float]:
    """Menor e maior retorno atingíveis sob soma = 1 e bounds (programação linear)."""
    mu = np.asarray(mu)
    n = len(mu)
    kw = dict(A_eq=np.ones((1, n)), b_eq=[1.0], bounds=bounds, method="highs")
    lo = linprog(mu, **kw)
    hi = linprog(-mu, **kw)
    if not (lo.success and hi.success):
        return float(mu.min()), float(mu.max())
    return float(mu @ lo.x), float(mu @ hi.x)


def efficient_frontier(mu, cov, bounds, k: int) -> pd.DataFrame:
    """Para K alvos de retorno, minimiza σ_p com restrição μ_p = alvo."""
    mu, cov = np.asarray(mu), np.asarray(cov)
    if check_feasible(bounds):
        return pd.DataFrame(columns=["Retorno", "Vol"])
    r_min, r_max = _return_range(mu, bounds)
    rows = []
    w_prev = _start(bounds)
    for target in np.linspace(r_min, r_max, k):
        cons = (_CONS_SUM, {"type": "eq", "fun": lambda w, t=target: w @ mu - t})
        res = minimize(lambda w: w @ cov @ w, w_prev, method="SLSQP", bounds=bounds,
                       constraints=cons, options=_OPTIONS)
        if res.success:
            w_prev = res.x
            rows.append({"Retorno": float(res.x @ mu), "Vol": float(np.sqrt(max(res.x @ cov @ res.x, 0))),
                         **{f"w{i}": x for i, x in enumerate(res.x)}})
    return pd.DataFrame(rows)
