"""Otimização de pesos com scipy SLSQP: long-only, soma = 1, bounds por ativo.

Restrição opcional de correlação: para cada par de ativos com correlação acima do máximo,
w_i + w_j ≤ teto (o peso máximo por ativo), ou seja, o par é tratado como um único ativo.
Restrições lineares extras são passadas como `lin = (A_ub, b_ub)`, significando A_ub @ w ≤ b_ub.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

import portfolio as P
import universe as U

_CONS_SUM = {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}
_OPTIONS = {"maxiter": U.OPT_MAX_ITER, "ftol": U.OPT_FTOL}

Linear = tuple[np.ndarray, np.ndarray] | None


@dataclass
class OptResult:
    weights: np.ndarray
    success: bool
    message: str


def make_bounds(n: int, w_min: float, w_max: float) -> list[tuple[float, float]]:
    return [(w_min, w_max)] * n


# --- Restrição de correlação --------------------------------------------------

def correlated_pairs(corr, max_corr: float) -> list[tuple[int, int]]:
    """Índices (i, j), i < j, dos pares com correlação acima do máximo."""
    c = np.asarray(corr)
    n = len(c)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if c[i, j] > max_corr + 1e-12]


def pair_constraints(n: int, pairs: list[tuple[int, int]], cap: float) -> Linear:
    """w_i + w_j ≤ cap para cada par. Retorna None se não houver pares."""
    if not pairs:
        return None
    A = np.zeros((len(pairs), n))
    for k, (i, j) in enumerate(pairs):
        A[k, i] = A[k, j] = 1.0
    return A, np.full(len(pairs), cap)


def _slsqp_lin(lin: Linear) -> tuple:
    if lin is None:
        return ()
    A, b = lin
    return ({"type": "ineq", "fun": lambda w: b - A @ w, "jac": lambda w: -A},)


# --- Viabilidade --------------------------------------------------------------

def check_feasible(bounds, lin: Linear = None) -> str | None:
    lo = sum(b[0] for b in bounds)
    hi = sum(b[1] for b in bounds)
    if lo > 1 + 1e-9 or hi < 1 - 1e-9:
        return (f"Bounds inviáveis para {len(bounds)} ativos: soma dos mínimos = {lo:.0%}, "
                f"soma dos máximos = {hi:.0%} (a soma dos pesos precisa ser 100%)")
    if lin is not None:
        n = len(bounds)
        res = linprog(np.zeros(n), A_ub=lin[0], b_ub=lin[1], A_eq=np.ones((1, n)), b_eq=[1.0],
                      bounds=bounds, method="highs")
        if not res.success:
            return (f"Restrição de correlação inviável: com {len(lin[1])} par(es) acima do limite, cada par "
                    f"somando no máximo {lin[1][0]:.0%}, não é possível chegar a 100%. "
                    "Aumente a correlação máxima, o peso máximo, ou troque ativos muito correlacionados.")
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


def _violation(w, bounds, lin: Linear) -> float:
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    v = max(abs(w.sum() - 1), np.max(lo - w, initial=0), np.max(w - hi, initial=0))
    if lin is not None:
        v = max(v, np.max(lin[0] @ w - lin[1], initial=0))
    return float(v)


def _run(fun, bounds, lin: Linear = None, extra_cons=()) -> OptResult:
    n = len(bounds)
    err = check_feasible(bounds, lin)
    if err:
        return OptResult(np.full(n, np.nan), False, err)
    res = minimize(fun, _start(bounds), method="SLSQP", bounds=bounds,
                   constraints=(_CONS_SUM, *_slsqp_lin(lin), *extra_cons), options=_OPTIONS)
    if not res.success or _violation(res.x, bounds, lin) > 1e-6:
        return OptResult(np.full(n, np.nan), False, str(res.message))
    w = np.clip(res.x, 0, None)
    return OptResult(w / w.sum(), True, str(res.message))


# --- Objetivos ----------------------------------------------------------------

def max_sharpe(mu, cov, rf: float, bounds, lin: Linear = None) -> OptResult:
    mu, cov = np.asarray(mu), np.asarray(cov)

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 0.0))
        return -(w @ mu - rf) / vol if vol > 0 else 0.0

    return _run(neg_sharpe, bounds, lin)


def min_vol(cov, bounds, lin: Linear = None) -> OptResult:
    cov = np.asarray(cov)
    return _run(lambda w: w @ cov @ w, bounds, lin)  # minimizar variância ≡ minimizar σ_p (mais estável)


def max_sortino(returns: pd.DataFrame, periods: int, rf: float, bounds, lin: Linear = None) -> OptResult:
    R = returns.to_numpy()
    mu = R.mean(axis=0) * periods

    def neg_sortino(w):
        rp = R @ w
        down = np.sqrt(np.mean(np.minimum(rp, 0.0) ** 2)) * np.sqrt(periods)
        return -(w @ mu - rf) / down if down > 0 else 0.0

    return _run(neg_sortino, bounds, lin)


def risk_parity(cov, bounds, lin: Linear = None) -> OptResult:
    cov = np.asarray(cov)
    n = len(cov)

    def objective(w):
        rc = P.risk_contribution(w, cov)
        return float(np.nansum((rc - 1 / n) ** 2)) * 1e3

    return _run(objective, bounds, lin)


def equal_weights(bounds, lin: Linear = None) -> OptResult:
    n = len(bounds)
    err = check_feasible(bounds, lin)
    if err:
        return OptResult(np.full(n, np.nan), False, err)
    w = _start(bounds)
    if _violation(w, bounds, lin) <= 1e-9:
        return OptResult(w, True, "Pesos iguais" + ("" if np.allclose(w, w[0]) else " (ajustados aos bounds)"))
    res = _run(lambda x: float(np.sum((x - 1 / n) ** 2)), bounds, lin)  # o mais próximo possível de 1/N
    if res.success:
        res.message = "Pesos o mais próximo possível de iguais (ajustados às restrições)"
    return res


def optimize(objective: str, mu, cov, returns: pd.DataFrame, periods: int, rf: float, bounds,
             lin: Linear = None) -> OptResult:
    if objective == "Max Sharpe":
        return max_sharpe(mu, cov, rf, bounds, lin)
    if objective == "Min Vol":
        return min_vol(cov, bounds, lin)
    if objective == "Max Sortino":
        return max_sortino(returns, periods, rf, bounds, lin)
    if objective == "Risk Parity":
        return risk_parity(cov, bounds, lin)
    if objective == "Igual":
        return equal_weights(bounds, lin)
    raise ValueError(f"Objetivo desconhecido: {objective}")


# --- Fronteira ----------------------------------------------------------------

def _return_range(mu, bounds, lin: Linear = None) -> tuple[float, float]:
    """Menor e maior retorno atingíveis sob soma = 1, bounds e restrições lineares."""
    mu = np.asarray(mu)
    n = len(mu)
    kw = dict(A_eq=np.ones((1, n)), b_eq=[1.0], bounds=bounds, method="highs")
    if lin is not None:
        kw.update(A_ub=lin[0], b_ub=lin[1])
    lo = linprog(mu, **kw)
    hi = linprog(-mu, **kw)
    if not (lo.success and hi.success):
        return float(mu.min()), float(mu.max())
    return float(mu @ lo.x), float(mu @ hi.x)


def efficient_frontier(mu, cov, bounds, k: int, lin: Linear = None) -> pd.DataFrame:
    """Para K alvos de retorno, minimiza σ_p com restrição μ_p = alvo."""
    mu, cov = np.asarray(mu), np.asarray(cov)
    if check_feasible(bounds, lin):
        return pd.DataFrame(columns=["Retorno", "Vol"])
    r_min, r_max = _return_range(mu, bounds, lin)
    rows = []
    w_prev = _start(bounds)
    for target in np.linspace(r_min, r_max, k):
        cons = (_CONS_SUM, *_slsqp_lin(lin), {"type": "eq", "fun": lambda w, t=target: w @ mu - t})
        res = minimize(lambda w: w @ cov @ w, w_prev, method="SLSQP", bounds=bounds,
                       constraints=cons, options=_OPTIONS)
        if res.success and _violation(res.x, bounds, lin) <= 1e-6:
            w_prev = res.x
            rows.append({"Retorno": float(res.x @ mu), "Vol": float(np.sqrt(max(res.x @ cov @ res.x, 0))),
                         **{f"w{i}": x for i, x in enumerate(res.x)}})
    return pd.DataFrame(rows)
