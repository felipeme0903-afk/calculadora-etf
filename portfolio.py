"""Métricas de carteira, correlação, covariância e contribuição ao risco — funções puras.

A série da carteira combina linearmente os log-retornos (r_p = Σ w_i r_i), de modo que
média(r_p) × períodos = W'μ e desvio(r_p) × √períodos = √(W'ΣW), exatamente como na otimização.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import metrics as M


def mu_vector(returns: pd.DataFrame, periods: int) -> pd.Series:
    return returns.mean() * periods


def cov_matrix(returns: pd.DataFrame, periods: int) -> pd.DataFrame:
    return returns.cov() * periods


def corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr(method="pearson")


def high_corr_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cols = list(corr.columns)
    pairs = [(cols[i], cols[j], corr.iloc[i, j])
             for i in range(len(cols)) for j in range(i + 1, len(cols)) if corr.iloc[i, j] > threshold]
    return pd.DataFrame(pairs, columns=["Ativo 1", "Ativo 2", "Correlação"])


def shrink_mu(mu: pd.Series, lam: float) -> pd.Series:
    """μ_ajustado = λ·μ_hist + (1−λ)·média(μ_hist)."""
    return lam * mu + (1 - lam) * mu.mean()


def portfolio_return(w, mu) -> float:
    return float(np.asarray(w) @ np.asarray(mu))


def portfolio_vol(w, cov) -> float:
    w = np.asarray(w)
    return float(np.sqrt(max(w @ np.asarray(cov) @ w, 0.0)))


def portfolio_sharpe(w, mu, cov, rf: float) -> float:
    return M.safe_div(portfolio_return(w, mu) - rf, portfolio_vol(w, cov))


def portfolio_returns(returns: pd.DataFrame, w) -> pd.Series:
    return pd.Series(returns.to_numpy() @ np.asarray(w), index=returns.index, name="Carteira")


def portfolio_index(returns: pd.DataFrame, w, base: float = 100.0) -> pd.Series:
    r = portfolio_returns(returns, w)
    idx = base * np.exp(r.cumsum())
    first = pd.Series([base], index=[returns.index[0] - pd.Timedelta(days=1)])
    return pd.concat([first, idx]).rename("Carteira")


def risk_contribution(w, cov) -> np.ndarray:
    """RC_i = w_i (Σw)_i / σ_p² — soma 1."""
    w = np.asarray(w)
    var = float(w @ np.asarray(cov) @ w)
    if var <= 0:
        return np.full(len(w), np.nan)
    return w * (np.asarray(cov) @ w) / var


def portfolio_metrics(returns: pd.DataFrame, w, bench_returns: pd.Series, periods: int, rf: float,
                      confidence: float) -> dict:
    r = portfolio_returns(returns, w)
    return M.metrics_from_returns(r, bench_returns, periods, rf, confidence)


def allocation(capital: float, w, last_prices: pd.Series, currencies: dict[str, str]) -> pd.DataFrame:
    w = np.asarray(w, dtype=float)
    alloc = capital * w
    prices = last_prices.to_numpy(dtype=float)
    shares = np.floor(np.divide(alloc, prices, out=np.full_like(alloc, np.nan), where=prices > 0))
    return pd.DataFrame({
        "Peso": w,
        "Valor alocado": alloc,
        "Preço (moeda base)": prices,
        "Nº de cotas": shares,
        "Valor investido": shares * prices,
        "Moeda original": [currencies.get(t, "") for t in last_prices.index],
    }, index=last_prices.index)
