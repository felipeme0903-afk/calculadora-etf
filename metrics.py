"""Métricas de risco, retorno e retorno ajustado ao risco — funções puras.

Convenções (usadas em todo o projeto):
- Retornos são log-retornos: r_t = ln(P_t / P_{t-1}).
- Retorno anualizado = média(r) × períodos/ano (taxa contínua).
- Taxa livre de risco informada em % a.a. (efetiva) é convertida para contínua: ln(1 + Rf).
- Divisões por zero retornam NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORMULAS = {
    "Retorno acumulado": "P_T / P_0 − 1",
    "Retorno anual": "média(ln(P_t/P_{t−1})) × períodos/ano",
    "CAGR": "(P_T/P_0)^(períodos/ano ÷ n) − 1",
    "Vol anual": "desvio(r) × √(períodos/ano)",
    "Beta": "cov(r_a, r_b) / var(r_b)",
    "Max DD": "min(P_t / max(P_0..P_t) − 1)",
    "Sharpe": "(R_anual − Rf) / Vol_anual",
    "Sortino": "(R_anual − Rf) / DesvioNeg_anual; DesvioNeg = √(média(min(r,0)²)) × √(períodos/ano)",
    "Treynor": "(R_anual − Rf) / Beta",
    "Calmar": "R_anual / |Max DD|",
    "Information Ratio": "média(r_a − r_b) × períodos/ano / (desvio(r_a − r_b) × √períodos/ano)",
    "Tracking error": "desvio(r_a − r_b) × √(períodos/ano)",
    "VaR": "quantil (1 − confiança) dos log-retornos do período",
}


def safe_div(a, b) -> float:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-15:
        return np.nan
    return a / b


def rf_continuous(rf_pct: float) -> float:
    """% a.a. efetiva -> taxa contínua anual."""
    return float(np.log1p(rf_pct / 100.0))


def log_returns(prices):
    return np.log(prices / prices.shift(1)).iloc[1:]


def cumulative_return(prices: pd.Series) -> float:
    p = prices.dropna()
    return safe_div(p.iloc[-1], p.iloc[0]) - 1 if len(p) > 1 else np.nan


def cagr(prices: pd.Series, periods: int) -> float:
    p = prices.dropna()
    n = len(p) - 1
    if n <= 0 or p.iloc[0] <= 0:
        return np.nan
    return (p.iloc[-1] / p.iloc[0]) ** (periods / n) - 1


def annualized_return(r: pd.Series, periods: int) -> float:
    r = r.dropna()
    return float(r.mean() * periods) if len(r) else np.nan


def annualized_vol(r: pd.Series, periods: int) -> float:
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else np.nan


def downside_deviation(r: pd.Series, periods: int) -> float:
    r = r.dropna()
    if not len(r):
        return np.nan
    return float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)) * np.sqrt(periods))


def beta(ra: pd.Series, rb: pd.Series) -> float:
    df = pd.concat([ra, rb], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    return safe_div(df.iloc[:, 0].cov(df.iloc[:, 1]), df.iloc[:, 1].var())


def drawdown_series(prices):
    return prices / prices.cummax() - 1


def max_drawdown(prices: pd.Series) -> float:
    p = prices.dropna()
    return float(drawdown_series(p).min()) if len(p) else np.nan


def sharpe(ann_ret: float, ann_vol: float, rf: float) -> float:
    return safe_div(ann_ret - rf, ann_vol)


def sortino(ann_ret: float, down_dev: float, rf: float) -> float:
    return safe_div(ann_ret - rf, down_dev)


def treynor(ann_ret: float, b: float, rf: float) -> float:
    return safe_div(ann_ret - rf, b)


def calmar(ann_ret: float, mdd: float) -> float:
    return safe_div(ann_ret, abs(mdd))


def tracking_error(ra: pd.Series, rb: pd.Series, periods: int) -> float:
    active = (ra - rb).dropna()
    return float(active.std(ddof=1) * np.sqrt(periods)) if len(active) > 1 else np.nan


def information_ratio(ra: pd.Series, rb: pd.Series, periods: int) -> float:
    active = (ra - rb).dropna()
    return safe_div(active.mean() * periods, tracking_error(ra, rb, periods))


def var_historical(r: pd.Series, confidence: float) -> float:
    r = r.dropna()
    return float(r.quantile(1 - confidence)) if len(r) else np.nan


def rolling_vol(r, window: int, periods: int):
    return r.rolling(window).std(ddof=1) * np.sqrt(periods)


def rolling_sharpe(r, window: int, periods: int, rf: float):
    mean = r.rolling(window).mean() * periods
    vol = rolling_vol(r, window, periods)
    return (mean - rf) / vol.where(vol > 0)


def metrics_from_returns(r: pd.Series, rb: pd.Series, periods: int, rf: float,
                         confidence: float, prices: pd.Series | None = None) -> dict:
    """Todas as métricas de uma série. Sem preços, reconstrói o índice por exp(cumsum(r))."""
    r = r.dropna()
    if not len(r):
        return {k: np.nan for k in metric_columns(confidence)}
    if prices is None:
        prices = pd.Series(np.exp(np.concatenate([[0.0], r.cumsum().to_numpy()])))
    ann_ret = annualized_return(r, periods)
    vol = annualized_vol(r, periods)
    mdd = max_drawdown(prices)
    b = beta(r, rb)
    return {
        "Retorno acumulado": cumulative_return(prices),
        "Retorno anual": ann_ret,
        "CAGR": cagr(prices, periods),
        "Vol anual": vol,
        "Beta": b,
        "Max DD": mdd,
        "Sharpe": sharpe(ann_ret, vol, rf),
        "Sortino": sortino(ann_ret, downside_deviation(r, periods), rf),
        "Treynor": treynor(ann_ret, b, rf),
        "Calmar": calmar(ann_ret, mdd),
        "Information Ratio": information_ratio(r, rb, periods),
        "Tracking error": tracking_error(r, rb, periods),
        f"VaR {confidence:.0%}": var_historical(r, confidence),
    }


def metric_columns(confidence: float) -> list[str]:
    return ["Retorno acumulado", "Retorno anual", "CAGR", "Vol anual", "Beta", "Max DD", "Sharpe",
            "Sortino", "Treynor", "Calmar", "Information Ratio", "Tracking error", f"VaR {confidence:.0%}"]


def asset_metrics_table(prices: pd.DataFrame, bench_prices: pd.Series, periods: int, rf: float,
                        confidence: float, missing: list[str] | None = None) -> pd.DataFrame:
    """Uma linha por ativo; tickers sem dados entram como linha de NaN."""
    rets = log_returns(prices)
    rb = log_returns(bench_prices)
    rows = {c: metrics_from_returns(rets[c], rb, periods, rf, confidence, prices[c]) for c in prices.columns}
    for m in missing or []:
        rows[m] = {k: np.nan for k in metric_columns(confidence)}
    return pd.DataFrame.from_dict(rows, orient="index")[metric_columns(confidence)]
