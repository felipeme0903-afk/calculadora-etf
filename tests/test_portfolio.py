import numpy as np

import metrics as M
import portfolio as P


def test_equal_weight_vol_between_min_and_max(prices):
    df, _ = prices
    r = M.log_returns(df)
    cov = P.cov_matrix(r, 252)
    w = np.full(df.shape[1], 1 / df.shape[1])
    vols = np.sqrt(np.diag(cov))
    assert vols.min() <= P.portfolio_vol(w, cov) <= vols.max()


def test_corr_diagonal_is_one(prices):
    df, _ = prices
    corr = P.corr_matrix(M.log_returns(df))
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.min().min() >= -1 and corr.max().max() <= 1


def test_series_consistent_with_matrix_formulas(prices):
    df, bench = prices
    r = M.log_returns(df)
    w = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    mu, cov = P.mu_vector(r, 252), P.cov_matrix(r, 252)
    pr = P.portfolio_returns(r, w)
    assert np.isclose(pr.mean() * 252, P.portfolio_return(w, mu))
    assert np.isclose(pr.std(ddof=1) * np.sqrt(252), P.portfolio_vol(w, cov))
    m = P.portfolio_metrics(r, w, M.log_returns(bench), 252, 0.05, 0.95)
    assert np.isclose(m["Sharpe"], P.portfolio_sharpe(w, mu, cov, 0.05))


def test_risk_contribution_sums_to_one(prices):
    df, _ = prices
    cov = P.cov_matrix(M.log_returns(df), 252)
    rc = P.risk_contribution(np.full(5, 0.2), cov)
    assert np.isclose(rc.sum(), 1.0)


def test_shrinkage():
    import pandas as pd
    mu = pd.Series([0.1, 0.3])
    assert np.allclose(P.shrink_mu(mu, 0.5), [0.15, 0.25])
    assert np.allclose(P.shrink_mu(mu, 1.0), mu)


def test_weekly_constraint_corr(prices):
    df, _ = prices
    r = M.log_returns(df)
    wk = P.weekly_returns(r)
    assert np.allclose(wk.sum(), r.sum())  # log-retornos somam
    assert np.allclose(P.constraint_corr(r, "D", False), P.corr_matrix(r))
    assert np.allclose(P.constraint_corr(r, "D", True), P.corr_matrix(wk))
    assert np.allclose(P.constraint_corr(r, "W", True), P.corr_matrix(r))
