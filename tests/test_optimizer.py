import numpy as np
import pytest

import metrics as M
import optimizer as O
import portfolio as P


@pytest.fixture
def inputs(prices):
    df, _ = prices
    r = M.log_returns(df)
    return r, P.mu_vector(r, 252), P.cov_matrix(r, 252)


@pytest.mark.parametrize("objective", ["Max Sharpe", "Min Vol", "Max Sortino", "Risk Parity", "Igual"])
def test_constraints_respected(inputs, objective):
    r, mu, cov = inputs
    bounds = O.make_bounds(5, 0.05, 0.40)
    res = O.optimize(objective, mu, cov, r, 252, 0.03, bounds)
    assert res.success, res.message
    assert np.isclose(res.weights.sum(), 1.0)
    assert (res.weights >= 0.05 - 1e-6).all() and (res.weights <= 0.40 + 1e-6).all()


def test_max_sharpe_beats_equal(inputs):
    r, mu, cov = inputs
    bounds = O.make_bounds(5, 0.05, 0.40)
    w = O.max_sharpe(mu, cov, 0.03, bounds).weights
    eq = np.full(5, 0.2)
    assert P.portfolio_sharpe(w, mu, cov, 0.03) >= P.portfolio_sharpe(eq, mu, cov, 0.03) - 1e-9


def test_min_vol_is_lowest(inputs):
    r, mu, cov = inputs
    bounds = O.make_bounds(5, 0.0, 1.0)
    w = O.min_vol(cov, bounds).weights
    assert P.portfolio_vol(w, cov) <= np.sqrt(np.diag(cov)).min() + 1e-9


def test_risk_parity_equalizes(inputs):
    r, mu, cov = inputs
    w = O.risk_parity(cov, O.make_bounds(5, 0.0, 1.0)).weights
    assert np.allclose(P.risk_contribution(w, cov), 0.2, atol=0.01)


def test_infeasible_bounds():
    res = O.optimize("Igual", np.zeros(8), np.eye(8), None, 252, 0.0, O.make_bounds(8, 0.05, 0.10))
    assert not res.success and "inviáveis" in res.message


def test_frontier(inputs):
    r, mu, cov = inputs
    bounds = O.make_bounds(5, 0.05, 0.40)
    fr = O.efficient_frontier(mu, cov, bounds, 20)
    assert len(fr) >= 15
    mv = P.portfolio_vol(O.min_vol(cov, bounds).weights, cov)
    assert fr["Vol"].min() >= mv - 1e-6
