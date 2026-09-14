import numpy as np
import pandas as pd

import metrics as M


def test_sharpe_matches_manual(prices):
    df, bench = prices
    p = df["A0"].to_numpy()
    r = np.log(p[1:] / p[:-1])
    rf = M.rf_continuous(10.0)
    manual = (r.mean() * 252 - np.log(1.10)) / (r.std(ddof=1) * np.sqrt(252))
    table = M.asset_metrics_table(df, bench, 252, rf, 0.95)
    assert np.isclose(table.loc["A0", "Sharpe"], manual)


def test_basic_metrics(prices):
    df, bench = prices
    rb = M.log_returns(bench)
    # benchmark contra ele mesmo
    assert np.isclose(M.beta(rb, rb), 1.0)
    assert np.isclose(M.tracking_error(rb, rb, 252), 0.0)
    assert np.isnan(M.information_ratio(rb, rb, 252))
    p = df["A1"]
    assert np.isclose(M.cumulative_return(p), p.iloc[-1] / p.iloc[0] - 1)
    dd = M.drawdown_series(p)
    assert dd.max() <= 0 and np.isclose(M.max_drawdown(p), dd.min())
    assert M.var_historical(M.log_returns(p), 0.99) < M.var_historical(M.log_returns(p), 0.95)


def test_division_by_zero_is_nan():
    flat = pd.Series([10.0] * 50, index=pd.bdate_range("2024-01-01", periods=50))
    r = M.log_returns(flat)
    assert M.annualized_vol(r, 252) == 0
    assert np.isnan(M.sharpe(0.0, 0.0, 0.1))
    assert np.isnan(M.beta(r, r))
    assert np.isnan(M.calmar(0.1, 0.0))


def test_missing_ticker_is_nan_row(prices):
    df, bench = prices
    table = M.asset_metrics_table(df, bench, 252, 0.0, 0.95, missing=["XPTO"])
    assert table.loc["XPTO"].isna().all()
