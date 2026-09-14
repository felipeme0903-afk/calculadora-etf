import numpy as np
import pandas as pd
import pytest

import data as D


def test_suffix_rule():
    assert D.to_yahoo("nasd11") == "NASD11.SA"
    assert D.to_yahoo("IVVB11") == "IVVB11.SA"
    assert D.to_yahoo("QQQ") == "QQQ"
    assert D.to_yahoo("^BVSP") == "^BVSP"
    assert D.to_yahoo("BRL=X") == "BRL=X"
    assert D.to_display("HASH11.SA") == "HASH11"


def test_currency_conversion():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"QQQ": [100.0, 110, 120], "BOVA11": [50.0, 55, 60]}, index=idx)
    fx = pd.Series([5.0, 5.0, 6.0], index=idx)
    cur = {"QQQ": "USD", "BOVA11": "BRL"}
    brl, _ = D.convert_to_base(prices, cur, fx, "BRL")
    assert np.allclose(brl["QQQ"], [500, 550, 720]) and np.allclose(brl["BOVA11"], prices["BOVA11"])
    usd, _ = D.convert_to_base(prices, cur, fx, "USD")
    assert np.allclose(usd["BOVA11"], [10, 11, 10]) and np.allclose(usd["QQQ"], prices["QQQ"])
    with pytest.raises(D.FxMissingError):
        D.convert_to_base(prices, cur, None, "BRL")


def test_align_ffill_limit():
    idx = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, 5, 6], "B": [1.0, 2, 3, 4, 5, 6]}, index=idx)
    aligned, removed = D.align(df)
    assert removed == 1 and aligned.notna().all().all()


def test_weekly_uses_real_last_date():
    idx = pd.bdate_range("2024-01-01", "2024-01-17")  # termina numa quarta
    w = D.resample(pd.DataFrame({"A": range(len(idx))}, index=idx, dtype=float), "W")
    assert w.index[-1] == pd.Timestamp("2024-01-17")
