import io

import numpy as np
import pandas as pd

import report as R


def test_csv_and_excel_roundtrip(tmp_path):
    df = pd.DataFrame({"Sharpe": [1.23456789, np.nan], "Moeda": ["USD", "BRL"]}, index=["QQQ", "NASD11"])
    back = pd.read_csv(io.BytesIO(R.csv_bytes(df)), sep=";", decimal=",", index_col=0, encoding="utf-8-sig")
    assert np.isclose(back.loc["QQQ", "Sharpe"], 1.23456789) and np.isnan(back.loc["NASD11", "Sharpe"])
    xl = pd.read_excel(io.BytesIO(R.excel_bytes({"a": df, "b": df})), sheet_name=None, index_col=0)
    assert set(xl) == {"a", "b"}


def test_pngs(prices):
    df, _ = prices
    assert R.fig_composition(pd.Series([0.5, 0.5], index=["A", "B"]))[:4] == b"\x89PNG"
    assert R.fig_corr(df.corr())[:4] == b"\x89PNG"
