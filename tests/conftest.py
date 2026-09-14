import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    """Preços sintéticos determinísticos: 5 ativos + benchmark, 600 dias úteis."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2022-01-03", periods=600)
    market = rng.normal(0.0004, 0.012, len(idx))
    cols = {}
    for i, (b, vol, drift) in enumerate([(1.2, .010, .0003), (0.8, .008, .0002), (1.0, .015, .0006),
                                          (0.3, .006, .0001), (1.5, .020, .0004)]):
        r = drift + b * market + rng.normal(0, vol, len(idx))
        cols[f"A{i}"] = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame(cols, index=idx)
    bench = pd.Series(100 * np.exp(np.cumsum(market)), index=idx, name="BENCH")
    return df, bench
