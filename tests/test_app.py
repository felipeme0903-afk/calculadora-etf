"""Teste de fumaça da interface com dados sintéticos (sem rede)."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import data as D
import streamlit as st
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def fake_download(monkeypatch, prices):
    df, bench = prices
    # marca os módulos como atualizados para o app não recarregá-los (o que desfaria os patches)
    import importlib
    for name in ("universe", "metrics", "portfolio", "optimizer", "data", "report"):
        mod = importlib.import_module(name)
        mod._src_mtime = Path(mod.__file__).stat().st_mtime

    def fake(symbols, period, force=False):
        rng = np.random.default_rng(len(symbols))
        out = pd.DataFrame(index=df.index)
        for i, s in enumerate(symbols):
            if s == "BRL=X":
                out[s] = 5.0 + np.cumsum(rng.normal(0, 0.01, len(df)))
            elif s.startswith("^"):
                out[s] = bench.values
            elif s == "SEMDADOS":
                out[s] = np.nan
            else:
                out[s] = df.iloc[:, i % df.shape[1]].values * (1 + 0.01 * i)
        return out, "fake"

    monkeypatch.setattr(D, "download_prices", fake)
    monkeypatch.setattr(D, "detect_currency", lambda s: "BRL" if s.endswith(".SA") else "USD")


def run_app():
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_runs_and_optimizes(fake_download):
    at = run_app()
    assert len(at.tabs) == 6
    assert not at.error, [e.value for e in at.error]
    # mudar N de ativos cria/remove linhas
    at.sidebar.slider(key="n_assets").set_value(8).run()
    assert "tk_7" in [t.key for t in at.sidebar.text_input]
    at.sidebar.slider(key="n_assets").set_value(5).run()
    assert "tk_5" not in [t.key for t in at.sidebar.text_input]
    # botão otimizar preenche os pesos
    before = [at.session_state[f"w_{i}"] for i in range(5)]
    next(b for b in at.sidebar.button if "Executar" in b.label).click().run()
    assert not at.exception
    after = [at.session_state[f"w_{i}"] for i in range(5)]
    assert before != after and abs(sum(after) - 100) < 0.1
    assert all(5 - 0.01 <= w <= 40 + 0.01 for w in after)


def test_missing_ticker_requires_confirmation(fake_download):
    at = run_app()
    at.sidebar.text_input(key="tk_0").set_value("SEMDADOS").run()
    assert any("sem dados" in e.value for e in at.error)
    assert len(at.tabs) == 0
    at.checkbox(key="confirm_missing").check().run()
    assert not at.exception and len(at.tabs) == 6


def test_export_generates_files(fake_download, monkeypatch, tmp_path):
    import universe as U
    monkeypatch.setattr(U, "OUTPUT_DIR", tmp_path)
    at = run_app()
    next(b for b in at.button if "Gerar tudo" in b.label).click().run()
    assert not at.exception, at.exception
    names = {p.name for p in tmp_path.iterdir()}
    for fn in U.OUTPUT_FILES.values():
        assert fn in names
    assert {"composicao.png", "fronteira.png", "correlacao.png"} <= names


def test_correlation_limit_applied_in_app(fake_download):
    import metrics as M
    import optimizer as O
    import portfolio as P
    at = run_app()
    at.sidebar.slider(key="max_corr").set_value(0.60).run()
    assert not at.exception, at.exception
    opt = at.session_state["_opt_hist"]
    assert opt["success"], opt["message"]
    tickers = list(opt["weights"])
    # recalcula os pares a partir dos mesmos dados sintéticos usados pelo app
    uni = D.load_universe(tickers, at.session_state["bench"], "5y", "BRL")
    corr = P.corr_matrix(M.log_returns(uni.prices[tickers]))
    pairs = O.correlated_pairs(corr, 0.60)
    assert pairs
    w = [opt["weights"][t] for t in tickers]
    cap = at.session_state["wbounds"][1] / 100
    assert all(w[i] + w[j] <= cap + 1e-6 for i, j in pairs)
