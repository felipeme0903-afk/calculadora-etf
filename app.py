"""Calculadora de Carteira de ETFs — interface Streamlit (única camada com st.*).

Executar:  streamlit run app.py
"""

from __future__ import annotations

import importlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as D
import metrics as M
import optimizer as O
import portfolio as P
import report as R
import universe as U


def _reload_stale_modules():
    """Recarrega módulos do projeto cujo arquivo mudou desde que foram carregados.

    Ao atualizar o código (git pull no Streamlit Cloud ou edição local), o Streamlit reexecuta
    o app.py mas pode manter na memória a versão antiga dos módulos importados, o que causa
    erros como "module 'universe' has no attribute ...". Ordem: dependências primeiro.
    """
    for mod in (U, M, P, O, D, R):
        mtime = Path(mod.__file__).stat().st_mtime
        if getattr(mod, "_src_mtime", None) != mtime:
            importlib.reload(mod)
            mod._src_mtime = mtime


_reload_stale_modules()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="Calculadora de Carteira de ETFs", page_icon="📈", layout="wide")

SS = st.session_state
SCENARIO_KEYS = ["n_assets", "wbounds", "capital", "period", "freq", "base", "bench", "rf",
                 "var_level", "roll_win", "objective", "n_frontier", "shrink_lambda", "max_corr"]


# ============================================================================
# Estado e callbacks
# ============================================================================

def _set_rows(items: list[tuple[str, float]]):
    items = items[: U.MAX_ASSETS]
    SS.n_assets = min(max(len(items), U.MIN_ASSETS), U.MAX_ASSETS)
    for i in range(U.MAX_ASSETS):
        tk, w = items[i] if i < len(items) else ("", 0.0)
        SS[f"tk_{i}"] = tk
        SS[f"w_{i}"] = float(w)
        SS[f"on_{i}"] = bool(tk)


def init_state():
    if SS.get("_init"):
        return
    SS._init = True
    _set_rows(list(U.PRESETS[U.DEFAULT_PRESET].items()))
    SS.preset_sel = U.DEFAULT_PRESET
    SS.wbounds = U.DEFAULT_WEIGHT_BOUNDS
    SS.capital = U.DEFAULT_CAPITAL
    SS.period = U.DEFAULT_PERIOD
    SS.freq = U.DEFAULT_FREQUENCY
    SS.base = U.DEFAULT_BASE_CURRENCY
    SS.bench = U.DEFAULT_BENCHMARK
    SS.rf = U.DEFAULT_RF[U.DEFAULT_BASE_CURRENCY]
    SS.var_level = U.DEFAULT_VAR_LEVEL
    SS.roll_win = U.DEFAULT_ROLLING_WINDOW
    SS.objective = U.DEFAULT_OBJECTIVE
    SS.n_frontier = U.DEFAULT_FRONTIER_POINTS
    SS.shrink_lambda = U.DEFAULT_SHRINKAGE
    SS.max_corr = U.DEFAULT_MAX_CORR
    SS.nonce = 0
    SS.force_pending = False
    SS.flash = []


def flash(kind: str, msg: str):
    SS.flash.append((kind, msg))


def cb_preset():
    _set_rows(list(U.PRESETS[SS.preset_sel].items()))
    flash("success", f"Preset '{SS.preset_sel}' carregado")


def cb_normalize():
    idx = [i for i in range(SS.n_assets) if SS[f"on_{i}"] and SS[f"tk_{i}"].strip()]
    total = sum(SS[f"w_{i}"] for i in idx)
    if total <= 0:
        flash("error", "Não há pesos positivos para normalizar")
        return
    for i in idx:
        SS[f"w_{i}"] = round(SS[f"w_{i}"] / total * 100, 4)


def cb_base():
    SS.rf = U.DEFAULT_RF[SS.base]


def cb_force():
    SS.nonce += 1
    SS.force_pending = True


def cb_load_csv():
    f = SS.get("csv_upload")
    if f is None:
        flash("error", "Selecione um CSV com colunas ticker, peso")
        return
    try:
        text = f.getvalue().decode("utf-8-sig")
        semicolon = ";" in text.splitlines()[0]
        df = pd.read_csv(io.StringIO(text), sep=";" if semicolon else ",",
                         decimal="," if semicolon else ".")
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df[["ticker", "peso"]].dropna()
        pesos = pd.to_numeric(df["peso"], errors="coerce").fillna(0.0)
        if pesos.sum() <= 1.0 + 1e-9:  # pesos em fração
            pesos = pesos * 100
        items = [(D.normalize_ticker(t), float(w)) for t, w in zip(df["ticker"], pesos)]
    except Exception as exc:
        flash("error", f"CSV inválido: {exc}")
        return
    if len(items) > U.MAX_ASSETS:
        flash("warning", f"CSV com {len(items)} ativos; mantidos os {U.MAX_ASSETS} primeiros")
    _set_rows(items)
    flash("success", f"Carteira carregada: {len(items[:U.MAX_ASSETS])} ativos")


def cb_load_scenario():
    f = SS.get("scenario_upload")
    if f is None:
        flash("error", "Selecione um arquivo JSON de cenário")
        return
    try:
        sc = json.loads(f.getvalue().decode("utf-8"))
        _set_rows([(D.normalize_ticker(a["ticker"]), float(a["peso"])) for a in sc["ativos"]])
        for i, a in enumerate(sc["ativos"][: U.MAX_ASSETS]):
            SS[f"on_{i}"] = bool(a.get("ativo", True))
        for k in SCENARIO_KEYS:
            if k in sc["parametros"]:
                v = sc["parametros"][k]
                SS[k] = tuple(v) if k == "wbounds" else v
    except Exception as exc:
        flash("error", f"Cenário inválido: {exc}")
        return
    flash("success", "Cenário carregado")


def cb_apply_weights(key: str):
    opt = SS.get(key)
    if not opt or not opt["success"]:
        flash("error", "Otimizador não convergiu; pesos atuais mantidos. " + (opt or {}).get("message", ""))
        return
    for i in range(SS.n_assets):
        tk = D.normalize_ticker(SS[f"tk_{i}"])
        if SS[f"on_{i}"] and tk in opt["weights"]:
            SS[f"w_{i}"] = round(opt["weights"][tk] * 100, 2)
    flash("success", f"Pesos preenchidos: {opt['label']}")


def scenario_json() -> str:
    ativos = [{"ticker": SS[f"tk_{i}"], "peso": SS[f"w_{i}"], "ativo": SS[f"on_{i}"]} for i in range(SS.n_assets)]
    params = {k: (list(SS[k]) if isinstance(SS[k], tuple) else SS[k]) for k in SCENARIO_KEYS}
    return json.dumps({"ativos": ativos, "parametros": params}, indent=2, ensure_ascii=False)


def portfolio_csv() -> bytes:
    rows = [{"ticker": SS[f"tk_{i}"], "peso": SS[f"w_{i}"]}
            for i in range(SS.n_assets) if SS[f"on_{i}"] and SS[f"tk_{i}"].strip()]
    return R.csv_bytes(pd.DataFrame(rows), index=False)


# ============================================================================
# Cálculo com cache
# ============================================================================

@st.cache_data(show_spinner="Baixando cotações…")
def cached_universe(tickers: tuple, bench: str, period: str, base: str, freq: str, nonce: int, _force: bool):
    return D.load_universe(list(tickers), bench, period, base, freq, force=_force)


@st.cache_data(show_spinner="Otimizando…")
def cached_optimization(returns: pd.DataFrame, objective: str, periods: int, rf: float,
                        bounds: tuple, lam: float, k: int, max_corr: float):
    tickers = list(returns.columns)
    mu = P.mu_vector(returns, periods)
    cov = P.cov_matrix(returns, periods)
    mu_s = P.shrink_mu(mu, lam)
    pairs = O.correlated_pairs(P.corr_matrix(returns), max_corr)
    lin = O.pair_constraints(len(tickers), pairs, bounds[0][1])  # par ≤ peso máximo por ativo

    def pack(res: O.OptResult, label: str):
        return {"weights": dict(zip(tickers, res.weights)), "success": res.success,
                "message": res.message, "label": label}

    hist = O.optimize(objective, mu, cov, returns, periods, rf, list(bounds), lin)
    if objective == "Max Sortino":  # Sortino usa a série; desloca a média para refletir o shrinkage
        shrunk_returns = returns + (mu_s - mu) / periods
        shr = O.optimize(objective, mu_s, cov, shrunk_returns, periods, rf, list(bounds), lin)
    else:
        shr = O.optimize(objective, mu_s, cov, returns, periods, rf, list(bounds), lin)
    frontier = O.efficient_frontier(mu, cov, list(bounds), k, lin)
    frontier = frontier.rename(columns={f"w{i}": t for i, t in enumerate(tickers)})
    return pack(hist, f"{objective} (μ histórico)"), pack(shr, f"{objective} (shrinkage λ={lam:.2f})"), frontier


# ============================================================================
# Gráficos Plotly
# ============================================================================

def plot_pie(weights: pd.Series, title: str):
    w = weights[weights > 1e-6]
    fig = go.Figure(go.Pie(labels=w.index, values=w.values, hole=0.35, sort=False,
                           texttemplate="%{label}<br>%{percent:.1%}"))
    fig.update_layout(title=title, height=380, margin=dict(t=50, b=10, l=10, r=10), showlegend=False)
    return fig


def plot_lines(df: pd.DataFrame, title: str, pct: bool = False, highlight=("Carteira", "Carteira otimizada", )):
    # SVG em vez de WebGL: com >1000 pontos o Plotly usa WebGL, que falha quando o navegador
    # não tem aceleração gráfica ou esgota os contextos WebGL (cada gráfico em cada aba abre um)
    fig = px.line(df, title=title, render_mode="svg")
    for tr in fig.data:
        tr.line.width = 3 if tr.name in highlight else 1.3
    fig.update_layout(height=450, legend_title_text="", hovermode="x unified", yaxis_title=None, xaxis_title=None)
    if pct:
        fig.update_yaxes(tickformat=".0%")
    return fig


def plot_corr(corr: pd.DataFrame):
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale="RdBu_r", text=np.round(corr.values, 2), texttemplate="%{text}"))
    fig.update_layout(height=120 + 55 * len(corr), yaxis_autorange="reversed", title="Correlação de Pearson (log-retornos)")
    return fig


def plot_frontier(frontier: pd.DataFrame, assets: pd.DataFrame, points: dict):
    fig = go.Figure()
    if not frontier.empty:
        fig.add_trace(go.Scatter(x=frontier["Vol"], y=frontier["Retorno"], mode="lines", name="Fronteira eficiente",
                                 line=dict(width=3)))
    fig.add_trace(go.Scatter(x=assets["Vol"], y=assets["Retorno"], mode="markers+text", text=assets.index,
                             textposition="top center", name="Ativos", marker=dict(size=9, color="gray")))
    for (name, (vol, ret)), sym in zip(points.items(), ["circle", "star", "diamond", "square"]):
        fig.add_trace(go.Scatter(x=[vol], y=[ret], mode="markers", name=name, marker=dict(size=16, symbol=sym)))
    fig.update_layout(height=520, xaxis_title="Volatilidade anual (σ)", yaxis_title="Retorno anual (μ)",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", title="Fronteira eficiente")
    return fig


def metric_column_config(columns) -> dict:
    ratio = {"Beta", "Sharpe", "Sortino", "Calmar", "Information Ratio"}
    cfg = {}
    for c in columns:
        key = "VaR" if c.startswith("VaR") else c
        if key in M.FORMULAS:
            cfg[c] = st.column_config.NumberColumn(c, help=M.FORMULAS[key],
                                                   format="%.2f" if c in ratio else "percent")
    return cfg


# ============================================================================
# Sidebar
# ============================================================================

init_state()
sb = st.sidebar
sb.title("⚙️ Parâmetros")

sb.header("Carteira")
sb.selectbox("Presets", list(U.PRESETS), key="preset_sel", on_change=cb_preset,
             help="Carrega tickers e pesos definidos em universe.py")
sb.slider("Número de ativos", U.MIN_ASSETS, U.MAX_ASSETS, key="n_assets",
          help="Regra do trabalho: 5 a 8 ETFs. Gera as linhas de input abaixo.")

h = sb.columns([0.6, 2, 1.6])
h[0].caption("Ativo")
h[1].caption("Ticker")
h[2].caption("Peso (%)")
for i in range(SS.n_assets):
    c = sb.columns([0.6, 2, 1.6], vertical_alignment="center")
    c[0].checkbox("ativo", key=f"on_{i}", label_visibility="collapsed",
                  help="Desmarque para excluir o ETF do cálculo sem apagar o input")
    c[1].text_input("Ticker", key=f"tk_{i}", label_visibility="collapsed", placeholder="ex.: QQQ, IVVB11",
                    help="Ticker terminado em dígito recebe .SA automaticamente")
    c[2].number_input("Peso", key=f"w_{i}", min_value=0.0, max_value=100.0, step=0.5, format="%.2f",
                      label_visibility="collapsed")

active_rows = [i for i in range(SS.n_assets) if SS[f"on_{i}"] and SS[f"tk_{i}"].strip()]
tickers_raw = [D.normalize_ticker(SS[f"tk_{i}"]) for i in active_rows]
weight_sum = sum(SS[f"w_{i}"] for i in active_rows)
sb.button("Normalizar pesos (soma = 100%)", on_click=cb_normalize, width="stretch")
if abs(weight_sum - 100) > U.WEIGHT_TOLERANCE:
    sb.error(f"Soma dos pesos = {weight_sum:.2f}% (≠ 100%)")
else:
    sb.caption(f"Soma dos pesos: {weight_sum:.2f}% ✔")
if not U.MIN_ASSETS <= len(active_rows) <= U.MAX_ASSETS:
    sb.error(f"{len(active_rows)} ativos ativos: a regra do trabalho exige de {U.MIN_ASSETS} a {U.MAX_ASSETS}")
dups = sorted({t for t in tickers_raw if tickers_raw.count(t) > 1})
if dups:
    sb.error(f"Tickers duplicados: {', '.join(dups)} (considerada só a primeira ocorrência)")

sb.slider("Peso mín. / máx. por ativo (%)", 0.0, 100.0, key="wbounds", step=1.0,
          help="Bounds do otimizador. Controles centrais: Max Sharpe histórico gera pesos extremos.")
sb.number_input(f"Capital ({SS.base})", min_value=0.0, step=10_000.0, key="capital", format="%.2f",
                help="Gera a alocação em valor e o nº de cotas (preço na moeda base)")

with sb.expander("Carregar / salvar carteira e cenário"):
    st.file_uploader("Carteira (CSV: ticker, peso)", type=["csv"], key="csv_upload")
    st.button("Carregar carteira do CSV", on_click=cb_load_csv, width="stretch")
    st.download_button("Baixar carteira (CSV)", portfolio_csv(), "carteira.csv", "text/csv", width="stretch")
    st.divider()
    st.download_button("Salvar cenário (JSON)", scenario_json(), "cenario.json", "application/json",
                       width="stretch")
    st.file_uploader("Cenário (JSON)", type=["json"], key="scenario_upload")
    st.button("Carregar cenário", on_click=cb_load_scenario, width="stretch")

sb.header("Dados")
sb.select_slider("Janela histórica", U.PERIODS, key="period")
sb.radio("Frequência dos retornos", list(U.FREQUENCIES), key="freq", horizontal=True,
         help="Semanal: último preço de cada sexta-feira; anualização por 52")
sb.radio("Moeda base", U.BASE_CURRENCIES, key="base", horizontal=True, on_change=cb_base,
         help="Converte todos os ativos e o benchmark via BRL=X")
sb.button("Forçar novo download", on_click=cb_force, width="stretch", help="Ignora o cache em data/")

sb.header("Risco")
sb.selectbox("Benchmark", U.BENCHMARKS, key="bench", accept_new_options=True)
sb.number_input(f"Taxa livre de risco (% a.a.) — {U.RF_LABEL[SS.base]}", step=0.05, format="%.2f", key="rf",
                help="Taxa efetiva anual; convertida para contínua ln(1+Rf) por coerência com log-retornos")
sb.selectbox("Nível de confiança do VaR", U.VAR_LEVELS, key="var_level", format_func=lambda x: f"{x:.0%}")
sb.slider("Janela móvel (dias)", *U.ROLLING_WINDOW_RANGE, key="roll_win")

sb.header("Otimização")
sb.radio("Objetivo", U.OBJECTIVES, key="objective")
sb.slider("λ do shrinkage", 0.0, 1.0, step=0.05, key="shrink_lambda",
          help="μ_aj = λ·μ_hist + (1−λ)·média(μ_hist). λ=1: histórico puro; λ=0: todos iguais")
sb.slider("Correlação máxima entre pares", *U.MAX_CORR_RANGE, step=0.01, key="max_corr",
          help="Para cada par de ativos com correlação acima deste valor, o otimizador limita "
               "w_i + w_j ≤ peso máximo por ativo (o par conta como um ativo só). 1,00 = sem restrição.")
sb.slider("Nº de pontos da fronteira", *U.FRONTIER_POINTS_RANGE, key="n_frontier")
sb.button("▶ Executar otimização", type="primary", on_click=cb_apply_weights, args=("_opt_hist",),
          width="stretch", help="Roda o SLSQP e preenche os pesos na tela")

# ============================================================================
# Página principal
# ============================================================================

st.title("📈 Calculadora de Carteira de ETFs")
for kind, msg in SS.flash:
    getattr(st, kind)(msg)
SS.flash = []

tickers = list(dict.fromkeys(tickers_raw))
if len(tickers) < 2:
    st.error("Informe ao menos 2 tickers ativos para calcular.")
    st.stop()

freq = U.FREQUENCIES[SS.freq]
periods = U.PERIODS_PER_YEAR[freq]
force = SS.force_pending
SS.force_pending = False
try:
    uni = cached_universe(tuple(tickers), D.normalize_ticker(SS.bench), SS.period, SS.base, freq, SS.nonce, force)
except D.FxMissingError as exc:
    st.error(f"⛔ {exc}. Cálculo bloqueado: não seguir sem conversão cambial.")
    st.stop()
except Exception as exc:
    st.error(f"Falha ao carregar dados: {exc}")
    st.stop()

for w in uni.warnings:
    st.warning(w)

rf_pct = float(SS.rf)
rf = M.rf_continuous(rf_pct)
conf = float(SS.var_level)
window = SS.roll_win if freq == "D" else max(4, round(SS.roll_win / 5))

asset_table = M.asset_metrics_table(uni.prices, uni.benchmark, periods, rf, conf, uni.missing)
if uni.missing:
    st.error(f"Tickers sem dados: {', '.join(uni.missing)} (linhas de NaN no relatório)")
    if not st.checkbox(f"Confirmo recalcular a carteira sem {', '.join(uni.missing)}", key="confirm_missing"):
        st.subheader("Métricas por ativo")
        st.dataframe(asset_table, column_config=metric_column_config(asset_table.columns))
        st.info("Marque a confirmação acima para calcular a carteira sem os tickers sem dados.")
        st.stop()
if uni.prices.shape[1] < 2:
    st.error("Menos de 2 ativos com dados: impossível calcular a carteira.")
    st.stop()

cols = list(uni.prices.columns)
w_input = pd.Series({D.normalize_ticker(SS[f"tk_{i}"]): SS[f"w_{i}"] / 100 for i in reversed(active_rows)})
w_cur = w_input.reindex(cols).fillna(0.0)
if w_cur.sum() <= 0:
    st.error("Todos os pesos dos ativos com dados são zero.")
    st.stop()
if abs(w_cur.sum() - 1) > U.WEIGHT_TOLERANCE / 100:
    st.warning(f"Pesos dos ativos com dados somam {w_cur.sum():.2%}; para os cálculos foram normalizados para 100%.")
w_cur = w_cur / w_cur.sum()

rets = M.log_returns(uni.prices)
rb = M.log_returns(uni.benchmark)
mu = P.mu_vector(rets, periods)
cov = P.cov_matrix(rets, periods)
corr = P.corr_matrix(rets)

lo, hi = SS.wbounds[0] / 100, SS.wbounds[1] / 100
bounds = tuple(O.make_bounds(len(cols), lo, hi))
max_corr = float(SS.max_corr)
corr_pairs = O.correlated_pairs(corr, max_corr)
corr_lin = O.pair_constraints(len(cols), corr_pairs, hi)
opt_hist, opt_shr, frontier = cached_optimization(rets, SS.objective, periods, rf, bounds,
                                                  float(SS.shrink_lambda), int(SS.n_frontier), max_corr)
SS._opt_hist, SS._opt_shr = opt_hist, opt_shr
w_opt = pd.Series(opt_hist["weights"]).reindex(cols)
w_shr = pd.Series(opt_shr["weights"]).reindex(cols)

# Tabelas comparativas
bench_name = uni.benchmark.name
comparison = pd.DataFrame({
    "Carteira atual": P.portfolio_metrics(rets, w_cur, rb, periods, rf, conf),
    f"Carteira otimizada ({SS.objective})": (P.portfolio_metrics(rets, w_opt.fillna(0), rb, periods, rf, conf)
                                             if opt_hist["success"] else {k: np.nan for k in M.metric_columns(conf)}),
    f"Benchmark ({bench_name})": M.metrics_from_returns(rb, rb, periods, rf, conf, uni.benchmark),
}).T
alloc = P.allocation(float(SS.capital), w_cur, uni.prices.iloc[-1], uni.currencies)

base100 = uni.prices / uni.prices.iloc[0] * 100
idx_cur = P.portfolio_index(rets, w_cur, uni.prices.index[0])
curves = pd.concat([idx_cur, uni.benchmark / uni.benchmark.iloc[0] * 100, base100], axis=1)
if opt_hist["success"]:
    curves.insert(1, "Carteira otimizada", P.portfolio_index(rets, w_opt, uni.prices.index[0]))

st.caption(
    f"Moeda base **{uni.base}** · {uni.n_obs} observações ({SS.freq.lower()}) de "
    f"{uni.prices.index[0]:%d/%m/%Y} a {uni.prices.index[-1]:%d/%m/%Y} · "
    f"{uni.rows_removed} linhas removidas no alinhamento · origem: {uni.source} · "
    f"Rf {rf_pct:.2f}% a.a. ({rf:.2%} contínua)"
)

tabs = st.tabs(["Resumo", "Ativos", "Correlação", "Otimização", "Drawdown", "Exportar"])

# ---------------------------------------------------------------- Resumo
with tabs[0]:
    cm = comparison.loc["Carteira atual"]
    k = st.columns(5)
    k[0].metric("Retorno anual", f"{cm['Retorno anual']:.2%}", help=M.FORMULAS["Retorno anual"])
    k[1].metric("Vol anual", f"{cm['Vol anual']:.2%}", help=M.FORMULAS["Vol anual"])
    k[2].metric("Sharpe", f"{cm['Sharpe']:.2f}", help=M.FORMULAS["Sharpe"])
    k[3].metric("Sortino", f"{cm['Sortino']:.2f}", help=M.FORMULAS["Sortino"])
    k[4].metric("Max DD", f"{cm['Max DD']:.2%}", help=M.FORMULAS["Max DD"])

    st.subheader("Carteira atual × otimizada × benchmark")
    st.dataframe(comparison, column_config=metric_column_config(comparison.columns))
    if not opt_hist["success"]:
        st.error(f"Otimizador não convergiu: {opt_hist['message']}. Pesos atuais mantidos.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_pie(w_cur, "Composição atual"), width="stretch")
    if opt_hist["success"]:
        c2.plotly_chart(plot_pie(w_opt, f"Composição otimizada ({SS.objective})"), width="stretch")

    st.plotly_chart(plot_lines(curves, f"Retorno acumulado (base 100, {uni.base})"), width="stretch")

    st.subheader(f"Alocação do capital ({uni.base} {SS.capital:,.2f})")
    st.dataframe(alloc, column_config={
        "Peso": st.column_config.NumberColumn(format="percent"),
        "Valor alocado": st.column_config.NumberColumn(format="%.2f"),
        "Preço (moeda base)": st.column_config.NumberColumn(format="%.2f"),
        "Nº de cotas": st.column_config.NumberColumn(format="%d", help="Cotas inteiras: floor(valor / preço)"),
        "Valor investido": st.column_config.NumberColumn(format="%.2f"),
    })

# ---------------------------------------------------------------- Ativos
with tabs[1]:
    st.subheader("Métricas por ativo")
    table = asset_table.copy()
    table.insert(0, "Peso", w_cur.reindex(table.index))
    table.insert(1, "Moeda original", [uni.currencies.get(t, "—") for t in table.index])
    st.dataframe(table, column_config={"Peso": st.column_config.NumberColumn(format="percent"),
                                       **metric_column_config(table.columns)})
    st.caption(f"VaR histórico por período ({SS.freq.lower()}), log-retorno. Passe o mouse no cabeçalho para ver a fórmula.")

# ---------------------------------------------------------------- Correlação
with tabs[2]:
    st.plotly_chart(plot_corr(corr), width="stretch")
    pairs = P.high_corr_pairs(corr, max_corr)
    if pairs.empty:
        st.success(f"Nenhum par com correlação > {max_corr:.2f}: a restrição de correlação não afeta o otimizador.")
    else:
        pairs["Soma dos pesos (atual)"] = [w_cur[a] + w_cur[b] for a, b in zip(pairs["Ativo 1"], pairs["Ativo 2"])]
        if opt_hist["success"]:
            pairs["Soma dos pesos (otimizada)"] = [w_opt[a] + w_opt[b] for a, b in zip(pairs["Ativo 1"], pairs["Ativo 2"])]
        pairs["Limite"] = hi
        st.warning(f"Pares com correlação > {max_corr:.2f} (possível overlap). No otimizador, cada par soma no "
                   f"máximo {hi:.0%} (peso máximo por ativo).")
        st.dataframe(pairs, hide_index=True, column_config={
            "Correlação": st.column_config.NumberColumn(format="%.3f"),
            **{c: st.column_config.NumberColumn(format="percent")
               for c in ("Soma dos pesos (atual)", "Soma dos pesos (otimizada)", "Limite")}})
        acima = pairs[pairs["Soma dos pesos (atual)"] > hi + 1e-9]
        if not acima.empty:
            st.error("A carteira atual excede o limite em: " +
                     ", ".join(f"{a} + {b}" for a, b in zip(acima["Ativo 1"], acima["Ativo 2"])))

# ---------------------------------------------------------------- Otimização
with tabs[3]:
    st.warning("⚠️ Max Sharpe sobre média histórica é sensível ao período; pesos extremos são esperados. "
               "Por isso os bounds mín./máx. são controles centrais, não opcionais. Compare com a versão com shrinkage.")
    for opt in (opt_hist, opt_shr):
        if not opt["success"]:
            st.error(f"{opt['label']}: não convergiu — {opt['message']}")

    weights_tbl = pd.DataFrame({"Atual": w_cur, "Otimizada (μ histórico)": w_opt,
                                f"Otimizada (shrinkage λ={SS.shrink_lambda:.2f})": w_shr})
    b1, b2 = st.columns(2)
    b1.button("Aplicar pesos otimizados (μ histórico)", on_click=cb_apply_weights, args=("_opt_hist",),
              width="stretch", type="primary")
    b2.button("Aplicar pesos com shrinkage", on_click=cb_apply_weights, args=("_opt_shr",), width="stretch")
    st.dataframe(weights_tbl, column_config={c: st.column_config.NumberColumn(format="percent") for c in weights_tbl})

    points = {"Carteira atual": (P.portfolio_vol(w_cur, cov), P.portfolio_return(w_cur, mu))}
    if opt_hist["success"]:
        points["Ótima (μ histórico)"] = (P.portfolio_vol(w_opt, cov), P.portfolio_return(w_opt, mu))
    if opt_shr["success"]:
        points["Ótima (shrinkage)"] = (P.portfolio_vol(w_shr, cov), P.portfolio_return(w_shr, mu))
    assets_pts = pd.DataFrame({"Vol": np.sqrt(np.diag(cov)), "Retorno": mu.values}, index=cols)
    if frontier.empty:
        st.error(O.check_feasible(list(bounds), corr_lin) or "Fronteira eficiente não pôde ser calculada.")
    st.plotly_chart(plot_frontier(frontier, assets_pts, points), width="stretch")
    st.caption("Retorno e σ pelo μ histórico; a carteira com shrinkage é plotada com o μ histórico para comparação."
               + (f" A fronteira respeita a restrição de correlação ({len(corr_pairs)} par(es) > {max_corr:.2f}); "
                  "a carteira atual pode ficar acima dela se não respeitar a restrição." if corr_pairs else ""))

    rc = pd.DataFrame({"Atual": P.risk_contribution(w_cur, cov)}, index=cols)
    if opt_hist["success"]:
        rc["Otimizada"] = P.risk_contribution(w_opt.to_numpy(), cov)
    fig_rc = px.bar(rc, barmode="group", title="Contribuição ao risco — RC_i = w_i(Σw)_i / σ_p²")
    fig_rc.update_layout(yaxis_tickformat=".0%", legend_title_text="", xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig_rc, width="stretch")

# ---------------------------------------------------------------- Drawdown
with tabs[4]:
    dd = pd.DataFrame({"Carteira": M.drawdown_series(idx_cur),
                       bench_name: M.drawdown_series(uni.benchmark)})
    if opt_hist["success"]:
        dd.insert(1, "Carteira otimizada", M.drawdown_series(curves["Carteira otimizada"]))
    st.plotly_chart(plot_lines(dd, "Drawdown", pct=True), width="stretch")

    rp = P.portfolio_returns(rets, w_cur)
    rvol = pd.DataFrame({"Carteira": M.rolling_vol(rp, window, periods), bench_name: M.rolling_vol(rb, window, periods)})
    rsh = pd.DataFrame({"Carteira": M.rolling_sharpe(rp, window, periods, rf),
                        bench_name: M.rolling_sharpe(rb, window, periods, rf)})
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_lines(rvol.dropna(), f"Volatilidade móvel ({window} períodos)", pct=True), width="stretch")
    c2.plotly_chart(plot_lines(rsh.dropna(), f"Sharpe móvel ({window} períodos)"), width="stretch")

# ---------------------------------------------------------------- Exportar
with tabs[5]:
    relatorio = table.copy()
    relatorio["Valor alocado"] = alloc["Valor alocado"].reindex(relatorio.index)
    relatorio["Nº de cotas"] = alloc["Nº de cotas"].reindex(relatorio.index)
    pesos_otimos = weights_tbl.copy()
    pesos_otimos.index.name = "ticker"
    export_tables = {
        "relatorio": (U.OUTPUT_FILES["relatorio"], R.fmt_table(relatorio)),
        "consolidado": (U.OUTPUT_FILES["consolidado"], R.fmt_table(comparison)),
        "correlacao": (U.OUTPUT_FILES["correlacao"], R.fmt_table(corr)),
        "pesos": (U.OUTPUT_FILES["pesos"], R.fmt_table(pesos_otimos)),
        "fronteira": (U.OUTPUT_FILES["fronteira"], R.fmt_table(frontier)),
    }
    st.caption("CSV no padrão do Excel em português (separador `;`, decimal `,`).")
    dl = st.columns(3)
    for n, (fn, df) in enumerate(export_tables.values()):
        dl[n % 3].download_button(f"⬇ {fn}", R.csv_bytes(df), fn, "text/csv", width="stretch")
    dl[len(export_tables) % 3].download_button(
        f"⬇ {U.OUTPUT_FILES['excel']}",
        R.excel_bytes({fn.removesuffix(".csv"): df for fn, df in export_tables.values()}),
        U.OUTPUT_FILES["excel"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch")

    st.divider()
    if st.button("Gerar tudo em outputs/ (CSV, Excel e PNGs)", type="primary"):
        with st.spinner("Gerando figuras…"):
            figs = {
                "composicao": R.fig_composition(w_cur),
                "retorno_acumulado": R.fig_lines(curves, f"Retorno acumulado (base 100, {uni.base})", "Base 100"),
                "drawdown": R.fig_lines(dd, "Drawdown", "", pct=True),
                "correlacao": R.fig_corr(corr),
                "fronteira": R.fig_frontier(frontier, points, assets_pts),
                "contribuicao_risco": R.fig_bars(rc["Atual"], "Contribuição ao risco (carteira atual)"),
                "vol_movel": R.fig_lines(rvol.dropna(), f"Volatilidade móvel ({window})", "", pct=True),
                "sharpe_movel": R.fig_lines(rsh.dropna(), f"Sharpe móvel ({window})", ""),
            }
            if opt_hist["success"]:
                figs["composicao_otimizada"] = R.fig_composition(w_opt, f"Composição otimizada ({SS.objective})")
            written = R.save_outputs(export_tables, figs)
            files = {p.name: p.read_bytes() for p in written}
            SS._zip = R.zip_bytes(files)
        st.success(f"{len(written)} arquivos gravados em {U.OUTPUT_DIR}")
    if SS.get("_zip"):
        st.download_button("⬇ Baixar tudo (.zip)", SS._zip, "calculadora_etf_outputs.zip", "application/zip")
