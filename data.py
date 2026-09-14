"""Download em lote, detecção de moeda, conversão cambial, alinhamento e cache.

Sem st.*: erros de dados viram NaN + warning (logging), nunca exceção não tratada.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import universe as U

log = logging.getLogger(__name__)


class FxMissingError(RuntimeError):
    """BRL=X ausente quando a conversão cambial é necessária."""


# --- Tickers ------------------------------------------------------------------

def normalize_ticker(t: str) -> str:
    return str(t).strip().upper()


def to_yahoo(t: str) -> str:
    """Ticker terminado em dígito (NASD11, IVVB11) recebe .SA; índices (^) e alfabéticos não."""
    t = normalize_ticker(t)
    if t and not t.startswith("^") and "." not in t and "=" not in t and t[-1].isdigit():
        return t + U.B3_SUFFIX
    return t


def to_display(symbol: str) -> str:
    s = normalize_ticker(symbol)
    return s[: -len(U.B3_SUFFIX)] if s.endswith(U.B3_SUFFIX) else s


# --- Cache --------------------------------------------------------------------

def _cache_path(symbols: list[str], period: str):
    key = "|".join(sorted(symbols)) + "|" + period
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return U.DATA_DIR / f"prices_{digest}.csv"


def _cache_fresh(path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < U.CACHE_MAX_AGE_HOURS * 3600


# --- Download -----------------------------------------------------------------

def _extract_adj_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Extrai Adj Close de um retorno de yf.download(group_by='ticker')."""
    out = {}
    if raw is None or raw.empty:
        return pd.DataFrame(columns=symbols, dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        for s in symbols:
            if s in lvl0 and "Adj Close" in raw[s].columns:
                out[s] = raw[s]["Adj Close"]
            elif "Adj Close" in lvl0 and s in raw["Adj Close"].columns:  # layout alternativo
                out[s] = raw["Adj Close"][s]
            else:
                out[s] = pd.Series(np.nan, index=raw.index)
    else:  # um único ticker sem MultiIndex
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        out[symbols[0]] = raw[col]
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
    return df.sort_index().astype(float)


def download_prices(symbols: list[str], period: str, force: bool = False) -> tuple[pd.DataFrame, str]:
    """Adj Close de todos os símbolos numa única chamada. Retorna (preços, origem)."""
    symbols = list(dict.fromkeys(symbols))
    U.DATA_DIR.mkdir(exist_ok=True)
    path = _cache_path(symbols, period)
    if not force and _cache_fresh(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(symbols) <= set(df.columns):
            log.info("Cache usado: %s", path.name)
            return df[symbols], f"cache ({path.name})"

    import yfinance as yf

    log.info("Baixando %d símbolos (%s): %s", len(symbols), period, symbols)
    try:
        raw = yf.download(symbols, period=period, interval="1d", auto_adjust=False,
                          group_by="ticker", threads=True, progress=False)
    except Exception as exc:  # rede, rate limit etc.
        log.warning("Falha no download: %s", exc)
        raw = pd.DataFrame()
    df = _extract_adj_close(raw, symbols)
    if not df.dropna(how="all").empty:
        df.to_csv(path)
    return df, "download"


def detect_currency(symbol: str) -> str:
    symbol = normalize_ticker(symbol)
    if symbol.endswith(U.B3_SUFFIX):
        return "BRL"
    disp = to_display(symbol)
    if disp in U.KNOWN_CURRENCY:
        return U.KNOWN_CURRENCY[disp]

    cache_file = U.DATA_DIR / "currencies.json"
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    if symbol in cache:
        return cache[symbol]

    currency = None
    try:
        import yfinance as yf
        currency = yf.Ticker(symbol).fast_info.get("currency")
    except Exception as exc:
        log.warning("Moeda de %s não detectada (%s)", symbol, exc)
    if not currency:
        log.warning("Moeda de %s desconhecida; assumindo USD", symbol)
        return "USD"
    currency = str(currency).upper()
    U.DATA_DIR.mkdir(exist_ok=True)
    cache[symbol] = currency
    cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return currency


# --- Conversão e alinhamento --------------------------------------------------

def convert_to_base(prices: pd.DataFrame, currencies: dict[str, str], fx: pd.Series | None,
                    base: str) -> tuple[pd.DataFrame, list[str]]:
    """USD->BRL: × BRL=X; BRL->USD: ÷ BRL=X. Outras moedas viram NaN com warning."""
    out = prices.copy()
    warnings = []
    for col in prices.columns:
        cur = currencies.get(col, "USD")
        if cur == base:
            continue
        if cur not in ("USD", "BRL"):
            msg = f"{col}: moeda {cur} não suportada para conversão; ativo ignorado"
            log.warning(msg)
            warnings.append(msg)
            out[col] = np.nan
            continue
        if fx is None or fx.dropna().empty:
            raise FxMissingError(f"{U.FX_TICKER} ausente: não é possível converter {col} de {cur} para {base}")
        fx_aligned = fx.reindex(out.index).ffill(limit=U.FFILL_LIMIT)
        out[col] = out[col] * fx_aligned if base == "BRL" else out[col] / fx_aligned
    return out, warnings


def align(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Alinha calendários EUA/B3: ffill limitado, depois dropna inner."""
    filled = prices.sort_index().ffill(limit=U.FFILL_LIMIT)
    aligned = filled.dropna(how="any")
    removed = len(prices) - len(aligned)
    log.info("Alinhamento: %d linhas brutas, %d removidas, %d restantes", len(prices), removed, len(aligned))
    return aligned, removed


def resample(prices: pd.DataFrame | pd.Series, freq: str):
    """Semanal: último preço de cada semana, indexado pela data real desse pregão."""
    if freq != "W":
        return prices
    weekly = prices.resample(U.WEEKLY_RULE).last()
    weekly.index = pd.DatetimeIndex(prices.index.to_series().resample(U.WEEKLY_RULE).last())
    return weekly.dropna(how="all")


# --- Orquestração -------------------------------------------------------------

@dataclass
class Universe:
    prices: pd.DataFrame            # ativos com dados, moeda base, alinhados (colunas = ticker de exibição)
    benchmark: pd.Series            # benchmark na moeda base, mesmo índice
    currencies: dict[str, str]      # moeda original por ticker de exibição (inclui benchmark)
    missing: list[str]              # tickers sem dados
    rows_raw: int
    rows_removed: int
    source: str
    base: str
    freq: str
    warnings: list[str] = field(default_factory=list)

    @property
    def n_obs(self) -> int:
        return len(self.prices)


def load_universe(tickers: list[str], benchmark: str, period: str, base: str,
                  freq: str = "D", force: bool = False) -> Universe:
    tickers = [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
    benchmark = normalize_ticker(benchmark)
    asset_syms = [to_yahoo(t) for t in tickers]
    bench_sym = to_yahoo(benchmark)
    symbols = list(dict.fromkeys(asset_syms + [bench_sym, U.FX_TICKER]))

    raw, source = download_prices(symbols, period, force=force)
    raw = raw.rename(columns=to_display)
    fx = raw[U.FX_TICKER] if U.FX_TICKER in raw.columns else None
    warnings: list[str] = []

    display_all = list(dict.fromkeys([to_display(s) for s in asset_syms] + [benchmark]))
    missing = [t for t in display_all if t not in raw.columns or raw[t].dropna().empty]
    for t in missing:
        msg = f"{t}: sem dados retornados"
        log.warning(msg)
        warnings.append(msg)
    if benchmark in missing:
        raise ValueError(f"Benchmark {benchmark} sem dados")

    available = [t for t in display_all if t not in missing]
    currencies = {t: detect_currency(to_yahoo(t)) for t in available}
    converted, conv_warn = convert_to_base(raw[available], currencies, fx, base)
    warnings += conv_warn
    unsupported = [c for c in converted.columns if converted[c].dropna().empty]
    missing += [c for c in unsupported if c not in missing]
    converted = converted.drop(columns=unsupported)

    aligned, removed = align(converted)
    aligned = resample(aligned, freq)
    asset_cols = [t for t in dict.fromkeys(to_display(s) for s in asset_syms) if t in aligned.columns]
    bench = aligned[benchmark].rename(benchmark)
    if len(aligned) < U.MIN_OBS_RELIABLE[freq]:
        warnings.append(f"Apenas {len(aligned)} observações alinhadas: métricas anualizadas são pouco confiáveis")
    return Universe(prices=aligned[asset_cols], benchmark=bench, currencies=currencies,
                    missing=[m for m in missing if m != benchmark], rows_raw=len(converted),
                    rows_removed=removed, source=source, base=base, freq=freq, warnings=warnings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    preset = U.PRESETS[U.DEFAULT_PRESET]
    uni = load_universe(list(preset), U.DEFAULT_BENCHMARK, U.DEFAULT_PERIOD, "BRL")
    print(f"Moeda base: {uni.base} | origem: {uni.source}")
    print("Moedas originais:", uni.currencies)
    print("Sem dados:", uni.missing)
    print(uni.prices.head())
    print(f"Linhas alinhadas: {uni.n_obs} (removidas {uni.rows_removed} de {uni.rows_raw})")
    print("NaN restantes:", int(uni.prices.isna().sum().sum() + uni.benchmark.isna().sum()))
