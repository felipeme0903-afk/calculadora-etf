"""Presets de ETFs, benchmarks e defaults de todos os parâmetros de cálculo.

Nenhum parâmetro de cálculo deve ficar hard-coded fora deste módulo.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# --- Carteira -----------------------------------------------------------------
MIN_ASSETS = 5
MAX_ASSETS = 8
DEFAULT_N_ASSETS = 6
DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_WEIGHT_BOUNDS = (5.0, 40.0)  # % mínimo / máximo por ativo
WEIGHT_TOLERANCE = 0.01  # tolerância (em p.p.) para considerar soma = 100%

# Presets: ticker de exibição (sem .SA) -> peso em %
PRESETS: dict[str, dict[str, float]] = {
    "IA Core": {
        "QQQ": 25.0, "SMH": 20.0, "AIQ": 15.0, "BOTZ": 10.0, "IGV": 15.0, "NASD11": 15.0,
    },
    "Semicondutores": {
        "SMH": 25.0, "SOXX": 25.0, "XSD": 15.0, "PSI": 15.0, "SOXQ": 20.0,
    },
    "Global + BR": {
        "ACWI": 20.0, "IVV": 15.0, "VWO": 10.0, "IVVB11": 15.0,
        "BOVA11": 20.0, "SMAL11": 10.0, "HASH11": 10.0,
    },
}
DEFAULT_PRESET = "IA Core"

# --- Dados --------------------------------------------------------------------
PERIODS = ["1y", "2y", "3y", "5y"]
DEFAULT_PERIOD = "5y"
FREQUENCIES = {"Diário": "D", "Semanal": "W"}
DEFAULT_FREQUENCY = "Diário"
PERIODS_PER_YEAR = {"D": 252, "W": 52}
WEEKLY_RULE = "W-FRI"
BASE_CURRENCIES = ["BRL", "USD"]
DEFAULT_BASE_CURRENCY = "BRL"
FX_TICKER = "BRL=X"  # BRL por 1 USD
B3_SUFFIX = ".SA"
FFILL_LIMIT = 2
CACHE_MAX_AGE_HOURS = 24
MIN_OBS_RELIABLE = {"D": 252, "W": 52}

# Moeda conhecida de alguns símbolos (evita consulta de rede); o resto é detectado.
KNOWN_CURRENCY = {
    "^NDX": "USD", "^GSPC": "USD", "^SOX": "USD", "^BVSP": "BRL",
    "ACWI": "USD", "URTH": "USD", "QQQ": "USD", "SMH": "USD", "SOXX": "USD",
}

# --- Risco --------------------------------------------------------------------
BENCHMARKS = ["^NDX", "^GSPC", "^SOX", "^BVSP", "ACWI", "URTH"]
DEFAULT_BENCHMARK = "^NDX"
# Taxa livre de risco em % a.a. — valores de referência; atualize para o valor corrente.
DEFAULT_RF = {"BRL": 14.90, "USD": 4.00}  # CDI / T-bill 3M
RF_LABEL = {"BRL": "CDI", "USD": "T-bill"}
VAR_LEVELS = [0.95, 0.99]
DEFAULT_VAR_LEVEL = 0.95
ROLLING_WINDOW_RANGE = (21, 252)
DEFAULT_ROLLING_WINDOW = 63
DEFAULT_MAX_CORR = 0.85  # pares acima disso: w_i + w_j ≤ peso máximo por ativo
MAX_CORR_RANGE = (0.50, 1.00)  # 1,00 = sem restrição

# --- Otimização ---------------------------------------------------------------
OBJECTIVES = ["Max Sharpe", "Min Vol", "Max Sortino", "Risk Parity", "Igual"]
DEFAULT_OBJECTIVE = "Max Sharpe"
FRONTIER_POINTS_RANGE = (20, 200)
DEFAULT_FRONTIER_POINTS = 50
DEFAULT_SHRINKAGE = 0.5  # lambda: mu_aj = lambda*mu_hist + (1-lambda)*média(mu_hist)
OPT_MAX_ITER = 500
OPT_FTOL = 1e-10

# --- Saídas -------------------------------------------------------------------
OUTPUT_FILES = {
    "relatorio": "relatorio_carteira.csv",
    "consolidado": "consolidado.csv",
    "correlacao": "correlacao.csv",
    "pesos": "pesos_otimos.csv",
    "fronteira": "fronteira.csv",
    "excel": "relatorio.xlsx",
}
