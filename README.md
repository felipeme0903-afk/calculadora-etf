# Calculadora de Carteira de ETFs

App local em Streamlit. Recebe de 5 a 8 ETFs (globais ou da B3), baixa as cotações, converte tudo para uma moeda base (BRL ou USD) e calcula métricas de risco, retorno e retorno ajustado ao risco. Também otimiza os pesos sob restrições definidas na tela.

## Instalação e uso

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Testes (sem rede, usam dados sintéticos): `.venv\Scripts\python -m pytest -q`
Teste de aceitação da etapa 1 (com rede): `.venv\Scripts\python data.py`

## Módulos

| Arquivo | Papel |
|---|---|
| `app.py` | Interface Streamlit (única camada com `st.*`) |
| `universe.py` | Presets de ETFs, benchmarks e defaults de todos os parâmetros |
| `data.py` | Download em lote, detecção de moeda, conversão cambial, alinhamento, cache |
| `metrics.py` | Funções puras: retornos, vol, beta, drawdown, Sharpe, Sortino, Treynor, Calmar, IR, VaR, TE |
| `portfolio.py` | Métricas de carteira, correlação, covariância, contribuição ao risco, alocação |
| `optimizer.py` | Max Sharpe, Min Vol, Max Sortino, Risk Parity, pesos iguais, fronteira eficiente (SLSQP) |
| `report.py` | Exportação CSV / Excel / PNG |
| `data/` | Cache de preços (fora do git) |
| `outputs/` | Relatórios gerados (fora do git) |

## Convenções de cálculo

- **Retornos:** log-retornos, `r_t = ln(P_t / P_{t-1})`.
- **Retorno anualizado:** `média(r) × 252` (ou `× 52` no modo semanal). É uma taxa contínua e vale para todas as métricas. O CAGR aparece só como coluna informativa.
- **Taxa livre de risco:** digitada em % a.a. efetiva e convertida para contínua, `ln(1 + Rf)`, para ficar coerente com os log-retornos. Os valores padrão em `universe.py` (CDI 14,90%, T-bill 4,00%) são só referência: **atualize-os**.
- **Série da carteira:** `r_p = Σ w_i r_i`. Assim, `média(r_p)×252 = W'μ` e `desvio(r_p)×√252 = √(W'ΣW)`, as mesmas fórmulas usadas no otimizador.
- **Sortino:** `DesvioNeg = √(média(min(r,0)²)) × √252`.
- **Conversão cambial:** ativo em USD com base BRL é multiplicado por `BRL=X`. Ativo em BRL com base USD é dividido por `BRL=X`.
- **Correlação máxima (otimizador):** para cada par de ativos com correlação acima do limite da tela, `w_i + w_j ≤ peso máximo somado por par` (default 30%). Com correlação máxima 1,00 não há restrição. Por padrão a correlação da restrição é medida em base **semanal**: ativos da B3 e dos EUA fecham em horários diferentes e a correlação diária entre eles sai subestimada (ex.: QQQ × NASD11 ≈ 0,68 diária e 0,92 semanal). A aba Correlação mostra, par a par, a soma dos pesos sem a restrição, com ela e na carteira atual. A restrição é por par: três ou mais ativos correlacionados entre si podem somar mais que o limite no total.
- **Alinhamento:** `ffill(limit=2)`, depois `dropna` (junção inner).
- **CSVs exportados:** padrão do Excel em português (separador `;`, decimal `,`).

## Formato do CSV de carteira

```
ticker;peso
QQQ;25
NASD11;15
```

A vírgula também serve como separador. Pesos que somam 1 são lidos como frações.
