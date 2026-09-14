"""Exportação CSV / Excel / PNG. Sem st.*."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import universe as U  # noqa: E402

log = logging.getLogger(__name__)

# CSV no padrão do Excel em português: separador ';' e decimal ','
CSV_KW = dict(sep=";", decimal=",", encoding="utf-8-sig")


def csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    return df.to_csv(index=index, sep=CSV_KW["sep"], decimal=CSV_KW["decimal"]).encode(CSV_KW["encoding"])


def excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31])
            ws = xw.sheets[name[:31]]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 40)
    return buf.getvalue()


# --- Figuras (Matplotlib, para colar no trabalho) -----------------------------

def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_composition(weights: pd.Series, title: str = "Composição da carteira") -> bytes:
    w = weights[weights > 1e-6]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(w.values, labels=w.index, autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.set_title(title)
    return _png(fig)


def fig_lines(df: pd.DataFrame, title: str, ylabel: str, pct: bool = False) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in df.columns:
        lw = 2.4 if c in ("Carteira", "Carteira otimizada") else 1.1
        ax.plot(df.index, df[c], label=c, linewidth=lw)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return _png(fig)


def fig_corr(corr: pd.DataFrame) -> bytes:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1 + 0.9 * n, 0.8 + 0.8 * n))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlação (log-retornos)")
    return _png(fig)


def fig_frontier(frontier: pd.DataFrame, points: dict[str, tuple[float, float]],
                 assets: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 6))
    if not frontier.empty:
        ax.plot(frontier["Vol"], frontier["Retorno"], label="Fronteira eficiente", linewidth=2)
    ax.scatter(assets["Vol"], assets["Retorno"], color="gray")
    for name, row in assets.iterrows():
        ax.annotate(name, (row["Vol"], row["Retorno"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    for (label, (vol, ret)), marker in zip(points.items(), ["o", "*", "D", "s"]):
        ax.scatter([vol], [ret], s=140, marker=marker, label=label, zorder=5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Volatilidade anual (σ)")
    ax.set_ylabel("Retorno anual (μ)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Fronteira eficiente")
    return _png(fig)


def fig_bars(values: pd.Series, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(values.index, values.values)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return _png(fig)


# --- Gravação -----------------------------------------------------------------

def save_outputs(tables: dict[str, tuple[str, pd.DataFrame]], figures: dict[str, bytes],
                 out_dir: Path | None = None) -> list[Path]:
    """tables: chave -> (nome do arquivo CSV, DataFrame). Grava CSVs, relatorio.xlsx e PNGs."""
    out_dir = Path(out_dir or U.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, df in tables.values():
        p = out_dir / filename
        p.write_bytes(csv_bytes(df))
        written.append(p)
    xlsx = out_dir / U.OUTPUT_FILES["excel"]
    xlsx.write_bytes(excel_bytes({Path(fn).stem: df for fn, df in tables.values()}))
    written.append(xlsx)
    for name, data in figures.items():
        p = out_dir / f"{name}.png"
        p.write_bytes(data)
        written.append(p)
    log.info("Saídas gravadas em %s (%d arquivos)", out_dir, len(written))
    return written


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    """Arredonda para exportação legível sem perder a natureza numérica."""
    return df.apply(lambda s: s.round(6) if pd.api.types.is_numeric_dtype(s) else s)
