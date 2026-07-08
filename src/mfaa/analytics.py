#!/usr/bin/env python3
"""
Professional performance analytics for the MFAA defensive framework.

Everything here operates on a daily NAV/equity ``pd.Series``. For a *defensive*
strategy the headline number is the **excess-over-cash Sharpe** (a client can
always hold T-bills), so metrics that can be computed net of a cash proxy are.
Nothing in this module fits or selects anything -- it only measures.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def _excess(rets: pd.Series, cash_rets: Optional[pd.Series]) -> pd.Series:
    if cash_rets is None:
        return rets
    return rets - cash_rets.reindex(rets.index).fillna(0.0)


def cash_returns_from_prices(cash_prices: Optional[pd.Series], index: pd.Index) -> Optional[pd.Series]:
    """Daily cash returns (e.g. from BIL) aligned to ``index``."""
    if cash_prices is None:
        return None
    return cash_prices.pct_change().reindex(index).fillna(0.0)


# --------------------------------------------------------------------------- #
# scalar metrics
# --------------------------------------------------------------------------- #
def cagr(equity: pd.Series) -> float:
    total = equity.iloc[-1] / equity.iloc[0]
    days = (equity.index[-1] - equity.index[0]).days or 1
    return float(total ** (365.0 / days) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    return float(-(equity / equity.cummax() - 1.0).min())


def ulcer_index(equity: pd.Series) -> float:
    """Root-mean-square drawdown (%). Penalises deep, long drawdowns."""
    dd = (equity / equity.cummax() - 1.0) * 100.0
    return float(np.sqrt((dd ** 2).mean()))


def tail_ratio(rets: pd.Series) -> float:
    """|95th pctile| / |5th pctile| of daily returns; > 1 means fatter right tail."""
    left = abs(np.percentile(rets, 5))
    right = abs(np.percentile(rets, 95))
    return float(right / left) if left > 0 else float("nan")


def sharpe(rets: pd.Series) -> float:
    return float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else float("nan")


def sortino(rets: pd.Series) -> float:
    downside = rets[rets < 0]
    if len(downside) < 2 or downside.std() == 0:
        return float("nan")
    return float(rets.mean() / downside.std() * np.sqrt(TRADING_DAYS))


def benchmark_stats(rets: pd.Series, bench_rets: pd.Series) -> Dict:
    """Beta, annualised alpha, alpha t-stat, R^2, Information Ratio, correlation."""
    common = rets.index.intersection(bench_rets.index)
    pr, br = rets.loc[common], bench_rets.loc[common]
    if len(common) < 30 or br.var() == 0:
        return {}
    beta = float(pr.cov(br) / br.var())
    resid = pr - beta * br
    alpha_daily = float(resid.mean())
    # OLS t-stat of the intercept (alpha)
    n = len(common)
    x = br.values
    sxx = float(((x - x.mean()) ** 2).sum())
    resid_full = pr.values - (alpha_daily + beta * x)
    dof = max(1, n - 2)
    s2 = float((resid_full ** 2).sum() / dof)
    se_alpha = float(np.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx))) if sxx > 0 else float("nan")
    t_alpha = alpha_daily / se_alpha if se_alpha and se_alpha > 0 else float("nan")
    active = pr - br
    info_ratio = float(active.mean() / active.std() * np.sqrt(TRADING_DAYS)) if active.std() > 0 else float("nan")
    corr = float(pr.corr(br))
    return {
        "beta": round(beta, 3),
        "alpha_ann_pct": round(alpha_daily * TRADING_DAYS * 100, 2),
        "alpha_tstat": round(float(t_alpha), 2),
        "r_squared": round(corr ** 2, 3),
        "information_ratio": round(info_ratio, 2),
        "correlation": round(corr, 2),
    }


def full_metrics(
    equity: pd.Series,
    cash_prices: Optional[pd.Series] = None,
    benchmark_prices: Optional[pd.Series] = None,
    label: str = "strategy",
) -> Dict:
    """A professional metric block: total + excess-of-cash + benchmark-relative."""
    rets = _daily_returns(equity)
    cash_rets = cash_returns_from_prices(cash_prices, rets.index)
    ex = _excess(rets, cash_rets)

    m = {
        "label": label,
        "cagr_pct": round(cagr(equity) * 100, 2),
        "ann_vol_pct": round(float(rets.std() * np.sqrt(TRADING_DAYS)) * 100, 2),
        "sharpe": round(sharpe(rets), 2),
        "sharpe_excess_cash": round(sharpe(ex), 2),
        "sortino_excess_cash": round(sortino(ex), 2),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "calmar": round(cagr(equity) / max_drawdown(equity), 2) if max_drawdown(equity) > 0 else float("nan"),
        "ulcer_index": round(ulcer_index(equity), 2),
        "tail_ratio": round(tail_ratio(rets), 2),
        "skew": round(float(rets.skew()), 2),
        "kurtosis": round(float(rets.kurt()), 2),
        "best_month_pct": None,
        "worst_month_pct": None,
        "pos_month_pct": None,
    }
    monthly = equity.groupby(equity.index.to_period("M")).last().pct_change().dropna()
    if len(monthly):
        m["best_month_pct"] = round(float(monthly.max()) * 100, 2)
        m["worst_month_pct"] = round(float(monthly.min()) * 100, 2)
        m["pos_month_pct"] = round(float((monthly > 0).mean()) * 100, 1)

    if benchmark_prices is not None:
        bench_rets = benchmark_prices.pct_change().reindex(rets.index).dropna()
        m.update({f"vs_bench_{k}": v for k, v in benchmark_stats(rets, bench_rets).items()})
    return m


# --------------------------------------------------------------------------- #
# time-series diagnostics
# --------------------------------------------------------------------------- #
def rolling_sharpe(equity: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    r = _daily_returns(equity)
    return (r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(TRADING_DAYS)).dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def crisis_table(
    curves: Dict[str, pd.Series],
    episodes: Optional[List[Tuple[str, str, str]]] = None,
) -> pd.DataFrame:
    """Peak-to-trough return of each curve over a set of named stress episodes.

    ``curves`` maps a label -> equity Series. ``episodes`` is a list of
    (name, start, end); defaults to the major post-2020 drawdowns.
    """
    if episodes is None:
        episodes = [
            ("COVID crash", "2020-02-19", "2020-03-23"),
            ("2022 bear", "2022-01-03", "2022-10-12"),
            ("2023 SVB scare", "2023-02-02", "2023-03-13"),
            ("2025 tariff shock", "2025-02-19", "2025-04-08"),
        ]
    rows = []
    for name, s, e in episodes:
        row = {"episode": name, "window": f"{s} .. {e}"}
        for label, curve in curves.items():
            seg = curve.loc[s:e]
            row[label] = round(float(seg.iloc[-1] / seg.iloc[0] - 1.0) * 100, 1) if len(seg) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# tearsheet
# --------------------------------------------------------------------------- #
def tearsheet(
    equity: pd.Series,
    benchmarks: Dict[str, pd.Series],
    out_path: str,
    title: str = "MFAA defensive framework",
    rolling_window: int = TRADING_DAYS,
) -> None:
    """Multi-panel institutional-style tearsheet PNG (growth / drawdown / rolling Sharpe / annual bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.32, wspace=0.22)

    # 1) growth of 10k (log)
    ax1 = fig.add_subplot(gs[0, :])
    (equity / equity.iloc[0]).plot(ax=ax1, color="#ff7f0e", lw=1.8, label="MFAA")
    for name, curve in benchmarks.items():
        c = curve.reindex(equity.index).dropna()
        (c / c.iloc[0]).plot(ax=ax1, lw=1.0, alpha=0.8, label=name)
    ax1.set_yscale("log")
    ax1.set_title(title, fontsize=13, weight="bold")
    ax1.set_ylabel("growth of 1 (log)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, ls="--", alpha=0.4)

    # 2) underwater / drawdown
    ax2 = fig.add_subplot(gs[1, :])
    dd = drawdown_series(equity) * 100
    ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("drawdown %")
    ax2.set_title("Underwater plot", fontsize=11)
    ax2.grid(True, ls="--", alpha=0.4)

    # 3) rolling 1y Sharpe
    ax3 = fig.add_subplot(gs[2, 0])
    rs = rolling_sharpe(equity, rolling_window)
    rs.plot(ax=ax3, color="#1f77b4", lw=1.2)
    ax3.axhline(0, color="k", lw=0.6)
    ax3.axhline(1, color="green", lw=0.6, ls="--", alpha=0.6)
    ax3.set_title(f"Rolling {rolling_window}d Sharpe", fontsize=11)
    ax3.grid(True, ls="--", alpha=0.4)

    # 4) annual returns bar
    ax4 = fig.add_subplot(gs[2, 1])
    yr = equity.groupby(equity.index.year).last()
    prev = yr.shift(1)
    prev.iloc[0] = equity.iloc[0]
    ann = (yr / prev - 1.0) * 100
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ann.values]
    ax4.bar([str(y) for y in ann.index], ann.values, color=colors)
    ax4.set_title("Annual return %", fontsize=11)
    ax4.tick_params(axis="x", rotation=45, labelsize=8)
    ax4.grid(True, ls="--", alpha=0.4, axis="y")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
