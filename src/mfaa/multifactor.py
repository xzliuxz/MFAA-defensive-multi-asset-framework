#!/usr/bin/env python3
"""
MFAA -- Multi-Factor Adaptive Allocation (defensive multi-asset framework).

A price-only, long-only multi-asset strategy whose rules and parameters are all
fixed ex-ante from published research -- pre-2020 except the BAA (2022) canary and
defensive-pool elements (see docs/methodology.md for sources, caveats and the
anti-overfitting protocol). Design goal: a *defensive*
allocation -- roughly half the volatility and drawdown of equities, low beta,
positive risk-managed carry -- not a return-maximiser.

Three factor sleeves, combined and then de-risked by a crash overlay:

1. XSMOM sleeve  -- cross-sectional multi-horizon momentum: rank risky assets
   by the average of 1/3/6/12-month returns ("13612" filter, Keller & Keuning
   2016; Jegadeesh & Titman 1993), keep the top N that also beat cash momentum
   (absolute momentum, Antonacci 2014), weight them by blended inverse volatility.
2. TSTREND sleeve -- time-series trend following with *fractional* participation:
   each risky asset earns its inverse-vol budget scaled by the fraction of a
   {6,10,12}-month SMA ensemble it trades above (Faber 2007; Moskowitz, Ooi &
   Pedersen 2012). Off-trend budget moves to the defensive asset.
3. Canary overlay -- breadth of fast "13612W" momentum across
   {SPY, EFA, EEM, AGG} scales the whole book toward the defensive asset
   (Keller & Keuning 2017 VAA / 2022 BAA canary logic).

**Ensembling as an anti-overfitting device.** The trend signal averages over
three lookbacks and the volatility estimate averages over three windows, so no
single hand-picked parameter drives the result -- a standard robustness practice
in the trend-following literature and the opposite of curve-fitting one magic
number. The defensive slot is itself chosen by relative momentum within a short-
duration pool (BAA-style). Final weights are scaled by a realised-volatility
target (Moreira & Muir 2017) with a hard leverage cap.

Signals are computed on month-end closes; positions apply from the next trading
day (no same-bar execution, no look-ahead). The backtest charges proportional
transaction costs on turnover plus financing on gross exposure above 1.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# --- Universe (all ETFs with inception <= mid-2007; mirrors the classic
# multi-asset universes of Faber 2007 GTAA / Butler et al. 2012 AAA /
# Keller's PAA-BAA papers; fixed ex-ante, never selected on post-2020 data).
RISKY_ASSETS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "LQD", "GLD", "DBC"]
DEFENSIVE_POOL = ["IEF", "TIP", "BIL"]
CANARY_ASSETS = ["SPY", "EFA", "EEM", "AGG"]
CASH_PROXY = "BIL"
ALL_TICKERS = sorted(set(RISKY_ASSETS + DEFENSIVE_POOL + CANARY_ASSETS))

MOM_HORIZONS = (1, 3, 6, 12)             # months; "13612" multi-horizon momentum
CANARY_WEIGHTS = (12.0, 4.0, 2.0, 1.0)   # "13612W" weighting for the canary
TREND_MONTHS = (6, 10, 12)               # SMA lookbacks ensembled for participation
VOL_WINDOWS = (21, 63, 126)              # daily-vol windows ensembled for weighting


def momentum_13612(monthly: pd.DataFrame, weighted: bool = False) -> pd.Series:
    """Multi-horizon momentum from a month-end close frame (last row = "now").

    Plain version averages the 1/3/6/12-month total returns; the weighted
    version uses Keller's 13612W weights (12,4,2,1)/19, which reacts faster
    and is used for the canary. Assets without 13 monthly closes get NaN.
    """
    if len(monthly) < 13:
        return pd.Series(np.nan, index=monthly.columns)
    p0 = monthly.iloc[-1]
    rets = [p0 / monthly.iloc[-1 - h] - 1.0 for h in MOM_HORIZONS]
    if weighted:
        w = np.array(CANARY_WEIGHTS)
        score = sum(r * wi for r, wi in zip(rets, w)) / w.sum()
    else:
        score = sum(rets) / len(rets)
    return score


def inverse_vol_weights(daily_rets: pd.DataFrame, assets: List[str], window: int = 63) -> pd.Series:
    """Inverse-volatility weights over ``assets`` for a single trailing window.

    Assets with fewer than window/2 observations are dropped. Returns weights
    summing to 1 (empty Series if nothing qualifies).
    """
    if not assets:
        return pd.Series(dtype=float)
    tail = daily_rets[assets].tail(window)
    vols = tail.std()
    counts = tail.count()
    vols = vols[(counts >= window // 2) & (vols > 0)]
    if vols.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / vols
    return inv / inv.sum()


def blended_inverse_vol_weights(
    daily_rets: pd.DataFrame, assets: List[str], windows: Sequence[int] = VOL_WINDOWS
) -> pd.Series:
    """Inverse-vol weights averaged across several windows (ensemble).

    Averaging the normalised inverse-vol weights over {21,63,126}d removes the
    dependence on any single volatility lookback. Result sums to 1.
    """
    if not assets:
        return pd.Series(dtype=float)
    parts = [inverse_vol_weights(daily_rets, assets, window=w) for w in windows]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.Series(dtype=float)
    blended = pd.concat(parts, axis=1).mean(axis=1)
    total = blended.sum()
    return blended / total if total > 0 else blended


def trend_participation(monthly: pd.DataFrame, assets: List[str], months: Sequence[int] = TREND_MONTHS) -> pd.Series:
    """Fraction of the SMA-lookback ensemble each asset trades above, in [0, 1].

    For every lookback L in ``months`` we test close > mean(last L monthly
    closes); the participation is the average of these booleans. A single 10m
    SMA is the special case ``months=(10,)``; the ensemble is smoother and does
    not hinge on one threshold.
    """
    present = [a for a in assets if a in monthly.columns]
    if not present:
        return pd.Series(dtype=float)
    closes = monthly[present]
    last = closes.iloc[-1]
    signals = []
    for lb in months:
        if len(closes) < lb:
            continue
        sma = closes.tail(lb).mean()
        signals.append((last > sma).astype(float))
    if not signals:
        return pd.Series(0.0, index=present)
    return pd.concat(signals, axis=1).mean(axis=1)


class MultiFactorEngine:
    """Computes MFAA target weights at a given month-end date (point-in-time)."""

    def __init__(
        self,
        prices: pd.DataFrame,
        top_n: int = 4,
        trend_months: Sequence[int] = TREND_MONTHS,
        vol_windows: Sequence[int] = VOL_WINDOWS,
        canary_mode: str = "graded",   # "graded" (n_bad/4 to defense) | "binary" (any bad -> all defense)
        sleeve_mix: float = 0.5,       # weight of XSMOM sleeve; (1 - mix) goes to TSTREND
    ):
        self.prices = prices
        self.top_n = top_n
        self.trend_months = tuple(trend_months)
        self.vol_windows = tuple(vol_windows)
        self.canary_mode = canary_mode
        self.sleeve_mix = sleeve_mix

    # -- point-in-time views ------------------------------------------------
    def _monthly_closes(self, date: pd.Timestamp) -> pd.DataFrame:
        """Month-end closes strictly up to ``date`` (the row at ``date`` included)."""
        hist = self.prices.loc[:date]
        return hist.groupby(hist.index.to_period("M")).last()

    def _daily_rets(self, date: pd.Timestamp) -> pd.DataFrame:
        return self.prices.loc[:date].pct_change()

    # -- building blocks ----------------------------------------------------
    def _eligible(self, monthly: pd.DataFrame, assets: List[str]) -> List[str]:
        """Assets with at least 13 month-end closes as of the signal date."""
        counts = monthly[[a for a in assets if a in monthly.columns]].count()
        return counts[counts >= 13].index.tolist()

    def defensive_asset(self, monthly: pd.DataFrame) -> str:
        """Best short-duration asset by 13612 momentum (BAA defensive pool)."""
        pool = self._eligible(monthly, DEFENSIVE_POOL)
        if not pool:
            return CASH_PROXY if CASH_PROXY in self.prices.columns else self.prices.columns[0]
        score = momentum_13612(monthly[pool].dropna(how="all"))
        score = score.dropna()
        if score.empty:
            return pool[-1]
        return str(score.idxmax())

    def canary_risk_fraction(self, monthly: pd.DataFrame) -> float:
        """Fraction of the book allowed in risk assets, from canary breadth."""
        canaries = self._eligible(monthly, CANARY_ASSETS)
        if not canaries:
            return 1.0
        score = momentum_13612(monthly[canaries].dropna(how="all"), weighted=True).dropna()
        if score.empty:
            return 1.0
        n_bad = int((score < 0).sum())
        if self.canary_mode == "binary":
            return 0.0 if n_bad > 0 else 1.0
        return 1.0 - n_bad / len(score)

    def xsmom_sleeve(self, monthly: pd.DataFrame, daily_rets: pd.DataFrame, defensive: str) -> pd.Series:
        """Cross-sectional momentum sleeve: top-N by 13612, gated vs cash momentum."""
        risky = self._eligible(monthly, RISKY_ASSETS)
        w = pd.Series(0.0, index=self.prices.columns)
        if not risky:
            w[defensive] = 1.0
            return w
        score = momentum_13612(monthly[risky]).dropna()
        cash_gate = 0.0
        if CASH_PROXY in monthly.columns and monthly[CASH_PROXY].count() >= 13:
            cash_gate = float(momentum_13612(monthly[[CASH_PROXY]]).iloc[0])
        winners = score[score > cash_gate].sort_values(ascending=False).head(self.top_n)
        k = len(winners)
        if k > 0:
            iv = blended_inverse_vol_weights(daily_rets, winners.index.tolist(), self.vol_windows)
            w[iv.index] += iv * (k / self.top_n)
        w[defensive] += (self.top_n - k) / self.top_n
        return w

    def tstrend_sleeve(self, monthly: pd.DataFrame, daily_rets: pd.DataFrame, defensive: str) -> pd.Series:
        """Trend sleeve: inverse-vol budgets scaled by fractional SMA participation."""
        risky = self._eligible(monthly, RISKY_ASSETS)
        w = pd.Series(0.0, index=self.prices.columns)
        if not risky:
            w[defensive] = 1.0
            return w
        budgets = blended_inverse_vol_weights(daily_rets, risky, self.vol_windows)
        if budgets.empty:
            w[defensive] = 1.0
            return w
        part = trend_participation(monthly, budgets.index.tolist(), self.trend_months)
        for asset, budget in budgets.items():
            frac = float(part.get(asset, 0.0))
            w[asset] += budget * frac
            w[defensive] += budget * (1.0 - frac)
        return w

    # -- combined target ----------------------------------------------------
    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        """Unlevered MFAA book at ``date`` (weights sum to 1, long-only)."""
        monthly = self._monthly_closes(date)
        daily_rets = self._daily_rets(date)
        defensive = self.defensive_asset(monthly)

        w_xs = self.xsmom_sleeve(monthly, daily_rets, defensive)
        w_ts = self.tstrend_sleeve(monthly, daily_rets, defensive)
        risky_book = self.sleeve_mix * w_xs + (1.0 - self.sleeve_mix) * w_ts

        risk_frac = self.canary_risk_fraction(monthly)
        book = risky_book * risk_frac
        book[defensive] += 1.0 - risk_frac
        return book.fillna(0.0)


def run_multifactor_backtest(
    prices_df: pd.DataFrame,
    start_date: str = "2008-07-01",
    end_date: Optional[str] = None,
    top_n: int = 4,
    trend_months: Sequence[int] = TREND_MONTHS,
    vol_windows: Sequence[int] = VOL_WINDOWS,
    canary_mode: str = "graded",
    sleeve_mix: float = 0.5,
    target_vol: float = 0.10,
    max_lev: float = 1.5,
    cost_bps: float = 10.0,
    benchmark: Optional[str] = "SPY",
    verbose: bool = True,
) -> Dict:
    """
    Run the MFAA backtest on a daily adjusted-close price frame.

    Monthly rebalance: signals use data up to and including the last trading
    day of each month; the resulting positions earn returns from the next
    trading day onward. ``cost_bps`` is charged on the turnover of levered
    target positions at each rebalance; financing (BIL-derived rate + 150bps)
    accrues daily on gross exposure above 1.

    Returns dict with 'equity' (Series), 'metrics' (dict), 'weights'
    (DataFrame of levered target weights per rebalance date), and 'rebalance_vol'
    (realised vol estimate used for sizing at each rebalance).
    """
    px = prices_df.loc[:end_date] if end_date else prices_df
    px = px.copy()
    vol_windows = tuple(vol_windows)
    size_window = max(vol_windows)

    engine = MultiFactorEngine(
        px, top_n=top_n, trend_months=trend_months, vol_windows=vol_windows,
        canary_mode=canary_mode, sleeve_mix=sleeve_mix,
    )
    daily_rets = px.pct_change().fillna(0.0)

    # Month-end trading days on/after start_date (signal dates).
    idx = px.loc[start_date:].index
    month_ends = idx.to_series().groupby(idx.to_period("M")).last().tolist()

    nav = [10000.0]
    dates = [month_ends[0]]
    prev_pos = pd.Series(0.0, index=px.columns)
    turnovers, weight_rows, vol_rows = [], [], []

    if verbose:
        print(
            f"MFAA | topN={top_n} trend={trend_months} volw={vol_windows} "
            f"canary={canary_mode} mix={sleeve_mix} tv={target_vol:.0%} cost={cost_bps}bps"
        )

    for i in range(len(month_ends) - 1):
        t, t_next = month_ends[i], month_ends[i + 1]

        book = engine.target_weights(t)

        # Volatility targeting on the unlevered book (realised, trailing window).
        active = book[book.abs() > 1e-10].index
        hist = daily_rets[active].loc[:t].tail(size_window)
        port_vol = float((hist * book[active]).sum(axis=1).std() * np.sqrt(252))
        if not np.isfinite(port_vol) or port_vol < 0.02:
            lev = 1.0
        else:
            lev = min(target_vol / port_vol, max_lev)
        w_final = book * lev

        # Financing rate proxy from BIL (as in the core engine).
        rf_rate = 0.02
        if CASH_PROXY in px.columns and px[CASH_PROXY].loc[:t].count() > 21:
            bil = px[CASH_PROXY].loc[:t].tail(22)
            rf_rate = max(0.0, (bil.iloc[-1] / bil.iloc[0]) ** (252 / 21) - 1.0)
        borrow_daily = max(0.0, float(w_final.abs().sum()) - 1.0) * (rf_rate + 0.015) / 252.0

        seg_rets = daily_rets.loc[t:t_next].iloc[1:]
        if seg_rets.empty:
            continue
        port_ret = (seg_rets[w_final.index] * w_final).sum(axis=1) - borrow_daily

        turnover = float((w_final - prev_pos).abs().sum())
        turnovers.append(turnover)
        prev_pos = w_final
        weight_rows.append(w_final.rename(t))
        vol_rows.append((t, port_vol, lev))

        seg_nav = nav[-1] * (1.0 - turnover * cost_bps / 1e4) * (1.0 + port_ret).cumprod()
        nav.extend(seg_nav.values)
        dates.extend(seg_nav.index)

    equity = pd.Series(nav, index=dates, name="equity")
    metrics = compute_metrics(equity, cost_bps=cost_bps, turnovers=turnovers)

    if benchmark and benchmark in px.columns:
        rets = equity.pct_change().dropna()
        bench = px[benchmark].pct_change().reindex(rets.index).dropna()
        common = rets.index.intersection(bench.index)
        if len(common) > 50 and bench.loc[common].var() > 0:
            pr, br = rets.loc[common], bench.loc[common]
            beta = float(pr.cov(br) / br.var())
            metrics.update({
                "benchmark": benchmark,
                "benchmark_beta": round(beta, 2),
                "benchmark_ann_alpha_pct": round(float(pr.mean() - beta * br.mean()) * 252 * 100, 2),
                "benchmark_corr": round(float(pr.corr(br)), 2),
            })

    if verbose:
        print(
            f"  {metrics['start']}..{metrics['end']} | CAGR {metrics['cagr']}% | "
            f"MaxDD {metrics['max_drawdown']}% | Sharpe {metrics['sharpe']} | "
            f"Sortino {metrics['sortino']} | vol {metrics['ann_vol_pct']}%"
        )

    return {
        "equity": equity,
        "metrics": metrics,
        "weights": pd.DataFrame(weight_rows) if weight_rows else pd.DataFrame(),
        "rebalance_vol": pd.DataFrame(vol_rows, columns=["date", "est_vol", "leverage"]).set_index("date")
        if vol_rows else pd.DataFrame(),
    }


def compute_metrics(equity: pd.Series, cost_bps: float = 0.0, turnovers=None) -> Dict:
    """Repo-convention performance metrics for an equity curve (Sharpe vs 0% cash)."""
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    days = (equity.index[-1] - equity.index[0]).days or 1
    cagr = (1.0 + total_ret) ** (365.0 / days) - 1.0
    dd = equity / equity.cummax() - 1.0
    max_dd = float(-dd.min())
    rets = equity.pct_change().dropna()
    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) else float("nan")
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
    downside = rets[rets < 0]
    sortino = (
        float(rets.mean() / downside.std() * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 0 else float("nan")
    )
    monthly = equity.groupby(equity.index.to_period("M")).last().pct_change().dropna()
    return {
        "cagr": round(cagr * 100, 2),
        "total_return": round(total_ret * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar_ratio": round(cagr / max_dd, 3) if max_dd > 0 else float("nan"),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "pos_month_pct": round(float((monthly > 0).mean() * 100), 1) if len(monthly) else float("nan"),
        "avg_turnover": round(float(np.mean(turnovers)), 2) if turnovers else 0.0,
        "cost_bps": cost_bps,
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
    }


def sharpe_excess(equity: pd.Series, cash_prices: Optional[pd.Series]) -> float:
    """Sharpe of daily returns in excess of a cash proxy (e.g. BIL)."""
    rets = equity.pct_change().dropna()
    if cash_prices is None:
        return float(rets.mean() / rets.std() * np.sqrt(252))
    cash = cash_prices.pct_change().reindex(rets.index).fillna(0.0)
    ex = rets - cash
    return float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else float("nan")


def block_bootstrap_sharpe_ci(
    rets: pd.Series, n_draws: int = 2000, block: int = 21, seed: int = 7,
) -> Dict:
    """Stationary-block bootstrap 95% CI for the annualised Sharpe ratio."""
    r = rets.dropna().values
    n = len(r)
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_draws)
    for d in range(n_draws):
        out = np.empty(n)
        pos = 0
        while pos < n:
            start = rng.integers(0, n)
            length = min(rng.geometric(1.0 / block), n - pos)
            chunk = np.take(r, np.arange(start, start + length), mode="wrap")
            out[pos:pos + length] = chunk
            pos += length
        s = out.std()
        sharpes[d] = out.mean() / s * np.sqrt(252) if s > 0 else np.nan
    sharpes = sharpes[np.isfinite(sharpes)]
    return {
        "sharpe_ci_low": round(float(np.percentile(sharpes, 2.5)), 2),
        "sharpe_ci_high": round(float(np.percentile(sharpes, 97.5)), 2),
        "p_sharpe_gt_0": round(float((sharpes > 0).mean()), 4),
        "p_sharpe_gt_1": round(float((sharpes > 1).mean()), 4),
    }


def deflated_sharpe(observed_sr_ann: float, trial_srs_ann: List[float], rets: pd.Series) -> Dict:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    ``observed_sr_ann``: annualised Sharpe of the selected strategy.
    ``trial_srs_ann``: annualised Sharpes of ALL variants evaluated during
    selection (the honest trial count). Skew/kurtosis come from the selected
    strategy's daily returns. Returns the DSR probability that the true
    Sharpe exceeds the expected-max-under-N-trials benchmark.
    """
    from scipy import stats

    r = rets.dropna()
    t_obs = len(r)
    sr_daily = observed_sr_ann / np.sqrt(252)
    trials_daily = np.array(trial_srs_ann) / np.sqrt(252)
    n_trials = len(trials_daily)
    var_trials = float(trials_daily.var(ddof=1)) if n_trials > 1 else 0.0

    emc = 0.5772156649015329
    if var_trials > 0 and n_trials > 1:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_star = np.sqrt(var_trials) * ((1 - emc) * z1 + emc * z2)
    else:
        sr_star = 0.0

    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily**2))
    dsr = float(stats.norm.cdf((sr_daily - sr_star) * np.sqrt(t_obs - 1) / denom))
    return {
        "n_trials": n_trials,
        "expected_max_sr_ann": round(float(sr_star * np.sqrt(252)), 2),
        "deflated_sharpe_prob": round(dsr, 4),
    }


def load_universe_prices(
    cache_path: str = "data/mfaa_prices.csv",
    start: str = "2004-01-01",
    refresh: bool = False,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Download (or read cached) daily adjusted closes for the MFAA universe."""
    from pathlib import Path

    tickers = tickers or ALL_TICKERS
    path = Path(cache_path)
    if path.exists() and not refresh:
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in px.columns]
        if not missing:
            return px[tickers]
        print(f"Cache missing {missing}; refreshing download.")

    import yfinance as yf

    print(f"Downloading {len(tickers)} tickers via yfinance from {start}...")
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    px = px[tickers].dropna(how="all")
    path.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(path)
    print(f"Saved {px.shape[0]} rows x {px.shape[1]} tickers to {path}")
    return px
