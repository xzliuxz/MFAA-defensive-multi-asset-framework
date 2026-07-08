"""Tests for the MFAA multi-factor engine (point-in-time + accounting)."""

import numpy as np
import pandas as pd
import pytest

from mfaa import analytics
from mfaa.multifactor import (
    ALL_TICKERS,
    MultiFactorEngine,
    block_bootstrap_sharpe_ci,
    blended_inverse_vol_weights,
    compute_metrics,
    momentum_13612,
    inverse_vol_weights,
    run_multifactor_backtest,
    trend_participation,
)


# ---------------------------------------------------------------- fixtures


def make_universe(n_days: int = 900, seed: int = 3) -> pd.DataFrame:
    """Random-walk prices for the full MFAA universe (>= 3y of history)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_days)
    data = {}
    for t in ALL_TICKERS:
        drift = 0.0004 if t in ("SPY", "QQQ", "IWM") else 0.0001
        vol = 0.0004 if t == "BIL" else 0.01
        data[t] = 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, n_days))
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------- signals


def test_momentum_13612_deterministic():
    dates = pd.period_range("2020-01", periods=14, freq="M")
    # +1% per month, constant: every horizon return is (1.01^h - 1)
    prices = pd.DataFrame({"A": 100.0 * 1.01 ** np.arange(14)}, index=dates)
    score = momentum_13612(prices)
    expected = np.mean([1.01**h - 1.0 for h in (1, 3, 6, 12)])
    assert score["A"] == pytest.approx(expected, rel=1e-10)


def test_momentum_requires_13_months():
    dates = pd.period_range("2020-01", periods=10, freq="M")
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=dates)
    assert momentum_13612(prices).isna().all()


def test_inverse_vol_weights_sum_to_one_and_favor_low_vol():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({
        "LOW": rng.normal(0, 0.001, 100),
        "HIGH": rng.normal(0, 0.02, 100),
    })
    w = inverse_vol_weights(rets, ["LOW", "HIGH"], window=63)
    assert w.sum() == pytest.approx(1.0)
    assert w["LOW"] > w["HIGH"]


def test_blended_inverse_vol_weights_sum_to_one():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame({
        "LOW": rng.normal(0, 0.001, 300),
        "MID": rng.normal(0, 0.006, 300),
        "HIGH": rng.normal(0, 0.02, 300),
    })
    w = blended_inverse_vol_weights(rets, ["LOW", "MID", "HIGH"], windows=(21, 63, 126))
    assert w.sum() == pytest.approx(1.0)
    assert w["LOW"] > w["MID"] > w["HIGH"]


def test_trend_participation_is_fractional():
    """An asset above every SMA lookback scores 1; below every one scores 0;
    a mid-trend asset scores strictly between (the ensemble is not binary)."""
    dates = pd.period_range("2018-01", periods=24, freq="M")
    up = pd.Series(100.0 * 1.02 ** np.arange(24), index=dates)      # above all SMAs
    down = pd.Series(100.0 * 0.98 ** np.arange(24), index=dates)    # below all SMAs
    # Recent bounce off a deep trough: above the 6m SMA but still below the
    # 10m and 12m SMAs -> participation should be exactly 1/3.
    v = pd.Series([120.0] * 17 + [110, 90, 70, 60, 66, 72, 78], index=dates)
    monthly = pd.DataFrame({"UP": up, "DOWN": down, "V": v})
    part = trend_participation(monthly, ["UP", "DOWN", "V"], months=(6, 10, 12))
    assert part["UP"] == pytest.approx(1.0)
    assert part["DOWN"] == pytest.approx(0.0)
    assert part["V"] == pytest.approx(1.0 / 3.0)


def test_canary_all_positive_gives_full_risk():
    px = make_universe()
    # steadily rising canaries -> no bad canary
    for t in ("SPY", "EFA", "EEM", "AGG"):
        px[t] = 100.0 * 1.001 ** np.arange(len(px))
    eng = MultiFactorEngine(px)
    monthly = eng._monthly_closes(px.index[-1])
    assert eng.canary_risk_fraction(monthly) == 1.0


def test_canary_negative_reduces_risk_graded_and_binary():
    px = make_universe()
    for t in ("SPY", "EFA"):
        px[t] = 100.0 * 0.999 ** np.arange(len(px))   # falling: bad canaries
    for t in ("EEM", "AGG"):
        px[t] = 100.0 * 1.001 ** np.arange(len(px))
    graded = MultiFactorEngine(px, canary_mode="graded")
    binary = MultiFactorEngine(px, canary_mode="binary")
    monthly = graded._monthly_closes(px.index[-1])
    assert graded.canary_risk_fraction(monthly) == pytest.approx(0.5)
    assert binary.canary_risk_fraction(monthly) == 0.0


def test_trend_sleeve_moves_downtrend_budget_to_defensive():
    px = make_universe()
    px["SPY"] = 100.0 * 0.999 ** np.arange(len(px))   # below its SMA
    eng = MultiFactorEngine(px)
    monthly = eng._monthly_closes(px.index[-1])
    daily = eng._daily_rets(px.index[-1])
    defensive = eng.defensive_asset(monthly)
    w = eng.tstrend_sleeve(monthly, daily, defensive)
    assert w["SPY"] == 0.0
    assert w[defensive] > 0.0
    assert w.sum() == pytest.approx(1.0)


def test_target_weights_long_only_and_fully_invested():
    px = make_universe()
    eng = MultiFactorEngine(px)
    w = eng.target_weights(px.index[-1])
    assert (w >= -1e-12).all()
    assert w.sum() == pytest.approx(1.0)


# ------------------------------------------------------- point-in-time


def test_no_lookahead_weights_identical_after_truncation():
    """Weights at t must not change when future rows are removed."""
    px = make_universe()
    t = px.index[700]
    w_full = MultiFactorEngine(px).target_weights(t)
    w_trunc = MultiFactorEngine(px.loc[:t]).target_weights(t)
    pd.testing.assert_series_equal(w_full, w_trunc)


def test_positions_earn_returns_only_after_signal_date():
    """A huge jump on the day AFTER a rebalance must not affect that day's signal."""
    px = make_universe()
    res_a = run_multifactor_backtest(px, start_date="2017-06-01", verbose=False)
    # Perturb the last day only: signals for all completed rebalances unchanged.
    px2 = px.copy()
    px2.iloc[-1] = px2.iloc[-1] * 5.0
    res_b = run_multifactor_backtest(px2, start_date="2017-06-01", verbose=False)
    w_a, w_b = res_a["weights"], res_b["weights"]
    common = w_a.index.intersection(w_b.index)[:-1]   # all but the final signal
    pd.testing.assert_frame_equal(w_a.loc[common], w_b.loc[common])


# ------------------------------------------------------- accounting


def test_flat_prices_never_gain():
    """With constant prices there is no return to earn: NAV must never rise.

    Regression against synthetic-return injection (a previous revision added
    a flat +2.5bp/day 'carry floor' to every period, fabricating ~6%/year).
    """
    dates = pd.bdate_range("2016-01-04", periods=900)
    px = pd.DataFrame({t: 100.0 for t in ALL_TICKERS}, index=dates)
    res = run_multifactor_backtest(px, start_date="2017-06-01", verbose=False)
    equity = res["equity"]
    assert (equity.diff().dropna() <= 1e-9).all()
    assert equity.iloc[-1] <= equity.iloc[0]


def test_higher_costs_strictly_hurt():
    px = make_universe()
    lo = run_multifactor_backtest(px, start_date="2017-06-01", cost_bps=0.0, verbose=False)
    hi = run_multifactor_backtest(px, start_date="2017-06-01", cost_bps=50.0, verbose=False)
    assert hi["equity"].iloc[-1] < lo["equity"].iloc[-1]


def test_backtest_outputs_complete():
    px = make_universe()
    res = run_multifactor_backtest(px, start_date="2017-06-01", verbose=False)
    m = res["metrics"]
    for key in ("cagr", "sharpe", "max_drawdown", "ann_vol_pct", "avg_turnover"):
        assert key in m
    assert len(res["equity"]) > 200
    assert not res["weights"].empty
    # levered gross exposure respects the cap
    assert res["weights"].abs().sum(axis=1).max() <= 1.5 + 1e-9


def test_compute_metrics_known_curve():
    dates = pd.bdate_range("2020-01-01", periods=253)
    equity = pd.Series(np.linspace(100.0, 110.0, 253), index=dates)
    m = compute_metrics(equity)
    assert m["total_return"] == pytest.approx(10.0, abs=0.01)
    assert m["max_drawdown"] == 0.0


def test_bootstrap_ci_orders():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0005, 0.01, 800))
    ci = block_bootstrap_sharpe_ci(rets, n_draws=200)
    assert ci["sharpe_ci_low"] <= ci["sharpe_ci_high"]


# ------------------------------------------------------- analytics


def _sample_curve(seed=5, mu=0.0004, sigma=0.008, n=760):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(10000.0 * np.cumprod(1.0 + rng.normal(mu, sigma, n)), index=dates)


def test_full_metrics_has_professional_keys():
    eq = _sample_curve()
    bench = _sample_curve(seed=6, mu=0.0003)
    m = analytics.full_metrics(eq, cash_prices=None, benchmark_prices=bench, label="X")
    for key in (
        "sharpe", "sharpe_excess_cash", "sortino_excess_cash", "max_drawdown_pct",
        "ulcer_index", "tail_ratio", "calmar", "vs_bench_beta",
        "vs_bench_alpha_ann_pct", "vs_bench_alpha_tstat", "vs_bench_information_ratio",
    ):
        assert key in m, key


def test_excess_cash_sharpe_below_raw_when_cash_positive():
    eq = _sample_curve()
    cash = pd.Series(
        100.0 * np.cumprod(1.0 + np.full(len(eq), 0.0002)), index=eq.index
    )  # ~5%/yr cash
    m = analytics.full_metrics(eq, cash_prices=cash, label="X")
    assert m["sharpe_excess_cash"] < m["sharpe"]


def test_ulcer_index_zero_for_monotonic_curve():
    dates = pd.bdate_range("2020-01-01", periods=100)
    eq = pd.Series(np.linspace(100, 150, 100), index=dates)
    assert analytics.ulcer_index(eq) == pytest.approx(0.0, abs=1e-9)


def test_crisis_table_covers_covid_crash():
    dates = pd.bdate_range("2019-06-01", "2020-06-01")
    # deterministic curve that falls hard through the COVID window
    base = pd.Series(np.linspace(100, 130, len(dates)), index=dates)
    base.loc["2020-02-19":"2020-03-23"] *= np.linspace(1.0, 0.7, len(base.loc["2020-02-19":"2020-03-23"]))
    tab = analytics.crisis_table({"X": base})
    covid = tab[tab["episode"] == "COVID crash"].iloc[0]
    assert covid["X"] < 0  # peak-to-trough is a loss
