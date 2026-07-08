#!/usr/bin/env python3
"""
MFAA experiment orchestrator -- the anti-overfitting protocol, executed.

Protocol (documented in docs/methodology.md):
0. NO data-driven selection. The configuration is fixed a priori to a neutral
   prior -- equal blend of the two sleeves (mix=0.5), graded canary, and a
   breadth of half the risky universe (top_n=5). This is deliberate: selecting
   "the best design-period cell" is itself a mild overfit (it tends to land on
   a cell that decays out of sample), so -- consistent with the ensembling
   philosophy -- we do not select at all.
1. DESIGN period 2008-07-01 .. 2019-12-31: the pre-registered variant grid is
   shown only as ROBUSTNESS evidence (the Sharpe range across all 18 cells and
   where the neutral prior sits), never to pick the config.
2. EVALUATION period 2020-01-01 .. today: run the fixed config once; that run is
   the headline. It is a *defensive* framework, so the headline is the
   excess-over-cash Sharpe alongside drawdown/vol vs SPY and 60/40.
3. Transparency: the full 18-variant grid on the evaluation period too (so the
   reader sees the neutral prior is mid-pack, not cherry-picked), plus
   one-at-a-time sensitivity, cost stress, a stationary block bootstrap CI for
   the Sharpe, the Deflated Sharpe Ratio (18 trials), a crisis drawdown table,
   and a professional tearsheet.

Usage:  python scripts/run_experiment.py [--refresh] [--out results/multifactor]
        (or, after `pip install -e .`:  mfaa-experiment  /  python -m mfaa)
"""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from mfaa import analytics
from mfaa.multifactor import (
    CASH_PROXY,
    block_bootstrap_sharpe_ci,
    deflated_sharpe,
    load_universe_prices,
    run_multifactor_backtest,
    sharpe_excess,
)

DESIGN_START, DESIGN_END = "2008-07-01", "2019-12-31"
# First signal on the last trading day of 2019 => positions are live for every
# trading day of 2020 onward (the evaluation window the audit doc reports).
OOS_START = "2019-12-15"

# Fixed neutral prior -- chosen a priori, NOT by searching any data:
#   sleeve_mix 0.5 = equal weight to momentum and trend (no view on which wins)
#   canary_mode graded = smooth proportional de-risking (Keller BAA default)
#   top_n 5 = hold half of the 10-asset risky universe (neutral breadth)
CONFIG = {"canary_mode": "graded", "top_n": 5, "sleeve_mix": 0.5}

# Pre-registered variant grid: shown for ROBUSTNESS only, never for selection.
GRID = {
    "canary_mode": ["graded", "binary"],
    "top_n": [3, 4, 5],
    "sleeve_mix": [0.0, 0.5, 1.0],
}

# One-at-a-time sensitivity around the selected variant (evaluation period).
# Each row perturbs a single knob; ensemble knobs are stress-tested by swapping
# the whole ensemble for a narrower or wider one.
SENSITIVITY = {
    "trend_months": [(10,), (3, 6, 9, 12)],       # single 10m SMA vs wide ensemble
    "vol_windows": [(63,), (21, 42, 63, 126, 252)],  # single 63d vs wide ensemble
    "target_vol": [0.08, 0.12],
    "cost_bps": [5.0, 20.0],
    "top_n": None,   # filled with chosen +/- 1 at runtime
}


def run_variant(px, start, end, verbose=False, **kw):
    return run_multifactor_backtest(
        px, start_date=start, end_date=end, benchmark=None, verbose=verbose, **kw
    )


def grid_table(px, start, end, label):
    rows = []
    for canary, topn, mix in itertools.product(
        GRID["canary_mode"], GRID["top_n"], GRID["sleeve_mix"]
    ):
        kw = dict(canary_mode=canary, top_n=topn, sleeve_mix=mix)
        m = run_variant(px, start, end, **kw)["metrics"]
        rows.append({
            "canary_mode": canary, "top_n": topn, "sleeve_mix": mix,
            "sharpe": m["sharpe"], "cagr": m["cagr"],
            "max_drawdown": m["max_drawdown"], "ann_vol_pct": m["ann_vol_pct"],
            "avg_turnover": m["avg_turnover"],
        })
        print(f"  [{label}] canary={canary:6} topN={topn} mix={mix:.1f} -> "
              f"Sharpe {m['sharpe']:5.2f} | CAGR {m['cagr']:6.2f}% | DD {m['max_drawdown']:5.2f}%")
    return pd.DataFrame(rows)


def daily_benchmark(px, cols_weights, start, end=None):
    """Daily-rebalanced fixed-weight benchmark equity curve."""
    rets = px[list(cols_weights)].loc[start:end].pct_change().dropna(how="all").fillna(0.0)
    port = sum(rets[c] * w for c, w in cols_weights.items())
    return 10000.0 * (1.0 + port).cumprod()


def yearly_returns(equity):
    yearly = equity.groupby(equity.index.year).last()
    prev = yearly.shift(1)
    prev.iloc[0] = equity.iloc[0]
    return ((yearly / prev - 1.0) * 100).round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="Re-download prices")
    ap.add_argument("--out", default="results/multifactor")
    ap.add_argument("--end", default=None, help="Optional hard end date for the OOS run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    px = load_universe_prices(refresh=args.refresh)
    cash_px = px[CASH_PROXY] if CASH_PROXY in px.columns else None
    print(f"Data: {px.index[0].date()} .. {px.index[-1].date()} | {px.shape[1]} tickers\n")

    chosen = dict(CONFIG)

    # ------------------------------------------------------------------ 1
    print(f"== Step 1: design-period ROBUSTNESS grid ({DESIGN_START} .. {DESIGN_END}) ==")
    print("   (shown to prove the neutral prior is not cherry-picked; NOT used to select)")
    design = grid_table(px, DESIGN_START, DESIGN_END, "design")
    design.to_csv(out / "design_grid.csv", index=False)
    dprior = design[(design.canary_mode == CONFIG["canary_mode"])
                    & (design.top_n == CONFIG["top_n"])
                    & (design.sleeve_mix == CONFIG["sleeve_mix"])]
    dprior_sharpe = float(dprior["sharpe"].iloc[0]) if len(dprior) else float("nan")
    print(f"\nFixed neutral prior {chosen}: design-period Sharpe {dprior_sharpe} "
          f"(grid range {design['sharpe'].min()}-{design['sharpe'].max()})\n")

    # ------------------------------------------------------------------ 2
    print(f"== Step 2: single OOS evaluation of the fixed prior ({OOS_START} .. now) ==")
    res = run_multifactor_backtest(
        px, start_date=OOS_START, end_date=args.end, verbose=True, **chosen
    )
    equity = res["equity"]
    rets = equity.pct_change().dropna()

    # ------------------------------------------------------------------ benchmarks
    spy = daily_benchmark(px, {"SPY": 1.0}, OOS_START, args.end)
    b6040 = daily_benchmark(px, {"SPY": 0.6, "IEF": 0.4}, OOS_START, args.end)

    # professional metric blocks (excess-over-cash headline for a defensive book)
    pro = analytics.full_metrics(equity, cash_prices=cash_px, benchmark_prices=px["SPY"], label="MFAA")
    pro_spy = analytics.full_metrics(spy, cash_prices=cash_px, label="SPY")
    pro_6040 = analytics.full_metrics(b6040, cash_prices=cash_px, benchmark_prices=px["SPY"], label="60/40")

    print(f"\n  MFAA  : Sharpe(excess-cash) {pro['sharpe_excess_cash']} | Sharpe {pro['sharpe']} | "
          f"CAGR {pro['cagr_pct']}% | vol {pro['ann_vol_pct']}% | MaxDD {pro['max_drawdown_pct']}% | "
          f"Ulcer {pro['ulcer_index']} | beta {pro.get('vs_bench_beta')} | "
          f"alpha {pro.get('vs_bench_alpha_ann_pct')}% (t={pro.get('vs_bench_alpha_tstat')})")
    print(f"  SPY   : Sharpe(excess-cash) {pro_spy['sharpe_excess_cash']} | CAGR {pro_spy['cagr_pct']}% | "
          f"MaxDD {pro_spy['max_drawdown_pct']}%")
    print(f"  60/40 : Sharpe(excess-cash) {pro_6040['sharpe_excess_cash']} | CAGR {pro_6040['cagr_pct']}% | "
          f"MaxDD {pro_6040['max_drawdown_pct']}%")

    # ------------------------------------------------------------------ 3
    print("\n== Step 3: transparency -- full grid on the evaluation period ==")
    oos_grid = grid_table(px, OOS_START, args.end, "OOS")
    oos_grid.to_csv(out / "oos_grid.csv", index=False)

    print("\n== Step 4: one-at-a-time sensitivity around the selected variant ==")
    sens_rows = []
    sens = dict(SENSITIVITY)
    sens["top_n"] = sorted({max(2, chosen["top_n"] - 1), chosen["top_n"] + 1})
    for param, values in sens.items():
        for v in values:
            kw = dict(chosen)
            kw[param] = v
            m = run_variant(px, OOS_START, args.end, **kw)["metrics"]
            sens_rows.append({"param": param, "value": str(v), "sharpe": m["sharpe"],
                              "cagr": m["cagr"], "max_drawdown": m["max_drawdown"]})
            print(f"  {param}={str(v):18} -> Sharpe {m['sharpe']:5.2f} | CAGR {m['cagr']:6.2f}% "
                  f"| DD {m['max_drawdown']:5.2f}%")
    pd.DataFrame(sens_rows).to_csv(out / "sensitivity.csv", index=False)

    print("\n== Step 5: bootstrap CI + Deflated Sharpe ==")
    ci = block_bootstrap_sharpe_ci(rets)
    dsr = deflated_sharpe(pro["sharpe"], oos_grid["sharpe"].tolist(), rets)
    print(f"  95% bootstrap Sharpe CI [{ci['sharpe_ci_low']}, {ci['sharpe_ci_high']}] | "
          f"P(SR>1) {ci['p_sharpe_gt_1']:.1%}")
    print(f"  Deflated Sharpe prob {dsr['deflated_sharpe_prob']:.1%} "
          f"(N={dsr['n_trials']} trials, E[maxSR] {dsr['expected_max_sr_ann']})")

    # ------------------------------------------------------------------ 6 crisis table
    print("\n== Step 6: crisis drawdown table (peak-to-trough %) ==")
    crisis = analytics.crisis_table({"MFAA": equity, "SPY": spy, "60/40": b6040})
    crisis.to_csv(out / "crisis_table.csv", index=False)
    print(crisis.to_string(index=False))

    yr = pd.DataFrame({
        "MFAA": yearly_returns(equity),
        "SPY": yearly_returns(spy.reindex(equity.index).dropna()),
        "60/40": yearly_returns(b6040.reindex(equity.index).dropna()),
    })
    yr = yr[yr.index >= 2020]
    yr.to_csv(out / "yearly_returns.csv")
    print("\nYearly returns (%):")
    print(yr.to_string())

    # ------------------------------------------------------------------ persist
    payload = {
        "strategy": "MFAA (multi-factor adaptive allocation) -- defensive framework",
        "config": chosen,
        "selection": "none -- fixed neutral prior, not tuned on any data",
        "design_period_robustness": {
            "start": DESIGN_START, "end": DESIGN_END,
            "prior_sharpe": dprior_sharpe,
            "grid_sharpe_min": float(design["sharpe"].min()),
            "grid_sharpe_max": float(design["sharpe"].max()),
        },
        "oos_headline": pro,
        "oos_robustness": {**ci, **dsr,
                           "sharpe_excess_bil": round(sharpe_excess(equity, cash_px), 2)},
        "benchmarks_oos": {"SPY": pro_spy, "60_40_SPY_IEF": pro_6040},
        "crisis_table": crisis.to_dict(orient="records"),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    equity.to_csv(out / "equity_curve.csv")
    res["weights"].round(4).to_csv(out / "weights.csv")
    res["rebalance_vol"].round(4).to_csv(out / "rebalance_vol.csv")

    analytics.tearsheet(
        equity, {"SPY": spy, "60/40": b6040}, str(out / "tearsheet.png"),
        title=f"MFAA defensive framework  2020..{equity.index[-1].date()}  "
              f"(Sharpe {pro['sharpe']}, excess-cash {pro['sharpe_excess_cash']}, "
              f"MaxDD {pro['max_drawdown_pct']}%)",
    )
    # keep the simple equity chart too
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    (equity / equity.iloc[0]).plot(color="#ff7f0e", linewidth=1.6, label="MFAA")
    (spy / spy.iloc[0]).reindex(equity.index).plot(color="#7f7f7f", linewidth=1.0, label="SPY")
    (b6040 / b6040.iloc[0]).reindex(equity.index).plot(color="#1f77b4", linewidth=1.0, label="60/40")
    plt.title(f"MFAA | excess-cash Sharpe {pro['sharpe_excess_cash']} | "
              f"MaxDD {pro['max_drawdown_pct']}% vs SPY {pro_spy['max_drawdown_pct']}%")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(out / "equity_curve.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved results to {out}/ (metrics.json, tearsheet.png, crisis_table.csv, ...)")


if __name__ == "__main__":
    main()
