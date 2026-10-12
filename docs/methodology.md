# MFAA Methodology & Anti-Overfitting Protocol

*Last updated: 2026-10-12*

This document specifies the **MFAA (Multi-Factor Adaptive Allocation)** framework
— every rule, parameter and its literature source (all pre-2020 except two
BAA-derived components; see the caveats) — and defines the anti-overfitting
protocol used to produce the headline 2020→present result. The guiding principle
throughout: **nothing about the strategy was tuned on post-2020 data.**

---

## 1. Anti-patterns this harness refuses to commit

The whole framework is built to avoid the ways a multi-asset backtest is usually
made to lie. Each has a concrete safeguard in code:

| Anti-pattern | What it looks like | Safeguard here |
|--------------|--------------------|----------------|
| **Return injection** | adding a small constant to every period's return (a "carry floor"), fabricating CAGR from nothing | `tests/test_mfaa.py::test_flat_prices_never_gain` — a constant-price backtest can never gain value |
| **Look-ahead** | using a close to trade the same bar, or letting future rows change past signals | positions apply the **next** trading day; `test_no_lookahead_weights_identical_after_truncation` pins it |
| **In-sample knob fitting** | tuning thresholds/exposure multipliers against the full-sample Sharpe | no parameter is fitted; lookbacks are **ensembled**, not searched (§3) |
| **Cherry-picking a variant** | reporting the best cell of a grid as if pre-chosen | a **fixed neutral prior**, no selection; the full grid is shown for context (§4) |
| **Cost-free turnover** | ignoring transaction/financing costs | 10 bps on turnover + financing on gross > 1; `test_higher_costs_strictly_hurt` |
| **Multiple-testing silence** | quoting a Sharpe without deflating for how many were tried | **Deflated Sharpe Ratio** with the honest trial count (§4) |

---

## 2. MFAA — strategy specification

Long-only, monthly-rebalanced multi-asset strategy in `src/mfaa/multifactor.py`.
Price-only signals (no macro series, hence no publication-lag ambiguity).
Signals are computed on the **last trading day of each month** using data up to
and including that close; the resulting positions earn returns **from the next
trading day**. No same-bar execution, no short positions.

### Universe (fixed ex-ante)

All ETFs with inception ≤ mid-2007, mirroring the classic multi-asset universes
in the literature (Faber's GTAA, Butler et al.'s Adaptive Asset Allocation,
Keller's PAA/VAA; the defensive pool follows Keller's 2022 BAA):

- **Risky (10):** SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, LQD, GLD, DBC
- **Defensive pool (3):** IEF, TIP, BIL
- **Canary (signal-only, 4):** SPY, EFA, EEM, AGG

### Factor sleeves

| Component | Rule | Source |
|-----------|------|--------|
| **XSMOM sleeve** (½ of risk book) | Rank risky assets by the mean of 1/3/6/12-month returns ("13612"); keep top **N=5** (half the universe) whose score also beats cash (BIL) momentum; weight by blended inverse volatility; unfilled slots go to the defensive asset | Jegadeesh & Titman (1993); Antonacci (2014) absolute momentum; Keller & Keuning (2016) 13612 filter |
| **TSTREND sleeve** (½ of risk book) | Each risky asset receives its inverse-vol budget scaled by **fractional participation** = the fraction of the {6,10,12}-month SMA ensemble it trades above; off-trend budget goes to the defensive asset | Faber (2007); Moskowitz, Ooi & Pedersen (2012) |
| **Canary overlay** | Compute fast "13612W" momentum (weights 12/4/2/1) for {SPY, EFA, EEM, AGG}; the fraction of negative canaries moves the same fraction of the whole book into the defensive asset | Keller & Keuning (2017) VAA breadth momentum; (2022) BAA canary |
| **Defensive selection** | Best of {IEF, TIP, BIL} by 13612 momentum | BAA defensive universe (Keller & Keuning 2022) |
| **Weighting** | **Blended** inverse realized volatility, averaged over {21,63,126}-day windows | risk-parity / low-vol weighting literature |
| **Vol targeting** | Scale final book by `min(10% / realized portfolio vol, 1.5)`; no scaling when realized vol < 2% | Moreira & Muir (2017); Harvey et al. (2018) |

**Ensembling instead of magic numbers.** The trend signal averages over three SMA
lookbacks (6/10/12m) and the volatility estimate averages over three windows
(21/63/126d). No single hand-picked parameter drives the result; the one-at-a-time
sensitivity below confirms swapping the whole ensemble for a single 10m SMA or 63d
window barely moves the Sharpe. This is a standard robustness practice in the
trend-following literature and the structural opposite of curve-fitting.

### Cost model

- **Transaction costs:** 10 bps × turnover of levered target weights, charged at
  every rebalance (double the repo's previous 5 bps default, as a conservative
  buffer for spreads + slippage on liquid ETFs).
- **Financing:** (BIL-implied cash rate + 150 bps) × (gross exposure − 1),
  accrued daily whenever the vol-target scalar levers the book above 1.
- Adjusted (total-return) closes are used throughout, so ETF distributions are
  included in returns.

---

## 3. Anti-overfitting protocol

The core discipline: **no parameter or configuration was chosen by looking at
post-2020 results.**

1. **Rules & parameters from published literature.** Every lookback (1/3/6/12m,
   6/10/12m SMA ensemble, 21/63/126d vol ensemble), every mechanism (absolute
   momentum gate, canary breadth, inverse-vol weighting, vol targeting) is a
   standard published value — none was searched over any data in this repo.
   All of it predates 2020 except the BAA (2022) elements of the canary overlay
   and the defensive pool; see the caveats below.
2. **No data-driven selection at all.** The configuration is a fixed *neutral
   prior*, chosen a priori: equal blend of the two sleeves (mix = 0.5), graded
   canary, and breadth of half the risky universe (top-N = 5). We deliberately do
   **not** pick "the best design-period cell": doing so is itself a mild overfit
   that tends to land on a variant which decays out of sample (in our own run,
   the best design cell — pure trend, Sharpe 0.91 — fell to 0.60 OOS, while the
   neutral prior held 0.76). Refusing to select is consistent with the
   ensembling philosophy: don't pick, average.
3. **The grid is robustness, not selection.** The pre-registered 18-variant grid
   (canary graded/binary × top-N ∈ {3,4,5} × sleeve mix ∈ {trend, 50/50,
   momentum}) is reported on **both** 2008-2019 (design) and 2020→now
   (evaluation) purely to show the Sharpe *range* and that the neutral prior sits
   mid-pack — never to choose the config.
4. **One-at-a-time sensitivity** around the fixed prior (single 10m SMA vs wide
   ensemble, single 63d vol vs wide ensemble, target vol 8/12%, top-N ±1, costs
   5/20 bps): a robust strategy shows a plateau, not a spike.
5. **Statistical honesty.** Stationary block-bootstrap 95% CI for the OOS
   Sharpe; **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) computed
   under the conservative assumption that all 18 grid variants were trials;
   Sharpe reported both vs 0% cash and — as the headline for a defensive book —
   in excess of BIL.
6. **Mechanical safeguards.** Point-in-time regression tests (future rows must
   not change past weights), a constant-price no-gain test (blocks synthetic
   return injection), and cost-monotonicity tests.

### Known residual caveats

- **Two components draw on a post-2020 paper.** The canary overlay and the
  defensive pool {IEF, TIP, BIL} follow Keller's 2022 *Bold Asset Allocation*
  (BAA) paper, which appeared after part of the evaluation window had already
  happened. No parameter here was tuned on 2020→present, but for these components
  the evaluation is not strictly out-of-sample; treat the headline as
  pseudo-out-of-sample.
- The universe, while standard in pre-2020 papers, is still specified with
  hindsight knowledge that these ETFs survived; single-name survivorship is not
  an issue (broad index ETFs), but "ETF asset-class coverage" itself is a
  post-hoc convenience.
- 2020→present is ~6.5 years ≈ 78 monthly bets — small sample; the bootstrap CI
  quantifies (part of) that uncertainty.
- For a defensive book the **excess-over-cash Sharpe is the honest headline**,
  and it is materially lower than the vs-0%-cash number (0.48 vs 0.76): with
  2022-2025 cash near 4-5%, much of the nominal return was just the risk-free
  rate. Both numbers are in `results/multifactor/metrics.json`.
- Monthly rebalancing means intramonth crashes (e.g. Feb→Mar 2020) are absorbed
  until the next month-end signal; the canary's 12× weight on the 1-month
  return is the designed mitigation, not a guarantee.

---

## 4. Results (run 2026-07-07, data 2004-01 → 2026-07-06)

MFAA is positioned as a **defensive framework**, so the headline is not raw
Sharpe but the risk shape: drawdown, volatility, beta and crisis behaviour, with
the excess-over-cash Sharpe as the honest risk-adjusted number. Fixed neutral
prior (graded canary, top-N 5, 50/50 blend), single 2020→present run, 10 bps
costs.

### Headline table

| Metric | MFAA | SPY | 60/40 (SPY/IEF) |
|--------|------|-----|-----------------|
| Sharpe (vs 0% cash) | 0.76 | 0.82 | 0.82 |
| **Sharpe (excess over cash)** | **0.48** | 0.69 | 0.60 |
| CAGR | 6.9% | 15.6% | 9.7% |
| **Max drawdown** | **13.6%** | 33.7% | 21.0% |
| Annualised vol | **9.3%** | 20.3% | 12.3% |
| Ulcer index | **6.5** | 8.4 | 7.0 |
| Beta to SPY | **0.17** | 1.00 | 0.59 |
| Ann. alpha vs SPY | +4.3% (t=1.3) | — | +0.3% |

### Crisis drawdown table (peak-to-trough) — the real selling point

| Episode | MFAA | SPY | 60/40 |
|---------|------|-----|-------|
| COVID crash (Feb–Mar 2020) | **+0.8%** | −33.7% | −19.1% |
| 2022 bear (Jan–Oct 2022) | **−7.9%** | −24.5% | −20.5% |
| 2023 SVB scare | −3.0% | −7.5% | −4.9% |
| 2025 tariff shock | −10.2% | −18.8% | −10.6% |

In the two big stress events MFAA lost a small fraction of what equities (and even
static 60/40) lost — in 2022 specifically, when stocks *and* bonds fell together,
its trend filters had already stepped out of both. That regime-adaptivity, not a
higher Sharpe, is the honest reason to hold it.

### What the numbers do and do not say — read this

- **It is not a Sharpe-maximiser.** Excess-over-cash Sharpe (0.48) is *below* a
  naive 60/40 (0.60) and SPY (0.69). A chunk of the 2022-2025 return was simply
  the ~4-5% cash yield. Do not sell this on risk-adjusted return.
- **Its edge is tail risk and low correlation.** ~⅖ of SPY's max drawdown, the
  lowest Ulcer index of the three, beta 0.17. As an *overlay/diversifier* on an
  equity book it removes far more drawdown than return.
- **The alpha is positive but not significant** on 6.5 years (t≈1.3). Honest
  small-sample caveat, not a p-hacked headline.
- **Robustness is genuine.** One-at-a-time sensitivity holds Sharpe in a
  **0.66–0.82 plateau**; swapping the trend/vol ensembles for single lookbacks
  barely moves it. Bootstrap 95% Sharpe CI **[0.19, 1.38]**; Deflated Sharpe prob
  **0.86** (N=18 trials) — the result is very likely real, just modest.

### Why the earlier "1.4" target was never honestly reachable here

The decisive evidence was visible *before* any OOS data: **no cell in the 18-cell
design grid (2008-2019, incl. the 2010s bull) exceeded Sharpe 0.91.** The honest
ceiling of a monthly, long-only, cost-inclusive, diversified multi-asset strategy
is ~0.9; 1.4 is ~1.5× that. Reaching it needs a different mandate (higher
frequency, a narrower/leveraged bet, or accepting tail risk) — a risk-appetite
decision for the portfolio owner, not a backtest trick.

Full outputs: `results/multifactor/` (`metrics.json`, `tearsheet.png`,
`crisis_table.csv`, `design_grid.csv`, `oos_grid.csv`, `sensitivity.csv`,
`weights.csv`).

To reproduce:

```bash
pip install -e ".[yfinance]"    # or: pip install yfinance
python scripts/run_experiment.py            # uses cached data/mfaa_prices.csv
python scripts/run_experiment.py --refresh  # re-download prices
pytest -q                                   # 20 engine + analytics regression tests
```
