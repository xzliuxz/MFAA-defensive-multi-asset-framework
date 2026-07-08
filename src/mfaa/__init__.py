"""
MFAA -- Multi-Factor Adaptive Allocation.

A defensive, long-only, monthly, price-only multi-asset ETF framework built from
factors published before 2020 (cross-sectional 13612 momentum + ensembled trend
following + canary crash overlay + inverse-vol weighting + vol targeting), packaged
with an honest anti-overfitting research harness (fixed neutral prior, Deflated
Sharpe Ratio, bootstrap CI, crisis table, tearsheet).

See ``docs/methodology.md`` for the full protocol and out-of-sample results.
"""

from .multifactor import (
    ALL_TICKERS,
    CANARY_ASSETS,
    DEFENSIVE_POOL,
    RISKY_ASSETS,
    MultiFactorEngine,
    block_bootstrap_sharpe_ci,
    blended_inverse_vol_weights,
    deflated_sharpe,
    inverse_vol_weights,
    load_universe_prices,
    momentum_13612,
    run_multifactor_backtest,
    sharpe_excess,
    trend_participation,
)
from . import analytics

__version__ = "0.3.0"
__all__ = [
    "MultiFactorEngine",
    "run_multifactor_backtest",
    "momentum_13612",
    "inverse_vol_weights",
    "blended_inverse_vol_weights",
    "trend_participation",
    "block_bootstrap_sharpe_ci",
    "deflated_sharpe",
    "sharpe_excess",
    "load_universe_prices",
    "analytics",
    "RISKY_ASSETS",
    "DEFENSIVE_POOL",
    "CANARY_ASSETS",
    "ALL_TICKERS",
]
