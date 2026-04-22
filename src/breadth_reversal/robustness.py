"""
Robustness checks (items 10, 11, 13, 14 from critique):
  * lookback_sensitivity — sensitivity to pct_window parameter
  * walk_forward         — rolling 5-year in-sample / 1-year OOS test
  * rolling_correlation  — 252-day rolling corr vs BH ES
  * capacity_analysis    — back-of-envelope capacity math
"""

import numpy as np
import pandas as pd

from .backtest import TRUE_SHARPE_KEY, run_backtest
from .compute_breadth import compute_signal


# ──────────────────── lookback sensitivity ────────────────────

def lookback_sensitivity(base_df, windows=(10, 20, 40, 60), pct_threshold=10,
                         diff_lag=1, ma_window=5, holding_period=1):
    """Re-run with different pct_window lookbacks and summarise key stats."""
    rows = []
    for w in windows:
        df = compute_signal(base_df.copy(), diff_lag=diff_lag, ma_window=ma_window,
                            pct_window=w, pct_threshold=pct_threshold,
                            holding_period=holding_period)
        df, stats = run_backtest(df)
        rows.append({
            'pct_window':            w,
            'Exposure (%)':          stats['Exposure (%)'],
            'Trades':                stats['Total Trades'],
            'Win Rate (%)':          stats['Win Rate (%)'],
            'CAGR (%)':              stats['CAGR (%)'],
            'Sharpe':                stats[TRUE_SHARPE_KEY],
            'Max DD (%)':            stats['Maximum Drawdown (%)'],
            'IR vs B&H':             stats['Information Ratio vs B&H ES'],
            'Avg Trade (%)':         stats['Avg Return per Trade (%)'],
        })
    return pd.DataFrame(rows)


# ──────────────────── walk-forward ────────────────────

def walk_forward(base_df, insample_years=5, oos_years=1,
                 thresholds=(5, 10, 15), ma_windows=(3, 5, 7),
                 diff_lag=1, pct_window=20, holding_period=1):
    """
    Rolling IS/OOS test.

    * IS window: `insample_years` of daily data used to sweep a small
      parameter grid and pick the highest-Sharpe combo.
    * OOS window: the next `oos_years` of daily data are scored using the
      IS-chosen parameters.
    * Roll forward by `oos_years` each iteration.

    Returns a DataFrame with one row per OOS window.
    """
    out = []
    start = base_df.index.min()
    end   = base_df.index.max()
    is_delta  = pd.DateOffset(years=insample_years)
    oos_delta = pd.DateOffset(years=oos_years)

    is_start = start
    while True:
        is_end  = is_start + is_delta
        oos_end = is_end + oos_delta
        if oos_end > end:
            break

        is_slice  = base_df.loc[is_start:is_end]
        oos_slice = base_df.loc[is_end:oos_end]
        if len(is_slice) < 252 or len(oos_slice) < 126:
            break

        # IS parameter sweep — pick the pair maximising Sharpe.
        best = None
        for th in thresholds:
            for mw in ma_windows:
                is_df = compute_signal(is_slice.copy(), diff_lag=diff_lag,
                                       ma_window=mw, pct_window=pct_window,
                                       pct_threshold=th, holding_period=holding_period)
                _, s = run_backtest(is_df)
                sharpe = s[TRUE_SHARPE_KEY]
                if best is None or (sharpe is not None and sharpe > best['sharpe']):
                    best = {'th': th, 'mw': mw, 'sharpe': sharpe}

        # OOS evaluation with IS-chosen params
        oos_df = compute_signal(oos_slice.copy(), diff_lag=diff_lag,
                                ma_window=best['mw'], pct_window=pct_window,
                                pct_threshold=best['th'], holding_period=holding_period)
        _, s = run_backtest(oos_df)
        out.append({
            'IS start':  is_start.date(),
            'IS end':    is_end.date(),
            'OOS end':   oos_end.date(),
            'IS threshold': best['th'],
            'IS ma_window': best['mw'],
            'IS Sharpe':    round(best['sharpe'], 3) if best['sharpe'] is not None else np.nan,
            'OOS Sharpe':   s[TRUE_SHARPE_KEY],
            'OOS CAGR (%)': s['CAGR (%)'],
            'OOS Max DD (%)': s['Maximum Drawdown (%)'],
            'OOS Trades':   s['Total Trades'],
        })
        is_start = is_start + oos_delta

    return pd.DataFrame(out)


# ──────────────────── rolling correlation ────────────────────

def rolling_correlation(df, window=252):
    """252-day rolling correlation of daily strategy PnL vs daily BH ES return."""
    return df['PnL'].rolling(window).corr(df['Return_Pct']).dropna()


# ──────────────────── capacity ────────────────────

def capacity_analysis(df, es_notional_per_contract_multiplier=50,
                      aum_grid_musd=(100, 250, 500, 1_000, 2_000),
                      moc_fraction_daily_volume=0.10,
                      es_daily_notional_usd_bn=500):
    """
    Back-of-envelope capacity math for the MOC execution window.

    Assumptions (stated in-report):
      * ES notional = price × $50 × contracts. We use the last observed
        ES close as the price reference.
      * Aggregate ES daily notional = $500B. MOC imbalance is conservatively
        taken as `moc_fraction_daily_volume` × $500B = $50B notional.
      * On a signal day, the strategy must execute its full AUM notional
        through the MOC window.

    Returns a DataFrame indexed by AUM with estimated MOC footprint.
    """
    last_close = df['ES_Close'].iloc[-1]
    contract_notional = last_close * es_notional_per_contract_multiplier
    moc_imbalance_usd = moc_fraction_daily_volume * es_daily_notional_usd_bn * 1e9

    rows = []
    for aum_m in aum_grid_musd:
        aum_usd = aum_m * 1e6
        contracts = aum_usd / contract_notional
        footprint = aum_usd / moc_imbalance_usd * 100
        rows.append({
            'AUM ($M)':             aum_m,
            'ES contracts / signal day': round(contracts, 0),
            '% of $50B MOC imbalance':   round(footprint, 2),
        })
    return pd.DataFrame(rows)
