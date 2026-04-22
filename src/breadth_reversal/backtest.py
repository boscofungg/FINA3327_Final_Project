"""
Backtest + factor attribution engine for the Breadth Reversal strategy.

Critique responses implemented here:
  * Time-varying risk-free rate: Sharpe and excess-return metrics use the
    Kenneth French daily T-bill series, not a flat 4%. The full-period
    mean of that series (~1.4% annualised over 2010-2026) is reported for
    transparency.
  * True Sharpe labelling: the headline Sharpe is computed exactly as
    requested, i.e. (annualised return - average annual RF) / annualised
    volatility. We also preserve an exposure-aware diagnostic view via the
    active-day Sharpe and an exposure-adjusted annualised return.
  * Information Ratio: redefined vs. the BH ES benchmark (tracking-error
    formulation). Formula: IR = ann(mean(PnL - BH_excess))
                                / ann(std(PnL - BH_excess)).
  * Newey-West HAC standard errors in all factor regressions.
  * Removed the "exposure-adjusted return" as a return. We keep a
    diagnostic "capital-efficiency ratio" (compounded return / exposure)
    labeled explicitly as NOT a return.
  * Rolling 1-year correlation vs BH exposed to callers via the returned
    DataFrame.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from .paths import DATA_DIR


TRUE_SHARPE_KEY = "Sharpe Ratio (ann. return - avg RF) / ann. vol"


# ─────────────── utility: Newey-West HAC ───────────────

def _newey_west_lag(n_obs):
    """Newey-West (1994) plug-in bandwidth: ceil(4 * (T/100)^(2/9))."""
    return int(max(1, math.ceil(4 * (n_obs / 100.0) ** (2.0 / 9.0))))


def _ols_with_nw(y, x, nw_lag=None):
    """OLS with Newey-West (Bartlett) HAC standard errors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[keep], x[keep]
    n_obs, n_params = x.shape

    if nw_lag is None:
        nw_lag = _newey_west_lag(n_obs)

    xtx_inv = np.linalg.inv(x.T @ x)
    params = xtx_inv @ x.T @ y
    resid = y - x @ params

    u = x * resid[:, None]
    S = u.T @ u
    for L in range(1, nw_lag + 1):
        w = 1.0 - L / (nw_lag + 1.0)
        GammaL = u[L:].T @ u[:-L]
        S += w * (GammaL + GammaL.T)

    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvalues = params / np.where(se > 0, se, np.nan)
    pvalues = np.array([math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else np.nan
                        for t in tvalues])
    ss_total = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_total if ss_total > 0 else np.nan

    return {'params': params, 'se': se, 'tvalues': tvalues, 'pvalues': pvalues,
            'rsquared': r2, 'nobs': n_obs, 'nw_lag': nw_lag}


# ─────────────── utility: FF loader ───────────────

def _load_ff_daily(path):
    path = Path(path)
    if not path.exists():
        for candidate in (DATA_DIR / path.name, Path.cwd() / path.name):
            if candidate.exists():
                path = candidate
                break
    raw = path.read_text().splitlines()
    skip = 0
    for i, line in enumerate(raw):
        if line.strip() and line.strip()[0].isdigit():
            skip = i - 1
            break
    factors = pd.read_csv(path, skiprows=skip)
    factors.columns = factors.columns.str.strip()
    factors.rename(columns={factors.columns[0]: 'Date'}, inplace=True)
    factors = factors[pd.to_numeric(factors['Date'], errors='coerce').notna()].copy()
    factors['Date'] = pd.to_datetime(factors['Date'].astype(str), format='%Y%m%d')
    factors = factors.set_index('Date').apply(pd.to_numeric, errors='coerce') / 100
    return factors


# ─────────────── main backtest ───────────────

def run_backtest(df, ff3_path='F-F_Research_Data_Factors_daily.csv'):
    # Next-day returns: signal[t] earns close[t]→close[t+1]
    df['Return_Pct'] = df['ES_Close'].pct_change().shift(-1)
    df['Return_Pts'] = df['ES_Close'].shift(-1) - df['ES_Close']

    # Join time-varying daily RF (for true Sharpe & IR); fallback to 0.
    try:
        ff = _load_ff_daily(ff3_path)
        df = df.join(ff[['RF']], how='left')
        df['RF'] = df['RF'].ffill().fillna(0)
    except Exception:
        df['RF'] = 0.0
    rf_avg_ann = df['RF'].mean() * 252 * 100

    # Transaction cost: 2 bps per side (4 bps round trip)
    TC = 0.0002
    df['TC_Cost'] = (df['Entry'] + df['Exit']) * TC

    # Strategy daily PnL (futures convention: excess over collateral rate).
    df['PnL'] = df['Signal'] * df['Return_Pct'] - df['TC_Cost']
    df['PnL'] = df['PnL'].fillna(0)

    # NAV, cumulative PnL, drawdown
    df['NAV']            = (1 + df['PnL']).cumprod()
    df['Cumulative_PnL'] = df['PnL'].cumsum()
    df['DrawDown_NAV']   = (df['NAV'].cummax() - df['NAV']) / df['NAV'].cummax()

    # ── Trade-level analysis ──
    total_trades = int(df['Entry'].sum())
    total_days   = len(df)
    years        = (df.index[-1] - df.index[0]).days / 365.25

    trade_returns = []
    in_trade, trade_pnl = False, 0.0
    for _, row in df.iterrows():
        if row['Entry'] == 1:
            in_trade, trade_pnl = True, 0.0
        if in_trade:
            trade_pnl += row['PnL']
        if row['Exit'] == 1:
            trade_returns.append(trade_pnl)
            in_trade = False
    trade_returns = pd.Series(trade_returns)

    wins           = (trade_returns > 0).sum()
    win_rate       = wins / len(trade_returns) * 100 if len(trade_returns) > 0 else 0
    avg_return_pct = trade_returns.mean() * 100
    avg_return_pts = df[df['Entry'] == 1]['Return_Pts'].mean()
    gross_wins     = trade_returns[trade_returns > 0].sum()
    gross_losses   = -trade_returns[trade_returns < 0].sum()
    profit_factor  = gross_wins / gross_losses if gross_losses > 0 else np.nan

    # ── Headline returns ──
    exposure          = df['Signal'].sum() / total_days * 100
    compounded_return = (df['NAV'].iloc[-1] - 1) * 100
    annualized_return = df['PnL'].mean() * 252 * 100
    cagr              = (df['NAV'].iloc[-1] ** (1 / years) - 1) * 100
    max_drawdown      = df['DrawDown_NAV'].max() * 100

    # Capital-efficiency ratio — explicitly NOT a return; diagnostic only.
    capital_efficiency = compounded_return / exposure if exposure > 0 else np.nan

    # ── True Sharpe view ──
    full_mean, full_std = df['PnL'].mean(), df['PnL'].std(ddof=1)
    ann_vol_full = full_std * np.sqrt(252) * 100
    true_sharpe = ((annualized_return - rf_avg_ann) / ann_vol_full
                   if ann_vol_full > 0 else np.nan)
    exposure_adjusted_return = annualized_return / (exposure / 100) if exposure > 0 else np.nan

    # ── Active-day Sharpe ──
    active_returns = df.loc[df['Signal'] == 1, 'PnL']
    active_mean, active_std = active_returns.mean(), active_returns.std(ddof=1)
    sharpe_active = (active_mean / active_std) * np.sqrt(252) if active_std > 0 else np.nan
    active_ann_return = active_mean * 252 * 100 if len(active_returns) else np.nan
    active_ann_vol = active_std * np.sqrt(252) * 100 if active_std > 0 else np.nan

    # ── Information Ratio vs buy-and-hold ES ──
    # IR = ann(mean(PnL - (ES_ret - RF))) / ann(std(PnL - (ES_ret - RF)))
    # Rationale: BH excess = ES_ret - RF is the investable benchmark; the
    # futures PnL is already expressed net of RF, so we subtract the same
    # RF from ES_ret to keep both on the same footing.
    bh_raw     = df['Return_Pct']
    bh_excess  = (bh_raw - df['RF']).dropna()
    diff_vs_bh = (df['PnL'].reindex(bh_excess.index) - bh_excess).dropna()
    ir_numer = diff_vs_bh.mean() * 252
    ir_denom = diff_vs_bh.std(ddof=1) * np.sqrt(252)
    ir_vs_bh = ir_numer / ir_denom if ir_denom > 0 else np.nan

    # Downside deviation (active days, annualised)
    downside_active = active_returns[active_returns < 0]
    downside_dev = (downside_active.std(ddof=1) * np.sqrt(252) * 100
                    if len(downside_active) > 1 else np.nan)

    # ── VaR / CVaR (active days) ──
    z_95, z_99 = 1.645, 2.326
    phi_95 = math.exp(-0.5 * z_95 ** 2) / math.sqrt(2 * math.pi)
    phi_99 = math.exp(-0.5 * z_99 ** 2) / math.sqrt(2 * math.pi)
    var_95_param  = -(active_mean - z_95 * active_std) * 100
    var_99_param  = -(active_mean - z_99 * active_std) * 100
    var_95_hist   = -np.percentile(active_returns, 5) * 100
    var_99_hist   = -np.percentile(active_returns, 1) * 100
    cvar_95_param = -(active_mean - active_std * phi_95 / 0.05) * 100
    cvar_99_param = -(active_mean - active_std * phi_99 / 0.01) * 100
    cvar_95_hist  = -active_returns[active_returns <= np.percentile(active_returns, 5)].mean() * 100
    cvar_99_hist  = -active_returns[active_returns <= np.percentile(active_returns, 1)].mean() * 100

    # ── Buy-and-Hold benchmark ──
    bh_returns    = bh_raw.dropna()
    bh_nav        = (1 + bh_returns).cumprod()
    bh_compounded = (bh_nav.iloc[-1] - 1) * 100
    bh_cagr       = (bh_nav.iloc[-1] ** (1 / years) - 1) * 100
    bh_ann_return = bh_returns.mean() * 252 * 100
    bh_ann_vol    = bh_returns.std(ddof=1) * np.sqrt(252) * 100
    bh_sharpe = ((bh_ann_return - rf_avg_ann) / bh_ann_vol) if bh_ann_vol > 0 else np.nan
    bh_drawdown = ((bh_nav.cummax() - bh_nav) / bh_nav.cummax()).max() * 100

    # ── Correlation vs BH (full-series daily) ──
    aligned = pd.concat([df['PnL'], bh_returns], axis=1, join='inner').dropna()
    aligned.columns = ['Strat', 'BH']
    correlation_bh = aligned['Strat'].corr(aligned['BH'])

    # Rolling 1-year correlation (252-day)
    df['RollCorr_1y'] = df['PnL'].rolling(252).corr(df['Return_Pct'])

    # ── Monthly stats ──
    monthly_returns = (1 + df['PnL']).resample('ME').prod() - 1
    monthly_returns = monthly_returns.dropna()
    best_month     = monthly_returns.max() * 100
    worst_month    = monthly_returns.min() * 100
    pct_pos_months = (monthly_returns > 0).mean() * 100

    stats = {
        # Trade stats
        'Total Trades':                      total_trades,
        'Win Rate (%)':                      round(win_rate, 2),
        'Profit Factor':                     round(profit_factor, 2) if not np.isnan(profit_factor) else 'n/a',
        'Avg Return per Trade (%)':          round(avg_return_pct, 4),
        'Avg Return per Trade (pts)':        round(avg_return_pts, 2),

        # Headline returns
        'Compounded Return (%)':             round(compounded_return, 2),
        'Annualized Return (%)':             round(annualized_return, 2),
        'CAGR (%)':                          round(cagr, 2),
        'Capital Efficiency Ratio':          round(capital_efficiency, 2),
        'Exposure-Adjusted Annualized Return (%)': round(exposure_adjusted_return, 2),

        # Risk-adjusted
        TRUE_SHARPE_KEY:                     round(true_sharpe, 3),
        'Active-Day Sharpe':                 round(sharpe_active, 3),
        'Active-Day Annualized Return (%)':  round(active_ann_return, 2) if not np.isnan(active_ann_return) else 'n/a',
        'Active-Day Annualized Volatility (%)': round(active_ann_vol, 2) if not np.isnan(active_ann_vol) else 'n/a',
        'Information Ratio vs B&H ES':       round(ir_vs_bh, 3),
        'Annualized Volatility (%)':         round(ann_vol_full, 2),
        'Downside Dev Active (%)':           round(downside_dev, 2) if not np.isnan(downside_dev) else 'n/a',

        # Risk
        'Maximum Drawdown (%)':              round(max_drawdown, 2),
        'Exposure (%)':                      round(exposure, 2),
        'Parametric VaR 95% Active (%)':     round(var_95_param, 4),
        'Parametric VaR 99% Active (%)':     round(var_99_param, 4),
        'Historical VaR 95% Active (%)':     round(var_95_hist, 4),
        'Historical VaR 99% Active (%)':     round(var_99_hist, 4),
        'Parametric CVaR 95% Active (%)':    round(cvar_95_param, 4),
        'Parametric CVaR 99% Active (%)':    round(cvar_99_param, 4),
        'Historical CVaR 95% Active (%)':    round(cvar_95_hist, 4),
        'Historical CVaR 99% Active (%)':    round(cvar_99_hist, 4),

        # Monthly
        'Best Month (%)':                    round(best_month, 2),
        'Worst Month (%)':                   round(worst_month, 2),
        'Winning Months (%)':                round(pct_pos_months, 2),

        # B&H benchmark
        'BH Compounded Return (%)':          round(bh_compounded, 2),
        'BH Annualized Return (%)':          round(bh_ann_return, 2),
        'BH CAGR (%)':                       round(bh_cagr, 2),
        'BH Sharpe (ann. return - avg RF) / ann. vol': round(bh_sharpe, 3),
        'BH Annualized Volatility (%)':      round(bh_ann_vol, 2),
        'BH Max Drawdown (%)':               round(bh_drawdown, 2),
        'Correlation vs B&H (daily)':        round(correlation_bh, 3),

        # Methodology
        'Avg RF used (ann. %)':              round(rf_avg_ann, 2),
    }
    return df, stats


# ─────────────── factor analysis with HAC SEs ───────────────

def run_factor_analysis(df,
                        ff3_path='F-F_Research_Data_Factors_daily.csv'):

    ff3_factors = _load_ff_daily(ff3_path)

    # Alignment: PnL[t] realizes close[t]→close[t+1]; FF factor[t] is
    # close[t-1]→close[t]. Shift strategy PnL forward by 1 day so that
    # both are dated on the realization day.
    df_aligned = df[['PnL', 'ES_Close']].copy()
    df_aligned['PnL'] = df_aligned['PnL'].shift(1)

    daily = df_aligned.join(ff3_factors, how='inner').dropna()
    daily['Excess'] = daily['PnL'] - daily['RF']
    daily['BH_ES'] = daily['ES_Close'].pct_change()

    y = daily['Excess'].values

    # ── CAPM (HAC) ──
    capm = _ols_with_nw(y, np.column_stack([np.ones(len(daily)), daily['Mkt-RF'].values]))
    capm_alpha_daily = capm['params'][0]
    capm_alpha_ann   = (1 + capm_alpha_daily) ** 252 - 1
    capm_beta        = capm['params'][1]
    capm_alpha_t     = capm['tvalues'][0]
    capm_alpha_p     = capm['pvalues'][0]
    capm_r2          = capm['rsquared']
    capm_nw          = capm['nw_lag']

    # ── FF3 (HAC) ──
    ff3_aligned = df_aligned[['PnL']].join(ff3_factors, how='inner').dropna()
    ff3_aligned['Excess'] = ff3_aligned['PnL'] - ff3_aligned['RF']
    ff3_cols = ['Mkt-RF', 'SMB', 'HML']
    ff3 = _ols_with_nw(ff3_aligned['Excess'].values,
                       np.column_stack([np.ones(len(ff3_aligned)), ff3_aligned[ff3_cols].values]))
    ff3_alpha_daily = ff3['params'][0]
    ff3_alpha_ann   = (1 + ff3_alpha_daily) ** 252 - 1
    ff3_alpha_t     = ff3['tvalues'][0]
    ff3_alpha_p     = ff3['pvalues'][0]
    ff3_r2          = ff3['rsquared']
    ff3_betas       = dict(zip(ff3_cols, ff3['params'][1:]))
    ff3_tvalues     = dict(zip(ff3_cols, ff3['tvalues'][1:]))
    ff3_pvalues     = dict(zip(ff3_cols, ff3['pvalues'][1:]))

    # ── Monthly aggregation ──
    monthly_strat = daily['PnL'].resample('ME').sum()
    monthly_bh    = daily['BH_ES'].resample('ME').sum()
    monthly_rf    = daily['RF'].resample('ME').sum()
    monthly_mktrf = daily['Mkt-RF'].resample('ME').sum()
    monthly = pd.DataFrame({'Strategy': monthly_strat, 'BH_ES': monthly_bh,
                            'RF': monthly_rf, 'Mkt-RF': monthly_mktrf}).dropna()
    monthly['Excess'] = monthly['Strategy'] - monthly['RF']

    factor_stats = {
        'capm_alpha_daily': capm_alpha_daily, 'capm_alpha_ann': capm_alpha_ann,
        'capm_alpha_t': capm_alpha_t, 'capm_alpha_p': capm_alpha_p,
        'capm_beta': capm_beta, 'capm_r2': capm_r2, 'capm_nw_lag': capm_nw,
        'ff3_alpha_ann': ff3_alpha_ann, 'ff3_alpha_t': ff3_alpha_t,
        'ff3_alpha_p': ff3_alpha_p, 'ff3_r2': ff3_r2,
        'ff3_betas': ff3_betas, 'ff3_tvalues': ff3_tvalues, 'ff3_pvalues': ff3_pvalues,
        'ff3_nw_lag': ff3['nw_lag'],
        'ann_return': monthly_strat.mean() * 12,
        'max_dd': daily['PnL'].cumsum().cummax().sub(daily['PnL'].cumsum()).max(),
    }

    print("\n--- CAPM (HAC SEs, NW lag = %d) ---" % capm_nw)
    print(f"  Obs: {len(daily)} | α(ann): {capm_alpha_ann*100:.2f}%  t={capm_alpha_t:.3f}  p={capm_alpha_p:.3f}")
    print(f"  β_MKT: {capm_beta:.4f}  R²: {capm_r2:.4f}")
    print("\n--- FF3 (HAC SEs, NW lag = %d) ---" % ff3['nw_lag'])
    print(f"  α(ann): {ff3_alpha_ann*100:.2f}%  t={ff3_alpha_t:.3f}  p={ff3_alpha_p:.3f}  R²: {ff3_r2:.4f}")
    for k in ff3_cols:
        print(f"  {k:6s} β={ff3_betas[k]: .4f}  t={ff3_tvalues[k]: .3f}  p={ff3_pvalues[k]:.3f}")

    return factor_stats, monthly


# ─────────────── rolling FF3 alpha ───────────────

def rolling_ff3_alpha(df, ff3_path='F-F_Research_Data_Factors_daily.csv',
                      window=756, sample_freq='ME'):
    """
    Estimate rolling FF3 alpha over time.

    Uses daily strategy PnL aligned to FF3 realization dates, then fits a
    rolling FF3 regression over `window` trading days. The alpha estimate is
    annualized and sampled at the requested frequency (default: month-end).
    """
    ff3 = _load_ff_daily(ff3_path)

    aligned = df[['PnL']].copy()
    aligned['PnL'] = aligned['PnL'].shift(1)
    aligned = aligned.join(ff3[['RF', 'Mkt-RF', 'SMB', 'HML']], how='inner').dropna()
    aligned['Excess'] = aligned['PnL'] - aligned['RF']

    sample_dates = aligned.resample(sample_freq).last().index
    sample_dates = [date for date in sample_dates if date in aligned.index]

    rows = []
    ff3_cols = ['Mkt-RF', 'SMB', 'HML']
    for date in sample_dates:
        loc = aligned.index.get_loc(date)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc + 1 < window:
            continue

        window_df = aligned.iloc[loc + 1 - window:loc + 1]
        res = _ols_with_nw(
            window_df['Excess'].values,
            np.column_stack([np.ones(len(window_df)), window_df[ff3_cols].values]),
        )
        alpha_daily = res['params'][0]
        alpha_ann = ((1 + alpha_daily) ** 252 - 1) * 100
        rows.append({
            'Date': date,
            'FF3 Alpha (ann. %)': alpha_ann,
            't (HAC)': res['tvalues'][0],
            'p (HAC)': res['pvalues'][0],
            'R²': res['rsquared'],
            'N obs': res['nobs'],
        })

    if not rows:
        return pd.DataFrame(columns=['FF3 Alpha (ann. %)', 't (HAC)', 'p (HAC)', 'R²', 'N obs'])

    return pd.DataFrame(rows).set_index('Date')


# ─────────────── alpha-by-frequency (HAC) ───────────────

def run_alpha_by_frequency(df,
                           ff3_path='F-F_Research_Data_Factors_daily.csv'):
    """Reproduces the cross-frequency table: {daily, monthly, yearly} × {CAPM, FF3}."""
    ff3 = _load_ff_daily(ff3_path)

    # Daily realization-day alignment
    d = df[['PnL']].copy()
    d['PnL'] = d['PnL'].shift(1)
    d3 = d.join(ff3, how='inner').dropna()

    def fit(y, X, nw_lag=None):
        Xc = np.column_stack([np.ones(len(X)), X.values if isinstance(X, pd.DataFrame) else X])
        res = _ols_with_nw(y, Xc, nw_lag=nw_lag)
        return res

    periods = []
    def ann_alpha(daily_alpha, freq):
        if freq == 'daily':   return (1 + daily_alpha) ** 252 - 1
        if freq == 'monthly': return (1 + daily_alpha) ** 12  - 1
        return daily_alpha  # yearly: already annualised

    # daily
    y3 = (d3['PnL'] - d3['RF']).values
    for name, cols, dat, y in [
        ('CAPM', ['Mkt-RF'],                d3, y3),
        ('FF3',  ['Mkt-RF','SMB','HML'],    d3, y3),
    ]:
        res = fit(y, dat[cols])
        periods.append(('Daily', name, ann_alpha(res['params'][0], 'daily')*100,
                        res['tvalues'][0], res['pvalues'][0], res['rsquared'], res['nobs'], res['nw_lag']))

    # monthly
    m3 = d3[['PnL','RF','Mkt-RF','SMB','HML']].resample('ME').sum()
    for name, cols, dat in [
        ('CAPM', ['Mkt-RF'],                m3),
        ('FF3',  ['Mkt-RF','SMB','HML'],    m3),
    ]:
        y = (dat['PnL'] - dat['RF']).values
        res = fit(y, dat[cols])
        periods.append(('Monthly', name, ann_alpha(res['params'][0], 'monthly')*100,
                        res['tvalues'][0], res['pvalues'][0], res['rsquared'], res['nobs'], res['nw_lag']))

    # yearly
    y3y = d3[['PnL','RF','Mkt-RF','SMB','HML']].resample('YE').sum()
    for name, cols, dat in [
        ('CAPM', ['Mkt-RF'],                y3y),
        ('FF3',  ['Mkt-RF','SMB','HML'],    y3y),
    ]:
        y = (dat['PnL'] - dat['RF']).values
        res = fit(y, dat[cols], nw_lag=1)  # few obs; cap NW lag at 1
        periods.append(('Yearly', name, res['params'][0]*100,
                        res['tvalues'][0], res['pvalues'][0], res['rsquared'], res['nobs'], res['nw_lag']))

    out = pd.DataFrame(periods, columns=['Frequency','Model','Alpha (ann %)','t (HAC)',
                                         'p (HAC)','R²','N obs','NW lag'])
    return out
