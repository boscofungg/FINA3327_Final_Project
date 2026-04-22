"""Generate all charts for the updated report."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from breadth_reversal.backtest import TRUE_SHARPE_KEY, rolling_ff3_alpha, run_backtest
from breadth_reversal.compute_breadth import compute_signal
from breadth_reversal.data_loader import load_data
from breadth_reversal.paths import GRAPHS_DIR
from breadth_reversal.robustness import lookback_sensitivity, rolling_correlation

GRAPHS_DIR.mkdir(exist_ok=True)

PARAMS = {'diff_lag': 1, 'ma_window': 5, 'pct_window': 20,
          'pct_threshold': 10.0, 'holding_period': 1}

df = load_data('Bloomberg_data.xlsx')
df = compute_signal(df, **PARAMS)
df, stats = run_backtest(df)

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.edgecolor':     '#333333',
    'axes.labelcolor':    '#333333',
    'xtick.color':        '#333333',
    'ytick.color':        '#333333',
    'axes.grid':          True,
    'grid.alpha':         0.3,
})
NAVY   = '#1F3864'
ORANGE = '#C55A11'
RED    = '#C0392B'

# B&H NAV
bh_ret = df['Return_Pct'].dropna()
bh_nav = (1 + bh_ret).cumprod()

# ── equity curve ──
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(df.index, df['NAV'],    color=NAVY,   lw=2, label='Strategy')
ax.plot(bh_nav.index, bh_nav,   color=ORANGE, lw=2, label='Buy-and-Hold ES', alpha=0.85)
ax.set_ylabel('NAV (starting = 1.00)', fontsize=11)
ax.set_title('Cumulative Equity Curve — Strategy vs Buy-and-Hold S&P 500 (ES)',
             fontsize=13, fontweight='bold', color=NAVY)
ax.legend(loc='upper left', frameon=False, fontsize=10)
summary_text = "\n".join([
    f"True Sharpe = {stats[TRUE_SHARPE_KEY]:.2f}",
    f"Avg RF (2010-2026) = {stats['Avg RF used (ann. %)']:.2f}%",
    f"Exposure-adjusted ann. return = {stats['Exposure-Adjusted Annualized Return (%)']:.2f}%",
    f"Active-day Sharpe = {stats['Active-Day Sharpe']:.2f}",
])
ax.text(
    0.99,
    0.02,
    summary_text,
    transform=ax.transAxes,
    ha='right',
    va='bottom',
    fontsize=9,
    color='#222222',
    bbox={'boxstyle': 'round,pad=0.4', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': '#CCCCCC'},
)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / 'equity_curve.png', dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

# ── drawdown ──
dd_strat = (df['NAV'].cummax() - df['NAV']) / df['NAV'].cummax() * 100
dd_bh    = (bh_nav.cummax() - bh_nav) / bh_nav.cummax() * 100
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(df.index, -dd_strat, 0, color=NAVY,   alpha=0.7, label='Strategy')
ax.fill_between(dd_bh.index, -dd_bh, 0, color=ORANGE, alpha=0.4, label='Buy-and-Hold')
ax.set_ylabel('Drawdown (%)', fontsize=11)
ax.set_title('Drawdown Comparison — Strategy vs Buy-and-Hold',
             fontsize=13, fontweight='bold', color=NAVY)
ax.legend(loc='lower left', frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / 'drawdown.png', dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

# ── monthly heatmap ──
monthly = (1 + df['PnL']).resample('ME').prod() - 1
monthly_tbl = monthly.to_frame('ret')
monthly_tbl['Year']  = monthly_tbl.index.year
monthly_tbl['Month'] = monthly_tbl.index.month
pivot = monthly_tbl.pivot(index='Year', columns='Month', values='ret') * 100
vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
fig, ax = plt.subplots(figsize=(11, 5.5))
im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(12))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=8, color='black' if abs(v) < vmax*0.6 else 'white')
ax.set_title('Monthly Strategy Returns (%)', fontsize=13, fontweight='bold', color=NAVY)
ax.grid(False)
plt.colorbar(im, ax=ax, shrink=0.8, label='Return (%)')
plt.tight_layout()
plt.savefig(GRAPHS_DIR / 'monthly_heatmap.png', dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

# ── trade-return distribution ──
trade_returns = []
in_trade, pnl = False, 0.0
for _, row in df.iterrows():
    if row['Entry'] == 1:
        in_trade, pnl = True, 0.0
    if in_trade:
        pnl += row['PnL']
    if row['Exit'] == 1:
        trade_returns.append(pnl * 100)
        in_trade = False
trade_returns = pd.Series(trade_returns)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.hist(trade_returns, bins=50, color=NAVY, alpha=0.85, edgecolor='white')
ax.axvline(trade_returns.mean(), color=ORANGE, linestyle='--', lw=2,
           label=f'Mean = {trade_returns.mean():.2f}%')
ax.axvline(0, color='black', lw=0.8, alpha=0.5)
ax.set_xlabel('Trade Return (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Trade Returns', fontsize=13, fontweight='bold', color=NAVY)
ax.legend(loc='upper right', frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / 'return_dist.png', dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

# ── bucket analysis ──
bucket_perf = df.groupby('Bucket', observed=True)['Return_Pct'].agg(['mean', 'count'])
bucket_perf['mean'] = bucket_perf['mean'] * 100
fig, ax = plt.subplots(figsize=(11, 4.5))
colors = [RED if i == 0 else NAVY for i in range(len(bucket_perf))]
bars = ax.bar(range(len(bucket_perf)), bucket_perf['mean'].values,
              color=colors, alpha=0.85, edgecolor='white')
ax.set_xticks(range(len(bucket_perf)))
ax.set_xticklabels(bucket_perf.index.astype(str), rotation=0)
ax.axhline(0, color='black', lw=0.8, alpha=0.5)
ax.set_xlabel('Breadth-Momentum Percentile Bucket', fontsize=11)
ax.set_ylabel('Mean Next-Day Return (%)', fontsize=11)
ax.set_title('Forward Return by Breadth-Momentum Percentile Bucket (Signal Zone Highlighted)',
             fontsize=12, fontweight='bold', color=NAVY)
for i, (b, v) in enumerate(zip(bars, bucket_perf['mean'].values)):
    ax.text(i, v + (0.015 if v >= 0 else -0.025),
            f'{v:.2f}%', ha='center',
            va='bottom' if v >= 0 else 'top', fontsize=9)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / 'buckets.png', dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

# ── NEW: rolling 1-year correlation vs BH ──
rc = rolling_correlation(df, window=252)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(rc.index, rc.values, color=NAVY, lw=1.6, label='1-year rolling ρ(Strategy, ES)')
ax.axhline(rc.mean(),   color=ORANGE, lw=1, ls='--', label=f'Mean = {rc.mean():.2f}')
ax.axhline(0,           color='black', lw=0.6, alpha=0.5)
# shade known stress windows
for (a, b, lbl) in [('2018-01', '2018-06', '2018 vol'),
                    ('2020-02', '2020-06', 'COVID'),
                    ('2022-01', '2022-12', '2022 rates'),
                    ('2025-02', '2025-06', '2025 drawdown')]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=RED, alpha=0.08)
ax.set_ylim(-0.1, 1.0)
ax.set_ylabel('Rolling Correlation', fontsize=11)
ax.set_title('Rolling 1-Year Correlation, Strategy vs Buy-and-Hold ES',
             fontsize=13, fontweight='bold', color=NAVY)
ax.legend(loc='upper right', frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / 'rolling_corr.png', dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

# ── NEW: rolling FF3 alpha ──
rolling_alpha = rolling_ff3_alpha(df, window=756, sample_freq='ME')
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(
    rolling_alpha.index,
    rolling_alpha['FF3 Alpha (ann. %)'],
    color=NAVY,
    lw=1.8,
    label='Rolling FF3 alpha (annualized)',
)
ax.axhline(0, color='black', lw=0.8, alpha=0.6)
ax.axhline(
    rolling_alpha['FF3 Alpha (ann. %)'].mean(),
    color=ORANGE,
    lw=1,
    ls='--',
    label=f"Mean = {rolling_alpha['FF3 Alpha (ann. %)'].mean():.2f}%",
)
ax.set_ylabel('Alpha (ann. %)', fontsize=11)
ax.set_title('Rolling FF3 Alpha — 3-Year Window, Sampled Monthly',
             fontsize=13, fontweight='bold', color=NAVY)
ax.legend(loc='upper right', frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / 'rolling_ff3_alpha.png', dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

# ── NEW: lookback sensitivity ──
base = load_data('Bloomberg_data.xlsx')
ls = lookback_sensitivity(base, windows=(10, 20, 40, 60))
fig, ax1 = plt.subplots(figsize=(11, 4.2))
x = np.arange(len(ls))
width = 0.35
bars_s = ax1.bar(x - width/2, ls['Sharpe'].values, width,
                 color=NAVY, alpha=0.85, label='True Sharpe')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{int(w)}-day' for w in ls['pct_window']])
ax1.set_ylabel('Sharpe Ratio', color=NAVY, fontsize=11)
ax1.axhline(0, color='black', lw=0.6, alpha=0.5)
ax1.set_ylim(0, max(ls['Sharpe'])*1.25)

ax2 = ax1.twinx()
bars_d = ax2.bar(x + width/2, ls['Max DD (%)'].values, width,
                 color=ORANGE, alpha=0.85, label='Max DD (%)')
ax2.set_ylabel('Max Drawdown (%)', color=ORANGE, fontsize=11)
ax2.grid(False)
ax2.set_ylim(0, max(ls['Max DD (%)'])*1.25)

for bars, vals, fmt in [(bars_s, ls['Sharpe'].values, '{:.2f}'),
                        (bars_d, ls['Max DD (%)'].values, '{:.1f}%')]:
    for b, v in zip(bars, vals):
        ax1_or_ax2 = ax1 if bars is bars_s else ax2
        ax1_or_ax2.text(b.get_x()+b.get_width()/2, v*1.03,
                        fmt.format(v), ha='center', va='bottom', fontsize=9)

ax1.set_title('Lookback Sensitivity — Percentile Window (pct_window)',
              fontsize=13, fontweight='bold', color=NAVY)
# unified legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / 'lookback_sensitivity.png', dpi=140, bbox_inches='tight', facecolor='white')
plt.close()

print(f'Charts saved to {GRAPHS_DIR}')
