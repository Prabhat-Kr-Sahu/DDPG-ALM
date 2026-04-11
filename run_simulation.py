"""
Simulation: DDPG-ALM vs Baselines  (2021-2025)
Reads from artifacts/ — no training required.
"""
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime as dt

SIM_START = '2021-01-01'
SIM_END   = '2025-04-30'

# ── 1. Load price data and build date index ───────────────────────────────────
data = pd.read_csv('artifacts/data.csv')
data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
data = data[(data['date'] >= SIM_START) & (data['date'] <= SIM_END)]
pivot = data.pivot_table(index='date', columns='tic', values='close').sort_index()
trade_dates = pivot.index.tolist()
print(f"Simulation period : {trade_dates[0]} -> {trade_dates[-1]}  ({len(trade_dates)} trading days)")
print(f"Tickers           : {pivot.shape[1]}")

# ── 2. Buy & Hold (equal weight, no rebalancing) ─────────────────────────────
bnh_daily = pivot.pct_change().fillna(0).mean(axis=1).values
ew_daily  = bnh_daily.copy()   # equal-weight rebalanced daily == same mean

# ── 3. DDPG-ALM returns (align to simulation dates) ──────────────────────────
ddpg_df = pd.read_csv('artifacts/df_daily_return.csv')
ddpg_df.columns = [c.strip() for c in ddpg_df.columns]
if 'date' not in ddpg_df.columns:
    ddpg_df = ddpg_df.rename(columns={ddpg_df.columns[0]: 'date',
                                       ddpg_df.columns[1]: 'daily_return'})
ddpg_df['date']         = pd.to_datetime(ddpg_df['date']).dt.strftime('%Y-%m-%d')
ddpg_df['daily_return'] = ddpg_df['daily_return'].astype(float)
# Align: pad with 0 for dates outside DDPG's trading period
ddpg_series = ddpg_df.set_index('date')['daily_return']
ddpg_daily  = ddpg_series.reindex(trade_dates).fillna(0).values
print(f"DDPG period       : {ddpg_df.date.min()} -> {ddpg_df.date.max()}  "
      f"(padded with 0 outside this range)")

# ── 4. Nifty 50 (^NSEI) ──────────────────────────────────────────────────────
headers  = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
start_ts = int(dt.strptime(SIM_START, '%Y-%m-%d').timestamp())
end_ts   = int(dt.strptime(SIM_END,   '%Y-%m-%d').timestamp())
try:
    r = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI',
                     params={'period1': start_ts, 'period2': end_ts, 'interval': '1d'},
                     headers=headers, timeout=15)
    res = r.json()['chart']['result']
    if res and res[0].get('timestamp'):
        ts  = res[0]['timestamp']
        adj = (res[0]['indicators'].get('adjclose', [{}])[0].get('adjclose')
               or res[0]['indicators']['quote'][0]['close'])
        nifty_dates = [dt.fromtimestamp(t).strftime('%Y-%m-%d') for t in ts]
        nifty_s     = pd.Series(adj, index=nifty_dates).pct_change().fillna(0)
        nifty_daily = nifty_s.reindex(trade_dates).fillna(0).values
        has_nifty   = True
        print(f"Nifty 50          : OK ({len(ts)} rows)")
    else:
        has_nifty = False
except Exception as e:
    has_nifty = False
    print(f"Nifty 50          : Failed — {e}")

# ── 5. Cumulative returns & drawdown helpers ──────────────────────────────────
def cum_ret(d):
    return (1 + np.array(d)).cumprod() - 1

def drawdown(d):
    v  = (1 + np.array(d)).cumprod()
    mx = np.maximum.accumulate(v)
    return (v - mx) / mx

def metrics(d, label):
    d       = np.array(d)
    ann_ret = (1 + d.mean()) ** 252 - 1
    ann_vol = d.std() * 252 ** 0.5
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd  = drawdown(d).min()
    return {'Strategy': label,
            'Ann. Return': f'{ann_ret*100:.2f}%',
            'Ann. Vol':    f'{ann_vol*100:.2f}%',
            'Sharpe':      f'{sharpe:.3f}',
            'Max DD':      f'{max_dd*100:.2f}%'}

# ── 6. Metrics table ──────────────────────────────────────────────────────────
rows = [
    metrics(ddpg_daily, 'DDPG-ALM'),
    metrics(bnh_daily,  'Buy & Hold'),
    metrics(ew_daily,   'Equal Weight'),
]
if has_nifty:
    rows.append(metrics(nifty_daily, 'Nifty 50'))

print()
print('=' * 62)
print('PERFORMANCE METRICS  (2021-2025)')
print('=' * 62)
print(pd.DataFrame(rows).to_string(index=False))
print('=' * 62)

# ── 7. Plot ───────────────────────────────────────────────────────────────────
dates = pd.to_datetime(trade_dates)

fig, axes = plt.subplots(3, 1, figsize=(14, 13))
fig.suptitle('DDPG-ALM vs Baselines  (2021–2025)', fontsize=14, fontweight='bold', y=1.01)

# — Cumulative Returns —
ax = axes[0]
ax.plot(dates, cum_ret(ddpg_daily) * 100, label='DDPG-ALM',     color='#3b82f6', lw=2.5)
ax.plot(dates, cum_ret(bnh_daily)  * 100, label='Buy & Hold',   color='#10b981', lw=1.8, ls='--')
ax.plot(dates, cum_ret(ew_daily)   * 100, label='Equal Weight', color='#f59e0b', lw=1.8, ls=':')
if has_nifty:
    ax.plot(dates, cum_ret(nifty_daily) * 100, label='Nifty 50', color='#ef4444', lw=1.8, ls='-.')
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Cumulative Returns')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# — Drawdown —
ax2 = axes[1]
for d, label, color, ls in [
    (ddpg_daily, 'DDPG-ALM',     '#3b82f6', '-'),
    (bnh_daily,  'Buy & Hold',   '#10b981', '--'),
    (ew_daily,   'Equal Weight', '#f59e0b', ':'),
] + ([(nifty_daily, 'Nifty 50', '#ef4444', '-.')] if has_nifty else []):
    dd = drawdown(d) * 100
    ax2.fill_between(dates, dd, 0, alpha=0.15, color=color)
    ax2.plot(dates, dd, label=label, color=color, lw=1.5, ls=ls)
ax2.set_ylabel('Drawdown (%)')
ax2.set_title('Drawdown')
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

# — Metrics bar chart —
ax3 = axes[2]
met_df = pd.DataFrame(rows)
x      = np.arange(len(met_df))
width  = 0.2
colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

ann_rets = [float(r.replace('%','')) for r in met_df['Ann. Return']]
sharpes  = [float(r) for r in met_df['Sharpe']]
max_dds  = [float(r.replace('%','')) for r in met_df['Max DD']]

b1 = ax3.bar(x - width, ann_rets, width, label='Ann. Return (%)', color=[c+'bb' for c in colors[:len(met_df)]])
b2 = ax3.bar(x,         sharpes,  width, label='Sharpe Ratio',    color=[c+'77' for c in colors[:len(met_df)]])
b3 = ax3.bar(x + width, max_dds,  width, label='Max Drawdown (%)',color=[c+'44' for c in colors[:len(met_df)]])
ax3.set_xticks(x); ax3.set_xticklabels(met_df['Strategy'], fontsize=10)
ax3.axhline(0, color='grey', lw=0.8)
ax3.set_title('Key Metrics Comparison')
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('artifacts/simulation_graphs.png', dpi=150, bbox_inches='tight')
print('\nSaved -> artifacts/simulation_graphs.png')
