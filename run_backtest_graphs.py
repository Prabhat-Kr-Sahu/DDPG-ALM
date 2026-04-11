"""
Full backtest + simulation graphs: DDPG-ALM vs Baselines (2021-2025)
Loads saved agent from artifacts/ddpg_agent and runs it on the backtest period.
No re-training required.
"""
# ── Imports ───────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings('ignore')
import os, sys, json
import numpy as np
import pandas as pd
import torch
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime as dt

sys.path.insert(0, os.path.dirname(__file__))

# ── Load all class/function definitions from notebook ────────────────────────
exec(open('nb_classes.py', encoding='utf-8').read())

print("Classes loaded.")

# ── 1. Load & process data ────────────────────────────────────────────────────
INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
              'close_30_sma', 'close_60_sma']

print("Loading data.csv ...")
df_raw = pd.read_csv('artifacts/data.csv')
df_raw = df_raw[df_raw['date'] >= '2021-01-01'].reset_index(drop=True)
print(f"Raw rows: {df_raw.shape[0]}  ({df_raw.date.min()} -> {df_raw.date.max()})")

print("Adding technical indicators ...")
df = add_tech(df_raw, INDICATORS)
df = df.ffill().bfill()

print("Computing covariance matrix (252-day lookback) ...")
df = df.sort_values(['date', 'tic'], ignore_index=True)
df.index = df.date.factorize()[0]

cov_list, return_list = [], []
lookback = 252
for i in range(lookback, len(df.index.unique())):
    data_lb  = df.loc[i - lookback:i, :]
    price_lb = data_lb.pivot_table(index='date', columns='tic', values='close')
    ret_lb   = price_lb.pct_change().dropna()
    cov_list.append(ret_lb.cov().values)
    return_list.append(ret_lb)

df_cov = pd.DataFrame({'date': df.date.unique()[lookback:],
                       'cov_list': cov_list,
                       'return_list': return_list})
df_final = df.merge(df_cov, on='date')
df_final = df_final.sort_values(['date', 'tic']).reset_index(drop=True)

# Historical volatility — must match notebook exactly:
# df has 30 rows per date; std() on a DataFrame gives Series(30,) per row
hist_vol_list = []
for i in range(len(df_final['return_list'])):
    returns = df_final['return_list'].values[i].std()   # Series of shape (30,)
    hist_vol_list.append(returns)
hist_vol = pd.DataFrame(np.array(hist_vol_list), df_final['date'])

print(f"Processed rows: {df_final.shape[0]}  first usable date: {df_final.date.min()}")

# ── 2. Backtest split: 2023-01-01 → 2025-04-30 ───────────────────────────────
BT_START = '2023-01-01'
BT_END   = '2025-04-30'

backtest_df  = data_split(df_final, BT_START, BT_END)
hist_vol_bt  = hist_vol[BT_START:BT_END]

stock_dim   = len(backtest_df.tic.unique())
state_space = stock_dim
TURBULENCE_THRESHOLD = 0.0020

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dim,
    "reward_scaling": 1e-4,
    "hist_vol": hist_vol_bt,
    "turbulence_threshold": TURBULENCE_THRESHOLD,
}

print(f"\nBacktest period: {BT_START} -> {BT_END}  ({backtest_df.shape[0]} rows, {stock_dim} stocks)")

e_bt_gym = StockPortfolioEnv(df=backtest_df, **env_kwargs)

# ── 3. Load saved DDPG agent ──────────────────────────────────────────────────
print("Loading saved agent from artifacts/ddpg_agent ...")
params = json.load(open('artifacts/params.json'))
params['Ahidden_dim'] = int(params['Ahidden_dim'])
params['Anum_layers']  = int(params['Anum_layers'])
params['Chidden_dim']  = int(params['Chidden_dim'])
params['Cnum_layers']  = int(params['Cnum_layers'])
params['batch_size']   = int(params['batch_size'])
params['buffer_size']  = int(params['buffer_size'])
params['warmup_steps'] = int(params['warmup_steps'])

# Create env just to init agent (env_full_train shape must match backtest)
agent = DDPGagent(e_bt_gym.get_sb_env()[0], params)

checkpoint = torch.load('artifacts/ddpg_agent', map_location='cpu', weights_only=False)
agent.actor.load_state_dict(checkpoint['actor_state_dict'])
agent.critic.load_state_dict(checkpoint['critic_state_dict'])
agent.cost_network.load_state_dict(checkpoint['cost_network_state_dict'])
agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
agent.cost_target.load_state_dict(checkpoint['cost_target_state_dict'])
agent.lambda_ = checkpoint['lambda_']
agent.rho     = checkpoint['rho']
print("Agent loaded.")

# ── 4. Run backtest ───────────────────────────────────────────────────────────
print("Running backtest ...")
account_memory, actions_memory, total_reward = agent.trade(
    e_bt_gym.get_sb_env()[0], e_bt_gym
)
ddpg_df = account_memory[0].copy()
ddpg_df['date'] = pd.to_datetime(ddpg_df['date']).dt.strftime('%Y-%m-%d')
print(f"Backtest complete. Days: {len(ddpg_df)}  Total reward: {float(np.array(total_reward).mean()):.4f}")

# Save updated returns
ddpg_df.to_csv('artifacts/df_daily_return.csv', index=False)
print("Saved updated returns -> artifacts/df_daily_return.csv")

# ── 5. Baselines ──────────────────────────────────────────────────────────────
trade_dates = ddpg_df['date'].tolist()

# Buy & Hold / Equal Weight
data_raw = pd.read_csv('artifacts/data.csv')
data_raw['date'] = pd.to_datetime(data_raw['date']).dt.strftime('%Y-%m-%d')
pivot = data_raw.pivot_table(index='date', columns='tic', values='close')
pivot = pivot[pivot.index.isin(trade_dates)].sort_index()
bnh_daily = pivot.pct_change().fillna(0).mean(axis=1).reindex(trade_dates).fillna(0).values
ew_daily  = bnh_daily.copy()

# Nifty 50
start_ts = int(dt.strptime(trade_dates[0],  '%Y-%m-%d').timestamp())
end_ts   = int(dt.strptime(trade_dates[-1], '%Y-%m-%d').timestamp())
headers  = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
has_nifty = False
try:
    r = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI',
                     params={'period1': start_ts, 'period2': end_ts, 'interval': '1d'},
                     headers=headers, timeout=15)
    res = r.json()['chart']['result']
    if res and res[0].get('timestamp'):
        ts  = res[0]['timestamp']
        adj = (res[0]['indicators'].get('adjclose', [{}])[0].get('adjclose')
               or res[0]['indicators']['quote'][0]['close'])
        nifty_s   = pd.Series(adj, index=[dt.fromtimestamp(t).strftime('%Y-%m-%d') for t in ts])
        nifty_daily = nifty_s.pct_change().fillna(0).reindex(trade_dates).fillna(0).values
        has_nifty = True
        print("Nifty 50: OK")
except Exception as e:
    print(f"Nifty 50 failed: {e}")

# ── 6. Compute stats ──────────────────────────────────────────────────────────
def cum_ret(d):
    return (1 + np.array(d)).cumprod() - 1

def drawdown(d):
    v = (1 + np.array(d)).cumprod()
    return (v - np.maximum.accumulate(v)) / np.maximum.accumulate(v)

def metrics(d, label):
    d = np.array(d)
    ann_ret = (1 + d.mean()) ** 252 - 1
    ann_vol = d.std() * 252 ** 0.5
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    return {'Strategy': label,
            'Ann. Return %': round(ann_ret * 100, 2),
            'Ann. Vol %':    round(ann_vol * 100, 2),
            'Sharpe':        round(sharpe, 3),
            'Max DD %':      round(drawdown(d).min() * 100, 2)}

ddpg_daily = ddpg_df['daily_return'].values
rows = [
    metrics(ddpg_daily, 'DDPG-ALM'),
    metrics(bnh_daily,  'Buy & Hold'),
    metrics(ew_daily,   'Equal Weight'),
]
if has_nifty:
    rows.append(metrics(nifty_daily, 'Nifty 50'))

met_df = pd.DataFrame(rows)
print('\n' + '='*62)
print(f'PERFORMANCE METRICS  ({BT_START} to {BT_END})')
print('='*62)
print(met_df.to_string(index=False))
print('='*62)

# ── 7. All Graphs — saved as 5 separate images ────────────────────────────────
dates  = pd.to_datetime(trade_dates)
colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
labels = ['DDPG-ALM', 'Buy & Hold', 'Equal Weight', 'Nifty 50']
styles = ['-', '--', ':', '-.']
series = [ddpg_daily, bnh_daily, ew_daily] + ([nifty_daily] if has_nifty else [])
n      = len(series)

TITLE = f'DDPG-ALM vs Baselines  ({BT_START} to {BT_END})'

# ── Image 1: Cumulative Returns ───────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(14, 5))
for i, (d, lbl) in enumerate(zip(series, labels[:n])):
    lw = 2.5 if i == 0 else 1.8
    ax1.plot(dates, cum_ret(d) * 100, label=lbl, color=colors[i], lw=lw, ls=styles[i])
ax1.axhline(0, color='grey', lw=0.8, ls='--')
ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
ax1.set_title(f'Cumulative Returns — {TITLE}', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('artifacts/graph_1_cumulative_returns.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print('Saved -> artifacts/graph_1_cumulative_returns.png')

# ── Image 2: Drawdown ─────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 5))
for i, (d, lbl) in enumerate(zip(series, labels[:n])):
    dd = drawdown(d) * 100
    ax2.fill_between(dates, dd, 0, alpha=0.15, color=colors[i])
    ax2.plot(dates, dd, label=lbl, color=colors[i], lw=1.5 if i == 0 else 1.2, ls=styles[i])
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.set_title(f'Drawdown — {TITLE}', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig('artifacts/graph_2_drawdown.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print('Saved -> artifacts/graph_2_drawdown.png')

# ── Image 3: Daily Returns Distribution ──────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
for i, (d, lbl) in enumerate(zip(series, labels[:n])):
    ax3.hist(np.array(d)*100, bins=60, alpha=0.45, color=colors[i], label=lbl, density=True)
ax3.set_xlabel('Daily Return (%)', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title(f'Daily Returns Distribution — {TITLE}', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10); ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig('artifacts/graph_3_daily_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print('Saved -> artifacts/graph_3_daily_distribution.png')

# ── Image 4: Rolling Sharpe (63-day) ─────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(14, 5))
window = 63
for i, (d, lbl) in enumerate(zip(series, labels[:n])):
    s = pd.Series(d)
    roll_sharpe = (s.rolling(window).mean() / s.rolling(window).std()) * (252**0.5)
    ax4.plot(dates, roll_sharpe, label=lbl, color=colors[i], lw=1.5 if i == 0 else 1.2, ls=styles[i])
ax4.axhline(0, color='grey', lw=0.8, ls='--')
ax4.set_ylabel('Rolling Sharpe', fontsize=12)
ax4.set_title(f'Rolling Sharpe Ratio (63-day) — {TITLE}', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11); ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig('artifacts/graph_4_rolling_sharpe.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print('Saved -> artifacts/graph_4_rolling_sharpe.png')

# ── Image 5: Metrics Bar Chart + Portfolio Weights (side by side) ─────────────
fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
fig5.suptitle(TITLE, fontsize=13, fontweight='bold')

x     = np.arange(n)
width = 0.25
ax5.bar(x - width, met_df['Ann. Return %'], width, label='Ann. Return %',
        color=[colors[i] + 'bb' for i in range(n)])
ax5.bar(x,         met_df['Sharpe'],        width, label='Sharpe',
        color=[colors[i] + '77' for i in range(n)])
ax5.bar(x + width, met_df['Max DD %'],      width, label='Max DD %',
        color=[colors[i] + '44' for i in range(n)])
ax5.set_xticks(x)
ax5.set_xticklabels(met_df['Strategy'], fontsize=10)
ax5.axhline(0, color='grey', lw=0.8)
ax5.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3, axis='y')

act_df = actions_memory[0].copy()
date_col = act_df.columns[0]
act_cols = act_df.columns[1:].tolist()
act_df[date_col] = pd.to_datetime(act_df[date_col])
top5 = act_df[act_cols].mean().nlargest(5).index.tolist()
for col in top5:
    ax6.plot(act_df[date_col], act_df[col], label=col.replace('.NS', ''), lw=1.5)
ax6.set_ylabel('Portfolio Weight', fontsize=11)
ax6.set_title('DDPG-ALM: Top-5 Stock Weights', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9, ncol=2); ax6.grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig('artifacts/graph_5_metrics_and_weights.png', dpi=150, bbox_inches='tight')
plt.close(fig5)
print('Saved -> artifacts/graph_5_metrics_and_weights.png')

print('\nAll 5 graphs saved to artifacts/')
