import pandas as pd
import os

ARTIFACTS = "artifacts"
BENCHMARK_TICKER = "^NSEI"


def load_ddpg_returns():
    path = os.path.join(ARTIFACTS, "df_daily_return.csv")
    df = pd.read_csv(path)
    df["daily_return"] = df["daily_return"].astype(float)
    return df  # columns: date, daily_return


def compute_cumulative_and_drawdown(daily_returns: pd.Series):
    cum = (1 + daily_returns).cumprod() - 1
    portfolio_val = (1 + daily_returns).cumprod()
    drawdown = (portfolio_val - portfolio_val.cummax()) / portfolio_val.cummax()
    return cum.tolist(), drawdown.tolist()


def compute_buy_and_hold(ddpg_dates: list):
    data = pd.read_csv(os.path.join(ARTIFACTS, "data.csv"))
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    pivot = data.pivot_table(index="date", columns="tic", values="close")
    pivot = pivot[pivot.index.isin(ddpg_dates)].sort_index()
    daily_returns = pivot.pct_change().fillna(0).mean(axis=1)
    daily_returns = daily_returns.reindex(ddpg_dates).fillna(0)
    return daily_returns


def compute_benchmark(start_date: str, end_date: str, ddpg_dates: list):
    try:
        import yfinance as yf
        raw = yf.download(BENCHMARK_TICKER, start=start_date, end=end_date, progress=False)
        if raw.empty:
            return None
        closes = raw["Close"].squeeze()
        closes.index = closes.index.strftime("%Y-%m-%d")
        daily_ret = closes.pct_change().fillna(0)
        daily_ret = daily_ret.reindex(ddpg_dates).fillna(0)
        return daily_ret
    except Exception:
        return None


def compute_metrics(daily_returns: pd.Series, label: str) -> dict:
    mean_d = daily_returns.mean()
    std_d = daily_returns.std()
    ann_return = (1 + mean_d) ** 252 - 1
    ann_vol = std_d * (252 ** 0.5)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cum_val = (1 + daily_returns).cumprod()
    drawdown = (cum_val - cum_val.cummax()) / cum_val.cummax()
    max_dd = drawdown.min()
    return {
        "strategy": label,
        "annualized_return": round(ann_return, 4),
        "annualized_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
    }


def build_comparison_response() -> dict:
    ddpg_df = load_ddpg_returns()
    dates = ddpg_df["date"].tolist()
    ddpg_ret = ddpg_df["daily_return"]

    bnh_ret = compute_buy_and_hold(dates)
    ew_ret = bnh_ret.copy()  # equal-weight daily rebalanced == same cross-sectional mean

    start, end = dates[0], dates[-1]
    bench_ret = compute_benchmark(start, end, dates)

    ddpg_cum, ddpg_dd = compute_cumulative_and_drawdown(ddpg_ret)
    bnh_cum, bnh_dd = compute_cumulative_and_drawdown(bnh_ret)
    ew_cum, ew_dd = compute_cumulative_and_drawdown(ew_ret)

    metrics = [
        compute_metrics(ddpg_ret, "DDPG-ALM"),
        compute_metrics(bnh_ret, "Buy & Hold"),
        compute_metrics(ew_ret, "Equal Weight"),
    ]

    result = {
        "dates": dates,
        "cumulative_returns": {
            "ddpg": ddpg_cum,
            "buy_and_hold": bnh_cum,
            "equal_weight": ew_cum,
        },
        "drawdowns": {
            "ddpg": ddpg_dd,
            "buy_and_hold": bnh_dd,
            "equal_weight": ew_dd,
        },
        "metrics": metrics,
    }

    if bench_ret is not None:
        b_cum, b_dd = compute_cumulative_and_drawdown(bench_ret)
        result["cumulative_returns"]["benchmark"] = b_cum
        result["drawdowns"]["benchmark"] = b_dd
        result["metrics"].append(compute_metrics(bench_ret, "Nifty 50"))
    else:
        result["cumulative_returns"]["benchmark"] = None
        result["drawdowns"]["benchmark"] = None

    return result
