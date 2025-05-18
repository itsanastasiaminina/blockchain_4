import numpy as np
import pandas as pd

def compute_volatility(df: pd.DataFrame, idx: int, window: int = 24) -> float:
    rets = df["price"].pct_change().dropna()
    start = max(0, idx - window)
    return rets.iloc[start:idx].std() if idx else 0.0


def compute_trade_pnl(results: pd.DataFrame) -> float:
    if results.empty:
        return 0.0
    return (results["value"] + results["fees"] - results["C_i"]).sum()


def compute_cagr(results: pd.DataFrame, init_cap: float) -> float:
    if results.empty:
        return 0.0
    start = results["start_time"].min()
    end = results["end_time"].max()
    days = max((end - start).days, 1)
    pnl = compute_trade_pnl(results)
    equity_final = init_cap + pnl
    if equity_final <= 0:
        return float("-inf")
    return (equity_final / init_cap) ** (365 / days) - 1


def compute_sharpe(equity: pd.Series, periods_per_year: int = 365 * 24) -> float:
    rets = equity.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return 0.0
    return rets.mean() / rets.std() * np.sqrt(periods_per_year)


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    drawdown = (cummax - equity) / cummax
    drawdown[cummax <= 0] = 0  
    return float(drawdown.clip(lower=0, upper=1).max())


def compute_capital_efficiency(results: pd.DataFrame) -> float:
    if results.empty:
        return 0.0
    return results["fees"].sum() / results["C_i"].mean()
