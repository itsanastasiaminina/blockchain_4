import numpy as np

def compute_volatility(df, idx, window=24):
    rets = df['price'].pct_change().dropna()
    start = max(0, idx-window)
    return rets.iloc[start:idx].std()


def compute_pnl(results, init_cap):
    if results.empty:
        return 0.0
    return results['value'].sum() + results['fees'].sum() - init_cap


def compute_cagr(results, init_cap):
    if results.empty:
        return 0.0
    days = (results['end_time'].max() - results['start_time'].min()).days
    net = compute_pnl(results, init_cap)
    return (1 + net/init_cap)**(365/days) - 1


def compute_sharpe(equity):
    rets = equity.pct_change().dropna()
    return rets.mean()/rets.std() if not rets.empty else 0.0


def compute_max_drawdown(equity):
    if equity.empty:
        return 0.0
    cm = equity.cummax()
    return ((cm - equity)/cm).max()


def compute_capital_efficiency(results):
    if results.empty:
        return 0.0
    return results['fees'].sum()/results['C_i'].mean()