from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from market import fetch_onchain_data
from strategy import TauResetMultiRangeStrategy
from metrics import (
    compute_pnl, compute_cagr, compute_sharpe,
    compute_max_drawdown, compute_capital_efficiency
)

REGIME_MA_WINDOW = 30  
OPTUNA_TRIALS = 5
INITIAL_CAPITAL = 100_000


def optimize_for_regime(df_reg, cap):
    df_reg = df_reg.reset_index(drop=True)
    def objective(trial):
        tau0_1 = trial.suggest_int('tau0_1', 6*3600, 48*3600, step=6*3600)
        tau0_2 = trial.suggest_int('tau0_2', 6*3600, 48*3600, step=6*3600)
        w1 = trial.suggest_float('w1', 0.005, 0.05, step=0.005)
        w2 = trial.suggest_float('w2', 0.005, 0.05, step=0.005)
        alpha = trial.suggest_float('alpha', 0.0, 2.0, step=0.1)
        batch_n = trial.suggest_int('batch_n', 1, 10)
        lambda_gas = trial.suggest_float('lambda_gas', 100, 1000, step=100)
        range_params = [
            {'tau0': tau0_1, 'w': w1},
            {'tau0': tau0_2, 'w': w2}
        ]
        strat = TauResetMultiRangeStrategy(df_reg, cap, range_params, alpha, batch_n)
        res = strat.run()
        pnl = compute_pnl(res, cap)
        resets = len(res) / len(range_params)
        return pnl - lambda_gas * resets

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    return study.best_params


if __name__ == '__main__':
    end = datetime(2025, 5, 15)
    start = end - timedelta(days=4)
    pool_addr = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    ticker = 'ETHUSDT'
    df = fetch_onchain_data(pool_addr, ticker, start, end)
    print(df)

    df['ma30'] = df['price'].rolling(window=REGIME_MA_WINDOW).mean()
    df = df.dropna(subset=['ma30'])
    df.reset_index(drop=True, inplace=True)
    df['regime'] = np.where(df['price'] > df['ma30'], 'bull', 'bear')

    best_params = {}
    for regime, df_reg in df.groupby('regime'):
        print(f"Optuna: {regime}...")
        best = optimize_for_regime(df_reg, INITIAL_CAPITAL)
        best_params[regime] = best
        print(f"Best params {regime}: {best}\n")

    df['segment_id'] = (df['regime'] != df['regime'].shift()).cumsum()
    cap = INITIAL_CAPITAL
    all_results = []

    for seg_id, df_seg in df.groupby('segment_id'):
        regime = df_seg['regime'].iloc[0]
        params = best_params[regime]
        range_params = [
            {'tau0': params['tau0_1'], 'w': params['w1']},
            {'tau0': params['tau0_2'], 'w': params['w2']}
        ]
        strat = TauResetMultiRangeStrategy(df_seg, cap, range_params, params['alpha'], params['batch_n'])
        res = strat.run()
        res = res.assign(regime=regime)
        res['pnl_i'] = res['value'] + res['fees'] - res['C_i']
        cap = cap + res['pnl_i'].sum()
        all_results.append(res)

    df_results = pd.concat(all_results).reset_index(drop=True)
    df_equity = df_results[['end_time', 'pnl_i']].sort_values('end_time')
    df_equity['equity'] = INITIAL_CAPITAL + df_equity['pnl_i'].cumsum()

    total_pnl = df_results['pnl_i'].sum()
    total_cagr = compute_cagr(df_results, INITIAL_CAPITAL)
    total_sharpe = compute_sharpe(df_equity['equity'])
    total_max_dd = compute_max_drawdown(df_equity['equity'])
    total_ce = compute_capital_efficiency(df_results)

    print(f"Итоговый PnL: {total_pnl:.2f}")
    print(f"Итоговый CAGR: {total_cagr:.2%}")
    print(f"Sharpe Ratio: {total_sharpe:.2f}")
    print(f"Max Drawdown: {total_max_dd:.2%}")
    print(f"Capital Efficiency: {total_ce:.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(df_equity['end_time'], df_equity['equity'], label='Equity')
    plt.title('Equity Curve (continuous)')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_curve_continuous.png')
    plt.close()

    df_price = df.set_index('timestamp')['price']
    df_results['price_start'] = df_results['start_time'].map(df_price)
    df_results['price_lower'] = df_results['price_start'] * (1 - df_results['w'] / 2)
    df_results['price_upper'] = df_results['price_start'] * (1 + df_results['w'] / 2)

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['price'], label='Market Price', linewidth=1)
    for _, row in df_results.iterrows():
        plt.fill_betweenx(
            [row['price_lower'], row['price_upper']],
            row['start_time'], row['end_time'],
            color='orange', alpha=0.1
        )
        plt.scatter(row['start_time'], row['price_start'], marker='^', c='green')
        plt.scatter(row['end_time'], df_price[row['end_time']], marker='v', c='red')

        plt.title('Price & Range Overlay')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(['Price', 'Ranges', 'Entry', 'Exit'])
        plt.tight_layout()
        plt.savefig('price_range_overlay.png')
        plt.close()

    cum_fees = df_results['fees'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(df_results['end_time'], cum_fees, color='purple')
    plt.title('Cumulative Fees Earned')
    plt.xlabel('Time')
    plt.ylabel('Fees')
    plt.tight_layout()
    plt.savefig('cumulative_fees.png')
    plt.close()

    range_width = df_results['price_upper'] - df_results['price_lower']
    plt.figure(figsize=(10, 5))
    plt.plot(df_results['end_time'], range_width, color='brown')
    plt.title('Range Width Over Time')
    plt.xlabel('Time')
    plt.ylabel('Width')
    plt.tight_layout()
    plt.savefig('range_width_over_time.png')
    plt.close()


    plt.figure(figsize=(10, 5))
    plt.plot(df_equity['end_time'], df_equity['equity'], label='Equity')
    plt.title('Equity Curve (continuous)')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_curve_continuous.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df_equity['end_time'], df_equity['equity'], label='Equity (all)')
    plt.title('Equity Curve (Continuous)')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_curve_continuous.png')
    plt.close()

    for regime in df_results['regime'].unique():
        df_seg = df_results[df_results['regime'] == regime]
        df_seg_eq = df_seg[['end_time', 'pnl_i']].sort_values('end_time')
        df_seg_eq['equity'] = (INITIAL_CAPITAL +
                              df_results[df_results['regime'] < regime]['pnl_i'].sum() +
                              df_seg_eq['pnl_i'].cumsum())
        plt.figure(figsize=(10, 5))
        plt.plot(df_seg_eq['end_time'], df_seg_eq['equity'], label=f'Equity ({regime})')
        plt.title(f'Equity Curve - {regime.capitalize()} Market')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'equity_curve_{regime}.png')
        plt.close()

    plt.figure(figsize=(8, 4))
    pd.DataFrame({r: df_results[df_results['regime'] == r]['pnl_i'].cumsum()
                   for r in df_results['regime'].unique()}).plot()
    plt.title('Cumulative PnL by Regime')
    plt.xlabel('Trade Index')
    plt.ylabel('Cumulative PnL')
    plt.tight_layout()
    plt.savefig('cumulative_pnl_by_regime.png')
    plt.close()
