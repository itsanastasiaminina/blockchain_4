from datetime import datetime, timedelta
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from market import fetch_onchain_data
from strategy import TauResetMultiRangeStrategy
from metrics import (
    compute_trade_pnl,
    compute_cagr,
    compute_sharpe,
    compute_max_drawdown,
    compute_capital_efficiency,
)

REGIME_MA_WINDOW = 30
OPTUNA_TRIALS = 30
INITIAL_CAPITAL = 100_000

_tmpdb = tempfile.NamedTemporaryFile(delete=False).name
storage = optuna.storages.RDBStorage(f"sqlite:///{_tmpdb}")


def _gas_spent(df_res: pd.DataFrame) -> float:
    gas_rows = df_res[(df_res["C_i"] == 0) & (df_res["value"] == 0) & (df_res["fees"] < 0)]
    return -gas_rows["fees"].sum()


def optimize_for_regime(df_reg: pd.DataFrame, cap: float):
    df_reg = df_reg.reset_index(drop=True)

    def objective(trial: optuna.trial.Trial):
        tau0 = trial.suggest_int("tau0", 2 * 86400, 7 * 86400, step=86400)
        w = trial.suggest_float("w", 0.05, 0.12, step=0.005)
        alpha = trial.suggest_float("alpha", 0.0, 0.5, step=0.05)
        batch_n = trial.suggest_int("batch_n", 3, 8)

        range_params = [{"tau0": tau0, "w": w}]
        strat = TauResetMultiRangeStrategy(df_reg, cap, range_params, alpha, batch_n)
        res = strat.run()
        pnl = compute_trade_pnl(res)
        gas_usd = _gas_spent(res)
        score = pnl / max(gas_usd, 1)  
        return score

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=False,
    )

    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=os.cpu_count())

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        complete = study.trials
    best = max(complete, key=lambda t: t.value)
    return best.params


def main():
    end = datetime(2025, 5, 15)
    start = end - timedelta(days=10)
    pool_addr = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    ticker = "ETHUSDT"

    df = fetch_onchain_data(pool_addr, ticker, start, end)

    df["ma30"] = df["price"].rolling(window=REGIME_MA_WINDOW).mean()
    df = df.dropna(subset=["ma30"]).reset_index(drop=True)
    df["regime"] = np.where(df["price"] > df["ma30"], "bull", "bear")

    best_params = {r: optimize_for_regime(g, INITIAL_CAPITAL) for r, g in df.groupby("regime")}

    df["segment_id"] = (df["regime"] != df["regime"].shift()).cumsum()
    cap = INITIAL_CAPITAL
    all_results = []

    for _, df_seg in df.groupby("segment_id"):
        regime = df_seg["regime"].iloc[0]
        p = best_params[regime]
        rp = [{"tau0": p["tau0"], "w": p["w"]}]
        strat = TauResetMultiRangeStrategy(df_seg, cap, rp, p["alpha"], p["batch_n"])
        res = strat.run()
        res = res.assign(regime=regime)
        res["pnl_i"] = res["value"] + res["fees"] - res["C_i"]
        cap += res["pnl_i"].sum()
        all_results.append(res)

    df_results = pd.concat(all_results, ignore_index=True)
    df_equity = df_results[["end_time", "pnl_i"]].sort_values("end_time")
    df_equity["equity"] = INITIAL_CAPITAL + df_equity["pnl_i"].cumsum()

    print("\n--- METRICS (10d) ---")
    print(f"PnL: {compute_trade_pnl(df_results):.2f}")
    print(f"CAGR: {compute_cagr(df_results, INITIAL_CAPITAL):.2%}")
    print(f"Sharpe: {compute_sharpe(df_equity['equity']):.2f}")
    print(f"Max DD: {compute_max_drawdown(df_equity['equity']):.2%}")
    print(f"Capital efficiency: {compute_capital_efficiency(df_results):.2%}")

    plt.figure(figsize=(10, 4))
    plt.plot(df_equity["end_time"], df_equity["equity"], label="Equity")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig("equity_curve_continuous.png"); plt.close()

    df_price = df.set_index("timestamp")["price"]
    df_results["price_start"] = df_results["start_time"].map(df_price)
    df_results["price_lower"] = df_results["price_start"] * (1 - df_results["w"] / 2)
    df_results["price_upper"] = df_results["price_start"] * (1 + df_results["w"] / 2)

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["price"], label="Price", linewidth=1)
    for _, r in df_results.iterrows():
        plt.fill_betweenx([r["price_lower"], r["price_upper"]], r["start_time"], r["end_time"], color="orange", alpha=0.1)
        plt.scatter(r["start_time"], r["price_start"], marker="^", c="green")
        plt.scatter(r["end_time"], df_price[r["end_time"]], marker="v", c="red")
    plt.legend(); plt.tight_layout(); plt.savefig("price_range_overlay.png"); plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df_results["end_time"], df_results["fees"].cumsum(), color="purple")
    plt.grid(); plt.tight_layout(); plt.savefig("cumulative_fees.png"); plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df_results["end_time"], df_results["price_upper"] - df_results["price_lower"], color="brown")
    plt.grid(); plt.tight_layout(); plt.savefig("range_width_over_time.png"); plt.close()

    for reg in df_results["regime"].unique():
        seg = df_results[df_results["regime"] == reg]
        seg_eq = seg[["end_time", "pnl_i"]].sort_values("end_time")
        seg_eq["equity"] = INITIAL_CAPITAL + seg_eq["pnl_i"].cumsum()
        plt.figure(figsize=(10, 4))
        plt.plot(seg_eq["end_time"], seg_eq["equity"], label=f"Equity ({reg})")
        plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"equity_curve_{reg}.png"); plt.close()

    pnl_by_reg = {r: df_results[df_results["regime"] == r]["pnl_i"].cumsum() for r in df_results["regime"].unique()}
    pd.DataFrame(pnl_by_reg).plot(figsize=(8, 4)); plt.grid(); plt.tight_layout(); plt.savefig("cumulative_pnl_by_regime.png"); plt.close()


if __name__ == "__main__":
    main()
