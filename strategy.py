import math
import numpy as np
import pandas as pd
from uniswap_v3 import compute_amounts, calc_fees
from metrics import compute_volatility
from gas_oracle import GasOracle

GAS_LIMIT_RESET = 180000
SIGMA_THRESHOLD = 0.04  
GAS_EFF_RATIO = 2     

class TauResetMultiRangeStrategy:
    def __init__(self, data, capital, range_params, alpha, batch_n):
        self.df = data.copy().reset_index(drop=True)
        self.capital = float(capital)
        self.range_params = range_params
        self.alpha = alpha
        self.batch_n = batch_n
        self.gas_oracle = GasOracle()
        self.positions = []

    def _price_outside_any(self, price):
        return any(price < p["L"] or price > p["U"] for p in self.positions)

    def _open_positions(self, price, idx):
        if compute_volatility(self.df, idx) > SIGMA_THRESHOLD:
            return False  
        self.positions.clear()
        inv_w = [1 / p["w"] for p in self.range_params]
        norm = sum(inv_w)
        for inv, p in zip(inv_w, self.range_params):
            c_i = self.capital * inv / norm
            l = price * (1 - p["w"] / 2)
            u = price * (1 + p["w"] / 2)
            x, y = compute_amounts(price, l, u, c_i)
            s = c_i / (math.sqrt(u) - math.sqrt(l))
            self.positions.append({
                "tau0": p["tau0"],
                "w": p["w"],
                "C_i": c_i,
                "L": l,
                "U": u,
                "x": x,
                "y": y,
                "S": s,
                "fees": 0.0,
                "open_idx": idx,
            })
        return True

    def _close_position(self, pos, p1, t_start, t_end):
        s = pos["S"]
        pa, pb = pos["L"], pos["U"]
        sp = math.sqrt(p1)
        sl, su = math.sqrt(pa), math.sqrt(pb)
        if p1 <= pa:
            amount0 = s * (su - sl) / (sl * su)
            amount1 = 0.0
        elif p1 >= pb:
            amount0 = 0.0
            amount1 = s * (su - sl)
        else:
            amount0 = s * (su - sp) / (sp * su)
            amount1 = s * (sp - sl)
        value = amount0 * p1 + amount1
        return {
            "start_time": t_start,
            "end_time": t_end,
            "tau0": pos["tau0"],
            "w": pos["w"],
            "C_i": pos["C_i"],
            "fees": pos["fees"],
            "value": value,
        }

    def run(self):
        df = self.df
        results = []
        pending = 0
        last_reset_time = None
        accrued_fees = 0.0
        for idx, row in df.iterrows():
            ts = row["timestamp"]
            price = row["price"]
            vol = row["volume"]
            tvl = row.get("tvl", np.nan)
            if not np.isfinite(tvl) or tvl <= 0:
                tvl = self.capital * 10
            sigma = compute_volatility(df, idx)
            tau_t = min(p["tau0"] * (1 + self.alpha * sigma) for p in self.range_params)
            if self.positions and self._price_outside_any(price):
                pending += 1
            elapsed = float("inf") if last_reset_time is None else (ts - last_reset_time).total_seconds()
            gas_price = self.gas_oracle.get_gas_price()
            gas_cost_usd = GAS_LIMIT_RESET * gas_price / 1e18 * price
            gas_ok = gas_price < self.gas_oracle.target_price
            need_reset = (
                last_reset_time is None
                or elapsed >= tau_t
                or (pending >= self.batch_n and gas_ok and self._price_outside_any(price) and accrued_fees >= GAS_EFF_RATIO * gas_cost_usd)
            )
            if need_reset:
                if self.positions:
                    for p in self.positions:
                        res = self._close_position(p, price, df.loc[p["open_idx"], "timestamp"], ts)
                        accrued_fees += res["fees"]
                        results.append(res)
                if gas_ok:
                    self.capital -= gas_cost_usd
                    results.append({
                        "start_time": ts,
                        "end_time": ts,
                        "tau0": 0,
                        "w": 0.0,
                        "C_i": 0.0,
                        "fees": -gas_cost_usd,
                        "value": 0.0,
                    })
                opened = self._open_positions(price, idx)
                if opened:
                    last_reset_time = ts
                    pending = 0
                    accrued_fees = 0.0
            if self.positions:
                for p in self.positions:
                    if p["L"] <= price <= p["U"]:
                        vol_range = vol * p["w"]
                        fee_add = calc_fees(vol_range) * (p["C_i"] / tvl)
                        p["fees"] += fee_add
                        accrued_fees += fee_add
        if self.positions:
            p1 = df.iloc[-1]["price"]
            t_end = df.iloc[-1]["timestamp"]
            for p in self.positions:
                results.append(self._close_position(p, p1, df.loc[p["open_idx"], "timestamp"], t_end))
        return pd.DataFrame(results)
