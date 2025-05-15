import numpy as np
import pandas as pd
from uniswap_v3 import compute_amounts, calc_fees
from metrics import compute_volatility
from gas_oracle import GasOracle

class TauResetMultiRangeStrategy:
    def __init__(
        self,
        data,
        capital,
        range_params,
        alpha,
        batch_n,
    ):
        self.df = data.copy().reset_index(drop=True)
        self.capital = capital
        self.range_params = range_params
        self.alpha = alpha
        self.batch_n = batch_n
        self.gas_oracle = GasOracle()
        self.positions = []

    def run(self):
        df = self.df
        results = []
        pending = 0
        last_reset_time = None
        N = len(self.range_params)
        inv_ws = [1/p['w'] for p in self.range_params]
        norm = sum(inv_ws)

        for idx, row in df.iterrows():
            current_time = row['timestamp']
            price = row['price']
            vol = row['volume']

            sigma = compute_volatility(df, idx)
            taus = [p['tau0'] * (1 + self.alpha * sigma) for p in self.range_params]
            tau_t = min(taus)

            if last_reset_time is None:
                elapsed = float('inf') 
            else:
                elapsed = (current_time - last_reset_time).total_seconds()

            if last_reset_time is None or (elapsed >= tau_t) or (pending >= self.batch_n and self.gas_oracle.get_gas_price() < self.gas_oracle.target_price):
                for pos in self.positions:
                    close_time = current_time
                    results.append(self._close(pos, pos['open_idx'], idx))
                self.positions.clear()

                for inv_w, params in zip(inv_ws, self.range_params):
                    C_i = self.capital * inv_w / norm
                    L = price * (1 - params['w']/2)
                    U = price * (1 + params['w']/2)
                    x, y = compute_amounts(price, L, U, C_i)
                    self.positions.append({
                        'tau0': params['tau0'],
                        'w': params['w'],
                        'C_i': C_i,
                        'L': L,
                        'U': U,
                        'x': x,
                        'y': y,
                        'fees': 0.0,
                        'open_idx': idx
                    })
                last_reset_time = current_time
                pending += 2 * N

            for pos in self.positions:
                if pos['L'] <= price <= pos['U']:
                    pos['fees'] += calc_fees(vol * (pos['C_i']/self.capital))

        final_idx = len(df) - 1
        for pos in self.positions:
            results.append(self._close(pos, pos['open_idx'], final_idx))

        return pd.DataFrame(results)

    def _close(self, pos, i0, i1):
        df = self.df
        p0 = df['price'].iat[i0]
        p1 = df['price'].iat[i1]
        R = p1 / p0
        IL = 2 * np.sqrt(R) / (1 + R) - 1
        return {
            'start_time': df['timestamp'].iat[i0],
            'end_time': df['timestamp'].iat[i1],
            'tau0': pos['tau0'],
            'w': pos['w'],
            'C_i': pos['C_i'],
            'fees': pos['fees'],
            'IL': IL,
            'value': pos['x'] * p1 + pos['y']
        }
