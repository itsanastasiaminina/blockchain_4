import requests

class GasOracle:
    def __init__(self, target_price_gwei: float = 50):
        self.api_url = "https://api.etherscan.io/api"
        self.target_price = target_price_gwei * 1e9  

    def get_gas_price(self) -> float:
        try:
            resp = requests.get(self.api_url, params={
                'module': 'gastracker',
                'action': 'gasoracle',
                'apikey': '4C84575H1X1DSI5RBRJSEDYNETJ54UB5MR'
            })
            resp.raise_for_status()
            json_data = resp.json()
            res = json_data.get("res", {}) if isinstance(json_data, dict) else {}
            if isinstance(res, dict):
                price_gwei = float(res.get("ProposeGasPrice", self.target_price / 1e9))
                return price_gwei * 1e9
        except Exception:
            pass
        return self.target_price
