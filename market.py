import time
import requests
from datetime import datetime
import pandas as pd
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3 import UniswapV3EthereumPoolHourDataLoader
from fractal.loaders.binance import BinanceHourPriceLoader

def fetch_onchain_data(
    pool_address: str,
    ticker: str,
    start: datetime,
    end: datetime,
    max_retries: int = 5,
    initial_backoff: float = 1.0
) -> pd.DataFrame:

    api_key = '47d58f4bc7d6f4270ff0029147ff03d8'

    pool_loader = UniswapV3EthereumPoolHourDataLoader(
        api_key=api_key,
        pool=pool_address,
        loader_type=LoaderType.CSV
    )
    df_pool = pool_loader.read(with_run=True)

    price_loader = BinanceHourPriceLoader(
        ticker=ticker,
        loader_type=LoaderType.CSV,
        start_time=start,
        end_time=end
    )

    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            df_price = price_loader.read(with_run=True)
            break
        except requests.exceptions.ReadTimeout as e:
            if attempt == max_retries:
                raise
            else:
                time.sleep(backoff)
                backoff *= 2  
    else:
        raise RuntimeError("Failed to fetch")

    df = df_price.join(df_pool[['volume']], how='inner')
    df = df.rename(columns={'close': 'price'})
    df = df.reset_index().rename(columns={'index': 'timestamp'})
    return df[['timestamp', 'price', 'volume']]