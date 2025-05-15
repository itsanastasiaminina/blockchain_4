import math

FEE_RATE = 0.003

def compute_amounts(price: float, L: float, U: float, C: float) -> tuple[float, float]:
    A = math.sqrt(L)
    B = math.sqrt(U)
    S = C / (B - A)
    y = S * (B - math.sqrt(price))
    x = S * (math.sqrt(price) - A) / price
    return x, y

def calc_fees(volume: float) -> float:
    return FEE_RATE * volume