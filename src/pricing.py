from typing import Tuple


def american_to_implied_prob(odds: int) -> float:
    if odds == 0:
        return 0.0
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def american_to_decimal(odds: int) -> float:
    if odds < 0:
        return 1.0 + 100.0 / (-odds)
    return 1.0 + odds / 100.0


def devig_two_way(p1: float, p2: float, method: str = "proportional") -> Tuple[float, float]:
    s = p1 + p2
    if s <= 0:
        return 0.5, 0.5

    method = (method or "proportional").lower()

    if method in ("proportional", "additive", "multiplicative", "mult"):
        return p1 / s, p2 / s

    if method in ("power", "pow"):
        lo, hi = 0.01, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            val = (p1 ** mid) + (p2 ** mid)
            if val > 1.0:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2
        a = p1 ** k
        b = p2 ** k
        z = a + b
        return a / z, b / z

    return p1 / s, p2 / s


def ev_per_1unit(p: float, odds: int) -> float:
    dec = american_to_decimal(odds)
    return p * dec - 1.0


def kelly_fraction(p: float, odds: int) -> float:
    dec = american_to_decimal(odds)
    b = dec - 1.0
    if b <= 0:
        return 0.0
    f = (p * (b + 1.0) - 1.0) / b
    return max(0.0, f)
