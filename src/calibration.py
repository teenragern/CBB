import os
import json
import math
import datetime as dt
from typing import Any, Dict, List, Tuple, Optional

CAL_PATH = os.getenv("CAL_PATH", "data/calibration.json")


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_iso(s: str) -> dt.datetime:
    if not s:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    ss = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(ss)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def default_calibration() -> Dict[str, Any]:
    return {
        "margin_bias": 0.0,
        "total_bias": 0.0,
        "sigma_margin_mult": 1.0,
        "sigma_total_mult": 1.0,
        "n_games": 0,
        "updated_utc": "",
    }


def load_calibration(path: str = CAL_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = default_calibration()
        base.update({k: data.get(k, base[k]) for k in base.keys()})
        return base
    except Exception:
        return default_calibration()


def save_calibration(cal: Dict[str, Any], path: str = CAL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cal = dict(cal)
    cal["updated_utc"] = _now_utc_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2, sort_keys=True)


def _winsorize(xs: List[float], p: float = 0.05) -> List[float]:
    if not xs:
        return xs
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    lo = xs_sorted[int(n * p)]
    hi = xs_sorted[int(n * (1 - p)) - 1]
    return [min(max(x, lo), hi) for x in xs]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(v, 0.0))


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return ys[mid] if n % 2 == 1 else 0.5 * (ys[mid - 1] + ys[mid])


def compute_calibration_from_db(
    con,
    max_games: int = 400,
    min_games: int = 50,
    winsor_p: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """
    Uses: results (final_margin, final_total, settled_ts_utc) + predictions (mu_margin, mu_total, sigma_*)
    Picks the latest prediction for each game at/after it is settled (best available).
    """
    rows = con.execute(
        """
        SELECT
          r.game_id, r.settled_ts_utc, r.final_margin, r.final_total,
          p.ts_utc, p.mu_margin, p.mu_total, p.sigma_margin, p.sigma_total
        FROM results r
        JOIN predictions p ON p.game_id = r.game_id
        """
    ).fetchall()

    by_game: Dict[int, Tuple[dt.datetime, Tuple]] = {}
    for row in rows:
        gid = int(row[0])
        settled_dt = _parse_iso(row[1])
        pred_dt = _parse_iso(row[4])

        # Use the latest prediction timestamp (most recent logged prediction for that game)
        # If you want "prediction before tip" only, we can tighten this later.
        key = (settled_dt, pred_dt)
        best = by_game.get(gid)
        if best is None or pred_dt > best[0]:
            by_game[gid] = (pred_dt, row)

    items = []
    for gid, (_, row) in by_game.items():
        settled_dt = _parse_iso(row[1])
        items.append((settled_dt, row))

    items.sort(key=lambda x: x[0])  # by settled time
    if len(items) > max_games:
        items = items[-max_games:]

    n = len(items)
    if n < min_games:
        return None

    margin_resid = []
    total_resid = []
    sig_m = []
    sig_t = []

    for _, row in items:
        final_margin = float(row[2])
        final_total = float(row[3])
        mu_margin = float(row[5])
        mu_total = float(row[6])
        sigma_margin = float(row[7]) if row[7] is not None else 0.0
        sigma_total = float(row[8]) if row[8] is not None else 0.0

        margin_resid.append(final_margin - mu_margin)  # actual - predicted
        total_resid.append(final_total - mu_total)
        if sigma_margin > 0:
            sig_m.append(sigma_margin)
        if sigma_total > 0:
            sig_t.append(sigma_total)

    margin_resid = _winsorize(margin_resid, p=winsor_p)
    total_resid = _winsorize(total_resid, p=winsor_p)

    mb = _mean(margin_resid)
    tb = _mean(total_resid)

    # Base sigmas from stored predictions
    base_m = _median(sig_m) if sig_m else 11.0
    base_t = _median(sig_t) if sig_t else 14.0

    sd_m = _std(margin_resid)
    sd_t = _std(total_resid)

    m_mult = sd_m / base_m if base_m > 0 else 1.0
    t_mult = sd_t / base_t if base_t > 0 else 1.0

    # Clip so it can’t go crazy
    mb = float(max(-3.0, min(3.0, mb)))
    tb = float(max(-6.0, min(6.0, tb)))
    m_mult = float(max(0.75, min(1.35, m_mult)))
    t_mult = float(max(0.75, min(1.35, t_mult)))

    return {
        "margin_bias": round(mb, 3),
        "total_bias": round(tb, 3),
        "sigma_margin_mult": round(m_mult, 3),
        "sigma_total_mult": round(t_mult, 3),
        "n_games": n,
    }


def update_calibration(con, alpha: float = 0.25) -> Dict[str, Any]:
    """
    Smooth update: new = (1-alpha)*old + alpha*estimate
    """
    old = load_calibration()
    est = compute_calibration_from_db(con)

    if est is None:
        # not enough data yet; keep old
        return old

    cal = dict(old)
    cal["margin_bias"] = round((1 - alpha) * float(old["margin_bias"]) + alpha * float(est["margin_bias"]), 3)
    cal["total_bias"] = round((1 - alpha) * float(old["total_bias"]) + alpha * float(est["total_bias"]), 3)
    cal["sigma_margin_mult"] = round((1 - alpha) * float(old["sigma_margin_mult"]) + alpha * float(est["sigma_margin_mult"]), 3)
    cal["sigma_total_mult"] = round((1 - alpha) * float(old["sigma_total_mult"]) + alpha * float(est["sigma_total_mult"]), 3)
    cal["n_games"] = int(est["n_games"])
    save_calibration(cal)
    return cal