import os
import json
import math
import datetime as dt
import sqlite3
from typing import Any, Dict, List, Optional

import config as cfg
from src.pricing import american_to_implied_prob, devig_two_way

UTC = dt.timezone.utc
GATE_PATH = os.getenv("CLV_GATE_PATH", "data/clv_gate.json")

MAJOR_BOOKS = {"draftkings", "fanduel", "caesars", "betmgm", "betrivers", "fanatics"}


def _now_utc_iso() -> str:
    return dt.datetime.now(UTC).isoformat()


def _parse_iso(s: str) -> dt.datetime:
    if not s:
        return dt.datetime.min.replace(tzinfo=UTC)
    ss = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(ss)
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return d.astimezone(UTC)


def default_gate() -> Dict[str, Any]:
    return {
        "stake_mult": 1.0,
        "avg_clv_mtm_pts": 0.0,
        "avg_clv_mtm_price": 0.0,
        "n": 0,
        "updated_utc": "",
    }


def load_gate(path: str = GATE_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = default_gate()
        base.update({k: data.get(k, base[k]) for k in base})
        return base
    except Exception:
        return default_gate()


def save_gate(g: Dict[str, Any], path: str = GATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = dict(g)
    out["updated_utc"] = _now_utc_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)


def _pick_latest_before(snaps: List[Dict[str, Any]], t: dt.datetime) -> Optional[Dict[str, Any]]:
    best = None
    for s in snaps:
        if s["ts_dt"] <= t:
            best = s
        else:
            break
    return best


def _fetch_snaps(con: sqlite3.Connection, game_id: int, vendor: str) -> List[Dict[str, Any]]:
    rows = con.execute(
        """
        SELECT ts_utc, vendor,
               spread_home, spread_home_odds, spread_away_odds,
               total, total_over_odds, total_under_odds
        FROM odds_snapshots
        WHERE game_id=?
        """,
        (game_id,),
    ).fetchall()

    v = (vendor or "").lower()
    snaps: List[Dict[str, Any]] = []
    for r in rows:
        vv = (r[1] or "").lower()
        if vv not in MAJOR_BOOKS:
            continue
        if v and vv != v:
            continue
        snaps.append(
            {
                "ts_dt": _parse_iso(r[0]),
                "vendor": vv,
                "spread_home": r[2],
                "spread_home_odds": r[3],
                "spread_away_odds": r[4],
                "total": r[5],
                "total_over_odds": r[6],
                "total_under_odds": r[7],
            }
        )
    snaps.sort(key=lambda x: x["ts_dt"])
    return snaps


def _clv_points(market: str, side: str, bet_line: float, snap: Dict[str, Any]) -> Optional[float]:
    if market == "spread":
        sh = snap.get("spread_home")
        if sh is None:
            return None
        sh = float(sh)
        if side == "home":
            return bet_line - sh
        if side == "away":
            return (-sh) - bet_line
        return None

    if market == "total":
        tl = snap.get("total")
        if tl is None:
            return None
        tl = float(tl)
        if side == "over":
            return tl - bet_line
        if side == "under":
            return bet_line - tl
        return None

    return None


def _line_matches(market: str, side: str, bet_line: float, snap: Dict[str, Any]) -> bool:
    eps = 1e-9
    if market == "spread":
        sh = snap.get("spread_home")
        if sh is None:
            return False
        sh = float(sh)
        if side == "home":
            return abs(sh - bet_line) < eps
        if side == "away":
            return abs((-sh) - bet_line) < eps
        return False

    if market == "total":
        tl = snap.get("total")
        if tl is None:
            return False
        return abs(float(tl) - bet_line) < eps

    return False


def _devig_prob_for_side(market: str, side: str, snap: Dict[str, Any]) -> Optional[float]:
    if market == "spread":
        oh = int(snap.get("spread_home_odds") or 0)
        oa = int(snap.get("spread_away_odds") or 0)
        if oh == 0 or oa == 0:
            return None
        ph = american_to_implied_prob(oh)
        pa = american_to_implied_prob(oa)
        ph_dv, pa_dv = devig_two_way(ph, pa, method=cfg.DEVIG_METHOD)
        return ph_dv if side == "home" else (pa_dv if side == "away" else None)

    if market == "total":
        oo = int(snap.get("total_over_odds") or 0)
        ou = int(snap.get("total_under_odds") or 0)
        if oo == 0 or ou == 0:
            return None
        po = american_to_implied_prob(oo)
        pu = american_to_implied_prob(ou)
        po_dv, pu_dv = devig_two_way(po, pu, method=cfg.DEVIG_METHOD)
        return po_dv if side == "over" else (pu_dv if side == "under" else None)

    return None


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return ys[mid] if (n % 2 == 1) else 0.5 * (ys[mid - 1] + ys[mid])

def _winsorize(xs: List[float], p: float) -> List[float]:
    """
    Clip the bottom/top p fraction to reduce outlier impact.
    p=0.10 means clip to 10th–90th percentile range.
    """
    if not xs:
        return xs
    ys = sorted(xs)
    n = len(ys)
    lo = ys[int(n * p)]
    hi = ys[int(n * (1 - p)) - 1]
    return [min(max(x, lo), hi) for x in xs]


def compute_gate(con: sqlite3.Connection, last_n: int = 30) -> Dict[str, Any]:
    now = dt.datetime.now(UTC)

    WINSOR_P = float(os.getenv("CLV_WINSOR_P", "0.10"))
    MIN_PTS_N = int(os.getenv("CLV_GATE_MIN_PTS_N", "12"))
    MIN_PRICE_N = int(os.getenv("CLV_GATE_MIN_PRICE_N", "8"))
    HARD_CLIP_PTS = float(os.getenv("CLV_GATE_HARD_CLIP_PTS", "10"))

    recs = con.execute(
        """
        SELECT game_id, ts_utc, market, side, line, vendor
        FROM recommendations
        ORDER BY ts_utc DESC
        LIMIT ?
        """,
        (last_n,),
    ).fetchall()

    mtm_pts: List[float] = []
    mtm_price: List[float] = []

    for (gid, ts_utc, market, side, line, vendor) in recs:
        gid = int(gid)
        rec_dt = _parse_iso(ts_utc)
        bet_line = float(line)
        v = (vendor or "").lower()

        snaps = _fetch_snaps(con, gid, v)
        if not snaps:
            continue

        entry = _pick_latest_before(snaps, rec_dt)
        mtm = _pick_latest_before(snaps, now)
        if not entry or not mtm:
            continue

        pts = _clv_points(market, side, bet_line, mtm)
        if pts is not None:
            pts = max(-HARD_CLIP_PTS, min(HARD_CLIP_PTS, float(pts)))
            mtm_pts.append(pts)

        if _line_matches(market, side, bet_line, entry) and _line_matches(market, side, bet_line, mtm):
            p0 = _devig_prob_for_side(market, side, entry)
            p1 = _devig_prob_for_side(market, side, mtm)
            if p0 is not None and p1 is not None:
                mtm_price.append(float(p1 - p0))

    if len(mtm_pts) == 0:
        return {
            "stake_mult": 0.50,
            "avg_clv_mtm_pts": 0.0,
            "avg_clv_mtm_price": 0.0,
            "n": len(recs),
            "n_pts": 0,
            "n_price": 0,
            "price_ok": False,
            "note": "no usable CLV samples; defaulting risk-off",
        }

    pts_used = _winsorize(mtm_pts, WINSOR_P) if len(mtm_pts) >= 5 else mtm_pts
    price_used = _winsorize(mtm_price, WINSOR_P) if len(mtm_price) >= 5 else mtm_price

    med_pts = _median(pts_used)
    mean_pts = _mean(pts_used)

    neg_rate = sum(1 for x in pts_used if x < 0) / len(pts_used) if pts_used else 0.0

    price_ok = len(price_used) >= MIN_PRICE_N
    med_price = _median(price_used) if price_ok else 0.0
    mean_price = _mean(price_used) if price_ok else 0.0

    if len(pts_used) < MIN_PTS_N:
        mult = 0.50
    else:
        if neg_rate >= 0.60:
            mult = 0.25
        elif neg_rate >= 0.53:
            mult = 0.50
        else:
            mult = 1.00

    return {
        "stake_mult": mult,
        "avg_clv_mtm_pts": round(med_pts, 3),
        "avg_clv_mtm_price": round(med_price, 5),
        "mean_clv_mtm_pts": round(mean_pts, 3),
        "mean_clv_mtm_price": round(mean_price, 5),
        "neg_rate_pts": round(neg_rate, 3),
        "n": len(recs),
        "n_pts": len(pts_used),
        "n_price": len(price_used),
        "price_ok": bool(price_ok),
        "winsor_p": WINSOR_P,
    }


def update_gate(con: sqlite3.Connection, last_n: int = 30) -> Dict[str, Any]:
    g = compute_gate(con, last_n=last_n)
    save_gate(g)
    return g