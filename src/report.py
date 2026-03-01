import datetime as dt
import os
from zoneinfo import ZoneInfo
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from rich import print
import config as cfg
from src.pricing import american_to_decimal, american_to_implied_prob, devig_two_way

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

MAJOR_BOOKS = {"draftkings", "fanduel", "caesars", "betmgm", "betrivers", "fanatics"}

CLOSE_MINUTES_BEFORE_TIP = int(os.getenv("CLOSE_MINUTES_BEFORE_TIP", "5"))
EST_GAME_DURATION_MIN = int(os.getenv("EST_GAME_DURATION_MIN", "125"))


def _is_bucket_tip_time(tip_dt_utc: dt.datetime) -> bool:
    return tip_dt_utc.minute == 0 and tip_dt_utc.second == 0 and tip_dt_utc.hour in (0, 2)


def parse_any_iso(s: str) -> dt.datetime:
    if not s:
        return dt.datetime.min.replace(tzinfo=UTC)
    ss = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(ss)
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return d.astimezone(UTC)


def fetch_snapshots(con: sqlite3.Connection, game_id: int) -> List[Dict[str, Any]]:
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

    snaps: List[Dict[str, Any]] = []
    for r in rows:
        snaps.append(
            {
                "ts_utc": r[0],
                "ts_dt": parse_any_iso(r[0]),
                "vendor": (r[1] or "").lower(),
                "spread_home": r[2],
                "spread_home_odds": r[3],
                "spread_away_odds": r[4],
                "total": r[5],
                "total_over_odds": r[6],
                "total_under_odds": r[7],
            }
        )

    snaps = [s for s in snaps if s["vendor"] in MAJOR_BOOKS]
    snaps.sort(key=lambda x: x["ts_dt"])
    return snaps


def pick_latest_before(snaps: List[Dict[str, Any]], t: dt.datetime) -> Optional[Dict[str, Any]]:
    best = None
    for s in snaps:
        if s["ts_dt"] <= t:
            best = s
        else:
            break
    return best


def clv_points(market: str, side: str, bet_line: float,
               spread_home: Optional[float], total_line: Optional[float]) -> Optional[float]:
    if market == "spread":
        if spread_home is None:
            return None
        sh = float(spread_home)
        if side == "home":
            return bet_line - sh
        if side == "away":
            close_away = -sh
            return close_away - bet_line
        return None

    if market == "total":
        if total_line is None:
            return None
        tl = float(total_line)
        if side == "over":
            return tl - bet_line
        if side == "under":
            return bet_line - tl
        return None

    return None


def grade_bet(market: str, side: str, line: float, home_score: float, away_score: float) -> str:
    margin = home_score - away_score
    total = home_score + away_score

    if market == "spread":
        if side == "home":
            thresh = -line
            if margin > thresh: return "W"
            if margin < thresh: return "L"
            return "P"
        if side == "away":
            if margin < line: return "W"
            if margin > line: return "L"
            return "P"

    if market == "total":
        if side == "over":
            if total > line: return "W"
            if total < line: return "L"
            return "P"
        if side == "under":
            if total < line: return "W"
            if total > line: return "L"
            return "P"

    return "?"


def pnl_units(result: str, stake: float, odds: int) -> float:
    if result == "P":
        return 0.0
    if result == "L":
        return -stake
    dec = american_to_decimal(int(odds))
    return stake * (dec - 1.0)


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
    """
    Returns devig implied probability for the bet side at this snapshot.
    """
    if market == "spread":
        oh = int(snap.get("spread_home_odds") or 0)
        oa = int(snap.get("spread_away_odds") or 0)
        if oh == 0 or oa == 0:
            return None
        ph = american_to_implied_prob(oh)
        pa = american_to_implied_prob(oa)
        ph_dv, pa_dv = devig_two_way(ph, pa, method=cfg.DEVIG_METHOD)
        if side == "home":
            return ph_dv
        if side == "away":
            return pa_dv
        return None

    if market == "total":
        oo = int(snap.get("total_over_odds") or 0)
        ou = int(snap.get("total_under_odds") or 0)
        if oo == 0 or ou == 0:
            return None
        po = american_to_implied_prob(oo)
        pu = american_to_implied_prob(ou)
        po_dv, pu_dv = devig_two_way(po, pu, method=cfg.DEVIG_METHOD)
        if side == "over":
            return po_dv
        if side == "under":
            return pu_dv
        return None

    return None


def report_last_n_days(con: sqlite3.Connection, days: int = 7, limit: int = 200) -> None:
    now_utc = dt.datetime.now(UTC)
    cutoff = now_utc - dt.timedelta(days=days)

    rows = con.execute(
        """
        SELECT
            r.game_id, r.ts_utc, r.market, r.side, r.line, r.odds, r.stake_units, r.vendor,
            g.start_time, g.status, g.home_team_name, g.away_team_name,
            res.settled_ts_utc, res.home_score, res.away_score
        FROM recommendations r
        JOIN games g ON g.game_id = r.game_id
        LEFT JOIN results res ON res.game_id = r.game_id
        ORDER BY r.ts_utc DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    filtered = []
    for row in rows:
        rec_dt = parse_any_iso(row[1])
        if rec_dt >= cutoff:
            filtered.append(row)

    if not filtered:
        print(f"No recommendations found in last {days} days (limit={limit}).")
        return

    settled = 0
    pending = 0
    settled_stake = 0.0
    pending_stake = 0.0
    total_pnl = 0.0

    clv_mtm_pts_vals: List[float] = []
    clv_close_pts_vals: List[float] = []
    clv_mtm_price_vals: List[float] = []
    clv_close_price_vals: List[float] = []

    out = []

    for (
        gid, rec_ts, market, side, line, odds, stake, rec_vendor,
        start_time, g_status, home, away,
        settled_ts, hs, ays
    ) in filtered:
        gid = int(gid)
        rec_dt = parse_any_iso(rec_ts)
        bet_line = float(line)

        snaps = fetch_snapshots(con, gid)

        # Vendor-specific snapshots (best practice)
        rv = (rec_vendor or "").lower()
        snaps_same = [s for s in snaps if s["vendor"] == rv]
        use_snaps = snaps_same if snaps_same else snaps

        entry_s = pick_latest_before(use_snaps, rec_dt)
        mtm_s = pick_latest_before(use_snaps, now_utc)

        close_s = None
        close_anchor = None

        if (g_status or "") == "post":
            tip_dt = parse_any_iso(start_time or "")
            tip_known = bool(start_time) and not _is_bucket_tip_time(tip_dt)

            if tip_known:
                close_anchor = tip_dt - dt.timedelta(minutes=CLOSE_MINUTES_BEFORE_TIP)
            elif settled_ts:
                settled_dt = parse_any_iso(settled_ts)
                close_anchor = settled_dt - dt.timedelta(minutes=(EST_GAME_DURATION_MIN + CLOSE_MINUTES_BEFORE_TIP))

            if close_anchor is not None:
                close_s = pick_latest_before(use_snaps, close_anchor)

        # ----- Points CLV -----
        clv_mtm_pts = None
        clv_close_pts = None

        if mtm_s:
            clv_mtm_pts = clv_points(market, side, bet_line, mtm_s.get("spread_home"), mtm_s.get("total"))
            if clv_mtm_pts is not None:
                clv_mtm_pts_vals.append(clv_mtm_pts)

        if close_s:
            clv_close_pts = clv_points(market, side, bet_line, close_s.get("spread_home"), close_s.get("total"))
            if clv_close_pts is not None:
                clv_close_pts_vals.append(clv_close_pts)

        # ----- PRICE CLV (devig prob) -----
        clv_mtm_price = None
        clv_close_price = None

        entry_p = None
        if entry_s and _line_matches(market, side, bet_line, entry_s):
            entry_p = _devig_prob_for_side(market, side, entry_s)

        if entry_p is not None and mtm_s and _line_matches(market, side, bet_line, mtm_s):
            mtm_p = _devig_prob_for_side(market, side, mtm_s)
            if mtm_p is not None:
                clv_mtm_price = mtm_p - entry_p
                clv_mtm_price_vals.append(clv_mtm_price)

        if entry_p is not None and close_s and _line_matches(market, side, bet_line, close_s):
            close_p = _devig_prob_for_side(market, side, close_s)
            if close_p is not None:
                clv_close_price = close_p - entry_p
                clv_close_price_vals.append(clv_close_price)

        # ----- Grade / PnL -----
        result = None
        pnl = None
        if hs is not None and ays is not None:
            settled += 1
            result = grade_bet(market, side, bet_line, float(hs), float(ays))
            pnl = pnl_units(result, float(stake), int(odds))
            settled_stake += float(stake)
            total_pnl += pnl
        else:
            pending += 1
            pending_stake += float(stake)

        out.append(
            {
                "match": f"{away} @ {home}",
                "market": market,
                "side": side,
                "line": bet_line,
                "odds": int(odds),
                "stake_u": float(stake),
                "clv_mtm_pts": None if clv_mtm_pts is None else round(clv_mtm_pts, 2),
                "clv_close_pts": None if clv_close_pts is None else round(clv_close_pts, 2),
                "clv_mtm_price": None if clv_mtm_price is None else round(clv_mtm_price, 4),
                "clv_close_price": None if clv_close_price is None else round(clv_close_price, 4),
                "result": result,
                "pnl": None if pnl is None else round(pnl, 2),
            }
        )

    avg_mtm_pts = (sum(clv_mtm_pts_vals) / len(clv_mtm_pts_vals)) if clv_mtm_pts_vals else 0.0
    avg_close_pts = (sum(clv_close_pts_vals) / len(clv_close_pts_vals)) if clv_close_pts_vals else 0.0
    avg_mtm_price = (sum(clv_mtm_price_vals) / len(clv_mtm_price_vals)) if clv_mtm_price_vals else 0.0
    avg_close_price = (sum(clv_close_price_vals) / len(clv_close_price_vals)) if clv_close_price_vals else 0.0

    roi = (total_pnl / settled_stake) if settled_stake > 0 else 0.0

    print(
        f"\nREPORT last {days} days | recs={len(out)} | settled={settled} pending={pending} "
        f"| stake_settled={settled_stake:.2f}u | stake_pending={pending_stake:.2f}u "
        f"| pnl={total_pnl:.2f}u | ROI={roi:.3f} "
        f"| avg_CLV_MTM_pts={avg_mtm_pts:.2f} | avg_CLV_CLOSE_pts={avg_close_pts:.2f} "
        f"| avg_CLV_MTM_price={avg_mtm_price:+.4f} | avg_CLV_CLOSE_price={avg_close_price:+.4f}"
    )

    for r in out[:25]:
        print(
            f"- {r['match']} | {r['market']} {r['side']} {r['line']} ({r['odds']}) "
            f"stake={r['stake_u']:.2f}u "
            f"CLVpts(mtm)={r['clv_mtm_pts']} CLVpts(close)={r['clv_close_pts']} "
            f"CLVprice(mtm)={r['clv_mtm_price']} CLVprice(close)={r['clv_close_price']} "
            f"res={r['result']} pnl={r['pnl']}"
        )
