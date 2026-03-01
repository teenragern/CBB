import datetime as dt
from zoneinfo import ZoneInfo
import sqlite3
from typing import Optional, List, Dict, Any

from src.pricing import american_to_decimal
pending_stake = 0.0
NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

def parse_any_iso(s: str) -> dt.datetime:
    """
    Accepts:
      - 2026-02-20T05:27:08.123456+00:00
      - 2026-02-20T01:00:00.000Z
    Returns timezone-aware datetime in UTC.
    """
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

    snaps = []
    for r in rows:
        snaps.append({
            "ts_utc": r[0],
            "ts_dt": parse_any_iso(r[0]),
            "vendor": r[1],
            "spread_home": r[2],
            "spread_home_odds": r[3],
            "spread_away_odds": r[4],
            "total": r[5],
            "total_over_odds": r[6],
            "total_under_odds": r[7],
        })
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
            # better if close spread becomes more negative; clv positive if you beat the close
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

def report_last_n_days(con: sqlite3.Connection, days: int = 7, limit: int = 200) -> None:
    global pending_stake
    pending_stake = 0.0  # Reset to 0 so we don't double-count on refresh

    rows = con.execute(
        """
        SELECT
            r.game_id, r.ts_utc, r.market, r.side, r.line, r.odds, r.stake_units,
            g.start_time, g.home_team_name, g.away_team_name,
            res.home_score, res.away_score
        FROM recommendations r
        JOIN games g ON g.game_id = r.game_id
        LEFT JOIN results res ON res.game_id = r.game_id
        WHERE r.ts_utc >= datetime('now', ?)
        ORDER BY r.ts_utc DESC
        LIMIT ?
        """,
        (f"-{days} days", limit),
    ).fetchall()

    if not rows:
        print("No recommendations found in that window.")
        return

    now_utc = dt.datetime.now(UTC)

    settled = 0
    pending = 0
    total_stake = 0.0
    total_pnl = 0.0

    clv_mtm_vals = []
    clv_close_vals = []

    out = []

    for (gid, rec_ts, market, side, line, odds, stake, start_time, home, away, hs, ays) in rows:
        gid = int(gid)
        rec_dt = parse_any_iso(rec_ts)
        start_dt = parse_any_iso(start_time or "")

        snaps = fetch_snapshots(con, gid)
        open_s = pick_latest_before(snaps, rec_dt)
        mtm_s  = pick_latest_before(snaps, now_utc)
        close_s = pick_latest_before(snaps, start_dt) if start_time else None

        clv_mtm = None
        clv_close = None
        if mtm_s:
            clv_mtm = clv_points(market, side, float(line), mtm_s.get("spread_home"), mtm_s.get("total"))
            if clv_mtm is not None:
                clv_mtm_vals.append(clv_mtm)
        if close_s:
            clv_close = clv_points(market, side, float(line), close_s.get("spread_home"), close_s.get("total"))
            if clv_close is not None:
                clv_close_vals.append(clv_close)

        result = None
        pnl = None
        # CHECK IF SETTLED OR PENDING
        if hs is not None and ays is not None:
            settled += 1
            result = grade_bet(market, side, float(line), float(hs), float(ays))
            pnl = pnl_units(result, float(stake), int(odds))
            total_stake += float(stake)
            total_pnl += pnl
        else:
            pending += 1
            pending_stake += float(stake) # Increment the global pending counter

        out.append({
            "match": f"{away} @ {home}",
            "market": market,
            "side": side,
            "line": float(line),
            "odds": int(odds),
            "stake_u": float(stake),
            "clv_mtm_pts": None if clv_mtm is None else round(clv_mtm, 2),
            "clv_close_pts": None if clv_close is None else round(clv_close, 2),
            "result": result,
            "pnl_u": None if pnl is None else round(pnl, 2),
            "rec_ts": rec_ts,
        })

    avg_clv_mtm = (sum(clv_mtm_vals) / len(clv_mtm_vals)) if clv_mtm_vals else 0.0
    avg_clv_close = (sum(clv_close_vals) / len(clv_close_vals)) if clv_close_vals else 0.0
    roi = (total_pnl / total_stake) if total_stake > 0 else 0.0

    # HEADER PRINT WITH PENDING STAKE
    print(
        f"\nREPORT last {days} days | recs={len(out)} | settled={settled} pending={pending} "
        f"| stake={total_stake:.2f}u | stake_pending={pending_stake:.2f}u | pnl={total_pnl:.2f}u | ROI={roi:.3f} "
        f"| avg_CLV_MTM_pts={avg_clv_mtm:.2f} | avg_CLV_CLOSE_pts={avg_clv_close:.2f}"
    )

    for r in out[:25]:
        print(
            f"- {r['match']} | {r['market']} {r['side']} {r['line']} ({r['odds']}) "
            f"stake={r['stake_u']:.2f}u CLV(mtm)={r['clv_mtm_pts']} CLV(close)={r['clv_close_pts']} "
            f"res={r['result']} pnl={r['pnl_u']}"
        )