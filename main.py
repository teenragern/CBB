from dotenv import load_dotenv
load_dotenv()

import argparse
import datetime as dt
from zoneinfo import ZoneInfo
import os
import io
import contextlib

from rich import print
import config as cfg

from src.telegram_notify import TelegramNotifier, format_recommendation_line

import re
from src.ODDS_API_KEY import OddsAPI
from src.bdl import BDLClient, arr
from src.ingest import fetch_games, fetch_team_stats
from src.model import fit_models
from src.db import (
    connect,
    init_db,
    upsert_games,
    insert_odds_snapshot,
    insert_prediction,
    insert_recommendation,
    settle_finished_games,
)
from src.scan import scan_game
from src.pricing import american_to_decimal
from src.timing_gate import timing_gate_decision

from src.calibration import load_calibration, update_calibration
from src.clv_gate import load_gate, update_gate

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc


def iso(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def today_ny() -> dt.date:
    return dt.datetime.now(NY).date()


def ny_date_from_iso_z(iso_z: str) -> dt.date:
    s = (iso_z or "").replace("Z", "+00:00")
    dtu = dt.datetime.fromisoformat(s)
    if dtu.tzinfo is None:
        dtu = dtu.replace(tzinfo=UTC)
    return dtu.astimezone(NY).date()


def parse_iso_z_to_utc_dt(iso_z: str) -> dt.datetime:
    s = (iso_z or "").replace("Z", "+00:00")
    dtu = dt.datetime.fromisoformat(s)
    if dtu.tzinfo is None:
        dtu = dtu.replace(tzinfo=UTC)
    return dtu.astimezone(UTC)


def _norm_team_name(name: str) -> str:
    s = (name or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = []
    for tok in s.split():
        if tok in {"university", "college", "the"}:
            continue
        if tok == "st":
            tok = "state"
        toks.append(tok)
    return " ".join(toks)


def _team_aliases(team: dict) -> set[str]:
    vals = {
        _norm_team_name(team.get("full_name") or ""),
        _norm_team_name(team.get("name") or ""),
    }
    return {v for v in vals if v}


def _match_odds_api_event(event: dict, todays_games: list[dict]) -> int | None:
    home_key = _norm_team_name(event.get("home_team") or "")
    away_key = _norm_team_name(event.get("away_team") or "")
    if not home_key or not away_key:
        return None

    try:
        event_tip = parse_iso_z_to_utc_dt(event.get("commence_time") or "")
    except Exception:
        event_tip = None

    best_gid = None
    best_delta = None

    for g in todays_games:
        home_aliases = _team_aliases(g.get("home_team") or {})
        away_aliases = _team_aliases(g.get("visitor_team") or {})
        if home_key not in home_aliases or away_key not in away_aliases:
            continue

        if event_tip is None:
            return int(g["id"])

        try:
            game_tip = parse_iso_z_to_utc_dt(g.get("date") or "")
            delta = abs((game_tip - event_tip).total_seconds())
        except Exception:
            delta = 0.0

        if delta > 6 * 3600:
            continue

        if best_delta is None or delta < best_delta:
            best_gid = int(g["id"])
            best_delta = delta

    return best_gid


def _extract_ml_offers(event: dict) -> list[dict]:
    out = []
    event_home = _norm_team_name(event.get("home_team") or "")
    event_away = _norm_team_name(event.get("away_team") or "")

    for book in event.get("bookmakers") or []:
        h2h = None
        for market in book.get("markets") or []:
            if (market.get("key") or "").lower() == "h2h":
                h2h = market
                break
        if not h2h:
            continue

        outcomes = h2h.get("outcomes") or []
        by_name = {_norm_team_name(o.get("name") or ""): o for o in outcomes}
        home = by_name.get(event_home)
        away = by_name.get(event_away)
        if not home or not away:
            continue

        out.append({
            "vendor": (book.get("key") or "").lower(),
            "updated_at": h2h.get("last_update") or book.get("last_update") or event.get("commence_time"),
            "ml_home_odds": int(home.get("price") or 0),
            "ml_away_odds": int(away.get("price") or 0),
            "source": "odds_api",
        })

    return out


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    m = n // 2
    if n % 2 == 1:
        return float(xs[m])
    return 0.5 * (float(xs[m - 1]) + float(xs[m]))


def _bdl_consensus_lines(rows: list[dict]) -> dict:
    spreads = [float(o["spread_home_value"]) for o in rows if o.get("spread_home_value") is not None]
    totals = [float(o["total_value"]) for o in rows if o.get("total_value") is not None]
    return {
        "spread": _median(spreads),
        "total": _median(totals),
    }


def _extract_alternate_offers(event: dict, consensus: dict) -> list[dict]:
    out = []
    event_home = _norm_team_name(event.get("home_team") or "")
    event_away = _norm_team_name(event.get("away_team") or "")

    consensus_spread = consensus.get("spread")
    consensus_total = consensus.get("total")

    for book in event.get("bookmakers") or []:
        vendor = (book.get("key") or "").lower()

        for market in book.get("markets") or []:
            mkey = (market.get("key") or "").lower()
            outcomes = market.get("outcomes") or []

            if mkey == "alternate_spreads":
                if consensus_spread is None:
                    continue

                by_name = {}
                for o in outcomes:
                    name_key = _norm_team_name(o.get("name") or "")
                    by_name.setdefault(name_key, []).append(o)

                home_rows = by_name.get(event_home, [])
                away_by_point = {}
                for o in by_name.get(event_away, []):
                    try:
                        away_by_point[round(float(o.get("point")), 3)] = o
                    except Exception:
                        continue

                seen = set()
                for ho in home_rows:
                    try:
                        home_point = float(ho.get("point"))
                        home_price = int(ho.get("price"))
                    except Exception:
                        continue

                    if abs(home_point - float(consensus_spread)) > 4.0:
                        continue

                    ao = away_by_point.get(round(-home_point, 3))
                    if not ao:
                        continue

                    key = round(home_point, 3)
                    if key in seen:
                        continue
                    seen.add(key)

                    try:
                        away_price = int(ao.get("price"))
                    except Exception:
                        continue

                    out.append({
                        "vendor": vendor,
                        "game_id": None,
                        "spread_home_value": home_point,
                        "spread_home_odds": home_price,
                        "spread_away_odds": away_price,
                        "total_value": None,
                        "total_over_odds": None,
                        "total_under_odds": None,
                        "updated_at": market.get("last_update") or book.get("last_update") or event.get("commence_time"),
                        "source": "odds_api_alternate",
                        "market_key": mkey,
                    })

            elif mkey == "alternate_totals":
                if consensus_total is None:
                    continue

                overs = {}
                unders = {}
                for o in outcomes:
                    nm = (o.get("name") or "").strip().lower()
                    try:
                        pt = float(o.get("point"))
                        price = int(o.get("price"))
                    except Exception:
                        continue
                    if nm == "over":
                        overs[round(pt, 3)] = price
                    elif nm == "under":
                        unders[round(pt, 3)] = price

                for pt_key, over_price in overs.items():
                    if pt_key not in unders:
                        continue
                    if abs(float(pt_key) - float(consensus_total)) > 8.0:
                        continue

                    out.append({
                        "vendor": vendor,
                        "game_id": None,
                        "spread_home_value": None,
                        "spread_home_odds": None,
                        "spread_away_odds": None,
                        "total_value": float(pt_key),
                        "total_over_odds": int(over_price),
                        "total_under_odds": int(unders[pt_key]),
                        "updated_at": market.get("last_update") or book.get("last_update") or event.get("commence_time"),
                        "source": "odds_api_alternate",
                        "market_key": mkey,
                    })

    return out


def apply_correlation_scaling(selected):
    by_gid = {}
    for g, r in selected:
        gid = int(g["id"])
        by_gid.setdefault(gid, []).append((g, r))

    out = []
    for gid, items in by_gid.items():
        if len(items) == 1:
            out += items
            continue

        stakes = [float(r.get("stake_units", 0.0)) for _, r in items]
        cap = max(stakes) if stakes else 0.0

        edges = [max(1e-6, float(r.get("edge", 0.0))) for _, r in items]
        s_edge = sum(edges)

        for (g, r), e in zip(items, edges):
            rr = dict(r)
            alloc = cap * (e / s_edge) if s_edge > 0 else 0.0
            rr["stake_units"] = min(float(rr.get("stake_units", 0.0)), alloc)
            rr["note"] = (rr.get("note", "") + f" | corr_cap={cap:.2f}").strip()
            out.append((g, rr))

    return out


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s_train = sub.add_parser("train")
    s_train.add_argument("--days", type=int, default=140)
    s_train.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (NY day)")

    s_scan = sub.add_parser("scan")
    s_scan.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (NY day)")
    s_scan.add_argument("--days", type=int, default=140)

    s_snap = sub.add_parser("snap")
    s_snap.add_argument("--hours", type=int, default=36)

    s_settle = sub.add_parser("settle")
    s_settle.add_argument("--days", type=int, default=14)

    s_report = sub.add_parser("report")
    s_report.add_argument("--days", type=int, default=7)

    s_close = sub.add_parser("close_snap")
    s_close.add_argument("--minutes_before", type=int, default=5)
    s_close.add_argument("--window", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()
    api = BDLClient()

    con = connect()
    init_db(con)

    notifier = TelegramNotifier()
    SEND_SLATE_SUMMARY = os.getenv("TELEGRAM_SEND_SLATE_SUMMARY", "1") == "1"
    SEND_REPORT = os.getenv("TELEGRAM_SEND_REPORT", "1") == "1"
    REPORT_DAYS = int(os.getenv("TELEGRAM_REPORT_DAYS", "2"))

    date_arg = getattr(args, "date", None)
    anchor = dt.date.fromisoformat(date_arg) if date_arg else today_ny()

    # -------------------------
    # SNAP
    # -------------------------
    if args.cmd == "snap":
        from src.snap import snap_upcoming_odds
        n = snap_upcoming_odds(api, con, hours_ahead=int(args.hours))
        print(f"[green]Saved odds snapshots:[/green] {n}")
        return

    if args.cmd == "close_snap":
        from src.snap import snap_close_odds
        n = snap_close_odds(api, con, minutes_before=int(args.minutes_before), window_min=int(args.window))
        print(f"[green]Saved close snapshots:[/green] {n}")
        return

    # -------------------------
    # REPORT (manual)
    # -------------------------
    if args.cmd == "report":
        from src.report import report_last_n_days
        report_last_n_days(con, days=int(args.days), limit=200)
        return

    # -------------------------
    # SETTLE + update calibration + update gate + Telegram nightly report
    # -------------------------
    if args.cmd == "settle":
        end = today_ny()
        start = end - dt.timedelta(days=int(args.days))
        seasons = sorted(set([start.year - 1, start.year, end.year]))

        games = fetch_games(api, iso(start), iso(end), seasons)
        upsert_games(con, games)

        n = settle_finished_games(con, games)
        print(f"[green]Settled/updated results:[/green] {n}")

        cal = update_calibration(con)
        print(f"[bold]CAL UPDATED[/bold] margin_bias={cal['margin_bias']:+.2f} total_bias={cal['total_bias']:+.2f} "
              f"sigM*{cal['sigma_margin_mult']:.2f} sigT*{cal['sigma_total_mult']:.2f} n={cal['n_games']}")

        gate = update_gate(con, last_n=30)
        print(f"[bold]GATE UPDATED[/bold] stake*{gate['stake_mult']:.2f} | clv_pts={gate.get('avg_clv_mtm_pts',0):+.3f} "
              f"| clv_price={gate.get('avg_clv_mtm_price',0):+.5f} | n={gate.get('n',0)}")

        if SEND_REPORT and notifier.enabled:
            from src.report import report_last_n_days
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                report_last_n_days(con, days=REPORT_DAYS, limit=200)
            text = buf.getvalue().strip()
            if text:
                notifier.send_long("NCAAB Nightly Report\n\n" + text)

        return

    # -------------------------
    # TRAIN / SCAN
    # -------------------------
    if args.cmd in ("train", "scan"):
        start_train = iso(anchor - dt.timedelta(days=int(args.days)))
        end_train = iso(anchor - dt.timedelta(days=1))
        seasons = sorted(set([anchor.year - 1, anchor.year]))

        todays_games = []
        if args.cmd == "scan":
            print(f"\n[bold]Scan day (NY):[/bold] {iso(anchor)}")

            fetch_start = anchor - dt.timedelta(days=1)
            fetch_end = anchor + dt.timedelta(days=2)
            print(f"[bold]Fetching games window:[/bold] {iso(fetch_start)} -> {iso(fetch_end)}")

            raw_games = api.fetch_all(
                "/games",
                params=[("start_date", iso(fetch_start)), ("end_date", iso(fetch_end))],
            )

            todays_games = [g for g in raw_games if ny_date_from_iso_z(g.get("date", "")) == anchor]
            print(f"[bold]Games raw:[/bold] {len(raw_games)} | [bold]NY-day:[/bold] {len(todays_games)}")

            upsert_games(con, todays_games)

        print(f"[bold]Training window:[/bold] {start_train} -> {end_train}")
        hist_games = fetch_games(api, start_train, end_train, seasons)
        hist_ts = fetch_team_stats(api, start_train, end_train, seasons)

        models = fit_models(hist_games, hist_ts)
        print(f"[bold]sigma_margin[/bold]={models.sigma_margin:.2f}  [bold]sigma_total[/bold]={models.sigma_total:.2f}")

        if args.cmd == "train":
            print("[green]Train OK.[/green]")
            return

        # Load calibration + gate for this scan run
        cal = load_calibration()
        print(f"[bold]CAL[/bold] margin_bias={cal['margin_bias']:+.2f} total_bias={cal['total_bias']:+.2f} "
              f"sigM*{cal['sigma_margin_mult']:.2f} sigT*{cal['sigma_total_mult']:.2f} n={cal['n_games']}")

        gate = load_gate()
        stake_mult = float(gate.get("stake_mult", 1.0))
        print(f"[bold]GATE[/bold] stake*{stake_mult:.2f} | clv_pts={gate.get('avg_clv_mtm_pts',0):+.3f} | clv_price={gate.get('avg_clv_mtm_price',0):+.5f}")

        # -------- SCAN ----------
        # todays_games already fetched above so injury auto-detection can be cached
        # before fit_models()/scan_game() read cfg.INJURY_OVERRIDES_PATH.

        # ---- pull ALL book offers for the slate (line shopping) ----
        odds_by_game = {}
        try:
            game_ids = [int(g["id"]) for g in todays_games]
            if game_ids:
                odds_rows = api.fetch_all("/odds", params=arr("game_ids", game_ids))
                for o in odds_rows:
                    gid = int(o.get("game_id") or o.get("id") or 0)
                    if gid:
                        odds_by_game.setdefault(gid, []).append(o)
                print(f"[green]Odds rows (by game_ids):[/green] {len(odds_rows)}")
            else:
                print("[yellow]No games for this NY day.[/yellow]")

            if todays_games:
                odds_api = OddsAPI()
                slate_events = odds_api.get_ncaab_odds(regions="us", markets="h2h", odds_format="american")
                ml_added = 0
                ml_unmatched = 0
                for event in slate_events:
                    gid = _match_odds_api_event(event, todays_games)
                    if gid is None:
                        ml_unmatched += 1
                        continue
                    ml_offers = _extract_ml_offers(event)
                    if not ml_offers:
                        continue
                    odds_by_game.setdefault(gid, []).extend(ml_offers)
                    ml_added += len(ml_offers)
                print(f"[green]Odds API ML offers merged:[/green] {ml_added} | unmatched events={ml_unmatched}")

                # ---- merge alternate spreads/totals from The Odds API once per slate ----
                try:
                    odds_api = OddsAPI()
                    slate_events = odds_api.get_ncaab_odds(regions="us", odds_format="american")

                    consensus_by_gid = {
                        int(g["id"]): _bdl_consensus_lines(odds_by_game.get(int(g["id"]), []))
                        for g in todays_games
                    }

                    alt_added = 0
                    alt_unmatched = 0
                    alt_skipped_no_consensus = 0

                    for event in slate_events:
                        gid = _match_odds_api_event(event, todays_games)
                        if gid is None:
                            alt_unmatched += 1
                            continue

                        consensus = consensus_by_gid.get(gid, {})
                        if consensus.get("spread") is None and consensus.get("total") is None:
                            alt_skipped_no_consensus += 1
                            continue

                        alt_offers = _extract_alternate_offers(event, consensus)
                        if not alt_offers:
                            continue

                        for o in alt_offers:
                            o["game_id"] = gid
                        odds_by_game.setdefault(gid, []).extend(alt_offers)
                        alt_added += len(alt_offers)

                    print(
                        f"[green]Odds API alternates merged:[/green] {alt_added} "
                        f"| unmatched_events={alt_unmatched} "
                        f"| no_consensus={alt_skipped_no_consensus}"
                    )
                except Exception as e:
                    print(f"[yellow]Odds API alternates unavailable (ok):[/yellow] {e}")

        except Exception as e:
            print(f"[yellow]Odds API moneyline unavailable (ok):[/yellow] {e}")

        now_utc = dt.datetime.now(UTC)

        candidates = []
        for g in todays_games:
            gid = int(g["id"])

            # minutes to tip (timing gate)
            try:
                game_dt = parse_iso_z_to_utc_dt(g.get("date", ""))
                minutes_to_tip = (game_dt - now_utc).total_seconds() / 60.0
            except Exception:
                minutes_to_tip = None

            # Skip if already started (optional)
            if cfg.TIME_GATE_SKIP_IF_STARTED and minutes_to_tip is not None and minutes_to_tip <= 0:
                continue

            late = (minutes_to_tip is not None and minutes_to_tip < cfg.TIME_GATE_MIN_MINUTES)
            late_min_edge = cfg.TIME_GATE_LATE_MIN_EDGE if late else None

            MAJOR_BOOKS = {"draftkings","fanduel","caesars","betmgm","betrivers","fanatics"}
            offers = [o for o in odds_by_game.get(gid, []) if (o.get("vendor") or "").lower() in MAJOR_BOOKS]

            def _median(xs):
                xs = sorted(xs)
                n = len(xs)
                if n == 0: return None
                m = n // 2
                return xs[m] if n % 2 == 1 else 0.5 * (xs[m-1] + xs[m])

            spreads = [float(o["spread_home_value"]) for o in offers if o.get("spread_home_value") is not None]
            totals = [float(o["total_value"]) for o in offers if o.get("total_value") is not None]
            med_spread = _median(spreads)
            med_total = _median(totals)

            def _is_outlier(o):
                if o.get("source") == "odds_api_alternate":
                    return False

                sh = o.get("spread_home_value")
                tl = o.get("total_value")
                if med_spread is not None and sh is not None and abs(float(sh) - med_spread) > 3.0:
                    return True
                if med_total is not None and tl is not None and abs(float(tl) - med_total) > 6.0:
                    return True
                return False

            offers = [o for o in offers if not _is_outlier(o)]

            for o in offers:
                vendor = (o.get("vendor") or "")
                if o.get("spread_home_value") is not None or o.get("total_value") is not None:
                    insert_odds_snapshot(con, gid, vendor, o)

            if not offers:
                pred_payload_ref, recs0 = scan_game(g, models, None, calibration=cal)
                insert_prediction(con, gid, pred_payload_ref)
                continue

            # ---- line shopping micro-upgrade ----
            # Step 1: For each (market, side), pick the best OFFER across books by:
            #   (a) best number, then (b) best odds, then (c) best EV as tiebreaker
            # Step 2: For each market (spread/total), pick the best SIDE by EV

            def line_score(mkt: str, side_: str, line_: float) -> float:
                if mkt == "moneyline":
                    return 0.0
                if mkt == "total" and side_ == "over":
                    return -float(line_)
                return float(line_)

            def odds_score(odds_: int) -> float:
                return float(american_to_decimal(int(odds_)))

            def is_better_offer(a: dict, b: dict) -> bool:
                sa = line_score(a["market"], a["side"], float(a["line"]))
                sb = line_score(b["market"], b["side"], float(b["line"]))
                if sa > sb + 1e-9:
                    return True
                if sb > sa + 1e-9:
                    return False

                oa = odds_score(int(a["odds"]))
                ob = odds_score(int(b["odds"]))
                if oa > ob + 1e-9:
                    return True
                if ob > oa + 1e-9:
                    return False

                return float(a.get("ev_per_unit", 0.0)) > float(b.get("ev_per_unit", 0.0))

            best_by_side = {}
            pred_payload_ref = None

            for o in offers:
                pred_payload, recs = scan_game(g, models, o, calibration=cal)
                if pred_payload_ref is None:
                    pred_payload_ref = pred_payload

                for r in recs:
                    if late_min_edge is not None and float(r.get("edge", 0.0)) < float(late_min_edge):
                        continue

                    key = (r["market"], r["side"])
                    cur = best_by_side.get(key)
                    if cur is None or is_better_offer(r, cur):
                        best_by_side[key] = r

            if pred_payload_ref is not None:
                insert_prediction(con, gid, pred_payload_ref)

            best_by_market = {}
            for (mkt, _side), r in best_by_side.items():
                cur = best_by_market.get(mkt)
                if cur is None or float(r.get("ev_per_unit", 0.0)) > float(cur.get("ev_per_unit", 0.0)):
                    best_by_market[mkt] = r

            for r in best_by_market.values():
                candidates.append((g, r))

        # Apply stake scaling from gate BEFORE selecting plays
        scaled = []
        for g, r in candidates:
            rr = dict(r)
            rr["stake_units"] = float(rr.get("stake_units", 0.0)) * stake_mult
            rr["note"] = (rr.get("note", "") + f" | gate×{stake_mult:.2f}").strip()
            scaled.append((g, rr))
        candidates = scaled

        # Rank by EV then edge
        candidates.sort(key=lambda gr: (gr[1].get("ev_per_unit", 0.0), gr[1].get("edge", 0.0)), reverse=True)

        # Exposure caps
        max_plays = getattr(cfg, "MAX_PLAYS", 12)
        max_daily_exposure = getattr(cfg, "MAX_DAILY_EXPOSURE", 0.12)
        cap = cfg.BANKROLL_UNITS * max_daily_exposure

        selected = []
        used = 0.0
        for g, r in candidates:
            if len(selected) >= max_plays:
                break
            stake = float(r.get("stake_units", 0.0))
            if used + stake > cap:
                continue
            selected.append((g, r))
            used += stake

        selected = apply_correlation_scaling(selected)
        used = sum(float(r.get("stake_units", 0.0)) for _, r in selected)

        print(f"\n[bold]Selected candidates:[/bold] {len(selected)} | exposure={used:.2f}u / cap={cap:.2f}u")

        if not selected:
            print("No edges passed filters (or exposure cap blocked them).")
            return

        def _reentry_line_score(mkt: str, side: str, line: float) -> float:
            if mkt == "total" and side == "over":
                return -float(line)
            return float(line)

        def allow_reentry(con, gid: int, mkt: str, side: str, new_line: float) -> bool:
            rows = con.execute(
                "SELECT side, line FROM recommendations WHERE game_id=? AND market=?",
                (gid, mkt),
            ).fetchall()

            if not rows:
                return True

            if len(rows) >= cfg.REENTRY_MAX_ENTRIES:
                return False

            if any((s or "").lower() != side for (s, _l) in rows):
                return False

            best_prev_score = max(_reentry_line_score(mkt, side, float(l)) for (_s, l) in rows)
            new_score = _reentry_line_score(mkt, side, float(new_line))

            improve = new_score - best_prev_score
            return improve >= cfg.REENTRY_MIN_IMPROVE

        bet_lines = []
        pass_lines = []

        for g, r in selected:
            gid = int(g["id"])

            if not allow_reentry(con, gid, r["market"], r["side"], float(r["line"])):
                pass_lines.append(format_recommendation_line(g, r, decision="PASS", reason="re-entry blocked"))
                continue

            action, why = timing_gate_decision(con, gid, r)
            if action == "WAIT":
                pass_lines.append(format_recommendation_line(g, r, decision="PASS", reason=f"WAIT — {why}"))
                continue

            r["note"] = (r.get("note", "") + f" | timing={why}").strip(" |")
            inserted = insert_recommendation(con, r)
            if inserted:
                bet_lines.append(format_recommendation_line(g, r, decision="BET"))
            else:
                pass_lines.append(format_recommendation_line(g, r, decision="PASS", reason="duplicate (already logged)"))

        print(f"\n[bold]New bets:[/bold] {len(bet_lines)}")
        print(f"[bold]Passes:[/bold] {len(pass_lines)}")

        if bet_lines:
            print("\n[green]BET[/green]")
            for line in bet_lines:
                print(line)

        if pass_lines:
            print("\n[yellow]PASS[/yellow]")
            for line in pass_lines[:5]:
                print(line)

        if notifier.enabled and SEND_SLATE_SUMMARY and (bet_lines or pass_lines):
            msg = f"NCAAB Bet / Pass ({iso(anchor)})\n\n"
            msg += f"BET ({len(bet_lines)}):\n" + ("\n\n".join(bet_lines) if bet_lines else "None")
            if pass_lines:
                msg += f"\n\nPASS ({len(pass_lines)}):\n" + "\n\n".join(pass_lines[:5])
                if len(pass_lines) > 5:
                    msg += f"\n\n(+{len(pass_lines)-5} more passes not shown)"
            notifier.send_long(msg)

        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
