import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich import print

import config as cfg
from src.features import norm_cdf, compute_rest_days, infer_game_type, load_injury_overrides, load_neutral_overrides
from src.pricing import american_to_implied_prob, devig_two_way, ev_per_1unit, kelly_fraction

def pick_latest_odds(odds_rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_game: Dict[int, List[Dict[str, Any]]] = {}
    for o in odds_rows:
        gid = int(o["game_id"])
        by_game.setdefault(gid, []).append(o)

    out: Dict[int, Dict[str, Any]] = {}
    for gid, rows in by_game.items():
        rows.sort(key=lambda r: r.get("updated_at", ""))
        rows = rows[::-1]  # newest first
        chosen = None
        for v in cfg.VENDOR_PREFERENCE:
            for r in rows:
                if (r.get("vendor") or "").lower() == v.lower():
                    chosen = r
                    break
            if chosen:
                break
        out[gid] = chosen or rows[0]
    return out

def blend_mu_with_market(mu: float, market_margin: float) -> float:
    w = cfg.MARKET_BLEND_W
    if abs(market_margin) >= 14:
        w = cfg.BIG_SPREAD_W14
    elif abs(market_margin) >= 10:
        w = cfg.BIG_SPREAD_W10
    return w * mu + (1.0 - w) * market_margin

def blend_prob(p_model: float, p_market_devig: float) -> float:
    w = cfg.PROB_BLEND_W
    return w * p_model + (1.0 - w) * p_market_devig

def stake_units_from_kelly(k: float) -> float:
    k_adj = max(0.0, k) * cfg.KELLY_MULT
    k_adj = min(k_adj, cfg.MAX_FRACTION)
    return cfg.BANKROLL_UNITS * k_adj

def lookup_sigma(bucket: Optional[Dict], x: float, fallback: float) -> float:
    if bucket is None:
        return fallback
    edges = bucket["edges"]
    sigs = bucket["sigmas"]
    for i in range(len(sigs)):
        if edges[i] <= x < edges[i + 1]:
            return float(sigs[i])
    return float(sigs[-1])

def sos_rating_profile(models, team_id: int) -> Dict[str, float]:
    ratings = getattr(models, "sos_ratings", {}) or {}
    row = ratings.get(int(team_id)) or {}
    return {
        "adj_off": float(row.get("adj_off", 1.0)),
        "adj_def": float(row.get("adj_def", 1.0)),
    }


def scan_game(game, models, odds=None, calibration=None):
    gid = int(game["id"])
    hid = int(game["home_team"]["id"])
    aid = int(game["visitor_team"]["id"])
    season = int(game["season"])

    inj = load_injury_overrides()
    neutral_map = load_neutral_overrides()
    is_neutral = neutral_map.get(gid, 0)
    inj_home = inj.get(hid, 0.0)
    inj_away = inj.get(aid, 0.0)

    rest_map = compute_rest_days([game])
    rest_home = rest_map.get((gid, hid), 7)
    rest_away = rest_map.get((gid, aid), 7)
    b2b_home = 1 if rest_home == 0 else 0
    b2b_away = 1 if rest_away == 0 else 0

    game_type = infer_game_type({"date": game.get("date"), "is_neutral": is_neutral})
    sos_home = sos_rating_profile(models, hid)
    sos_away = sos_rating_profile(models, aid)

    P_hat = float(models.pace_model.predict(pd.DataFrame([{
        "season": season, "game_type": game_type, "home_team_id": hid, "away_team_id": aid,
        "is_neutral": is_neutral,
        "rest_home": rest_home, "rest_away": rest_away,
        "b2b_home": b2b_home, "b2b_away": b2b_away,
        "inj_home": inj_home, "inj_away": inj_away,
    }]))[0])

    ppp_h = float(models.ppp_model.predict(pd.DataFrame([{
        "season": season, "game_type": game_type, "off_team_id": hid, "def_team_id": aid, "is_home": 1,
        "rest": rest_home, "b2b": b2b_home,
        "inj_off": inj_home, "inj_def": inj_away,
        "adj_off_off": sos_home["adj_off"], "adj_def_off": sos_home["adj_def"],
        "adj_off_def": sos_away["adj_off"], "adj_def_def": sos_away["adj_def"],
    }]))[0])

    ppp_a = float(models.ppp_model.predict(pd.DataFrame([{
        "season": season, "game_type": game_type, "off_team_id": aid, "def_team_id": hid, "is_home": 0,
        "rest": rest_away, "b2b": b2b_away,
        "inj_off": inj_away, "inj_def": inj_home,
        "adj_off_off": sos_away["adj_off"], "adj_def_off": sos_away["adj_def"],
        "adj_off_def": sos_home["adj_off"], "adj_def_def": sos_home["adj_def"],
    }]))[0])

    mu_home = P_hat * ppp_h
    mu_away = P_hat * ppp_a
    mu_margin = mu_home - mu_away
    mu_total = mu_home + mu_away

    cal = calibration or {}
    margin_bias = float(cal.get("margin_bias", 0.0))
    total_bias = float(cal.get("total_bias", 0.0))
    sig_m_mult = float(cal.get("sigma_margin_mult", 1.0))
    sig_t_mult = float(cal.get("sigma_total_mult", 1.0))

    mu_margin_adj = mu_margin + margin_bias
    mu_total_adj = mu_total + total_bias

    sigma_m_base = lookup_sigma(models.sigma_margin_by_abs, abs(mu_margin_adj), models.sigma_margin)
    sigma_t_base = lookup_sigma(models.sigma_total_by_mu, mu_total_adj, models.sigma_total)

    sigma_margin_used = sigma_m_base * sig_m_mult
    sigma_total_used = sigma_t_base * sig_t_mult

    pred_payload = {
        "game_id": gid,
        "mu_margin": mu_margin,
        "mu_total": mu_total,
        "sigma_margin": models.sigma_margin,
        "sigma_total": models.sigma_total,
        "fair_spread_home": mu_margin,
        "fair_total": mu_total,
        "mu_margin_adj": mu_margin_adj,
        "mu_total_adj": mu_total_adj,
        "sigma_margin_used": sigma_margin_used,
        "sigma_total_used": sigma_total_used,
        "calibration": cal,
    }

    recs: List[Dict[str, Any]] = []

    if odds:
        vendor = (odds.get("vendor") or "").lower()
        spread_home = float(odds.get("spread_home_value") or 0.0)
        market_margin = -spread_home

        mu_used = blend_mu_with_market(mu_margin_adj, market_margin)

        sigma_used = sigma_margin_used * (1.10 if abs(market_margin) >= 10 else 1.00)

        thresh = -spread_home
        p_cover_home = 1.0 - norm_cdf(thresh, mu_used, sigma_used)

        ph = american_to_implied_prob(int(odds.get("spread_home_odds") or 0))
        pa = american_to_implied_prob(int(odds.get("spread_away_odds") or 0))
        ph_dv, pa_dv = devig_two_way(ph, pa, method=cfg.DEVIG_METHOD)

        odds_home = int(odds.get("spread_home_odds") or 0)
        if odds_home:
            p_final = blend_prob(p_cover_home, ph_dv)
            edge = p_final - ph_dv
            ev = ev_per_1unit(p_final, odds_home)
            k = kelly_fraction(p_final, odds_home)
            stake = stake_units_from_kelly(k)

            if p_final >= cfg.MIN_P and edge >= cfg.MIN_EDGE and ev >= cfg.MIN_EV:
                recs.append({
                    "game_id": gid,
                    "market": "spread",
                    "side": "home",
                    "line": spread_home,
                    "odds": odds_home,
                    "p_model": p_cover_home,
                    "p_market_devig": ph_dv,
                    "p_final": p_final,
                    "edge": edge,
                    "ev_per_unit": ev,
                    "kelly_f": k,
                    "stake_units": stake,
                    "vendor": vendor,
                    "note": f"mu_blend={mu_used:.2f} sigma={sigma_used:.2f} w={cfg.PROB_BLEND_W:.2f}",
                })

        odds_away = int(odds.get("spread_away_odds") or 0)
        p_cover_away = 1.0 - p_cover_home
        if odds_away:
            p_final = blend_prob(p_cover_away, pa_dv)
            edge = p_final - pa_dv
            ev = ev_per_1unit(p_final, odds_away)
            k = kelly_fraction(p_final, odds_away)
            stake = stake_units_from_kelly(k)

            if p_final >= cfg.MIN_P and edge >= cfg.MIN_EDGE and ev >= cfg.MIN_EV:
                recs.append({
                    "game_id": gid,
                    "market": "spread",
                    "side": "away",
                    "line": -spread_home,
                    "odds": odds_away,
                    "p_model": p_cover_away,
                    "p_market_devig": pa_dv,
                    "p_final": p_final,
                    "edge": edge,
                    "ev_per_unit": ev,
                    "kelly_f": k,
                    "stake_units": stake,
                    "vendor": vendor,
                    "note": f"mu_blend={mu_used:.2f} sigma={sigma_used:.2f} w={cfg.PROB_BLEND_W:.2f}",
                })

        ml_home_odds = int(odds.get("ml_home_odds") or 0)
        ml_away_odds = int(odds.get("ml_away_odds") or 0)
        if ml_home_odds and ml_away_odds:
            p_win_home = 1.0 - norm_cdf(0.0, mu_margin_adj, sigma_margin_used)
            p_win_away = 1.0 - p_win_home

            pmh = american_to_implied_prob(ml_home_odds)
            pma = american_to_implied_prob(ml_away_odds)
            pmh_dv, pma_dv = devig_two_way(pmh, pma, method=cfg.DEVIG_METHOD)

            p_final_home = blend_prob(p_win_home, pmh_dv)
            edge_home = p_final_home - pmh_dv
            ev_home = ev_per_1unit(p_final_home, ml_home_odds)
            k_home = kelly_fraction(p_final_home, ml_home_odds)
            stake_home = stake_units_from_kelly(k_home)
            if p_final_home >= cfg.MIN_P and edge_home >= cfg.MIN_EDGE and ev_home >= cfg.MIN_EV:
                recs.append({
                    "game_id": gid,
                    "market": "moneyline",
                    "side": "home",
                    "line": 0.0,
                    "odds": ml_home_odds,
                    "p_model": p_win_home,
                    "p_market_devig": pmh_dv,
                    "p_final": p_final_home,
                    "edge": edge_home,
                    "ev_per_unit": ev_home,
                    "kelly_f": k_home,
                    "stake_units": stake_home,
                    "vendor": vendor,
                    "note": f"pwin={p_win_home:.3f} muM={mu_margin_adj:.2f} sigmaM={sigma_margin_used:.2f} w={cfg.PROB_BLEND_W:.2f}",
                })

            p_final_away = blend_prob(p_win_away, pma_dv)
            edge_away = p_final_away - pma_dv
            ev_away = ev_per_1unit(p_final_away, ml_away_odds)
            k_away = kelly_fraction(p_final_away, ml_away_odds)
            stake_away = stake_units_from_kelly(k_away)
            if p_final_away >= cfg.MIN_P and edge_away >= cfg.MIN_EDGE and ev_away >= cfg.MIN_EV:
                recs.append({
                    "game_id": gid,
                    "market": "moneyline",
                    "side": "away",
                    "line": 0.0,
                    "odds": ml_away_odds,
                    "p_model": p_win_away,
                    "p_market_devig": pma_dv,
                    "p_final": p_final_away,
                    "edge": edge_away,
                    "ev_per_unit": ev_away,
                    "kelly_f": k_away,
                    "stake_units": stake_away,
                    "vendor": vendor,
                    "note": f"pwin={p_win_away:.3f} muM={-mu_margin_adj:.2f} sigmaM={sigma_margin_used:.2f} w={cfg.PROB_BLEND_W:.2f}",
                })

        total_line = float(odds.get("total_value") or 0.0)
        mu_total_used = cfg.MARKET_BLEND_W * mu_total_adj + (1.0 - cfg.MARKET_BLEND_W) * total_line

        p_over = 1.0 - norm_cdf(total_line, mu_total_used, sigma_total_used)
        po = american_to_implied_prob(int(odds.get("total_over_odds") or 0))
        pu = american_to_implied_prob(int(odds.get("total_under_odds") or 0))
        po_dv, pu_dv = devig_two_way(po, pu, method=cfg.DEVIG_METHOD)

        over_odds = int(odds.get("total_over_odds") or 0)
        under_odds = int(odds.get("total_under_odds") or 0)

        if over_odds:
            p_final = blend_prob(p_over, po_dv)
            edge = p_final - po_dv
            ev = ev_per_1unit(p_final, over_odds)
            k = kelly_fraction(p_final, over_odds)
            stake = stake_units_from_kelly(k)
            if p_final >= cfg.MIN_P and edge >= cfg.MIN_EDGE and ev >= cfg.MIN_EV:
                recs.append({
                    "game_id": gid,
                    "market": "total",
                    "side": "over",
                    "line": total_line,
                    "odds": over_odds,
                    "p_model": p_over,
                    "p_market_devig": po_dv,
                    "p_final": p_final,
                    "edge": edge,
                    "ev_per_unit": ev,
                    "kelly_f": k,
                    "stake_units": stake,
                    "vendor": vendor,
                    "note": f"muT_blend={mu_total_used:.2f} sigmaT={sigma_total_used:.2f} w={cfg.PROB_BLEND_W:.2f}",
                })

        p_under = 1.0 - p_over
        if under_odds:
            p_final = blend_prob(p_under, pu_dv)
            edge = p_final - pu_dv
            ev = ev_per_1unit(p_final, under_odds)
            k = kelly_fraction(p_final, under_odds)
            stake = stake_units_from_kelly(k)
            if p_final >= cfg.MIN_P and edge >= cfg.MIN_EDGE and ev >= cfg.MIN_EV:
                recs.append({
                    "game_id": gid,
                    "market": "total",
                    "side": "under",
                    "line": total_line,
                    "odds": under_odds,
                    "p_model": p_under,
                    "p_market_devig": pu_dv,
                    "p_final": p_final,
                    "edge": edge,
                    "ev_per_unit": ev,
                    "kelly_f": k,
                    "stake_units": stake,
                    "vendor": vendor,
                    "note": f"muT_blend={mu_total_used:.2f} sigmaT={sigma_total_used:.2f} w={cfg.PROB_BLEND_W:.2f}",
                })

    return pred_payload, recs
