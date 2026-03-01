import subprocess
import pandas as pd
import streamlit as st

from src.db import connect, init_db

st.set_page_config(page_title="NCAAB Bot UI", layout="wide")


def run_cmd(args):
    p = subprocess.run(args, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr


def load_df(query, params=()):
    con = connect()
    init_db(con)
    try:
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()


st.title("NCAAB Betting Bot UI")

tab1, tab2, tab3 = st.tabs(["Run Bot", "Today's Picks", "History"])

with tab1:
    st.subheader("Run Commands")

    col1, col2, col3 = st.columns(3)

    train_days = col1.number_input("Train days", min_value=30, max_value=365, value=140)
    scan_days = col2.number_input("Scan training window", min_value=30, max_value=365, value=140)
    report_days = col3.number_input("Report days", min_value=1, max_value=60, value=7)

    date_str = st.text_input("Scan date (optional, YYYY-MM-DD)", value="")

    c1, c2, c3 = st.columns(3)
    if c1.button("Train"):
        args = ["python", "main.py", "train", "--days", str(train_days)]
        if date_str.strip():
            args += ["--date", date_str.strip()]
        code, out, err = run_cmd(args)
        st.code(out + ("\n" + err if err else ""))

    if c2.button("Scan"):
        args = ["python", "main.py", "scan", "--days", str(scan_days)]
        if date_str.strip():
            args += ["--date", date_str.strip()]
        code, out, err = run_cmd(args)
        st.code(out + ("\n" + err if err else ""))

    if c3.button("Report"):
        code, out, err = run_cmd(["python", "main.py", "report", "--days", str(report_days)])
        st.code(out + ("\n" + err if err else ""))

    st.divider()

    col4, col5, col6 = st.columns(3)

    snap_hours = col4.number_input("Snap hours ahead", min_value=1, max_value=72, value=36)
    close_minutes = col5.number_input("Close snap minutes_before", min_value=1, max_value=60, value=5)
    close_window = col6.number_input("Close snap window", min_value=1, max_value=30, value=2)

    d1, d2, d3 = st.columns(3)

    if d1.button("Snap Odds"):
        code, out, err = run_cmd(["python", "main.py", "snap", "--hours", str(snap_hours)])
        st.code(out + ("\n" + err if err else ""))

    if d2.button("Close Snap"):
        code, out, err = run_cmd([
            "python", "main.py", "close_snap",
            "--minutes_before", str(close_minutes),
            "--window", str(close_window),
        ])
        st.code(out + ("\n" + err if err else ""))

    if d3.button("Settle"):
        code, out, err = run_cmd(["python", "main.py", "settle", "--days", "14"])
        st.code(out + ("\n" + err if err else ""))


with tab2:
    st.subheader("Today's Picks")

    picks = load_df("""
        SELECT
            r.ts_utc,
            g.away_team_name || ' @ ' || g.home_team_name AS matchup,
            r.market,
            r.side,
            r.line,
            r.odds,
            r.edge,
            r.ev_per_unit,
            r.stake_units,
            r.vendor,
            r.note
        FROM recommendations r
        JOIN games g ON g.game_id = r.game_id
        ORDER BY r.ts_utc DESC
        LIMIT 200
    """)

    st.dataframe(picks, use_container_width=True)


with tab3:
    st.subheader("History / Results")

    history = load_df("""
        SELECT
            r.ts_utc,
            g.away_team_name || ' @ ' || g.home_team_name AS matchup,
            r.market,
            r.side,
            r.line,
            r.odds,
            r.stake_units,
            res.home_score,
            res.away_score
        FROM recommendations r
        JOIN games g ON g.game_id = r.game_id
        LEFT JOIN results res ON res.game_id = r.game_id
        ORDER BY r.ts_utc DESC
        LIMIT 300
    """)

    st.dataframe(history, use_container_width=True)
