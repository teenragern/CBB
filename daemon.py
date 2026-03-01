import os
import time
import subprocess
import datetime as dt
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

SNAP_EVERY_MIN = int(os.getenv("SNAP_EVERY_MIN", "10"))
SETTLE_EVERY_MIN = int(os.getenv("SETTLE_EVERY_MIN", "240"))
SNAP_HOURS_AHEAD = int(os.getenv("SNAP_HOURS_AHEAD", "36"))

SCAN_EVERY_MIN = int(os.getenv("SCAN_EVERY_MIN", "30"))
SCAN_ACTIVE_START = int(os.getenv("SCAN_ACTIVE_START", "10"))
SCAN_ACTIVE_END = int(os.getenv("SCAN_ACTIVE_END", "23"))

CLOSE_SNAP_EVERY_SEC = int(os.getenv("CLOSE_SNAP_EVERY_SEC", "60"))
SLEEP_SECONDS = int(os.getenv("DAEMON_SLEEP_SECONDS", "20"))

def now_ny():
    return dt.datetime.now(NY)

def run_cmd(args):
    ts = now_ny().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\n[{ts}] Running: {' '.join(args)}", flush=True)
    p = subprocess.run(args, text=True, capture_output=True)
    if p.stdout:
        print(p.stdout, flush=True)
    if p.stderr:
        print(p.stderr, flush=True)
    print(f"[{ts}] Exit code: {p.returncode}", flush=True)
    return p.returncode

def main():
    last_snap = None
    last_settle = None
    last_scan = None
    last_close_snap = None

    while True:
        t = now_ny()

        if last_snap is None or (t - last_snap).total_seconds() >= SNAP_EVERY_MIN * 60:
            run_cmd(["python", "main.py", "snap", "--hours", str(SNAP_HOURS_AHEAD)])
            last_snap = t

        in_active = SCAN_ACTIVE_START <= t.hour < SCAN_ACTIVE_END
        if in_active and (last_scan is None or (t - last_scan).total_seconds() >= SCAN_EVERY_MIN * 60):
            run_cmd(["python", "main.py", "scan", "--date", t.date().isoformat()])
            last_scan = t

        if last_close_snap is None or (t - last_close_snap).total_seconds() >= CLOSE_SNAP_EVERY_SEC:
            run_cmd(["python", "main.py", "close_snap", "--minutes_before", "5", "--window", "2"])
            last_close_snap = t

        if last_settle is None or (t - last_settle).total_seconds() >= SETTLE_EVERY_MIN * 60:
            run_cmd(["python", "main.py", "settle", "--days", "14"])
            last_settle = t

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
