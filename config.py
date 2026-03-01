import os

# Units-based bankroll (e.g., 100 units bankroll)
BANKROLL_UNITS = float(os.getenv("BANKROLL_UNITS", "100"))

# "Aggressive but capped" sizing
KELLY_MULT = float(os.getenv("KELLY_MULT", "0.50"))       # 0.50 Kelly
MAX_FRACTION = float(os.getenv("MAX_FRACTION", "0.03"))   # max 3% bankroll per bet

# Filters
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.020"))
MIN_EV = float(os.getenv("MIN_EV", "0.020"))
MIN_P = float(os.getenv("MIN_P", "0.520"))

# Market anchoring (market is strong baseline)
MARKET_BLEND_W = float(os.getenv("MARKET_BLEND_W", "0.60"))   # 0.60 model / 0.40 market
BIG_SPREAD_W10 = float(os.getenv("BIG_SPREAD_W10", "0.45"))   # if |spread|>=10 use smaller model weight
BIG_SPREAD_W14 = float(os.getenv("BIG_SPREAD_W14", "0.35"))   # if |spread|>=14 use even smaller model weight

# Vendor preference order for odds
VENDOR_PREFERENCE = tuple(os.getenv(
    "VENDOR_PREFERENCE",
    "draftkings,fanduel,caesars,betmgm"
).split(","))

PROB_BLEND_W = float(os.getenv("PROB_BLEND_W", "0.65"))  # 0.65 model / 0.35 market

TIME_GATE_MIN_MINUTES = int(os.getenv("TIME_GATE_MIN_MINUTES", "60"))   # <60 min to tip = "late"
TIME_GATE_LATE_MIN_EDGE = float(os.getenv("TIME_GATE_LATE_MIN_EDGE", "0.040"))  # stricter edge late
TIME_GATE_SKIP_IF_STARTED = os.getenv("TIME_GATE_SKIP_IF_STARTED", "1") == "1"

REENTRY_MIN_IMPROVE = float(os.getenv("REENTRY_MIN_IMPROVE", "1.5"))  # require 1.5 pts better number
REENTRY_MAX_ENTRIES = int(os.getenv("REENTRY_MAX_ENTRIES", "2"))      # initial + 1 re-entry

RECENCY_HALFLIFE_DAYS = int(os.getenv("RECENCY_HALFLIFE_DAYS", "45"))

SIGMA_BUCKETS = int(os.getenv("SIGMA_BUCKETS", "6"))
DEVIG_METHOD = os.getenv("DEVIG_METHOD", "power")

INJURY_OVERRIDES_PATH = os.getenv("INJURY_OVERRIDES_PATH", "data/injuries_override.json")
NEUTRAL_OVERRIDES_PATH = os.getenv("NEUTRAL_OVERRIDES_PATH", "data/neutral_overrides.json")

MAX_PLAYS = int(os.getenv("MAX_PLAYS", "12"))
MAX_DAILY_EXPOSURE = float(os.getenv("MAX_DAILY_EXPOSURE", "0.08"))  # 8% of bankroll per slate/day
