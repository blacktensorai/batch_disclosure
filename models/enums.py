# models/enums.py
from enum import Enum

# --------------------------
# Impact Score
class Impact(str, Enum):
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"

# --------------------------
# Sentiment / Tone
class Tone(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CAUTIOUS = "cautious"

# --------------------------
# Forecast Types (unified)
class ForecastType(str, Enum):
    INTENT = "INTENT"              # plan / intention / will / aims
    TIMING = "TIMING"              # timeline, schedule, soon
    CONTRACTUAL = "CONTRACTUAL"    # contracts, JV, MOU, deals
    GUIDANCE = "GUIDANCE"          # revenue/EBITDA guidance, forecast
    REGULATORY = "REGULATORY"      # approvals, filings, FDA, ASX
    STRATEGY = "STRATEGY"          # growth strategy, expansion
    HINTS = "HINTS"                # vague forward commentary
    MILESTONES = "MILESTONES"     # milestones
    DEALS = "DEALS"               # any new deals/contracts

# --------------------------
# Filing Types (canonical)
class FilingType(str, Enum):
    ASX_ANNUAL = "annual"
    ASX_QUARTERLY = "quarterly"
    ASX_INVESTOR = "investor"
    SEC_10Q = "SEC_10Q"