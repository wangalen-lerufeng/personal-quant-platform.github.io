from datetime import date, datetime, time
from zoneinfo import ZoneInfo


_MAINLAND_EQUITY_SESSIONS = ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0)))
_MAINLAND_FUTURES_SESSIONS = (
    (time(9, 0), time(11, 30)),
    (time(13, 30), time(15, 0)),
    (time(21, 0), time(23, 0)),
)


class MarketCalendar:
    def __init__(self, holidays: set[date] | None = None):
        self.holidays = holidays or set()

    def is_open(self, market: str, moment: datetime) -> bool:
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("moment must be timezone-aware")
        local_moment = moment.astimezone(ZoneInfo("Asia/Shanghai"))
        local_date = local_moment.date()
        if local_date in self.holidays:
            return False
        if local_moment.weekday() >= 5:
            return False
        local_time = local_moment.time()
        if market == "mainland_equity":
            return any(start <= local_time < end for start, end in _MAINLAND_EQUITY_SESSIONS)
        if market == "mainland_futures":
            return any(start <= local_time < end for start, end in _MAINLAND_FUTURES_SESSIONS)
        if market == "mt5_broker_symbols":
            return True
        raise ValueError(f"unknown market: {market}")


def is_trading_time(market: str, moment: datetime, holidays: set[date] | None = None) -> bool:
    return MarketCalendar(holidays).is_open(market, moment)
