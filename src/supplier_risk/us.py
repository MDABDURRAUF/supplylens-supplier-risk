"""United States reference data: Census regions, federal holidays and metro locations."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# U.S. Census Bureau regions (50 states and the District of Columbia).
CENSUS_REGIONS = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
STATE_REGION = {state: region for region, states in CENSUS_REGIONS.items() for state in states}

# Freight hubs used by the sample data: metro -> (state, latitude, longitude). Coordinates are approximate.
METROS = {
    "Los Angeles": ("CA", 34.05, -118.24), "Seattle": ("WA", 47.61, -122.33), "Phoenix": ("AZ", 33.45, -112.07),
    "Salt Lake City": ("UT", 40.76, -111.89), "Denver": ("CO", 39.74, -104.99), "Dallas": ("TX", 32.78, -96.80),
    "Houston": ("TX", 29.76, -95.37), "Kansas City": ("MO", 39.10, -94.58), "Minneapolis": ("MN", 44.98, -93.27),
    "Chicago": ("IL", 41.88, -87.63), "Detroit": ("MI", 42.33, -83.05), "Columbus": ("OH", 39.96, -83.00),
    "Memphis": ("TN", 35.15, -90.05), "Nashville": ("TN", 36.16, -86.78), "Atlanta": ("GA", 33.75, -84.39),
    "Miami": ("FL", 25.76, -80.19), "Charlotte": ("NC", 35.23, -80.84), "Philadelphia": ("PA", 39.95, -75.17),
    "Newark": ("NJ", 40.74, -74.17), "Boston": ("MA", 42.36, -71.06),
}


def region_of(state: str) -> str:
    """Census region of a two-letter state code."""
    try:
        return STATE_REGION[str(state).strip().upper()]
    except KeyError:
        raise ValueError(f"'{state}' is not a US state code") from None


def great_circle_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between points given in degrees (arrays allowed)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th given weekday of a month (Monday = 0); n = -1 means the last one."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def federal_holidays(years) -> dict[dt.date, str]:
    """US federal holidays for the given year or years, on their actual dates: ``{date: name}``."""
    years = [years] if isinstance(years, (int, np.integer)) else list(years)
    out = {}
    for y in years:
        y = int(y)
        out.update({
            dt.date(y, 1, 1): "New Year's Day",
            _nth_weekday(y, 1, 0, 3): "Birthday of Martin Luther King, Jr.",
            _nth_weekday(y, 2, 0, 3): "Washington's Birthday",
            _nth_weekday(y, 5, 0, -1): "Memorial Day",
            dt.date(y, 6, 19): "Juneteenth National Independence Day",
            dt.date(y, 7, 4): "Independence Day",
            _nth_weekday(y, 9, 0, 1): "Labor Day",
            _nth_weekday(y, 10, 0, 2): "Columbus Day",
            dt.date(y, 11, 11): "Veterans Day",
            _nth_weekday(y, 11, 3, 4): "Thanksgiving Day",
            dt.date(y, 12, 25): "Christmas Day",
        })
    return out


def holidays_between(start: pd.Series, end: pd.Series) -> np.ndarray:
    """For each row, the number of federal holidays from ``start`` to ``end`` inclusive."""
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    years = range(int(start.dt.year.min()), int(end.dt.year.max()) + 1)
    days = np.sort(np.array([np.datetime64(d) for d in federal_holidays(years)], dtype="datetime64[D]"))
    lo = np.searchsorted(days, start.to_numpy().astype("datetime64[D]"), side="left")
    hi = np.searchsorted(days, end.to_numpy().astype("datetime64[D]"), side="right")
    return hi - lo


def is_peak_season(dates: pd.Series) -> np.ndarray:
    """Whether a date falls in the year-end shipping peak (15 November to 31 December)."""
    dates = pd.to_datetime(dates)
    return (((dates.dt.month == 11) & (dates.dt.day >= 15)) | (dates.dt.month == 12)).to_numpy()
