"""Canonical plant clock for all semiturn aggregates and registry records.

All six records share an operational date at midnight. The operational day
starts at 06:00 and ends at 06:00 the next calendar day. These are conservative
close times; actual source publication latency is a separate requirement.
"""
from types import MappingProxyType
from typing import Iterable

import pandas as pd


SEMITURN_END_OFFSET_HOURS = MappingProxyType({1:10, 2:14, 3:18, 4:22, 5:26, 6:30})


def semiturn_end_case(column: str) -> str:
    """DuckDB interval expression; unknown semiturns yield NULL, never a guess."""
    branches = ' '.join(f'WHEN {s} THEN INTERVAL {h} HOUR'
                        for s,h in SEMITURN_END_OFFSET_HOURS.items())
    return f'CASE try_cast({column} AS INTEGER) {branches} END'


def semiturn_available_at(dates: Iterable, semiturns: Iterable) -> pd.DatetimeIndex:
    """Return close timestamps: e.g. 2026-05-01/ST6 -> 2026-05-02 06:00."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    turns = list(semiturns)
    if len(dates) != len(turns) or dates.isna().any():
        raise AssertionError('Invalid or unaligned semiturn dates')
    if any(turn not in SEMITURN_END_OFFSET_HOURS for turn in turns):
        raise AssertionError('Unknown semiturn; availability cannot be established')
    return dates + pd.to_timedelta([SEMITURN_END_OFFSET_HOURS[s] for s in turns], unit='h')
