from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable


def label_future_event(timestamp: datetime, future_starts: Iterable[datetime], horizon_minutes: int) -> int:
    """Implementa exactamente la regla (t, t+h]. Útil para tests unitarios."""
    upper = timestamp + timedelta(minutes=horizon_minutes)
    return int(any(timestamp < event_time <= upper for event_time in future_starts))


def target_expression(horizon_minutes: int, next_event_time: str = "next_ea.inicio_ea") -> str:
    """Expresión SQL sin inclusión del instante actual."""
    horizon = int(horizon_minutes)
    return (
        f"CASE WHEN {next_event_time} > c.ts "
        f"AND {next_event_time} <= c.ts + INTERVAL {horizon} MINUTE "
        "THEN 1 ELSE 0 END"
    )
