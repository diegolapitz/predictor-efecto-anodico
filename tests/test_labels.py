from datetime import datetime, timedelta

from src.labels import label_future_event


def test_target_interval_is_open_at_t_and_closed_at_horizon():
    timestamp = datetime(2026, 1, 1, 12, 0)
    assert label_future_event(timestamp, [timestamp], 15) == 0
    assert label_future_event(timestamp, [timestamp + timedelta(seconds=1)], 15) == 1
    assert label_future_event(timestamp, [timestamp + timedelta(minutes=15)], 15) == 1
    assert label_future_event(timestamp, [timestamp + timedelta(minutes=15, seconds=1)], 15) == 0


def test_targets_are_monotonic_by_horizon():
    timestamp = datetime(2026, 1, 1, 12, 0)
    events = [timestamp + timedelta(minutes=27)]
    assert [label_future_event(timestamp, events, h) for h in (15, 30, 60)] == [0, 1, 1]

