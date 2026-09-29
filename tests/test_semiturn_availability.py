from src.features import SEMITURN_END_OFFSET_HOURS


def test_semiturn_records_are_available_at_end_not_start():
    assert SEMITURN_END_OFFSET_HOURS == {1: 10, 2: 14, 3: 18, 4: 22, 5: 26, 6: 30}
    assert SEMITURN_END_OFFSET_HOURS[1] != 6
    assert SEMITURN_END_OFFSET_HOURS[5] > 24

