"""Coverage tables and capability scenarios."""
import numpy as np
import pandas as pd

from reach120 import coverage, display, scenarios


def test_ward_table_counts_women_by_threshold():
    women = np.array([[10, 20, 30], [40, 50, 0]], np.float32)
    minutes = np.array([[10, 45, 90], [130, np.nan, 20]], np.float32)
    ward_ids = np.array([[1, 1, 2], [2, 2, 0]], np.int32)
    wards = pd.DataFrame({"ward_id": [1, 2], "ward": ["A", "B"], "lga": ["X", "X"]})
    t = coverage.ward_table(women, minutes, ward_ids, wards).set_index("ward")
    assert t.loc["A", "women"] == 30 and t.loc["A", "within_30"] == 10 and t.loc["A", "within_60"] == 30
    assert t.loc["B", "women"] == 120 and t.loc["B", "within_120"] == 30
    assert t.loc["B", "no_route"] == 50 and t.loc["B", "beyond_target"] == 90
    assert abs(t.loc["B", "share_beyond_target"] - 0.75) < 1e-9
    tot = coverage.totals(t.reset_index())
    assert tot["women"] == 150 and tot["beyond_target"] == 90


def test_classes_follow_the_bands():
    minutes = np.array([[5, 30, 31, 60], [61, 120, 121, np.nan]], np.float32)
    inside = np.ones_like(minutes, bool)
    inside[0, 0] = False
    np.testing.assert_array_equal(display.classify(minutes, inside), [[0, 1, 2, 2], [3, 3, 4, 5]])


def test_scenarios_are_nested_and_respect_status():
    df = pd.DataFrame({
        "facility_type": ["General Hospital", "Cottage Hospital", "Hospital", "General Hospital", "Primary Health Center"],
        "functional": ["Functional", "Unknown", "", "Closed", "Functional"],
        "facility_ownership": ["Public", "Public", "Private", "Public", "Public"],
    })
    flags = scenarios.add_flags(df)
    assert flags["cap_strict"].tolist() == [True, False, False, False, False]
    assert flags["cap_base"].tolist() == [True, True, False, False, False]
    assert flags["cap_loose"].tolist() == [True, True, True, False, False]
    assert scenarios.nested_check() == []
