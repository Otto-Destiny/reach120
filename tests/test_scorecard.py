"""Facility-name matching for the capability scorecard."""
from reach120 import scorecard


def test_shared_generic_words_do_not_make_a_match():
    assert scorecard.similarity("Rikky Hospital", "Charity Hospital") < 0.85
    assert scorecard.similarity("Crest Hospital", "Kent Hospital") < 0.85


def test_same_distinctive_name_matches():
    assert scorecard.similarity("Veta Hospital", "Veta Hospital Ltd") == 1.0
    assert scorecard.similarity("Bluesea Hospital", "Bluesea Hospital And Maternity") == 1.0
    assert scorecard.similarity("Ajeromi General Hospital", "Ajeromi Ifelodun General Hospital") == 1.0


def test_acceptance_rules():
    assert scorecard.acceptable(4000, 0.1, same_code=True) == "code"
    assert scorecard.acceptable(1200, 0.9, same_code=False) == "strong name"
    assert scorecard.acceptable(1200, 0.75, same_code=False) is None
    assert scorecard.acceptable(250, 0.75, same_code=False) == "name and distance"
    assert scorecard.acceptable(40, 0.45, same_code=False) == "very close"
