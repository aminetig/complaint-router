import random

import numpy as np
import pytest

from complaint_router.data import Reservoir, split_for_date, text_hash
from complaint_router.metrics import (
    choose_threshold,
    classification_metrics,
    coverage_curve,
    real_mix_accuracy,
    routing_metrics,
)


@pytest.mark.parametrize(
    ("date", "split"),
    [
        ("2021-12-31", None),
        ("2022-01-01", "train"),
        ("2023-06-30", "train"),
        ("2023-07-01", "val"),
        ("2023-09-30T12:00:00", "val"),
        ("2023-10-01", "test"),
        ("2024-02-29", "test"),
        ("2024-03-01", None),
        ("", None),
    ],
)
def test_split_for_date(date, split):
    assert split_for_date(date) == split


def test_reservoir_keeps_at_most_size_items():
    r = Reservoir(10, random.Random(0))
    for i in range(1000):
        r.add(i)
    assert len(r.items) == 10
    assert r.seen == 1000
    assert len(set(r.items)) == 10


def test_reservoir_is_uniform():
    # Chaque élément d'un flux de 20 doit être gardé environ 5 fois sur 20 (taille 5)
    counts = np.zeros(20)
    rng = random.Random(1)
    for _ in range(4000):
        r = Reservoir(5, rng)
        for i in range(20):
            r.add(i)
        counts[r.items] += 1
    assert np.allclose(counts / 4000, 0.25, atol=0.03)


def test_text_hash_ignores_case():
    assert text_hash("Late FEE on my card") == text_hash("late fee on my card")
    assert text_hash("late fee") != text_hash("late fees")


def test_classification_metrics():
    m = classification_metrics(["cards", "cards", "mortgage"], ["cards", "mortgage", "mortgage"])
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["per_class"]["cards"]["recall"] == 0.5
    assert m["per_class"]["mortgage"]["precision"] == 0.5


def test_choose_threshold_reaches_target():
    confidence = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([1, 1, 1, 1, 0, 1, 0, 0], dtype=bool)
    t = choose_threshold(confidence, correct, target=0.8)
    routed = routing_metrics(confidence, correct, t)
    assert routed["justesse_automatisee"] >= 0.8
    # 6 premières : 5 justes sur 6 = 83 % ; 7 premières : 5 sur 7 = 71 %
    assert t == 0.6
    assert routed["part_automatisee"] == 0.75


def test_coverage_curve_matches_threshold():
    # Même exemple : à 80 % de justesse, on peut automatiser 6 cas sur 8
    confidence = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([1, 1, 1, 1, 0, 1, 0, 0], dtype=bool)
    curve = coverage_curve(confidence, correct, points=8, target=0.8)
    assert curve["part_a_la_cible"] == 0.75
    assert curve["couverture"][-1] == 1.0
    assert curve["justesse"][-1] == 0.625


def test_choose_threshold_when_target_unreachable():
    assert choose_threshold(np.array([0.9, 0.8]), np.array([False, False])) == 1.0


def test_real_mix_accuracy_weights_recall_by_share():
    per_class = {label: {"recall": 0.5} for label in
                 ("credit_reporting", "debt_collection", "cards", "bank_account", "mortgage",
                  "money_transfer", "personal_loan", "vehicle_loan", "student_loan")}
    per_class["credit_reporting"]["recall"] = 1.0
    assert real_mix_accuracy(per_class, {"credit_reporting": 3, "cards": 1}) == 0.875
