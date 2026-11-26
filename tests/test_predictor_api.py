"""Tests du modèle servi et de l'API. Ils utilisent le vrai modèle versionné dans `models/`."""

from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parent.parent / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "model_meta.json").exists(),
                                reason="modèle absent : lancer `make export`")

CARD = ("I was charged an annual fee on my credit card even though the bank told me the card had "
        "no fee. I called twice and they refused to refund it.")
MORTGAGE = ("My mortgage servicer added a late fee although my payment was made on time through "
            "autopay, and my escrow account shows a shortage.")


@pytest.fixture(scope="module")
def predictor():
    from complaint_router.predictor import Predictor

    return Predictor(MODELS)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_probabilities_are_valid(predictor):
    proba = predictor.predict_proba([CARD, MORTGAGE, "short"])
    assert proba.shape == (3, len(predictor.labels))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()


def test_obvious_complaints_are_routed_correctly(predictor):
    card, mortgage = predictor.predict([CARD, MORTGAGE])
    assert card.label == "cards"
    assert mortgage.label == "mortgage"
    assert len(card.top3) == 3
    assert card.top3[0]["label"] == card.label


def test_batch_and_single_predictions_agree(predictor):
    # Une réclamation doit recevoir exactement le même score seule ou dans un lot : en 8 bits,
    # un lot avec remplissage changerait l'échelle de quantification (voir predictor.py)
    long_text = CARD + " " + MORTGAGE * 3
    together = predictor.predict_proba([CARD, long_text, MORTGAGE])
    alone = np.vstack([predictor.predict_proba([t]) for t in (CARD, long_text, MORTGAGE)])
    np.testing.assert_array_equal(together, alone)


def test_auto_route_follows_threshold(predictor):
    p = predictor.predict([CARD])[0]
    assert p.auto_route == (p.confidence >= predictor.threshold)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_endpoint(client):
    r = client.post("/predict", json={"text": CARD})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "cards"
    assert body["team"] == "Cartes bancaires"
    assert 0 < body["confidence"] <= 1
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize("payload", [{"text": "too short"}, {"text": ""}, {}, {"txt": CARD}])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_endpoint(client):
    r = client.post("/predict/batch", json={"texts": [CARD, MORTGAGE]})
    assert r.status_code == 200
    assert [x["label"] for x in r.json()["results"]] == ["cards", "mortgage"]


def test_batch_limit(client):
    assert client.post("/predict/batch", json={"texts": [CARD] * 65}).status_code == 422


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Complaint Router" in r.text
