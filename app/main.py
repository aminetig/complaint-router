"""API de routage des réclamations.

- POST /predict        une réclamation : équipe destinataire, confiance, et décision de routage
- POST /predict/batch  jusqu'à 64 réclamations d'un coup
- GET  /health         état du service (pour Docker et la supervision)
- GET  /model          informations sur le modèle servi
- GET  /               page de démonstration
- GET  /docs           documentation interactive, générée par FastAPI

Lancer : uvicorn app.main:app --port 8000
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StringConstraints

from complaint_router.labels import TEAMS_FR
from complaint_router.predictor import Prediction, Predictor

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
STATIC = Path(__file__).resolve().parent / "static"

ComplaintText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=20,
                                                 max_length=20_000)]
EXAMPLE = ("I was charged an annual fee on my credit card even though the bank told me the card "
           "had no fee. I called twice and they refused to refund it.")


class ComplaintIn(BaseModel):
    text: ComplaintText = Field(examples=[EXAMPLE])


class BatchIn(BaseModel):
    texts: list[ComplaintText] = Field(min_length=1, max_length=64)


class Candidate(BaseModel):
    label: str
    team: str
    probability: float


class RoutingOut(BaseModel):
    label: str = Field(description="Catégorie prédite")
    team: str = Field(description="Équipe destinataire")
    confidence: float = Field(description="Probabilité de la catégorie prédite, de 0 à 1")
    auto_route: bool = Field(description="Vrai si la confiance dépasse le seuil de routage auto")
    decision: str
    top3: list[Candidate]


class PredictOut(RoutingOut):
    latency_ms: float
    model_version: str


class BatchOut(BaseModel):
    results: list[RoutingOut]
    latency_ms: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est chargé une seule fois, au démarrage, et partagé par toutes les requêtes
    threads = int(os.environ.get("ORT_THREADS", "0")) or None
    app.state.predictor = Predictor(MODEL_DIR, threads=threads)
    # Échauffement : le tout premier calcul d'ONNX Runtime est lent (300 ms mesurées contre 20).
    # On le fait au démarrage pour que le premier client n'attende pas.
    app.state.predictor.predict([EXAMPLE])
    yield


app = FastAPI(
    title="Complaint Router",
    description="Routage automatique des réclamations clients bancaires vers la bonne équipe.",
    version="1.0.0",
    lifespan=lifespan,
)


def to_out(p: Prediction) -> dict:
    decision = (f"Routée automatiquement vers « {p.team} »" if p.auto_route
                else "Confiance insuffisante : à vérifier par un conseiller")
    return {**p.__dict__, "decision": decision}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.predictor.meta["version"]}


@app.get("/model")
def model_info(request: Request):
    meta = request.app.state.predictor.meta
    return {
        "version": meta["version"],
        "encoder": meta["encoder"],
        "max_tokens": meta["max_tokens"],
        "threshold": meta["threshold"],
        "teams": TEAMS_FR,
    }


@app.post("/predict", response_model=PredictOut)
def predict(body: ComplaintIn, request: Request):
    predictor: Predictor = request.app.state.predictor
    start = time.perf_counter()
    prediction = predictor.predict([body.text])[0]
    return {**to_out(prediction), "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "model_version": predictor.meta["version"]}


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(body: BatchIn, request: Request):
    predictor: Predictor = request.app.state.predictor
    start = time.perf_counter()
    predictions = predictor.predict(body.texts)
    return {"results": [to_out(p) for p in predictions],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "model_version": predictor.meta["version"]}
