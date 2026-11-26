"""Prédiction en production, avec le modèle exporté en ONNX.

ONNX est un format standard de réseau de neurones. ONNX Runtime l'exécute sans PyTorch : l'image
Docker de l'API pèse quelques centaines de Mo au lieu de plusieurs Go, et démarre en une seconde.
Le découpage du texte en jetons utilise la bibliothèque `tokenizers` (écrite en Rust), avec
exactement le même vocabulaire qu'à l'entraînement.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .labels import TEAMS_FR
from .text import clean_text


@dataclass
class Prediction:
    label: str
    team: str
    confidence: float
    auto_route: bool
    top3: list[dict]


class Predictor:
    def __init__(self, model_dir: str | Path, threads: int | None = None):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        self.labels: list[str] = self.meta["labels"]
        self.threshold: float = self.meta["threshold"]

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.meta["max_tokens"])

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_dir / self.meta["onnx_file"]), options, providers=["CPUExecutionProvider"]
        )

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probabilités des catégories, une réclamation à la fois.

        Pas de lot avec remplissage (padding) : en quantification dynamique, l'échelle des
        entiers 8 bits est calculée sur tout le tenseur d'entrée. Dans un lot, les autres textes
        et le remplissage changeraient le score (jusqu'à plusieurs points mesurés). Ainsi, une
        réclamation reçoit toujours le même score, et la même décision, quel que soit le lot.
        """
        if not texts:
            return np.empty((0, len(self.labels)), dtype=np.float32)
        return np.vstack([self._proba_one(text) for text in texts])

    def _proba_one(self, text: str) -> np.ndarray:
        encoding = self.tokenizer.encode(clean_text(text))
        inputs = {"input_ids": np.array([encoding.ids], dtype=np.int64),
                  "attention_mask": np.array([encoding.attention_mask], dtype=np.int64)}
        logits = self.session.run(["logits"], inputs)[0]
        logits = logits - logits.max(axis=1, keepdims=True)  # softmax numériquement stable
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, texts: list[str]) -> list[Prediction]:
        results = []
        for proba in self.predict_proba(texts):
            order = np.argsort(-proba)
            best = self.labels[order[0]]
            confidence = float(proba[order[0]])
            results.append(Prediction(
                label=best,
                team=TEAMS_FR[best],
                confidence=round(confidence, 4),
                # Sous le seuil, la réclamation part vers un humain plutôt que d'être mal routée
                auto_route=confidence >= self.threshold,
                top3=[
                    {"label": self.labels[i], "team": TEAMS_FR[self.labels[i]],
                     "probability": round(float(proba[i]), 4)}
                    for i in order[:3]
                ],
            ))
        return results
