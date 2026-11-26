"""Modèle de référence : TF-IDF et régression logistique.

Avant un réseau de neurones, on mesure ce que fait une méthode simple, rapide et éprouvée. Si le
modèle de deep learning ne fait pas mieux, il ne vaut pas sa complexité.

TF-IDF transforme un texte en vecteur : chaque mot (ou paire de mots) reçoit un poids d'autant
plus fort qu'il est fréquent dans ce texte et rare dans les autres. La régression logistique
apprend ensuite un poids par mot et par catégorie.

Usage : python -m complaint_router.baseline
"""

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .config import MLFLOW_EXPERIMENT, MLFLOW_URI, PROCESSED_DIR, REPORTS_DIR, SEED
from .metrics import (
    choose_threshold,
    classification_metrics,
    coverage_curve,
    plot_confusion,
    real_mix_accuracy,
    real_shares,
    routing_metrics,
    save_report,
)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def build(c: float):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
        LogisticRegression(C=c, max_iter=3000, random_state=SEED),
    )


def main() -> dict:
    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Réglage de la régularisation C sur la validation (jamais sur le test)
    best = None
    for c in (1.0, 4.0, 16.0):
        model = build(c).fit(train.text, train.label)
        f1 = classification_metrics(val.label, model.predict(val.text))["macro_f1"]
        print(f"C={c:>4} : F1 macro validation = {f1:.4f}")
        if best is None or f1 > best[1]:
            best = (c, f1, model)
    c, _, model = best

    classes = list(model.classes_)
    proba_val = model.predict_proba(val.text)
    correct_val = np.array(classes)[proba_val.argmax(1)] == val.label.to_numpy()
    threshold = choose_threshold(proba_val.max(1), correct_val)

    started = time.perf_counter()
    proba = model.predict_proba(test.text)
    batch_ms = (time.perf_counter() - started) * 1000 / len(test)
    pred = np.array(classes)[proba.argmax(1)]
    correct = pred == test.label.to_numpy()

    single = test.text.head(200).tolist()
    started = time.perf_counter()
    for text in single:
        model.predict_proba([text])
    single_ms = (time.perf_counter() - started) * 1000 / len(single)

    metrics = classification_metrics(test.label, pred)
    report = {
        "modele": "TF-IDF (mots et paires de mots) + régression logistique",
        "parametres": {"C": c, "ngram_range": [1, 2], "max_features": 200_000},
        "texte_lu": "texte entier",
        "test": metrics,
        "justesse_flux_reel": real_mix_accuracy(metrics["per_class"], real_shares()),
        "routage": routing_metrics(proba.max(1), correct, threshold),
        "courbe_couverture": coverage_curve(proba.max(1), correct),
        "latence_ms": {"une_reclamation": round(single_ms, 2), "par_lot": round(batch_ms, 3)},
    }
    path = save_report("baseline", report)
    figure = REPORTS_DIR / "confusion_baseline.png"
    plot_confusion(test.label, pred, figure, "Référence TF-IDF : matrice de confusion (test)")

    with mlflow.start_run(run_name="tfidf-logreg"):
        mlflow.log_params(report["parametres"])
        mlflow.log_metrics({
            "test_macro_f1": metrics["macro_f1"],
            "test_accuracy": metrics["accuracy"],
            "part_automatisee": report["routage"]["part_automatisee"],
        })
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(figure))

    print(f"Test : F1 macro {metrics['macro_f1']:.4f}, justesse {metrics['accuracy']:.4f}")
    print(f"Routage : {report['routage']}")
    return report


if __name__ == "__main__":
    main()
