"""Mesures de qualité communes à tous les modèles, pour les comparer à armes égales."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import REPORTS_DIR
from .labels import LABELS, TEAMS_FR

TARGET_ACCURACY = 0.95  # justesse exigée sur les réclamations routées sans humain


def classification_metrics(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro",
                                         zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i, label in enumerate(LABELS)
        },
    }


def real_mix_accuracy(per_class: dict, shares: dict) -> float:
    """Justesse attendue sur le vrai flux, où les catégories ne sont pas équilibrées.

    Le jeu de test est équilibré (même nombre d'exemples par catégorie). En production, une
    catégorie représente une part p du flux : la justesse attendue est la somme des rappels
    pondérés par ces parts.
    """
    total = sum(shares.values())
    return round(sum(per_class[c]["recall"] * shares.get(c, 0) / total for c in LABELS), 4)


def choose_threshold(confidence: np.ndarray, correct: np.ndarray, target=TARGET_ACCURACY) -> float:
    """Seuil de confiance le plus bas qui garde la justesse des cas automatisés à la cible.

    On trie les prédictions de la plus sûre à la moins sûre, et on cherche jusqu'où on peut
    descendre en gardant `target` de justesse parmi celles qu'on automatise. Choisi sur la
    validation, appliqué tel quel au test.
    """
    order = np.argsort(-confidence)
    cumulative = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(cumulative >= target)
    if len(ok) == 0:
        return 1.0
    return float(confidence[order][ok.max()])


def routing_metrics(confidence: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    """Ce que vit le métier : part des réclamations routées seules, et leur justesse."""
    auto = confidence >= threshold
    return {
        "seuil": round(threshold, 4),
        "part_automatisee": round(float(auto.mean()), 4),
        "justesse_automatisee": round(float(correct[auto].mean()) if auto.any() else 0.0, 4),
        "vers_un_humain": round(float(1 - auto.mean()), 4),
    }


def coverage_curve(confidence: np.ndarray, correct: np.ndarray, points: int = 50,
                   target=TARGET_ACCURACY) -> dict:
    """Justesse des cas automatisés selon la part automatisée, du plus sûr au moins sûr.

    `part_a_la_cible` : la plus grande part qu'on pourrait automatiser en gardant `target` de
    justesse, si le seuil était réglé directement sur ce jeu. Sert à comparer deux modèles à
    justesse égale ; le seuil réellement utilisé, lui, est choisi sur la validation.
    """
    order = np.argsort(-confidence)
    cumulative = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    idx = np.linspace(0, len(order) - 1, points).astype(int)
    ok = np.flatnonzero(cumulative >= target)
    return {
        "couverture": [round(float((i + 1) / len(order)), 4) for i in idx],
        "justesse": [round(float(cumulative[i]), 4) for i in idx],
        "part_a_la_cible": round(float((ok.max() + 1) / len(order)), 4) if len(ok) else 0.0,
    }


def plot_confusion(y_true, y_pred, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS), normalize="true")
    names = [TEAMS_FR[label] for label in LABELS]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "#1a1a1a")
    ax.set_xlabel("Catégorie prédite")
    ax.set_ylabel("Vraie catégorie")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_report(name: str, report: dict) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def real_shares() -> dict:
    """Répartition réelle des catégories sur la période de test (calculée par `data.py`)."""
    summary = json.loads((REPORTS_DIR / "data_summary.json").read_text(encoding="utf-8"))
    return summary["repartition_reelle"]["test"]
