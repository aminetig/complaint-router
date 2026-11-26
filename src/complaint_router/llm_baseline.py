"""Comparaison avec un LLM généraliste utilisé sans entraînement (« zero-shot »).

Question posée : faut-il entraîner un petit modèle spécialisé, ou demander simplement à un LLM ?
On donne au LLM la liste des équipes avec une description, puis la réclamation. Sa réponse est
contrainte par un schéma JSON : il ne peut répondre qu'une des 9 catégories.

Le LLM tourne en local avec Ollama : les réclamations, qui contiennent des données personnelles,
ne quittent pas la machine. C'est la contrainte d'une banque, qui ne peut pas envoyer ses données
clients à un service externe sans précautions.

Usage : python -m complaint_router.llm_baseline [--model qwen2.5:3b] [--per-class 20]
Prérequis : Ollama installé et lancé, et `ollama pull qwen2.5:3b`.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from .config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR, SEED
from .labels import DESCRIPTIONS_EN, LABELS
from .metrics import classification_metrics, save_report
from .predictor import Predictor

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_CHARS = 1000  # environ 220 jetons : plus que les 128 lus par notre modèle

SYSTEM = (
    "You route customer complaints received by a bank to the team that handles them.\n"
    "Teams:\n"
    + "\n".join(f"- {label}: {DESCRIPTIONS_EN[label]}" for label in LABELS)
    + "\nRead the complaint and answer with the id of the single best team."
    # Indispensable avec une sortie contrainte : sans cette consigne, le modèle ne « sait » pas
    # qu'il doit écrire du JSON, et la contrainte déforme sa réponse (voir docs/journal.md)
    + '\nRespond in JSON with this exact shape: {"team": "<team id>"}'
)
SCHEMA = {
    "type": "object",
    "properties": {"team": {"type": "string", "enum": list(LABELS)}},
    "required": ["team"],
}


def classify(text: str, model: str) -> dict:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "system": SYSTEM,  # identique à chaque appel : Ollama réutilise son calcul
            "prompt": f"Complaint:\n{text[:MAX_CHARS]}",
            "format": SCHEMA,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 24, "seed": SEED},
        },
        timeout=900,
    )
    response.raise_for_status()
    data = response.json()
    try:
        team = json.loads(data["response"])["team"]
    except (json.JSONDecodeError, KeyError):
        team = "invalid"
    return {
        "team": team if team in LABELS else "invalid",
        "seconds": data.get("total_duration", 0) / 1e9,
        "prompt_tokens": data.get("prompt_eval_count", 0),
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()

    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    sample = (test.groupby("label").sample(n=args.per_class, random_state=SEED)
              .reset_index(drop=True))

    # Reprise possible : les réponses déjà obtenues sont gardées dans un fichier. Son nom contient
    # une empreinte du prompt : si le prompt change, les anciennes réponses ne sont pas reprises.
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    prompt_id = hashlib.blake2b(f"{SYSTEM}{SCHEMA}{MAX_CHARS}".encode(), digest_size=4).hexdigest()
    cache_path = ARTIFACTS_DIR / f"llm_{args.model.replace(':', '_')}_{prompt_id}.jsonl"
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["complaint_id"]] = row

    started = time.time()
    with cache_path.open("a", encoding="utf-8") as cache:
        for i, row in enumerate(sample.itertuples(), start=1):
            if row.complaint_id in done:
                continue
            result = {"complaint_id": row.complaint_id, **classify(row.text, args.model)}
            done[row.complaint_id] = result
            cache.write(json.dumps(result) + "\n")
            cache.flush()
            if i % 10 == 0:
                print(f"{i}/{len(sample)} ({time.time() - started:.0f} s)", flush=True)

    results = [done[cid] for cid in sample.complaint_id]
    llm_pred = [r["team"] for r in results]
    llm_seconds = [r["seconds"] for r in results]

    # Notre modèle, sur exactement les mêmes réclamations
    predictor = Predictor(MODELS_DIR, threads=2)  # Ollama utilise lui aussi les 2 cœurs
    ours_ms = []
    ours_pred = []
    for text in sample.text:
        start = time.perf_counter()
        ours_pred.append(predictor.predict([text])[0].label)
        ours_ms.append((time.perf_counter() - start) * 1000)

    llm_metrics = classification_metrics(sample.label, llm_pred)
    ours_metrics = classification_metrics(sample.label, ours_pred)
    report = {
        "llm": args.model,
        "reclamations": len(sample),
        "par_categorie": args.per_class,
        "texte_lu_llm": f"{MAX_CHARS} premiers caractères",
        "llm_resultats": {
            "macro_f1": llm_metrics["macro_f1"],
            "accuracy": llm_metrics["accuracy"],
            "reponses_invalides": llm_pred.count("invalid"),
            "secondes_par_reclamation": {
                "p50": round(float(np.percentile(llm_seconds, 50)), 2),
                "p95": round(float(np.percentile(llm_seconds, 95)), 2),
            },
            "jetons_lus_moyenne": round(float(np.mean([r["prompt_tokens"] for r in results]))),
            "par_categorie": {k: v["f1"] for k, v in llm_metrics["per_class"].items()},
        },
        "notre_modele_memes_reclamations": {
            "macro_f1": ours_metrics["macro_f1"],
            "accuracy": ours_metrics["accuracy"],
            "millisecondes_par_reclamation": {
                "p50": round(float(np.percentile(ours_ms, 50)), 1),
                "p95": round(float(np.percentile(ours_ms, 95)), 1),
            },
            "par_categorie": {k: v["f1"] for k, v in ours_metrics["per_class"].items()},
        },
        "rapport_de_vitesse": round(
            float(np.median(llm_seconds)) * 1000 / float(np.median(ours_ms))
        ),
        "machine": "Intel Core i5-7300U (2 cœurs, 2017), sans carte graphique",
    }
    save_report(f"llm_{args.model.replace(':', '_').replace('.', '_')}", report)
    print(json.dumps({k: v for k, v in report.items() if "par_categorie" not in str(k)},
                     indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
