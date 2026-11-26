"""Publie la démo en ligne, gratuitement, sur Hugging Face Spaces.

Spaces construit l'image Docker du projet et la sert à l'adresse
https://huggingface.co/spaces/<compte>/complaint-router.

Prérequis :
1. un compte sur https://huggingface.co ;
2. un jeton d'accès en écriture : Settings > Access Tokens > Create new token (Write) ;
3. ce jeton dans la variable HF_TOKEN.

Usage : make space HF_TOKEN=hf_xxx
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "Dockerfile",
    "requirements-serve.txt",
    "app/__init__.py",
    "app/main.py",
    "app/static/index.html",
    "src/complaint_router/__init__.py",
    "src/complaint_router/labels.py",
    "src/complaint_router/text.py",
    "src/complaint_router/predictor.py",
    "models/model_meta.json",
    "models/tokenizer.json",
]

CARD = """---
title: Complaint Router
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
license: mit
short_description: Routage des réclamations bancaires par NLP (MiniLM, ONNX)
---

Démo du projet [complaint-router](https://github.com/aminetig/complaint-router) : un modèle
MiniLM affiné en PyTorch sur de vraies réclamations, quantifié en 8 bits et servi par ONNX
Runtime derrière une API FastAPI.
"""


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Variable HF_TOKEN manquante : voir l'en-tête de ce fichier")
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/complaint-router"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
    onnx_file = meta["onnx_file"]
    for path in [*FILES, f"models/{onnx_file}"]:
        api.upload_file(path_or_fileobj=str(ROOT / path), path_in_repo=path, repo_id=repo_id,
                        repo_type="space")
        print(f"envoyé : {path}")
    api.upload_file(path_or_fileobj=CARD.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space")
    print(f"Démo : https://huggingface.co/spaces/{repo_id} (construction : 2 à 5 minutes)")


if __name__ == "__main__":
    main()
