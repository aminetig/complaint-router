"""Téléchargement des réclamations (925 Mo).

Le CFPB a retiré le texte des réclamations de ses données publiques : en novembre 2026, ni son
API ni son export complet ne contiennent plus la colonne « Consumer complaint narrative ». On
utilise donc une copie publique faite en février 2024, sous licence CC0 (domaine public) :
1,69 million de réclamations avec leur texte.

Usage : python -m complaint_router.download
"""

from huggingface_hub import hf_hub_download

from .config import RAW_DIR

REPO = "BEE-spoke-data/consumer-finance-complaints"
FILES = [f"has-text/train-0000{i}-of-00003.parquet" for i in range(3)]


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=RAW_DIR)
        print(f"OK : {path}")


if __name__ == "__main__":
    download()
