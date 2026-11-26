"""Préparation des données : de 1,7 million de réclamations aux jeux train / validation / test.

Étapes, en une seule lecture des fichiers (on ne charge jamais tout en mémoire) :
1. garder les réclamations de la période étudiée, et les ramener à nos 9 catégories ;
2. nettoyer le texte et écarter les textes trop courts ;
3. tirer au hasard un échantillon plafonné par catégorie et par période (échantillonnage
   « réservoir ») ;
4. retirer les doublons, et surtout tout texte de validation ou de test déjà présent dans la
   période d'entraînement : beaucoup de réclamations sont des modèles copiés-collés, et les
   laisser fausserait l'évaluation (fuite de données).

Usage : python -m complaint_router.data
"""

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import (
    MIN_CHARS,
    PER_CLASS,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    SEED,
    TEST_PERIOD,
    TRAIN_PERIOD,
    VAL_PERIOD,
)
from .labels import EXCLUDED_PRODUCTS, LABELS, label_for_product
from .text import clean_text

COLUMNS = ["Date received", "Product", "Consumer complaint narrative", "Complaint ID"]
SPLITS = ("train", "val", "test")
PERIODS = {"train": TRAIN_PERIOD, "val": VAL_PERIOD, "test": TEST_PERIOD}


def split_for_date(date: str) -> str | None:
    """Le jeu (train, val ou test) d'une réclamation selon sa date, ou None hors période."""
    day = date[:10]
    for name, (start, end) in PERIODS.items():
        if start <= day <= end:
            return name
    return None


class Reservoir:
    """Échantillon uniforme de taille fixe sur un flux de longueur inconnue (algorithme R).

    Chaque élément du flux a la même probabilité d'être gardé, sans connaître la longueur du
    flux à l'avance et sans le garder en mémoire.
    """

    def __init__(self, size: int, rng: random.Random):
        self.size = size
        self.rng = rng
        self.items: list = []
        self.seen = 0

    def add(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.size:
                self.items[j] = item


def text_hash(text: str) -> bytes:
    """Empreinte courte d'un texte, insensible à la casse, pour repérer les doublons."""
    return hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest()


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, oversample: float = 1.3):
    files = sorted(raw_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"Aucun fichier Parquet dans {raw_dir} : lancez d'abord `make download`")

    rng = random.Random(SEED)
    # Un peu plus que le plafond, pour compenser les doublons retirés ensuite
    reservoirs = {
        (split, label): Reservoir(int(PER_CLASS[split] * oversample), rng)
        for split in SPLITS
        for label in LABELS
    }
    train_hashes: set[bytes] = set()
    stats: Counter = Counter()
    unknown: Counter = Counter()
    natural = {split: Counter() for split in SPLITS}  # répartition réelle, avant plafonnement
    started = time.time()

    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20_000, columns=COLUMNS):
            cols = batch.to_pydict()
            for date, product, text, cid in zip(*(cols[c] for c in COLUMNS), strict=True):
                stats["lues"] += 1
                split = split_for_date(date or "")
                if split is None:
                    continue
                label = label_for_product(product)
                if label is None:
                    stats["produit_ecarte"] += 1
                    if product not in EXCLUDED_PRODUCTS:
                        unknown[product] += 1
                    continue
                text = clean_text(text or "")
                if len(text) < MIN_CHARS:
                    stats["texte_trop_court"] += 1
                    continue
                h = text_hash(text)
                if split == "train":
                    train_hashes.add(h)
                natural[split][label] += 1
                reservoirs[(split, label)].add((h, int(cid), date[:10], text))
        print(f"{path.name} : {stats['lues']:,} réclamations lues ({time.time() - started:.0f} s)")

    rows = []
    for split in SPLITS:
        seen: set[bytes] = set()
        for label in LABELS:
            kept = 0
            for h, cid, date, text in reservoirs[(split, label)].items:
                if h in seen:
                    stats[f"doublon_{split}"] += 1
                    continue
                if split != "train" and h in train_hashes:
                    stats[f"fuite_evitee_{split}"] += 1  # texte déjà vu pendant l'entraînement
                    continue
                if kept == PER_CLASS[split]:
                    break
                seen.add(h)
                kept += 1
                rows.append({"split": split, "complaint_id": cid, "date": date,
                             "label": label, "text": text})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        part = df[df.split == split].drop(columns="split").reset_index(drop=True)
        part.to_parquet(out_dir / f"{split}.parquet", index=False)

    summary = {
        "periodes": PERIODS,
        "plafond_par_categorie": PER_CLASS,
        "compteurs": dict(stats),
        "produits_inconnus": dict(unknown.most_common(20)),
        "repartition_reelle": {s: dict(natural[s].most_common()) for s in SPLITS},
        "exemples_gardes": {
            s: (df[df.split == s].label.value_counts()
                .reindex(LABELS).fillna(0).astype(int).to_dict())
            for s in SPLITS
        },
        "longueur_mots_mediane": int(df.text.str.split().str.len().median()),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for split in SPLITS:
        print(f"{split:5} : {int((df.split == split).sum()):>6,} exemples")
    print(json.dumps(summary["compteurs"], ensure_ascii=False))
    if unknown:
        print("Produits inconnus (à ajouter dans labels.py ?) :", dict(unknown.most_common(5)))
    return summary


if __name__ == "__main__":
    prepare()
