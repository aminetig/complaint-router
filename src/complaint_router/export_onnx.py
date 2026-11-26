"""Export du modèle en ONNX, quantification en 8 bits, puis vérifications.

La quantification stocke chaque poids sur 1 octet (un entier de -128 à 127, avec un facteur
d'échelle) au lieu de 4 (un nombre à virgule) : le modèle est environ 4 fois plus petit, et les
calculs sur entiers sont plus rapides sur processeur. On vérifie que la qualité ne baisse presque
pas, et on mesure le temps de réponse des trois versions sur la même machine.

Usage : python -m complaint_router.export_onnx
"""

import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from . import __version__
from .config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR
from .labels import LABELS
from .metrics import (
    choose_threshold,
    classification_metrics,
    coverage_curve,
    real_mix_accuracy,
    real_shares,
    routing_metrics,
    save_report,
)
from .model import ComplaintClassifier
from .predictor import Predictor

ONNX_FILE = "complaint_router.int8.onnx"
THREADS = 2  # même nombre de fils pour les trois versions : comparaison de vitesse équitable


def percentiles(values: list[float]) -> dict:
    return {"p50": round(float(np.percentile(values, 50)), 1),
            "p95": round(float(np.percentile(values, 95)), 1)}


def export() -> dict:
    torch.set_num_threads(THREADS)
    meta = json.loads((ARTIFACTS_DIR / "model_meta.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR / "tokenizer")
    model = ComplaintClassifier(meta["encoder"], len(LABELS))
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "best.pt", map_location="cpu"))
    model.eval()

    fp32_path = ARTIFACTS_DIR / "complaint_router.fp32.onnx"
    sample = tokenizer(["My card was charged twice", "I never received my loan statement"],
                       padding=True, return_tensors="pt")
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # Taille du lot et longueur du texte variables
        dynamic_axes={"input_ids": {0: "batch", 1: "tokens"},
                      "attention_mask": {0: "batch", 1: "tokens"},
                      "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    MODELS_DIR.mkdir(exist_ok=True)
    int8_path = MODELS_DIR / ONNX_FILE
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # Fichiers servis par l'API
    shutil.copy(ARTIFACTS_DIR / "tokenizer" / "tokenizer.json", MODELS_DIR / "tokenizer.json")
    served_meta = {
        "version": __version__,
        "encoder": meta["encoder"],
        "labels": list(LABELS),
        "max_tokens": meta["max_tokens"],
        "threshold": round(meta["threshold"], 4),
        "onnx_file": ONNX_FILE,
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(served_meta, indent=2), encoding="utf-8")

    int8 = Predictor(MODELS_DIR, threads=THREADS)

    # La quantification change un peu les scores de confiance : la version servie a donc son
    # propre seuil, choisi comme toujours sur la validation (jamais sur le test)
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    val_proba = int8.predict_proba(val.text.tolist())
    val_correct = np.array(LABELS)[val_proba.argmax(1)] == val.label.to_numpy()
    served_meta["threshold"] = round(choose_threshold(val_proba.max(1), val_correct), 4)
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(served_meta, indent=2), encoding="utf-8")
    int8.threshold = served_meta["threshold"]

    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    texts = test.text.tolist()
    fp32 = Predictor(_fp32_dir(fp32_path, served_meta), threads=THREADS)

    # 1. Fidélité de l'export : PyTorch et ONNX 32 bits doivent donner les mêmes scores
    enc = tokenizer(texts[:64], truncation=True, max_length=meta["max_tokens"], padding=True,
                    return_tensors="pt")
    with torch.no_grad():
        torch_proba = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), -1).numpy()
    max_gap = float(np.abs(torch_proba - fp32.predict_proba(texts[:64])).max())

    # 2. Qualité du modèle quantifié sur tout le test
    proba = int8.predict_proba(texts)  # une à une, comme dans l'API
    pred = np.array(LABELS)[proba.argmax(1)]
    truth = test.label.to_numpy()
    torch_pred = np.array(LABELS)[np.load(ARTIFACTS_DIR / "test_proba_torch.npy").argmax(1)]
    metrics = classification_metrics(truth, pred)

    # 3. Temps de réponse pour UNE réclamation (le cas d'une API), sur 200 textes
    def single_latency(fn) -> dict:
        for text in texts[:5]:
            fn(text)  # échauffement
        times = []
        for text in texts[:200]:
            start = time.perf_counter()
            fn(text)
            times.append((time.perf_counter() - start) * 1000)
        return percentiles(times)

    def torch_one(text: str):
        e = tokenizer([text], truncation=True, max_length=meta["max_tokens"], return_tensors="pt")
        with torch.no_grad():
            model(e["input_ids"], e["attention_mask"])

    sizes = {
        "pytorch_fp32": (ARTIFACTS_DIR / "best.pt").stat().st_size,
        "onnx_fp32": fp32_path.stat().st_size,
        "onnx_int8": int8_path.stat().st_size,
    }
    report = {
        "ecart_max_pytorch_onnx": max_gap,
        "test_int8": metrics,
        "justesse_flux_reel_int8": real_mix_accuracy(metrics["per_class"], real_shares()),
        "accord_avec_pytorch": round(float((pred == torch_pred).mean()), 4),
        "routage_int8": routing_metrics(proba.max(1), pred == truth, served_meta["threshold"]),
        "courbe_couverture_int8": coverage_curve(proba.max(1), pred == truth),
        "taille_mo": {k: round(v / 1e6, 1) for k, v in sizes.items()},
        "latence_ms_une_reclamation": {
            "pytorch_fp32": single_latency(torch_one),
            "onnx_fp32": single_latency(lambda t: fp32.predict_proba([t])),
            "onnx_int8": single_latency(lambda t: int8.predict_proba([t])),
        },
        "machine": "Intel Core i5-7300U (2 cœurs, 2017), sans carte graphique",
    }
    save_report("export", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("test_int8", "courbe_couverture_int8")},
                     indent=2, ensure_ascii=False))
    print(f"Test int8 : F1 macro {metrics['macro_f1']:.4f}, justesse {metrics['accuracy']:.4f}, "
          f"part routable à 95 % : {report['courbe_couverture_int8']['part_a_la_cible']:.1%}")
    return report


def _fp32_dir(fp32_path, served_meta):
    """Dossier temporaire au format attendu par Predictor, pour la version 32 bits."""
    folder = ARTIFACTS_DIR / "onnx_fp32"
    folder.mkdir(exist_ok=True)
    shutil.copy(MODELS_DIR / "tokenizer.json", folder / "tokenizer.json")
    target = folder / fp32_path.name
    if not target.exists() or target.stat().st_mtime < fp32_path.stat().st_mtime:
        shutil.copy(fp32_path, target)
    (folder / "model_meta.json").write_text(
        json.dumps({**served_meta, "onnx_file": fp32_path.name}), encoding="utf-8"
    )
    return folder


if __name__ == "__main__":
    export()
