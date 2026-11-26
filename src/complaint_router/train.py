"""Affinage du modèle en PyTorch, avec une boucle d'entraînement écrite à la main.

À chaque pas : le modèle prédit un lot de réclamations, on mesure son erreur (entropie croisée),
on calcule dans quel sens bouger chaque poids pour la réduire (rétropropagation), puis
l'optimiseur AdamW déplace les poids d'un petit pas. Le pas grandit au début (« warmup ») puis
diminue jusqu'à zéro, ce qui stabilise l'apprentissage d'un modèle pré-entraîné.

À la fin de chaque passage complet sur les données (une « époque »), on mesure le F1 sur la
validation et on garde la meilleure version (arrêt anticipé si ça ne progresse plus).

Usage :
  python -m complaint_router.train                    # entraînement complet
  python -m complaint_router.train --max-steps 20     # test de vitesse
"""

import argparse
import json
import os
import random
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import (
    ARTIFACTS_DIR,
    ENCODER,
    MAX_TOKENS,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEED,
)
from .labels import LABELS
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(ids: list[list[int]], labels: list[int], pad_id: int, batch_size: int,
                shuffle: bool) -> DataLoader:
    """Lots de taille fixe, complétés (padding) seulement jusqu'au texte le plus long du lot."""
    items = list(zip(ids, labels, strict=True))

    def collate(batch):
        longest = max(len(x) for x, _ in batch)
        input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, (x, _) in enumerate(batch):
            input_ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return input_ids, mask, torch.tensor([y for _, y in batch])

    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=generator)


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for input_ids, mask, _ in loader:
        logits = model(input_ids.to(device), mask.to(device))
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def plot_training(history: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(history["step"], history["loss"], color="#1F3A5F")
    a.set_title("Erreur d'entraînement (moyenne glissante)")
    a.set_xlabel("pas")
    b.plot(range(1, len(history["val_f1"]) + 1), history["val_f1"], marker="o", color="#1F3A5F")
    b.set_title("F1 macro sur la validation")
    b.set_xlabel("époque")
    b.set_xticks(range(1, len(history["val_f1"]) + 1))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train(args) -> dict:
    from .model import ComplaintClassifier

    seed_everything(SEED)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_to_id = {label: i for i, label in enumerate(LABELS)}

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    splits = {s: pd.read_parquet(PROCESSED_DIR / f"{s}.parquet") for s in ("train", "val", "test")}
    if args.train_limit:
        splits["train"] = splits["train"].sample(n=args.train_limit, random_state=SEED)
    loaders = {}
    for name, df in splits.items():
        ids = tokenizer(df.text.tolist(), truncation=True, max_length=args.max_tokens)["input_ids"]
        y = [label_to_id[label] for label in df.label]
        size = args.batch_size if name == "train" else 64
        loaders[name] = make_loader(ids, y, tokenizer.pad_token_id, size, shuffle=name == "train")

    model = ComplaintClassifier(args.encoder, len(LABELS), dropout=args.dropout).to(device)
    # Pas de « weight decay » sur les biais et les normalisations (tenseurs à 1 dimension) :
    # c'est l'usage pour les Transformers
    decay, no_decay = [], []
    for param in model.parameters():
        (no_decay if param.ndim == 1 else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    steps_per_epoch = len(loaders["train"])
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss()

    params = {
        "encoder": args.encoder, "max_tokens": args.max_tokens, "batch_size": args.batch_size,
        "lr": args.lr, "epochs": args.epochs, "dropout": args.dropout,
        "train_size": len(splits["train"]), "device": str(device), "threads": args.threads,
    }
    print(json.dumps(params), flush=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    best_path = ARTIFACTS_DIR / "best.pt"
    history = {"step": [], "loss": [], "val_f1": []}
    best_f1, bad_epochs, step = -1.0, 0, 0
    started = time.time()

    with mlflow.start_run(run_name=f"finetune-{args.encoder.split('/')[-1]}"):
        mlflow.log_params(params)
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = None
            for input_ids, mask, y in loaders["train"]:
                logits = model(input_ids.to(device), mask.to(device))
                loss = loss_fn(logits, y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # évite les pas trop grands
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()
                if step % 25 == 0 or step == total_steps:
                    elapsed = time.time() - started
                    rate = step * args.batch_size / elapsed
                    eta = (total_steps - step) * elapsed / step / 60
                    print(f"époque {epoch} pas {step}/{total_steps} erreur {running:.4f} "
                          f"{rate:.1f} textes/s, reste ~{eta:.0f} min", flush=True)
                    history["step"].append(step)
                    history["loss"].append(round(running, 4))
                    mlflow.log_metric("train_loss", running, step=step)
                if args.max_steps and step >= args.max_steps:
                    print(f"Vitesse : {step * args.batch_size / (time.time() - started):.1f} "
                          "textes/s", flush=True)
                    return {}

            proba = predict_proba(model, loaders["val"], device)
            val_true = [LABELS[i] for _, _, y in loaders["val"] for i in y.tolist()]
            f1 = classification_metrics(val_true, [LABELS[i] for i in proba.argmax(1)])["macro_f1"]
            history["val_f1"].append(f1)
            mlflow.log_metric("val_macro_f1", f1, step=epoch)
            print(f"== époque {epoch} : F1 macro validation {f1:.4f}", flush=True)
            if f1 > best_f1:
                best_f1, bad_epochs = f1, 0
                torch.save(model.state_dict(), best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print("Arrêt anticipé : la validation ne progresse plus", flush=True)
                    break

        # Évaluation finale avec la meilleure version
        model.load_state_dict(torch.load(best_path, map_location=device))
        val_proba = predict_proba(model, loaders["val"], device)
        val_true = np.array([LABELS[i] for _, _, y in loaders["val"] for i in y.tolist()])
        val_pred = np.array(LABELS)[val_proba.argmax(1)]
        threshold = choose_threshold(val_proba.max(1), val_pred == val_true)

        test_proba = predict_proba(model, loaders["test"], device)
        test_true = splits["test"].label.to_numpy()
        test_pred = np.array(LABELS)[test_proba.argmax(1)]
        correct = test_pred == test_true
        metrics = classification_metrics(test_true, test_pred)
        np.save(ARTIFACTS_DIR / "test_proba_torch.npy", test_proba)

        report = {
            "modele": f"{args.encoder}, affiné en PyTorch",
            "parametres": params,
            "texte_lu": f"{args.max_tokens} premiers jetons (environ 90 mots)",
            "duree_entrainement_min": round((time.time() - started) / 60, 1),
            "meilleur_f1_validation": best_f1,
            "test": metrics,
            "justesse_flux_reel": real_mix_accuracy(metrics["per_class"], real_shares()),
            "routage": routing_metrics(test_proba.max(1), correct, threshold),
            "courbe_couverture": coverage_curve(test_proba.max(1), correct),
            "historique": history,
        }
        path = save_report("transformer", report)
        plot_confusion(test_true, test_pred, REPORTS_DIR / "confusion_transformer.png",
                       "MiniLM affiné : matrice de confusion (test)")
        plot_training(history, REPORTS_DIR / "training_curve.png")
        mlflow.log_metrics({
            "test_macro_f1": metrics["macro_f1"], "test_accuracy": metrics["accuracy"],
            "part_automatisee": report["routage"]["part_automatisee"],
        })
        for artifact in (path, REPORTS_DIR / "confusion_transformer.png",
                         REPORTS_DIR / "training_curve.png"):
            mlflow.log_artifact(str(artifact))

    tokenizer.save_pretrained(ARTIFACTS_DIR / "tokenizer")
    meta = {"encoder": args.encoder, "labels": list(LABELS), "max_tokens": args.max_tokens,
            "threshold": threshold, "val_macro_f1": best_f1}
    (ARTIFACTS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Test : F1 macro {metrics['macro_f1']:.4f}, justesse {metrics['accuracy']:.4f}")
    print(f"Routage : {report['routage']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Un fil par cœur physique : avec l'hyperthreading, deux fils sur un même cœur se gênent
    # (mesuré : 4,1 textes/s avec 2 fils contre 3,2 avec 4 sur un processeur 2 cœurs)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--train-limit", type=int, default=0, help="sous-échantillon (tests)")
    parser.add_argument("--max-steps", type=int, default=0, help="arrêt après N pas (vitesse)")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
