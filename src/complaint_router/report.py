"""Tableau comparatif final et graphiques, à partir des rapports JSON de chaque étape.

Usage : python -m complaint_router.report  (écrit reports/RESULTS.md et deux graphiques)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import REPORTS_DIR  # noqa: E402
from .labels import LABELS, TEAMS_FR  # noqa: E402

BLUE, GREY, ORANGE = "#1F3A5F", "#9AA3AF", "#C2410C"


def load(name: str) -> dict | None:
    path = REPORTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(x: float) -> str:
    return dec(x * 100, 1) + " %"


def dec(x: float, digits: int = 3) -> str:
    """Nombre à virgule, à la française."""
    return f"{x:.{digits}f}".replace(".", ",")


def main() -> None:
    base, tr, ex = load("baseline"), load("transformer"), load("export")
    llms = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(REPORTS_DIR.glob("llm_*.json"))]
    data = load("data_summary")

    # 1. Courbe : justesse des cas automatisés selon la part automatisée
    served = "MiniLM affiné, version servie (ONNX 8 bits)"
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for curve, name, color in ((base["courbe_couverture"], "TF-IDF + régression logistique", GREY),
                               (ex["courbe_couverture_int8"], served, BLUE)):
        ax.plot([c * 100 for c in curve["couverture"]], [j * 100 for j in curve["justesse"]],
                label=name, color=color, linewidth=2)
        reach = curve["part_a_la_cible"] * 100
        ax.plot([reach], [95], "o", color=color)
        # Étiquette en bas à gauche du point : zone vide, sous les deux courbes
        ax.annotate(f"{reach:.0f} %", (reach, 95), xytext=(-30, -16), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold")
    ax.axhline(95, color=ORANGE, linestyle="--", linewidth=1)
    ax.text(99, 95.4, "objectif : 95 % de justesse", color=ORANGE, fontsize=9, ha="right")
    ax.set_xlabel("Part des réclamations routées automatiquement (%)")
    ax.set_ylabel("Justesse sur ces réclamations (%)")
    ax.set_ylim(70, 100.5)
    ax.legend(loc="lower left")
    ax.set_title("Plus le modèle est sûr de lui, plus il a raison")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "coverage.png", dpi=130)
    plt.close(fig)

    # 2. F1 par catégorie
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(LABELS))
    ax.barh([i + 0.2 for i in y], [base["test"]["per_class"][c]["f1"] for c in LABELS], 0.4,
            color=GREY, label="TF-IDF")
    ax.barh([i - 0.2 for i in y], [ex["test_int8"]["per_class"][c]["f1"] for c in LABELS], 0.4,
            color=BLUE, label=served)
    ax.set_yticks(list(y), [TEAMS_FR[c] for c in LABELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 sur le test")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "per_class_f1.png", dpi=130)
    plt.close(fig)

    # 3. Tableau
    lat = ex["latence_ms_une_reclamation"]
    rows = [
        ("TF-IDF + régression logistique", base["test"], base["justesse_flux_reel"],
         base["courbe_couverture"], base["routage"],
         f"{base['latence_ms']['une_reclamation']:.0f} ms", "–"),
        ("MiniLM affiné, PyTorch 32 bits", tr["test"], tr["justesse_flux_reel"],
         tr["courbe_couverture"], tr["routage"], f"{lat['pytorch_fp32']['p50']:.0f} ms",
         f"{ex['taille_mo']['pytorch_fp32']:.0f} Mo"),
        ("**MiniLM affiné, ONNX 8 bits (servi)**", ex["test_int8"], ex["justesse_flux_reel_int8"],
         ex["courbe_couverture_int8"], ex["routage_int8"], f"{lat['onnx_int8']['p50']:.0f} ms",
         f"{ex['taille_mo']['onnx_int8']:.0f} Mo"),
    ]
    n_test = f"{sum(data['exemples_gardes']['test'].values()):,}".replace(",", " ")
    lines = [
        "# Résultats",
        "",
        f"Test : {n_test} réclamations d'octobre 2023 à février 2024, jamais vues à "
        "l'entraînement, équilibrées entre les 9 catégories.",
        "",
        "| Modèle | F1 macro | Justesse attendue sur le vrai flux | Part routable à 95 % de "
        "justesse | Avec le seuil choisi sur la validation | Temps de réponse (médiane) "
        "| Taille |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, metrics, real, curve, routing, latency, size in rows:
        lines.append(
            f"| {name} | {dec(metrics['macro_f1'])} | {pct(real)} | "
            f"{pct(curve['part_a_la_cible'])} | {pct(routing['part_automatisee'])} routées, "
            f"{pct(routing['justesse_automatisee'])} justes | {latency} | {size} |"
        )
    lines += [
        "",
        "- **Part routable à 95 % de justesse :** la plus grande part des réclamations qu'on peut "
        "router sans humain en gardant 95 % de justesse, le seuil étant réglé sur le test. C'est "
        "la comparaison à armes égales entre les modèles (`coverage.png`).",
        "- **Avec le seuil choisi sur la validation :** ce qui se passe vraiment en production, où "
        "le seuil est fixé à l'avance. Entre l'été et l'hiver 2023, les réclamations changent un "
        "peu : sur le test, la justesse de ces cas passe sous les 95 % visés.",
        "- **Justesse attendue sur le vrai flux :** le test est équilibré, alors que 69 % des "
        "vraies réclamations concernent les rapports de crédit. Cette colonne pondère le rappel "
        "de chaque catégorie par sa part réelle.",
    ]
    for llm in llms:
        ours = llm["notre_modele_memes_reclamations"]
        res = llm["llm_resultats"]
        lines += [
            "",
            f"## LLM sans entraînement : {llm['llm']}, sur {llm['reclamations']} réclamations",
            "",
            "| | F1 macro | Justesse | Temps par réclamation (médiane) |",
            "| --- | --- | --- | --- |",
            f"| {llm['llm']} (Ollama, local) | {dec(res['macro_f1'])} | {pct(res['accuracy'])} | "
            f"{dec(res['secondes_par_reclamation']['p50'], 1)} s |",
            f"| MiniLM affiné, ONNX 8 bits | {dec(ours['macro_f1'])} | {pct(ours['accuracy'])} | "
            f"{ours['millisecondes_par_reclamation']['p50']:.0f} ms |",
            "",
            f"Notre modèle répond {llm['rapport_de_vitesse']} fois plus vite, "
            "sur le même processeur.",
        ]
    lines += ["", f"Machine : {ex['machine']}.", ""]
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
