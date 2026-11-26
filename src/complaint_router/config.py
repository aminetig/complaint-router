"""Chemins et paramètres du projet."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"  # fichiers Parquet téléchargés (non versionnés)
PROCESSED_DIR = ROOT / "data" / "processed"  # jeux train / val / test (non versionnés)
ARTIFACTS_DIR = ROOT / "artifacts"  # points de sauvegarde PyTorch (non versionnés)
MODELS_DIR = ROOT / "models"  # modèle servi par l'API : ONNX, tokenizer, métadonnées (versionné)
REPORTS_DIR = ROOT / "reports"  # métriques et graphiques (versionnés)
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "complaint-router"

SEED = 42

# Découpage dans le temps : on apprend sur le passé et on évalue sur une période plus récente,
# comme en production. Un découpage aléatoire mélangerait les époques et surestimerait la qualité.
TRAIN_PERIOD = ("2022-01-01", "2023-06-30")
VAL_PERIOD = ("2023-07-01", "2023-09-30")
TEST_PERIOD = ("2023-10-01", "2024-02-29")

# Exemples gardés par catégorie : les catégories sont très déséquilibrées (les rapports de crédit
# font plus de la moitié des réclamations), on plafonne donc chacune pour équilibrer.
PER_CLASS = {"train": 1200, "val": 250, "test": 500}
MIN_CHARS = 80  # textes trop courts pour être exploitables

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 128
