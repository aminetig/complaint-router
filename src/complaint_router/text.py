"""Nettoyage du texte des réclamations.

La même fonction sert à l'entraînement et à l'API : le modèle voit toujours le texte dans le même
format. Un écart entre les deux (« training-serving skew ») ferait baisser la qualité en
production sans que les tests de l'entraînement le voient.
"""

import re

# Le CFPB masque les informations personnelles par des X : « XXXX », « XX/XX/XXXX »...
# Une suite de masques devient un seul « XXXX » : ça libère de la place pour le vrai contenu,
# car le modèle ne lit que les premiers mots du texte.
_MASKS = re.compile(r"(?:\bX{2,}\b[\s/\-.,]*)+")
_AMOUNT = re.compile(r"\{\$([\d,]+(?:\.\d+)?)\}")  # « {$1200.00} » devient « $1200.00 »
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _AMOUNT.sub(r"$\1", text)
    text = _MASKS.sub("XXXX ", text)
    return _SPACES.sub(" ", text).strip()
