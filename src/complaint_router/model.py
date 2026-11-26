"""Le réseau de neurones : un encodeur de texte pré-entraîné, suivi d'une couche de classification.

- L'encodeur (MiniLM, 6 couches de Transformer, 22 millions de paramètres) a déjà appris le sens
  des phrases anglaises sur des millions de textes. Il transforme chaque mot en un vecteur de 384
  nombres qui tient compte du contexte.
- On fait la moyenne de ces vecteurs (« mean pooling ») pour obtenir un seul vecteur par texte.
- Une couche linéaire transforme ce vecteur en 9 scores, un par catégorie.

L'affinage (fine-tuning) ajuste tous ces poids, encodeur compris, sur nos réclamations.
"""

import torch
from torch import nn
from transformers import AutoModel


class ComplaintClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Moyenne des vecteurs des vrais mots, sans les positions de remplissage (padding)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(self.dropout(pooled))  # scores bruts (logits), un par catégorie
