# Fiche du modèle

Ce que fait le modèle, sur quelles données il a appris, ce qu'il vaut et où il se trompe.

## En bref

| | |
| --- | --- |
| Tâche | Lire une réclamation client (en anglais) et choisir l'équipe qui doit la traiter, parmi 9 |
| Modèle | `sentence-transformers/all-MiniLM-L6-v2` (6 couches, 22,7 millions de paramètres), affiné en PyTorch, avec une couche de classification ajoutée |
| Version servie | ONNX, quantifiée en entiers 8 bits : 23 Mo, 43 ms par réclamation sur un processeur de portable de 2017. Chaque réclamation est traitée seule, pour que son score ne dépende jamais des autres (voir le journal) |
| Texte lu | Les 128 premiers jetons, soit environ 90 mots. La moitié des réclamations font plus de 149 mots |
| Sortie | L'équipe, un score de confiance, les 3 équipes les plus probables, et la décision : routage automatique ou conseiller |
| Version | 1.0.0 |

## Les 9 équipes

| Identifiant | Équipe | Part du vrai flux (oct. 2023 à fév. 2024) |
| --- | --- | --- |
| `credit_reporting` | Rapports de crédit | 69 % |
| `cards` | Cartes bancaires | 8 % |
| `debt_collection` | Recouvrement | 7 % |
| `bank_account` | Comptes bancaires | 6 % |
| `mortgage` | Prêts immobiliers | 3 % |
| `student_loan` | Prêts étudiants | 2 % |
| `money_transfer` | Virements et paiements | 2 % |
| `vehicle_loan` | Crédits auto | 1 % |
| `personal_loan` | Prêts personnels | 1 % |

## Données

- **Source :** les réclamations publiées par le CFPB, l'autorité américaine de protection des consommateurs en matière financière. Copie publique de février 2024, sous licence CC0 (domaine public), avec 1,69 million de réclamations.
- **Étiquette :** le produit choisi **par le consommateur** quand il dépose sa réclamation, regroupé en 9 équipes. Ce n'est pas une étiquette vérifiée par un expert (voir « Limites »).
- **Découpage dans le temps :**

  | Jeu | Période | Réclamations |
  | --- | --- | --- |
  | Entraînement | janv. 2022 à juin 2023 | 10 800, soit 1 200 par équipe |
  | Validation | juil. à sept. 2023 | 2 191 |
  | Test | oct. 2023 à fév. 2024 | 4 476 |

- **Nettoyage :**
  - suppression des masques d'anonymisation (`XXXX`) répétés ;
  - montants remis en clair ;
  - textes de moins de 80 caractères écartés ;
  - doublons supprimés ;
  - 361 textes de validation et de test retirés, car déjà présents dans la période d'entraînement (lettres types).

## Entraînement

- **Boucle :** écrite à la main en PyTorch.
- **Réglages :** AdamW avec un pas de 8e-5, warmup sur 10 % des pas puis décroissance linéaire, écrêtage des gradients à 1,0, lots de 16.
- **Durée :** 3 époques, soit 2 025 pas, en 2 h 17 sur 2 cœurs, sans carte graphique.
- **Choix de la version gardée :** le F1 macro sur la validation, qui est passé de 0,778 à 0,796 puis à 0,802 au fil des 3 époques.
- **Suivi des essais :** MLflow (`make mlflow`).

## Résultats sur le test

Le test porte sur 4 476 réclamations jamais vues, postérieures à l'entraînement.

| | F1 macro | Justesse | Justesse attendue sur le vrai flux |
| --- | --- | --- | --- |
| Référence TF-IDF + régression logistique | 0,756 | 75,5 % | 77,3 % |
| MiniLM affiné, PyTorch 32 bits | 0,769 | 76,9 % | 76,6 % |
| **MiniLM affiné, ONNX 8 bits (servi)** | **0,769** | **76,9 %** | **76,4 %** |

**Routage automatique.** Le seuil de confiance (0,955) est choisi sur la validation pour viser 95 % de justesse sur les réclamations routées sans humain. Sur le test :
- le modèle servi route seul **49 %** des réclamations ;
- ces réclamations routées seules sont justes à **93,0 %** ;
- l'écart avec la cible de 95 % vient de la dérive entre l'été et l'hiver 2023.

**À justesse égale, le modèle route seul presque deux fois plus que la référence.** À 95 % de justesse, la version servie peut traiter seule 41,6 % des réclamations, contre 23,1 % pour TF-IDF (`reports/coverage.png`). Ce chiffre règle le seuil directement sur le test : il sert à comparer les modèles, pas à promettre un résultat en production.

**F1 par équipe (version servie) :**

| Équipe | F1 |
| --- | --- |
| Prêts étudiants | 0,93 |
| Prêts immobiliers | 0,89 |
| Crédits auto | 0,81 |
| Virements | 0,75 |
| Prêts personnels | 0,73 |
| Cartes | 0,72 |
| Rapports de crédit | 0,72 |
| Comptes | 0,72 |
| Recouvrement | 0,66 |

**Effet de la quantification :**
- le F1 perd 0,0004 ;
- 95,2 % des prédictions sont identiques à celles du modèle 32 bits ;
- les changements touchent des cas où le modèle hésitait déjà.

## Où il se trompe

L'analyse porte sur le modèle 32 bits, dont la version servie partage 95 % des réponses. Les deux confusions principales représentent 30 % des erreurs :
- **Virements et Comptes bancaires** (17 % des erreurs). Un virement contesté part d'un compte : la même histoire peut être déposée dans l'une ou l'autre catégorie.
- **Recouvrement et Rapports de crédit** (13 % des erreurs). Une dette en recouvrement apparaît sur le rapport de crédit, et le consommateur choisit l'une ou l'autre.

Les erreurs les plus sûres sont souvent des **erreurs d'étiquette**. Par exemple, une lettre qui cite la loi sur les rapports de crédit (Fair Credit Reporting Act) et s'adresse à un « bureau de crédit », mais que le consommateur a classée en « recouvrement ». Le modèle prédit « rapports de crédit » à 99 % : il a probablement raison.

## Limites

- **Étiquettes bruitées :** l'étiquette est le choix du consommateur. Une partie des « erreurs » mesurées n'en sont pas, ce qui plafonne la justesse mesurable.
- **Texte tronqué :** le modèle ne lit que les 128 premiers jetons, un compromis imposé par l'entraînement sur processeur. La référence TF-IDF lit tout le texte, ce qui explique qu'elle fasse aussi bien sur l'équipe la plus fréquente (rapports de crédit).
- **Anglais seulement**, et réclamations américaines de 2022 à 2024.
- **Dérive dans le temps :** la qualité baisse entre la validation et le test. En production, il faudrait suivre la justesse chaque mois sur un échantillon vérifié par des conseillers, et réentraîner régulièrement.
- **Pas de détection hors sujet :** une réclamation qui ne concerne aucune des 9 équipes sera tout de même classée. Le seuil de confiance l'enverra souvent, mais pas toujours, vers un conseiller.

## Usage prévu

Aide au tri : router seules les réclamations sûres et envoyer les autres à un conseiller, avec les 3 équipes les plus probables pour l'aider. Le modèle n'est pas fait pour décider seul du traitement d'une réclamation, ni pour juger de son bien-fondé.
