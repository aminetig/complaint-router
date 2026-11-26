# Résultats

Test : 4 476 réclamations d'octobre 2023 à février 2024, jamais vues à l'entraînement, équilibrées entre les 9 catégories.

| Modèle | F1 macro | Justesse attendue sur le vrai flux | Part routable à 95 % de justesse | Avec le seuil choisi sur la validation | Temps de réponse (médiane) | Taille |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + régression logistique | 0,756 | 77,3 % | 23,1 % | 27,5 % routées, 94,4 % justes | 3 ms | – |
| MiniLM affiné, PyTorch 32 bits | 0,769 | 76,6 % | 39,8 % | 53,2 % routées, 92,5 % justes | 59 ms | 91 Mo |
| **MiniLM affiné, ONNX 8 bits (servi)** | 0,769 | 76,4 % | 41,6 % | 49,1 % routées, 93,0 % justes | 43 ms | 23 Mo |

- **Part routable à 95 % de justesse :** la plus grande part des réclamations qu'on peut router sans humain en gardant 95 % de justesse, le seuil étant réglé sur le test. C'est la comparaison à armes égales entre les modèles (`coverage.png`).
- **Avec le seuil choisi sur la validation :** ce qui se passe vraiment en production, où le seuil est fixé à l'avance. Entre l'été et l'hiver 2023, les réclamations changent un peu : sur le test, la justesse de ces cas passe sous les 95 % visés.
- **Justesse attendue sur le vrai flux :** le test est équilibré, alors que 69 % des vraies réclamations concernent les rapports de crédit. Cette colonne pondère le rappel de chaque catégorie par sa part réelle.

## LLM sans entraînement : qwen2.5:3b, sur 180 réclamations

| | F1 macro | Justesse | Temps par réclamation (médiane) |
| --- | --- | --- | --- |
| qwen2.5:3b (Ollama, local) | 0,561 | 56,1 % | 23,4 s |
| MiniLM affiné, ONNX 8 bits | 0,771 | 77,2 % | 36 ms |

Notre modèle répond 643 fois plus vite, sur le même processeur.

Machine : Intel Core i5-7300U (2 cœurs, 2017), sans carte graphique.
