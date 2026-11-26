# Choix techniques

Chaque choix, avec l'alternative écartée et la raison.

## 1. Des données réelles, depuis une copie publique

- **Choix :** une copie publique de la base des réclamations du CFPB, faite en février 2024 : 1,69 million de réclamations avec leur texte, sous licence CC0 (domaine public).
- **Alternative écartée :** l'API officielle du CFPB.
- **Raison :** en novembre 2026, l'API et l'export complet ne contiennent plus le texte des réclamations. Des données simulées auraient été plus faciles, mais un modèle de texte ne vaut que par la réalité de son texte : fautes, colères, lettres types, informations masquées.

## 2. Un découpage dans le temps, pas au hasard

- **Choix :** apprentissage sur janvier 2022 à juin 2023, réglages sur juillet à septembre 2023, test sur octobre 2023 à février 2024.
- **Alternative écartée :** un tirage aléatoire des trois jeux.
- **Raison :** en production, le modèle est toujours appliqué à des réclamations plus récentes que celles de son apprentissage. Un tirage aléatoire mélangerait les époques et surestimerait la qualité. La référence TF-IDF le montre : F1 de 0,800 sur la validation, mais 0,756 sur le test, plus éloigné dans le temps.

## 3. Des catégories équilibrées, et deux façons de mesurer

- **Choix :** 1 200 exemples d'entraînement par catégorie. Le modèle est jugé sur le **F1 macro**, la moyenne des F1 de chaque catégorie, et sur la **justesse attendue sur le vrai flux**.
- **Alternative écartée :** garder la répartition réelle, où les rapports de crédit font 68 % des réclamations.
- **Raison :** avec la répartition réelle, un modèle qui répondrait toujours « rapports de crédit » aurait 68 % de justesse sans rien apprendre. L'équilibrage oblige le modèle à bien traiter les petites catégories, comme les prêts personnels, qui font 1 % du flux. La justesse attendue sur le vrai flux pondère ensuite le rappel de chaque catégorie par sa part réelle.

## 4. Chasser les doublons entre les périodes

- **Choix :** tout texte de validation ou de test déjà présent dans la période d'entraînement est retiré.
- **Alternative écartée :** un simple dédoublonnage à l'intérieur de chaque jeu.
- **Raison :** beaucoup de consommateurs envoient la même lettre type, trouvée en ligne. Ce contrôle a retiré 361 réclamations de validation et de test. Sans lui, le modèle aurait été noté sur des textes qu'il connaissait déjà, et la mesure aurait été trop optimiste.

## 5. Un petit modèle, MiniLM, plutôt qu'un gros

- **Choix :** `all-MiniLM-L6-v2` : 6 couches, 22 millions de paramètres.
- **Alternative écartée :** BERT-base, 110 millions de paramètres, ou RoBERTa.
- **Raison :** l'entraînement se fait sur un processeur de portable de 2017, sans carte graphique. MiniLM est environ 5 fois plus rapide que BERT-base, pour une qualité proche sur la classification de phrases. En production, c'est aussi ce qui permet de répondre en quelques millisecondes sans GPU. Le script accepte n'importe quel encodeur : `--encoder roberta-base` sur une machine avec GPU, sans rien changer d'autre.

## 6. Une boucle d'entraînement écrite à la main

- **Choix :** la boucle PyTorch est écrite à la main : lots, rétropropagation, écrêtage des gradients, warmup, arrêt anticipé.
- **Alternative écartée :** la classe `Trainer` de Hugging Face, qui fait tout en une ligne.
- **Raison :** on contrôle et on comprend chaque étape. On peut expliquer pourquoi le pas d'apprentissage commence petit, ou pourquoi on écrête les gradients. Seuls l'encodeur pré-entraîné et son tokenizer viennent de Hugging Face.

## 7. Un seuil de confiance : automatiser seulement quand c'est sûr

- **Choix :** sous un seuil de confiance, la réclamation part vers un conseiller au lieu d'être routée automatiquement. Le seuil est choisi sur la validation, pour garantir 95 % de justesse sur les cas automatisés.
- **Alternative écartée :** router toutes les réclamations.
- **Raison :** une erreur de routage coûte cher, car la réclamation est traitée en retard par une équipe qui doit la renvoyer. Le bon indicateur métier n'est pas « 85 % de justesse », mais « X % des réclamations traitées sans humain, avec 95 % de justesse ».

## 8. ONNX et quantification pour la production

- **Choix :** le modèle est exporté au format ONNX, quantifié en entiers 8 bits, et exécuté par ONNX Runtime.
- **Alternative écartée :** servir directement le modèle PyTorch.
- **Raison :** l'image Docker de l'API n'embarque ni PyTorch ni transformers, soit 676 Mo une fois installés, même en version processeur. ONNX Runtime et tokenizers n'en font que 56. Le modèle est aussi 4 fois plus petit, et plus rapide sur processeur. La perte de qualité est mesurée, pas supposée.

## 9. Une réclamation à la fois dans le modèle servi

- **Choix :** le modèle 8 bits traite chaque réclamation seule, sans remplissage, même quand l'API en reçoit un lot.
- **Alternative écartée :** des lots complétés par remplissage, la méthode habituelle pour gagner en débit.
- **Raison :** en quantification dynamique, l'échelle des entiers 8 bits est commune à tout le lot. Le score d'une réclamation dépendait donc de ses voisines : 2,1 % des réclamations de validation changeaient de catégorie (voir le journal). Une banque doit pouvoir expliquer pourquoi une réclamation a été routée, et la même réclamation doit toujours recevoir la même décision. Le débit perdu est acceptable : environ 3 secondes pour un lot de 64.

## 10. Un LLM local pour la comparaison

- **Choix :** Qwen2.5 3B, exécuté par Ollama sur la même machine. Sa réponse est contrainte par un schéma JSON qui n'autorise que les 9 catégories.
- **Alternative écartée :** un LLM en ligne, accessible par API.
- **Raison :** une banque ne peut pas envoyer les réclamations de ses clients, avec leurs données personnelles, à un service externe sans précautions lourdes. La comparaison se fait donc dans les mêmes conditions : même machine, données qui ne sortent pas. Le schéma JSON évite d'avoir à interpréter des réponses en texte libre.

## 11. FastAPI et une page HTML, plutôt que Gradio ou Streamlit

- **Choix :** une API FastAPI qui sert aussi une page de démonstration en HTML et JavaScript simples.
- **Alternative écartée :** Gradio ou Streamlit.
- **Raison :** une seule image Docker sert à la fois l'API utilisée par les autres applications et la démonstration. Elle n'a pas de dépendance lourde, et la documentation interactive est générée automatiquement (`/docs`).
