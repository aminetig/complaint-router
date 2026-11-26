# Journal de construction

Les problèmes réellement rencontrés, avec leur cause et leur solution.

## Le texte des réclamations a disparu des données officielles

- **Symptôme :** l'API du CFPB répond, mais aucune réclamation ne contient son texte. Le filtre `has_narrative=true` renvoie le même nombre de résultats qu'avec `false`, ou sans filtre : il est ignoré sans message d'erreur.
- **Vérification :** j'ai lu le début de l'export complet de 364 Mo, sans le télécharger en entier. Il suffit de décompresser les premiers octets du fichier ZIP. La colonne « Consumer complaint narrative » n'existe plus.
- **Solution :** une copie publique faite en février 2024, sous licence CC0, trouvée sur Hugging Face. Elle contient 1,69 million de réclamations avec leur texte.
- **Leçon :** vérifier les données elles-mêmes, pas la documentation. Un filtre ignoré en silence aurait pu passer inaperçu.

## Des lettres types qui faussaient l'évaluation

- **Constat :** 361 réclamations de validation et de test avaient exactement le même texte qu'une réclamation de la période d'entraînement. Ce sont des modèles de lettres circulant en ligne, surtout pour contester un rapport de crédit.
- **Conséquence :** dans la catégorie « rapports de crédit », un tiers de l'échantillon de validation était déjà connu.
- **Solution :** une empreinte (hash) de chaque texte de la période d'entraînement. Tout texte de validation ou de test qui a une empreinte connue est écarté.

## 2 fils d'exécution plus rapides que 4

- **Mesure :** 4,1 textes par seconde avec 2 fils, contre 3,2 avec 4.
- **Cause :** le processeur a 2 cœurs physiques et 4 « logiques » (hyperthreading). Les calculs de matrices occupent déjà toute la capacité d'un cœur : deux fils sur le même cœur se gênent plus qu'ils ne s'aident.

## La qualité baisse entre la validation et le test

- **Constat :** la référence TF-IDF a un F1 de 0,800 sur la validation (été 2023), mais de 0,756 sur le test (hiver 2023-2024).
- **Explication :** les réclamations évoluent avec le temps : nouveaux produits, nouvelles arnaques, renommage des catégories en 2023. C'est la dérive des données (« data drift »), et c'est précisément ce que le découpage dans le temps permet de mesurer.
- **En production :** suivre la qualité mois par mois, et réentraîner régulièrement.

## Vérifier l'export ONNX avant les 2 heures d'entraînement

- **Contexte :** dans les versions récentes de PyTorch, `torch.onnx.export` utilise par défaut un nouveau moteur (« dynamo »), qui déclare autrement les dimensions variables. L'ancien moteur (`dynamo=False`) enregistre les opérations au passage d'un exemple, et il est éprouvé sur les modèles Hugging Face.
- **Précaution :** avant de lancer l'entraînement, j'ai exporté le modèle non entraîné avec l'ancien moteur. L'export prend 4 secondes, et l'écart avec PyTorch est inférieur à 0,000001 avec deux tailles de lot et de texte différentes. Une incompatibilité découverte après 2 heures d'entraînement aurait coûté bien plus cher.
- **Après l'entraînement :** l'écart entre PyTorch et ONNX est au plus de 0,0000013 sur les probabilités, mesuré sur 64 vraies réclamations.

## Le seuil de confiance doit suivre la quantification

- **Constat :** le premier export reprenait le seuil choisi pour le modèle 32 bits (0,959). Or la quantification modifie légèrement les scores : avec ce seuil, la version servie routait seule 47,7 % des réclamations au lieu de 53 %.
- **Solution :** l'export recalcule le seuil sur la validation avec le modèle 8 bits lui-même. Après la correction décrite juste en dessous, ce seuil vaut 0,955. Règle générale : tout réglage dépendant des scores se refait avec le modèle réellement servi.

## En 8 bits, le score d'une réclamation dépendait de ses voisines

C'est le problème le plus instructif du projet.

- **Premier signe :** un test vérifiait qu'une réclamation obtient le même score seule ou dans un lot. Avec le modèle quantifié, il échouait de peu : un écart de 0,0007 entre deux textes courts.
- **Deuxième signe, plus net :** la capture d'écran de la démo affichait 80 % de confiance pour une réclamation. Le même texte, dans un lot de 5 textes de longueurs différentes, obtenait 70 %.
- **Cause :** en quantification « dynamique », les poids sont stockés une fois pour toutes en entiers 8 bits. Les valeurs intermédiaires, elles, sont converties en entiers à chaque appel, avec **une seule échelle pour tout le lot**. Les autres textes du lot, et le remplissage (padding) ajouté aux textes courts, déplacent cette échelle.
- **Mesure sur les 2 191 réclamations de validation**, par lots de 64 contre une à la fois :
  - écart de probabilité médian de 0,005, et jusqu'à 0,28 ;
  - **2,1 % des réclamations changeaient de catégorie** selon leurs voisines.

  Avec le modèle 32 bits, l'écart est exactement nul : le remplissage est donc bien masqué, et le problème vient uniquement de la quantification.
- **Pourquoi c'est grave :**
  - **Évaluation faussée :** le seuil et les résultats de la version servie avaient été calculés par lots de 64, alors que l'API traite les réclamations une par une. On mesurait donc un autre comportement que celui de la production.
  - **Audit impossible :** pour une banque, la même réclamation doit toujours recevoir la même décision.
- **Solution :** la version servie traite chaque réclamation seule, sans remplissage. Le seuil et toutes les mesures ont été refaits dans ces conditions, et le test exige maintenant des scores strictement identiques. Le prix à payer : un lot de 64 réclamations prend 64 fois le temps d'une seule, soit environ 3 secondes, ce qui reste acceptable pour ce service.
- **Leçon :** une optimisation (ici, la quantification) peut changer des propriétés qu'on croyait acquises. Il faut évaluer le modèle exactement comme il sera servi.

## Les mesures de vitesse varient d'une exécution à l'autre

- **Constat :** quatre exécutions de l'export ont donné, pour le modèle servi, de 35 à 44 ms par réclamation (médiane). Le chiffre publié, 43 ms, vient de la dernière.
- **Cause :** la mesure se fait sur un portable où tournent aussi un éditeur et un navigateur.
- **Ce qui reste stable :** le rapport entre les versions. ONNX 8 bits est 1,4 à 1,6 fois plus rapide que PyTorch à chaque fois. Pour un chiffre de référence, il faudrait mesurer sur une machine dédiée.
- **Dans Docker**, sous Linux, la même API répond en 14 à 28 ms.

## Les erreurs « les plus sûres » sont souvent des erreurs d'étiquette

- **Constat :** dans les erreurs où le modèle est sûr à 99 %, on trouve des lettres qui citent la loi sur les rapports de crédit (Fair Credit Reporting Act) et s'adressent à un « bureau de crédit », mais que le consommateur a classées en « recouvrement ».
- **Explication :** l'étiquette est le produit choisi par le consommateur, pas une vérité vérifiée. Les deux confusions principales (virements et comptes, recouvrement et rapports de crédit) font 30 % des erreurs, et ce sont justement les catégories qui se recouvrent dans la réalité.
- **Conséquence :** la justesse mesurée sous-estime la justesse réelle. En production, il faudrait faire vérifier un échantillon par des conseillers pour connaître le vrai taux d'erreur.

## Le LLM était désavantagé par mon prompt

- **Premier résultat :** Qwen2.5 3B obtenait un F1 de 0,43, contre 0,77 pour notre modèle sur les mêmes 180 réclamations. L'écart paraissait trop beau pour être publié sans vérification.
- **Indice :** le LLM répondait « cartes » pour 63 réclamations sur 180, y compris pour des prêts immobiliers et des crédits auto.
- **Diagnostic :** j'ai reposé 3 de ces cas **sans** contrainte de format. Le LLM a répondu juste sur 2 des 3. Le problème venait donc de la sortie contrainte par schéma JSON. Le prompt ne demandait pas de JSON : le modèle ne « savait » pas qu'il devait en écrire, et la contrainte, appliquée mot à mot pendant la génération, déformait sa réponse. La documentation d'Ollama recommande justement de demander le JSON dans le prompt.
- **Correction :** une ligne dans le prompt, qui donne la forme exacte attendue. Sur 6 cas auparavant tous faux, 3 sont devenus justes.
- **Nouvelle mesure complète :** le F1 du LLM passe de 0,43 à **0,56**, et sa justesse de 43 % à 56 %. Notre modèle reste devant, à 0,77, sur les mêmes 180 réclamations, mais l'écart réel était bien plus petit que le premier chiffre.
- **Précaution ajoutée :** le fichier qui garde les réponses déjà obtenues porte une empreinte du prompt. Si le prompt change, les anciennes réponses ne sont plus reprises par erreur.
- **Leçon :** quand une comparaison donne un écart spectaculaire, il faut d'abord chercher ce qui désavantage l'adversaire. Une comparaison n'a de valeur que si chaque concurrent est utilisé correctement.

## Piège de pandas 3

- **Constat :** `groupby(...).apply(...)` ne transmet plus la colonne de regroupement à la fonction appliquée. L'échantillonnage par catégorie aurait perdu la colonne `label`.
- **Solution :** `groupby("label").sample(n=...)`, qui est fait pour ça et plus lisible.

## L'installation d'Ollama semblait bloquée

- **Symptôme :** la commande d'installation silencieuse ne rendait pas la main.
- **Cause :** l'installeur lance Ollama à la fin, et `Start-Process -Wait` attend aussi ce programme, qui ne s'arrête jamais.
- **Solution :** vérifier que l'installation était terminée (fichiers présents, serveur qui répond sur le port 11434), puis arrêter l'attente.
