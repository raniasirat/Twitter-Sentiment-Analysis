# Twitter Sentiment Analysis
## Analyse de sentiments sur Twitter avec le Machine Learning et le NLP

## 1. Présentation du projet

Ce projet consiste à développer un système intelligent capable d'analyser automatiquement les tweets et de classifier les sentiments exprimés envers différentes entités (marques, produits, jeux vidéo, personnalités, etc.).

L'objectif est de construire un pipeline complet de Machine Learning, depuis le chargement et le nettoyage des données jusqu'à l'entraînement, l'optimisation et l'évaluation des modèles.

Le système identifie quatre catégories de sentiments :

- Positive : sentiment positif.
- Negative : sentiment négatif.
- Neutral : sentiment neutre.
- Irrelevant : tweet non pertinent par rapport à l'entité.

## 2. Objectifs du projet

- Analyser et comprendre les données textuelles issues de Twitter.
- Réaliser un audit de qualité et une analyse exploratoire des données.
- Nettoyer et préparer les tweets pour la modélisation.
- Transformer les données textuelles en représentations numériques avec TF-IDF.
- Comparer plusieurs algorithmes de classification.
- Optimiser les hyperparamètres du meilleur modèle.
- Évaluer les performances à l'aide de métriques statistiques et de visualisations.

## 3. Technologies et outils utilisés

| Technologie | Utilisation |
|---|---|
| Python | Langage de programmation |
| Pandas et NumPy | Manipulation et traitement des données |
| Scikit-learn | Machine Learning et évaluation |
| TF-IDF | Vectorisation des données textuelles |
| Matplotlib et Seaborn | Visualisation des données |
| Plotly | Visualisations interactives |
| WordCloud | Génération de nuages de mots |
| Jupyter Notebook | Environnement de développement |

## 4. Architecture du pipeline

### 4.1. Chargement des données
- Importation des fichiers CSV.
- Attribution des noms de colonnes.
- Séparation des données d'entraînement et de test.

### 4.2. Audit de qualité des données
- Détection des valeurs manquantes.
- Identification des doublons.
- Vérification de la cohérence des classes.

### 4.3. Analyse exploratoire des données (EDA)
- Distribution des sentiments.
- Analyse de la longueur des tweets.
- Étude des unigrammes et bigrammes.
- Génération de nuages de mots.
- Analyse des sentiments par entité.

### 4.4. Prétraitement du texte
- Conversion en minuscules.
- Suppression des URLs et des mentions.
- Nettoyage de la ponctuation.
- Suppression des tweets vides.
- Encodage des catégories.

### 4.5. Vectorisation TF-IDF

Transformation des tweets en vecteurs numériques avec les paramètres suivants :

- Nombre maximal de caractéristiques : 50 000.
- N-grammes : unigrammes et bigrammes (1, 2).
- Fréquence minimale : 2.
- TF sublinéaire activé.

### 4.6. Comparaison des modèles

Cinq algorithmes de classification sont comparés :

- Logistic Regression
- Multinomial Naive Bayes
- Complement Naive Bayes
- Linear Support Vector Classifier (Linear SVC)
- SGD Classifier

La comparaison est réalisée avec une validation croisée stratifiée à 5 plis (Stratified K-Fold), en utilisant le F1-Score Macro comme métrique principale.

### 4.7. Optimisation des hyperparamètres

Utilisation de HalvingGridSearchCV pour rechercher efficacement les meilleures configurations du modèle sélectionné.

### 4.8. Évaluation finale

Le modèle final est évalué sur un jeu de test indépendant à travers :

- Accuracy, Precision, Recall et F1-Score.
- Matrice de confusion.
- Courbes ROC.
- Courbes Précision-Rappel.
- Importance des caractéristiques TF-IDF.

## 5. Résultats obtenus

Le modèle Linear SVC a obtenu les meilleures performances lors de la comparaison initiale.

Après optimisation des hyperparamètres, les résultats suivants ont été obtenus sur le jeu de test :

| Métrique | Résultat |
|---|---:|
| Accuracy | 97,80 % |
| Macro Precision | 97,79 % |
| Macro Recall | 97,81 % |
| Macro F1-Score | 97,79 % |
| Weighted F1-Score | 97,80 % |

**Meilleur modèle :** Linear SVC

**F1-Macro en validation croisée après optimisation :** 89,90 %

**Meilleurs hyperparamètres :**
- C = 1.0
- Max Features = 50 000
- Min DF = 2
- N-gram Range = (1, 2)

## 6. Structure du projet

```text
Twitter-Sentiment-Analysis/
│
├── twitter_sentiment_final (1).ipynb
├── twitter_training.csv
├── twitter_validation.csv
└── README.md
```

Remarque : les fichiers CSV du dataset sont à télécharger séparément depuis Kaggle.

## 7. Installation et exécution

### Étape 1 : Cloner le repository

```bash
git clone https://github.com/raniasirat/Twitter-Sentiment-Analysis.git
```

### Étape 2 : Accéder au dossier

```bash
cd Twitter-Sentiment-Analysis
```

### Étape 3 : Installer les dépendances

```bash
pip install numpy pandas matplotlib seaborn plotly wordcloud scikit-learn jupyter
```

### Étape 4 : Télécharger le dataset

Télécharger les fichiers twitter_training.csv et twitter_validation.csv depuis Kaggle et les placer dans le dossier du projet.

### Étape 5 : Lancer Jupyter Notebook

```bash
jupyter notebook
```

Ouvrir ensuite le fichier twitter_sentiment_final (1).ipynb et exécuter les cellules dans l'ordre.

## 8. Source des données

Dataset utilisé : Twitter Entity Sentiment Analysis

Source : [Kaggle – Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

## 9. Perspectives d'amélioration

- Intégrer des modèles avancés de NLP comme BERT ou RoBERTa.
- Améliorer la détection des sentiments à l'aide des emojis et du contexte.
- Développer une interface web interactive pour tester le modèle.
- Mettre en place une classification en deux étapes pour mieux traiter les tweets non pertinents.
- Déployer le modèle sous forme d'une API.

## 10. Réalisation

**Projet académique — Techniques d'Intelligence Artificielle**

- Formation : Master M1 — Intelligence Artificielle et Objets Connectés
- Année universitaire : 2025–2026

**Réalisé par : Rania Sirat**


