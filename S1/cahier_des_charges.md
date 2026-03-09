# Cahier des Charges - Projet Informatique avec Tests Utilisateurs

# 1. Contexte et Objectifs du Projet

## 1.1. Contexte

Ce projet s'inscrit dans le cadre de l'UE **AMS Projet 1** (Semestre 0) et répond au **Sujet 2 : Factory de Mots de Passe**, proposé par M. Haddad et M. Morchid.

La sécurité informatique repose en grande partie sur la robustesse des mots de passe. Cependant, la création de mots de passe à la fois complexes et mémorisables reste un défi pour les utilisateurs. L'intelligence artificielle, et plus spécifiquement les réseaux de neurones récurrents (**RNN/LSTM**), offre une opportunité de générer des séquences de caractères qui imitent des structures linguistiques tout en introduisant de l'entropie.

Ce projet vise à explorer ces techniques en commençant par la génération de texte littéraire (style Shakespeare) pour valider l'approche, avant de l'appliquer à la génération de mots de passe sécurisés.

## 1.2. Objectifs

-   **Développer une solution logicielle** : Créer un scripte Python capable d'entraîner un modèle d'IA et de générer du texte.
-   **Maîtriser les architectures RNN** : Passer de l'utilisation de GRU à LSTM pour améliorer la mémoire à long terme du modèle.
-   **Tester et optimiser** : Évaluer la qualité des séquences générées et les performances de l'entraînement (support GPU).

# 2. Périmètre du Projet

## 2.1. Fonctionnalités attendues

-   **Lecture et traitement des données** : Chargement et pré-traitement de corpus textuels (ex: Shakespeare).
-   **Split Train/Validation/Test** : Division automatique du corpus pour une évaluation rigoureuse.
-   **Entraînement de modèle** : Entraînement d'un réseau LSTM caractère par caractère.
-   **Validation et Early Stopping** : Évaluation périodique sur le jeu de validation, sauvegarde du meilleur modèle.
-   **Génération de texte** : Production de séquences autonomes à partir d'une amorce (_seed_).
-   **Sauvegarde/Chargement** : Persistance des modèles entraînés pour réutilisation.
-   **Support Multi-plateforme** : Utilisation transparente du CPU, CUDA (Nvidia) ou MPS (Apple Silicon).

## 2.2. Technologies utilisées

-   **Langage** : Python 3.9+
-   **Framework IA** : PyTorch
-   **Architecture** : LSTM (Long Short-Term Memory)
-   **Accélération matérielle** : Apple MPS (Metal Performance Shaders) / CUDA

# 3. Public Cible

## 3.1. Profil des utilisateurs

-   **Étudiants et Enseignants** : Pour la démonstration pédagogique des RNN.
-   **Chercheurs en IA** : Pour l'expérimentation sur la génération de séquences.
-   **Administrateurs Système** (Futur) : Pour la génération de wordlists ou de mots de passe robustes.

## 3.2. Nombre d’utilisateurs pour les tests

-   **Phase 1 (Interne)** : 2 utilisateurs (l'équipe de développement) pour valider le code.
-   **Phase 2 (Démonstration)** : Enseignant et classe lors de la soutenance.

# 4. Scénarios de Tests Utilisateurs

## 4.1. Objectifs des tests utilisateurs

-   Valider que le script s'exécute sans erreur sur différentes machines.
-   Vérifier que le modèle apprend effectivement (baisse de la _loss_).
-   Juger qualitativement la cohérence du texte généré.

## 4.2. Méthodologie de tests

-   **Tests fonctionnels** : Exécution des commandes d'entraînement et de génération.
-   **Tests de performance** : Mesure du temps d'entraînement sur CPU vs GPU.

## 4.3. Scénarios de tests

-   **Scénario 1 : Entraînement Rapide**

    -   Lancer un entraînement court (500 époques).
    -   Vérifier que la _loss_ diminue.
    -   Vérifier la validation loss et la perplexité.
    -   Vérifier qu'un fichier modèle `.pt` est créé.

-   **Scénario 2 : Évaluation Test Set**

    -   Après entraînement, vérifier que le modèle est évalué sur le test set.
    -   Vérifier que la perplexité de test est cohérente avec la validation.

-   **Scénario 3 : Génération Continue**
    -   Charger le modèle entraîné.
    -   Fournir une amorce "THE".
    -   Générer 500 caractères et vérifier la structure (mots anglais, ponctuation).

# 5. Critères de Réussite et Évaluations

## 5.1. Indicateurs de performance (KPI)

| Indicateur            | Seuil de réussite | Description                                       |
| --------------------- | ----------------- | ------------------------------------------------- |
| **Loss (Train)**      | < 2.0             | Fonction de perte sur le jeu d'entraînement       |
| **Loss (Validation)** | < 2.5             | Fonction de perte sur le jeu de validation        |
| **Perplexité (Test)** | < 15              | Mesure standard de qualité des modèles de langage |
| **Temps d'exécution** | < 15 min          | Pour 20 000 époques sur GPU                       |
| **Utilisation GPU**   | Actif             | Le processus doit utiliser MPS ou CUDA            |

## 5.2. Feedback des utilisateurs

-   Analyse des erreurs lors de l'exécution (logs).
-   Inspection visuelle du texte généré : présence de vrais mots anglais, respect de la syntaxe de base (majuscules, points).

# 6. Contraintes et Risques

## 6.1. Contraintes techniques

-   **Ressources de calcul** : L'entraînement des LSTM est gourmand en calcul ; un GPU est fortement recommandé.
-   **Taille de mémoire** : Le chargement de gros corpus peut saturer la RAM.
-   **Compatibilité** : PyTorch doit être correctement installé avec les drivers spécifiques (MPS/CUDA).

## 6.2. Risques potentiels

-   **Sur-apprentissage (Overfitting)** : Le modèle recrache le texte par cœur au lieu de générer. → _Mitigation : Split train/validation/test + sauvegarde du meilleur modèle sur validation._
-   **Explosion du gradient** : Instabilité numérique. → _Mitigation : Gradient Clipping (pas encore implémenté mais envisagé)._
-   **Texte incohérent** : Si l'entraînement est trop court, le résultat peut être du "bruit" aléatoire.

# 7. Planning Prévisionnel

## 7.1. Phases du projet

-   **Analyse et Prise en main** : Compréhension du code `articleGeneration.py` initial.
-   **Modification Architecture** : Remplacement GRU par LSTM, ajout support GPU.
-   **Expérimentation** : Tuning des hyperparamètres (layers, hidden size).
-   **Rédaction** : Documentation et Rapport.

## 7.2. Livrables

-   Code source : `articleGeneration_LSTM.py`.
-   Rapport technique : `rapport_cahier_des_charges.pdf`.
-   Modèle entraîné : `models/lstmOptimized_3_256.pt`.
-   Cahier des charges (ce document).

# 8. Équipe du Projet

-   **AMOROS Gaël** : Développeur Principal & Rédacteur
-   **CHAHBI Thaha** : Développeur & Rédacteur

# 9. Annexes

## 9.1. Split des Données

Le corpus est automatiquement divisé selon les proportions suivantes :

| Ensemble       | Ratio | Utilisation                                               |
| -------------- | ----- | --------------------------------------------------------- |
| **Train**      | 80%   | Entraînement du modèle                                    |
| **Validation** | 10%   | Détection de l'overfitting, sauvegarde du meilleur modèle |
| **Test**       | 10%   | Évaluation finale (perplexité)                            |

Pour le corpus Shakespeare (~1.1 Mo) :

-   Train : ~892 000 caractères
-   Validation : ~111 500 caractères
-   Test : ~111 500 caractères

## 9.2. Métrique de Perplexité

La **perplexité** mesure la capacité du modèle à prédire le caractère suivant :

$$\text{Perplexité} = e^{\text{CrossEntropyLoss}}$$

-   Perplexité = 1 : Prédiction parfaite (impossible en pratique)
-   Perplexité = 100 : Le modèle est équivalent à un choix aléatoire parmi 100 caractères
-   **Objectif** : Perplexité < 15 sur le test set

## 9.3. Ressources

-   **Code source** : Voir le dépôt du projet.
-   **Données** : Fichier `data/shakespeare.txt`.
