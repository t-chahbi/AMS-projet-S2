# Factory de Mots de Passe - Génération de Texte (LSTM)

Ce projet implémente un réseau de neurones récurrents de type **LSTM (Long Short-Term Memory)** pour la génération de texte caractère par caractère. Il sert de base préliminaire à la création d'une "Factory de mots de passe".

## 📋 Prérequis

-   **Python 3.9+**
-   **PyTorch** (avec support MPS pour macOS ou CUDA pour NVIDIA recommandé)
-   **Unidecode**

### Installation des dépendances

```bash
pip install torch unidecode
```

## 🚀 Utilisation

Le script principal est `articleGeneration_LSTM.py`. Il permet à la fois d'entraîner le modèle et de générer du texte.

### 1. Entraînement du modèle

Pour entraîner le modèle sur le corpus Shakespeare (ou tout autre fichier texte) :

```bash
python3 articleGeneration_LSTM.py -d data/shakespeare.txt \
    --trainEval train \
    --max_epochs 20000 \
    --hidden_size 256 \
    --num_layers 3
```

**Arguments principaux :**

-   `-d`, `--trainingData` : Chemin vers le fichier texte d'entraînement.
-   `--trainEval train` : Mode entraînement.
-   `--max_epochs` : Nombre d'itérations (recommandé : 20000+ pour de bons résultats).
-   `--hidden_size` : Taille de la couche cachée (défaut: 256).
-   `--num_layers` : Nombre de couches LSTM (défaut: 3).
-   `-r`, `--run` : Préfixe du nom de fichier de sauvegarde (défaut: `lstmOptimized`).

_Le meilleur modèle (basé sur la perte minimale) est automatiquement sauvegardé dans le dossier `models/`._

### 2. Génération de texte (Évaluation)

Une fois un modèle entraîné, vous pouvez l'utiliser pour générer du texte :

```bash
python3 articleGeneration_LSTM.py -d data/shakespeare.txt \
    --trainEval eval \
    --length 500 \
    --run lstmOptimized \
    --hidden_size 256 \
    --num_layers 3
```

**Arguments spécifiques :**

-   `--trainEval eval` : Mode évaluation.
-   `--length` : Nombre de caractères à générer.
-   _Note : Assurez-vous d'utiliser les mêmes paramètres `--hidden_size` et `--num_layers` que lors de l'entraînement pour que le modèle se charge correctement._

### 3. Compilation du Rapport

Le rapport technique est rédigé en LaTeX. Pour le compiler en PDF :

```bash
pdflatex rapport_cahier_des_charges.tex
```

_(Nécessite une distribution LaTeX comme TeX Live ou MacTeX)._

## 📂 Structure du projet

-   `articleGeneration_LSTM.py` : Script principal (Modèle LSTM).
-   `articleGeneration.py` : Script original (Obsolète - GRU).
-   `data/` : Dossier contenant le corpus (`shakespeare.txt`).
-   `models/` : Dossier où sont sauvegardés les modèles `.pt`.
-   `rapport_cahier_des_charges.tex` : Code source du rapport.
-   `CERI_paper_kit/` : Template LaTeX utilisé.

## 👥 Auteurs

-   **AMOROS Gaël**
-   **CHAHBI Thaha**
