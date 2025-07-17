# 🎓 Résumé de Soutenance - GaiaSense Vision

## 🎯 Présentation du Projet (2 minutes)

**GaiaSense Vision** est un système IA hybride end-to-end qui combine :
- **Deep Learning** (SSD MobileNet V2) pour la détection d'objets dans les images
- **Machine Learning classique** (CatBoost) pour l'analyse de données tabulaires
- **Architecture de production** avec API REST et interface web

### Problématique Résolue
Détection automatique de champignons sains/contaminés avec classification des substrats pour l'agriculture.

---

## 🧠 Partie Machine Learning & Deep Learning (5-7 minutes)

### 1. 🔥 Modèle de Vision (Deep Learning)

**Architecture Choisie : SSD MobileNet V2**
- **Pourquoi SSD ?** Single Shot Detector = détection d'objets en une seule passe
- **Pourquoi MobileNet V2 ?** Optimisé pour la production (léger, rapide)
- **Avantages** : Balance parfaite entre performance et efficacité

**Spécifications Techniques :**
```
- Framework : TensorFlow 2.x
- Taille du modèle : ~23MB
- Temps d'inférence : ~150ms par image
- Format de sortie : TensorFlow SavedModel
- Input : Images RGB 320x320
```

**Performances Obtenues :**
- **F1-Score : 60%** sur le test set
- **Précision** : Détection correcte des champignons contaminés
- **Robustesse** : Fonctionne sur différents types d'images

**Processus d'Entraînement :**
1. **Dataset** : Photos 
2. **Preprocessing** : Redimensionnement, normalisation, augmentation de données
3. **Transfer Learning** : Fine-tuning depuis un modèle pré-entraîné
4. **Validation** : Split train/validation/test avec métriques TensorBoard

### 2. 🐱 Modèle Tabulaire (Machine Learning)

**Algorithme Choisi : CatBoost**
- **Pourquoi CatBoost ?** Gère naturellement les variables catégorielles
- **Avantages** : Robuste, peu de preprocessing, bonnes performances

**Spécifications Techniques :**
```
- Framework : CatBoost
- Sérialisation : Joblib
- Features : Données de substrat, conditions environnementales
- Temps d'inférence : ~50ms par prédiction
```

**Performances Obtenues :**
- **Accuracy : 85%** en validation croisée
- **Stabilité** : Résultats reproductibles
- **Interprétabilité** : Features importance analysées

**Processus d'Entraînement :**
1. **Feature Engineering** : Sélection et transformation des variables
2. **Validation Croisée** : K-fold pour évaluation robuste
3. **Hyperparameter Tuning** : Optimisation des paramètres
4. **Feature Importance** : Analyse des variables les plus importantes

### 3. 📊 Évaluation et Métriques

**Métriques Utilisées :**
- **Vision** : F1-Score, Précision, Rappel (sklearn.metrics)
- **Tabulaire** : Accuracy, Validation croisée, Courbes ROC
- **Monitoring** : TensorBoard pour le suivi d'entraînement

**Outils d'Évaluation :**
```python
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
```

---

## 🏗️ Architecture Technique (3 minutes)

### Système Hybride Intelligent
```
┌─────────────────┐    ┌─────────────────┐
│  IMAGE INPUT    │    │  TABULAR DATA   │
│                 │    │                 │
│  SSD MobileNet  │    │    CatBoost     │
│     V2 (DL)     │    │      (ML)       │
│                 │    │                 │
│  F1: 60%        │    │   Acc: 85%      │
│  150ms          │    │   50ms          │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌─────────────┐
              │  FASTAPI    │
              │  BACKEND    │
              │             │
              │ Versioning  │
              │  System     │
              └─────────────┘
                     │
              ┌─────────────┐
              │   FLASK     │
              │  FRONTEND   │
              │             │
              │ User Interface │
              └─────────────┘
```

### Technologies Clés
- **TensorFlow** : Deep Learning
- **CatBoost** : Machine Learning
- **FastAPI** : API REST avec auto-documentation
- **SQLite** : Stockage des prédictions
- **Flask** : Interface utilisateur

---

## 💡 Choix Techniques Justifiés (2 minutes)

### Pourquoi cette Stack ?

1. **SSD MobileNet V2** au lieu d'autres modèles :
   - Plus efficace qu'un CNN classique
   - Meilleur compromis performance/vitesse que YOLO
   - Optimisé pour la production

2. **CatBoost** au lieu d'autres ML :
   - Gère mieux les catégorielles que RandomForest
   - Plus stable que XGBoost
   - Moins de preprocessing requis

3. **Architecture Hybride** :
   - Exploite le meilleur des deux mondes (DL + ML)
   - Modularité : chaque modèle peut évoluer indépendamment
   - Scalabilité : facile d'ajouter de nouveaux modèles

---

## 🚀 Déploiement et Production (2 minutes)

### Système de Versioning Automatique
```python
# Gestion automatique des versions de modèles
class ModelVersionManager:
    def save_model(self, model, version_info)
    def load_latest_model(self)
    def rollback_model(self, version)
```

### API REST Robuste
- **FastAPI** avec documentation Swagger automatique
- **Gestion d'erreurs** complète
- **Validation** des inputs
- **Logging** des prédictions

### Interface Utilisateur
- **Flask** avec design responsive
- **Upload d'images** en temps réel
- **Visualisation** des résultats
- **Page About** avec informations techniques

---

## 📈 Résultats et Impact (1 minute)

### Métriques de Performance
- **Vision** : 60% F1-Score (acceptable pour un prototype)
- **Tabulaire** : 85% Accuracy (très bon pour des données réelles)
- **Vitesse** : <200ms pour une prédiction complète
- **Stabilité** : Système robuste en production

### Valeur Ajoutée
- **Automatisation** d'un processus manuel
- **Scalabilité** pour traiter des volumes importants
- **Précision** suffisante pour l'aide à la décision
- **Extensibilité** pour d'autres cas d'usage

---

## 🎯 Points Forts à Mettre en Avant

1. **Approche Hybride** : Combinaison intelligente DL + ML
2. **Choix Techniques** : Justifiés par les contraintes de production
3. **Architecture Complète** : De l'entraînement au déploiement
4. **Qualité du Code** : Documentation, tests, bonnes pratiques
5. **Résultats Concrets** : Métriques réelles sur données réelles

---

## 🗣️ Points de Discussion Possibles

**Questions Potentielles du Jury :**

1. **"Pourquoi SSD et pas YOLO ?"**
   - Réponse : SSD plus stable, MobileNet optimisé mobile, bon compromis

2. **"60% F1-Score, c'est suffisant ?"**
   - Réponse : Pour un prototype oui, amélioration possible avec plus de données

3. **"Pourquoi CatBoost ?"**
   - Réponse : Gère naturellement les catégorielles, robuste, peu de preprocessing

4. **"Comment vous gérez la montée en charge ?"**
   - Réponse : Architecture modulaire, API REST, possibilité Docker

5. **"Évolutions futures ?"**
   - Réponse : Plus de données, modèles plus complexes, déploiement cloud

---

## 🎪 Démonstration Live (2 minutes)

### Scénario de Démo
1. **Lancer le système** : `python start.py`
2. **Accéder à l'interface** : http://localhost:5000
3. **Upload d'une image** : Montrer la détection en temps réel
4. **Expliquer les résultats** : Confidence, bounding boxes
5. **Montrer l'API** : Documentation Swagger
6. **Page About** : Informations techniques et équipe

### Commandes Utiles
```bash
# Lancement
python start.py

# Arrêt propre
python stop.py

# URLs importantes
http://localhost:5000      # Interface
http://localhost:8000/docs # API Swagger
```

---

## 🏆 Conclusion (1 minute)

**GaiaSense Vision** démontre :
- **Maîtrise technique** : DL + ML + Architecture
- **Approche pragmatique** : Choix justifiés pour la production
- **Vision produit** : Système complet et utilisable
- **Qualité professionnelle** : Code, documentation, tests

**Compétences développées :**
- Deep Learning avec TensorFlow
- Machine Learning avec CatBoost
- Architecture logicielle
- API REST et interfaces web
- Déploiement et versioning

---

*Préparé pour la soutenance Alyra - 2025*
*Durée recommandée : 15-20 minutes + questions*
