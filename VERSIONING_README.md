# Système de Versioning Gaia Vision

## Vue d'ensemble

Système complet de versioning pour les modèles ML/DL du projet Gaia Vision avec logging automatique, incrémentation des versions, et gestion des rollbacks.

## Modèles supportés

- **SSD MobileNet V2** (Deep Learning) - Détection de contamination
- **CatBoost Classifier** (Machine Learning) - Prédiction basée sur les features

## Utilisation

### Script simple (recommandé)

bash
# Afficher l'aide
./versioning.sh help

# Voir le statut actuel
./versioning.sh status

# Déployer une nouvelle version SSD
./versioning.sh deploy ssd

# Déployer une nouvelle version CatBoost  
./versioning.sh deploy catboost

# Déployer les deux modèles
./versioning.sh deploy both

# Lister toutes les versions
./versioning.sh list

# Voir les logs récents
./versioning.sh logs

# Nettoyer les anciennes versions (garde 3 dernières)
./versioning.sh cleanup


### Script avancé

bash
# Déployer un modèle spécifique
python model_versioning.py deploy dl
python model_versioning.py deploy ml

# Lister les versions par type
python model_versioning.py list dl
python model_versioning.py list ml

# Rollback vers une version spécifique
python model_versioning.py rollback dl v1.3_20250716_132518
python model_versioning.py rollback ml v1.2_20250716_202522

# Nettoyer avec options avancées
python model_versioning.py cleanup dl --keep 5
python model_versioning.py cleanup ml --keep 3

# Statut complet du système
python model_versioning.py status


## État actuel

### Versions déployées
- **SSD MobileNet V2:** v1.5 (22.83 MB, SavedModel)
- **CatBoost:** v1.3 (0.62 MB, Joblib)

### Statistiques
- **Total versions SSD:** 7
- **Total versions CatBoost:** 4
- **Espace utilisé:** ~264 MB

## Structure des dossiers


api/models/
├── dl_model/
│   ├── current -> versions/v1.5_20250716_202917/saved_model
│   └── versions/
│       ├── v1.0_20250711_143245/
│       ├── v1.1_20250711_143319/
│       ├── v1.2_20250711_143350/
│       ├── v1.3_20250716_132518/
│       ├── v1.3_20250716_202207/
│       ├── v1.4_20250716_202612/
│       └── v1.5_20250716_202917/
│           ├── saved_model/
│           └── metadata.json
├── ml_model/
│   ├── current -> versions/v1.3_20250716_202927/model_catboost_best.joblib
│   └── versions/
│       ├── v1.0_20250711_145216/
│       ├── v1.1_20250716_202207/
│       ├── v1.2_20250716_202522/
│       └── v1.3_20250716_202927/
│           ├── model_catboost_best.joblib
│           └── metadata.json
└── model_version_manager.py


## Logging

Tous les logs sont automatiquement sauvegardés dans :

logs/versioning_YYYYMMDD_HHMMSS.log


### Exemple de log

2025-07-16 20:29:17,571 - ModelVersioning - INFO - 🚀 DÉPLOIEMENT MODÈLE DL
2025-07-16 20:29:17,571 - ModelVersioning - INFO - 📁 Source: /api/models/dl_model/versions/v1.4_20250716_202612/saved_model
2025-07-16 20:29:17,571 - ModelVersioning - INFO - 🔢 Prochaine version: v1.5
2025-07-16 20:29:17,585 - ModelVersioning - INFO - ✅ DÉPLOIEMENT RÉUSSI!
2025-07-16 20:29:17,585 - ModelVersioning - INFO -    Version: v1.5
2025-07-16 20:29:17,585 - ModelVersioning - INFO -    Taille: 22.83 MB
2025-07-16 20:29:17,585 - ModelVersioning - INFO -    Format: SavedModel


## Fonctionnalités

### Versioning automatique
- Incrémentation automatique des versions
- Support multi-format (SavedModel, Keras, Joblib)
- Métadonnées complètes pour chaque version

### Déploiement robuste
- Détection automatique du format de modèle
- Copie sécurisée avec vérification d'intégrité
- Liens symboliques automatiques vers `current`

### Gestion des rollbacks
- Rollback vers n'importe quelle version
- Validation de l'existence des versions
- Mise à jour automatique des liens

### Nettoyage intelligent
- Conservation des N dernières versions
- Suppression sécurisée des anciennes versions
- Préservation de la version actuelle

### Monitoring complet
- Logging détaillé de toutes les opérations
- Statut en temps réel du système
- Historique des déploiements

## Métadonnées sauvegardées

Chaque version contient :
json
{
  "version": "1.5",
  "timestamp": "2025-07-16T20:29:17.571070",
  "model_type": "dl",
  "deployed_at": "2025-07-16 20:29:17",
  "deployment_id": "20250716_202917",
  "architecture": "SSD MobileNet V2",
  "input_size": "320x320",
  "classes": ["background", "healthy", "contaminated"],
  "detection_threshold": 0.12,
  "framework": "TensorFlow Object Detection API",
  "model_file": "saved_model",
  "model_size_bytes": 23945728,
  "model_size_mb": 22.83,
  "model_format": "SavedModel",
  "deployment_reason": "Déploiement automatique via CLI"
}


## Exemples d'usage

### Déploiement quotidien
bash
# Déployer les nouvelles versions
./versioning.sh deploy both

# Vérifier le déploiement
./versioning.sh status

# Voir les logs en cas de problème
./versioning.sh logs


### Rollback d'urgence
bash
# Lister les versions disponibles
python model_versioning.py list dl

# Rollback vers une version stable
python model_versioning.py rollback dl v1.4_20250716_202612

# Vérifier le rollback
./versioning.sh status


### Maintenance
bash
# Nettoyer les anciennes versions
./versioning.sh cleanup

# Vérifier l'espace libéré
./versioning.sh status


## Points importants

1. **Sauvegarde automatique** : Chaque déploiement crée une copie complète
2. **Liens symboliques** : Le dossier `current` pointe toujours vers la version active
3. **Format detection** : Support automatique SavedModel/Keras/Joblib
4. **Logging complet** : Toutes les opérations sont tracées
5. **Rollback sécurisé** : Validation avant changement de version

## Support

En cas de problème :
1. Vérifier les logs : `./versioning.sh logs`
2. Vérifier le statut : `./versioning.sh status`
3. Rollback si nécessaire : `python model_versioning.py rollback [type] [version]`

---

**Système développé pour Gaia Vision - Détection de contamination fongique**  
*Versioning automatique, déploiement sécurisé, monitoring complet* 🍄
