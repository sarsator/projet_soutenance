# Gaia Vision API - Documentation Technique

Hey ! Voici la doc de mon API. J'ai essayé de faire quelque chose de propre et professionnel, avec une orchestration intelligente de deux modèles différents.

## L'idée derrière l'architecture

Mon système combine deux approches complémentaires :

1. **CatBoost pour les paramètres** : Race, substrat, hygrométrie, CO2... Parfait pour les données structurées
2. **Vision avec SSD MobileNet V2** : Détection d'objets pour identifier les champignons contaminés/sains

### Comment ça marche concrètement ?

J'ai pas fait un simple vote entre les deux modèles. Mon approche est plus smart :

1. **Premier passage** : CatBoost analyse les paramètres de culture
2. **Décision intelligente** : Si CatBoost détecte un risque OU n'est pas confiant → on fait l'analyse d'image
3. **Résultat final** : Combinaison réfléchie des deux sources d'info

Pourquoi cette approche ? Parce que l'analyse d'image coûte plus cher en ressources, donc autant l'éviter quand c'est pas nécessaire !

## Installation et setup

### Étape 1 : Récupérer le code
bash
git clone <votre-repo>
cd gaia_vision


### Étape 2 : Installer tout ce qu'il faut
bash
pip install -r requirements.txt


### Étape 3 : Configuration (optionnelle)
bash
cp .env.example .env
# Modifiez le .env si vous voulez changer des trucs


### Étape 4 : Vérifier que les modèles sont là
- Modèle CatBoost : `api/models/ml_model/model_catboost_best.joblib`
- Modèle Vision : `api/models/dl_model/current/` (SSD MobileNet V2)

Si ils sont pas là, il faut les entraîner avec les notebooks !

## � Lancement

### Juste l'API
bash
python -m api.run_api


### Tout le système (API + Interface)
bash
python run_gaia_vision.py


### Développement avec reload
bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000


## Endpoints

### GET `/status`
Vérification du statut de l'API

### GET `/health`
Vérification de l'état des modèles ML

### GET `/`
Documentation de base de l'API

### POST `/predict-image`
Prédiction complète avec image (CatBoost + SSD Vision)

**Paramètres :**
- `Authorization` (header) : `Bearer <API_KEY>`
- `race_champignon` (form) : Race du champignon
- `type_substrat` (form) : Type de substrat
- `jours_inoculation` (form) : Nombre de jours depuis l'inoculation
- `hygrometrie` (form) : Taux d'hygrométrie (%)
- `co2_ppm` (form) : Taux de CO2 en PPM
- `commentaire` (form) : Commentaire optionnel
- `image` (file) : Image à analyser

### POST `/predict-parameters-only`
Prédiction basée uniquement sur les paramètres (CatBoost seul)

**Paramètres :**
- Mêmes paramètres que `/predict-image` sans le fichier image

## Exemple d'utilisation

python
import requests

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "votre_clé_api"

# Prédiction avec image
headers = {"Authorization": f"Bearer {API_KEY}"}
data = {
    "race_champignon": "Pleurotus ostreatus",
    "type_substrat": "paille",
    "jours_inoculation": 12,
    "hygrometrie": 85.5,
    "co2_ppm": 800,
    "commentaire": "Test"
}
files = {"image": open("image.jpg", "rb")}

response = requests.post(
    f"{API_URL}/predict-image",
    headers=headers,
    data=data,
    files=files
)

result = response.json()
print(f"Prédiction: {result['prediction']}")
print(f"Confiance: {result['confidence']}")


## Réponse API

json
{
  "prediction": "sain",
  "confidence": 0.85,
  "details": {
    "catboost_prediction": {
      "prediction": 0,
      "confidence": 0.75,
      "model_type": "catboost"
    },
    "vision_prediction": {
      "prediction": "sain",
      "confidence": 0.95,
      "model_type": "ssd_object_detection",
      "detection_summary": {
        "total_detections": 3,
        "contaminated_count": 0,
        "healthy_count": 3,
        "max_healthy_score": 0.95
      }
    },
    "models_used": ["catboost", "ssd_vision"],
    "analysis_steps": ["catboost_analysis", "ssd_vision_analysis"]
  },
  "input_parameters": {
    "race_champignon": "Pleurotus ostreatus",
    "type_substrat": "paille",
    "jours_inoculation": 12,
    "hygrometrie": 85.5,
    "co2_ppm": 800,
    "commentaire": "Test",
    "image_file": "abc123.jpg"
  }
}


## Tests

Lancer les tests automatiques :
bash
python test_api.py


## Configuration avancée

Éditez le fichier `api/config.py` pour :
- Chemins des modèles
- Seuils de confiance
- Taille d'image
- Classes de prédiction

## Structure du projet


api/
├── main.py              # Point d'entrée FastAPI
├── config.py            # Configuration centralisée
├── run_api.py           # Script de lancement
├── models/              # Wrappers des modèles ML
│   ├── catboost_model.py
│   ├── vision_model.py
│   └── __init__.py
└── utils/               # Services et utilitaires
    ├── prediction_service.py
    └── __init__.py


## Logs

Les logs sont affichés dans la console avec différents niveaux :
- `INFO` : Opérations normales
- `WARNING` : Problèmes non bloquants
- `ERROR` : Erreurs nécessitant attention

## Sécurité

- Authentification par clé API
- Validation des paramètres d'entrée
- Gestion sécurisée des fichiers uploadés
- Limitation de taille des fichiers

## Production

Pour un déploiement en production :

1. Utilisez un serveur WSGI comme Gunicorn
2. Configurez un reverse proxy (Nginx)
3. Utilisez une vraie base de données
4. Implémentez du monitoring et des métriques
5. Configurez SSL/TLS

bash
# Exemple avec Gunicorn
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker

