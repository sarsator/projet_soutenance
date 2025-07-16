# 📖 Guide d'Utilisation - GaiaSense Vision

Salut ! Alors voici comment utiliser mon système. J'ai essayé de faire ça le plus simple possible, mais il y a quand même quelques petites choses à savoir.

## 🌱 Comment démarrer le système

### La méthode facile (que je recommande)
```bash
python run_gaia_vision.py
```

Boom ! Ça lance tout d'un coup. J'ai fait ce script parce que c'était pénible de lancer l'API et le frontend séparément à chaque fois.

### La méthode manuelle (si tu veux plus de contrôle)
```bash
# Premier terminal pour l'API
python api/run_api.py

# Deuxième terminal pour l'interface web
python frontend/app.py
```

Personnellement je préfère la première méthode, mais parfois on a besoin de debug et là c'est mieux d'avoir les logs séparés.

## 🌐 Où aller une fois que c'est lancé

- **L'interface principale** : http://localhost:5000 (c'est là que ça se passe !)
- **L'API REST** : http://localhost:8000 (pour les développeurs)
- **Documentation auto** : http://localhost:8000/docs (FastAPI fait ça tout seul, c'est magique)

## 📝 Comment utiliser l'interface web

### Étape 1 : Remplir le formulaire

Alors voici ce qu'il faut renseigner :

**Les trucs obligatoires :**
- **Race de champignon** : Liste déroulante avec les espèces supportées
- **Type de substrat** : Le support de culture (paille, sciure, etc.)
- **Date d'inoculation** : Quand vous avez inoculé (le système calcule les jours automatiquement)
- **Hygrométrie** : Le taux d'humidité en % (genre 85.5)
- **CO2 PPM** : Concentration de CO2 (souvent autour de 800)
- **Image** : Photo de votre champignon (jpg, png, ce que vous voulez)

**Optionnel :**
- **Commentaire** : Si vous voulez ajouter des notes

### Étape 2 : Analyser et comprendre les résultats

Une fois que vous cliquez sur "Analyser", mon système fait ses calculs et vous donne :

- **Le verdict final** : "sain" ou "contaminé" 
- **Le score de confiance** : Entre 0 et 1 (plus c'est proche de 1, plus c'est sûr)
- **Le détail de l'analyse** :
  - Ce que dit CatBoost sur les paramètres
  - Ce que dit le modèle vision sur l'image (si nécessaire)
  - Quels modèles ont été utilisés

### Comment ça marche sous le capot ?

Mon approche c'est :
1. **D'abord** : CatBoost analyse les paramètres de culture
2. **Si besoin** : Si CatBoost détecte un risque OU n'est pas très confiant → on fait l'analyse d'image
3. **Enfin** : On combine tout ça pour donner un résultat final

C'est plus intelligent qu'un simple vote, parce que ça évite de faire de la vision quand c'est pas nécessaire.

## 🔧 Pour les développeurs : Utiliser l'API directement

### Authentification

J'ai mis une clé d'API simple pour sécuriser un minimum :
```bash
Authorization: Bearer gaia-vision-test-key-2025
```

### Les endpoints utiles

#### Prédiction complète (avec image)
```bash
curl -X POST "http://localhost:8000/predict-image" \
  -H "Authorization: Bearer gaia-vision-test-key-2025" \
  -F "race_champignon=Pleurotus ostreatus" \
  -F "type_substrat=paille" \
  -F "jours_inoculation=12" \
  -F "hygrometrie=85.5" \
  -F "co2_ppm=800" \
  -F "commentaire=Mon test" \
  -F "image=@photo_champignon.jpg"
```

#### Analyse des paramètres seulement
```bash
curl -X POST "http://localhost:8000/predict-parameters-only" \
  -H "Authorization: Bearer gaia-vision-test-key-2025" \
  -F "race_champignon=Pleurotus ostreatus" \
  -F "type_substrat=paille" \
  -F "jours_inoculation=12" \
  -F "hygrometrie=85.5" \
  -F "co2_ppm=800"
```

#### Vérifier que tout fonctionne
```bash
curl -X GET "http://localhost:8000/status"
curl -X GET "http://localhost:8000/health"
```

## 🧪 Tests et validation

### Tests Automatiques
```bash
# Test complet du système
python test_complet.py

# Test rapide de l'API
python quick_test.py

# Test de conversion de date
python test_date_conversion.py

# Test du formulaire frontend
python test_frontend_form.py
```

## 🛠️ Résolution de Problèmes

### Problèmes Courants

**1. Erreur "Invalid API key"**
- Vérifier la clé API dans le fichier `.env`
- S'assurer que frontend et API utilisent la même clé

**2. Erreur "Input should be a valid integer"**
- ✅ **Corrigé**: Le frontend convertit automatiquement les dates en nombre de jours

**3. Modèles non chargés**
- Vérifier que les fichiers de modèles existent:
  - `api/models/ml_model/model_catboost_best.joblib`
  - `api/models/dl_model/final_model.keras`

**4. Port déjà utilisé**
```bash
# Nettoyer les processus
pkill -f "python.*api"
pkill -f "python.*frontend"
```

### Logs et Debug

Les logs sont affichés dans la console avec :
- `INFO`: Opérations normales
- `WARNING`: Problèmes non critiques
- `ERROR`: Erreurs nécessitant attention

## 🔐 Sécurité

- Authentification par clé API
- Validation des paramètres d'entrée
- Gestion sécurisée des fichiers uploadés
- Limitation de taille des fichiers

## 📈 Performance

- Préchargement des modèles au démarrage
- Cache des prédictions (si implémenté)
- Gestion optimisée des ressources

## 🚀 Déploiement en Production

Pour un déploiement en production :
1. Configurer une vraie clé API sécurisée
2. Utiliser un serveur WSGI (Gunicorn)
3. Configurer un reverse proxy (Nginx)
4. Mettre en place du monitoring
5. Configurer SSL/TLS

---

**Gaia Vision** - Système d'analyse de contamination de champignons par IA
Version 1.0.0 - Juillet 2025
