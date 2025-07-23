# 🌱 GaiaSense Vision - Mon Projet de Fin d'Études

Salut ! Bienvenue sur mon projet de soutenance pour Alyra. J'ai développé GaiaSense Vision, une solution IA complète qui combine machine learning classique et deep learning pour l'analyse d'images.

## 💁 Qu'est-ce que c'est ?

En gros, c'est un système intelligent qui peut :
- **Analyser des images** avec un modèle SSD MobileNet V2 pour la détection d'objets
- **Traiter des données tabulaires** avec CatBoost (mon algo favori pour ce type de données)
- **Déployer automatiquement** les modèles avec un système de versioning maison
- **Servir le tout** via une API REST moderne avec FastAPI
- **Proposer une interface** web intuitive pour les utilisateurs

## 🌱 Pourquoi ce projet ?

J'ai voulu créer quelque chose de concret qui pourrait vraiment être utilisé en entreprise. Pas juste un proof of concept, mais un vrai système de production avec :

🌿 **Système de versioning automatique** des modèles  
🌿 **API robuste** avec gestion d'erreurs  
🌿 **Interface utilisateur** responsive et moderne  
🌿 **Tests automatisés** pour la fiabilité  
🌿 **Documentation complète** pour la maintenance  
🌿 **Déploiement simple** avec Docker (coming soon)  

## 🏗️ Architecture

```
gaia_vision/
├── � training/          # Notebooks de développement des modèles
├── 🔌 api/              # Backend FastAPI avec système de versioning
├── � frontend/         # Interface web avec Flask
├── 📊 data_trie/        # Pipeline de données et preprocessing
└── 🖼️ api/images_a_traiter/ # Dataset d'images pour les tests
```

## 🛠️ Technologies utilisées

**Machine Learning :**
- 🐱 **CatBoost** pour les données tabulaires (gère naturellement les catégorielles)
- 🔥 **SSD MobileNet V2** pour la détection d'objets (optimisé pour la production)
- 📊 **Scikit-learn** pour les métriques et la validation

**Backend :**
- ⚡ **FastAPI** pour l'API (auto-documentation Swagger incluse)
- 🗄️ **SQLite** pour les prédictions et logs
- 📦 **Joblib/TensorFlow SavedModel** pour la sérialisation des modèles

**Frontend :**
- 🌶️ **Flask** pour l'interface web
- 🎨 **CSS/JavaScript** vanilla (pas de framework lourd)
- 📱 **Design responsive** mobile-friendly

**DevOps :**
- 🐍 **Python 3.10+** 
- 📝 **Jupyter** pour le développement iteratif
- 🔧 **Git** pour le versioning du code
- 📋 **Requirements.txt** pour la reproductibilité

## 🌱 Quick Start

1. **Clone le repo**
```bash
git clone [votre-repo]
cd gaia_vision
```

2. **Install les dépendances**
```bash
pip install -r requirements.txt
```

3. **Lance le système complet**
```bash
python start.py
```

4. **Go to** http://localhost:5000 et test !

5. **Arrêt propre du système**
```bash
python stop.py
```

## 📈 Performances

**Modèle Vision (SSD MobileNet V2) :**
- Détection d'objets : champignons sains/contaminés

**Modèle ML (CatBoost) :**
- Features importantes : analysées et documentées

## 🎓 Pour la soutenance

Ce projet montre mes compétences en :
- **Développement IA end-to-end** (de l'idée au déploiement)
- **Architecture logicielle** (API, versioning, tests)
- **Choix technologiques** justifiés et pragmatiques
- **Expérience utilisateur** (interface intuitive)
- **Qualité du code** (documentation, structure, bonnes pratiques)

## 👥 Équipe & Remerciements

**Équipe de développement :**
- **Développeur principal** : Développeur & Consultant IA
- **Maxime Cadieux** : Consultant IA - [maxcadieux@gmail.com](mailto:maxcadieux@gmail.com)
- **Namisita Bakayoko** : Consultant IA - [naminsita.bakayoko@gmail.com](mailto:naminsita.bakayoko@gmail.com)

**Remerciements spéciaux :**
- **CENTER FOR AGRIC-BUSINESS INNOVATIONS** - [WhatsApp: +237 98257837](https://wa.me/237982578375)
- *Merci pour leur temps précieux et la fourniture des photos d'entraînement essentielles au développement des modèles IA*

## 🤝 Contact

Des questions ? Des suggestions ? N'hésitez pas !

---

*Développé avec 🌱 pour Alyra - 2025*
