# � GaiaSense Vision - Mon Projet de Fin d'Études

Salut ! Bienvenue sur mon projet de soutenance pour Alyra. J'ai développé GaiaSense Vision, une solution IA complète qui combine machine learning classique et deep learning pour l'analyse d'images.

## 🎯 Qu'est-ce que c'est ?

En gros, c'est un système intelligent qui peut :
- **Analyser des images** avec un modèle EfficientNetB0 fine-tuné
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
- 🔥 **EfficientNetB0** pour la vision (excellent rapport performance/taille)
- 📊 **Scikit-learn** pour les métriques et la validation

**Backend :**
- ⚡ **FastAPI** pour l'API (auto-documentation Swagger incluse)
- 🗄️ **SQLite** pour les prédictions et logs
- 📦 **Joblib/Keras** pour la sérialisation des modèles

**Frontend :**
- 🌶️ **Flask** pour l'interface web
- 🎨 **CSS/JavaScript** vanilla (pas de framework lourd)
- 📱 **Design responsive** mobile-friendly

**DevOps :**
- 🐍 **Python 3.8+** 
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

3. **Lance l'API**
```bash
cd api
python main.py
```

4. **Lance le frontend** (dans un autre terminal)
```bash
cd frontend
python app.py
```

5. **Go to** http://localhost:5000 et test !

## 📈 Performances

**Modèle Vision (EfficientNetB0) :**
- Accuracy : ~92% sur le test set
- Temps d'inférence : ~200ms par image
- Taille du modèle : ~56MB

**Modèle ML (CatBoost) :**
- Accuracy : ~89% en validation croisée
- Temps d'inférence : ~50ms par prédiction
- Features importantes : analysées et documentées

## 🎓 Pour la soutenance

Ce projet montre mes compétences en :
- **Développement IA end-to-end** (de l'idée au déploiement)
- **Architecture logicielle** (API, versioning, tests)
- **Choix technologiques** justifiés et pragmatiques
- **Expérience utilisateur** (interface intuitive)
- **Qualité du code** (documentation, structure, bonnes pratiques)

## 🤝 Contact

Des questions ? Des suggestions ? N'hésitez pas !

---

*Développé avec 🌱 pour Alyra - 2025*
