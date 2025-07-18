# 🎤 Questions Probables du Jury - Soutenance GaiaSense Vision

## 🧠 Questions Techniques - Machine Learning & Deep Learning

### 🔥 Deep Learning (SSD MobileNet V2)

**Q1 : "Pourquoi avoir choisi SSD MobileNet V2 plutôt que YOLO ou R-CNN ?"**
- **Réponse** : SSD utilise un CNN (MobileNet V2) comme backbone avec des couches de détection multi-échelles. MobileNet V2 est optimisé pour la production (léger, rapide). SSD offre un bon compromis vitesse/précision avec détection en une seule passe, contrairement à R-CNN qui nécessite plusieurs passes.

**Q2 : "Comment avez-vous géré l'overfitting avec si peu de données ?"**
- **Réponse** : Transfer learning depuis un modèle pré-entraîné, data augmentation, early stopping, dropout dans les couches finales.

**Q3 : "60% de F1-Score, c'est suffisant pour un système de production ?"**
- **Réponse** : Pour un prototype oui, c'est un bon point de départ. Amélioration possible avec plus de données annotées, fine-tuning plus poussé, ou ensemble de modèles.

**Q4 : "Expliquez-moi le processus de Transfer Learning que vous avez utilisé."**
- **Réponse** : Modèle pré-entraîné sur COCO → gel des couches de base → fine-tuning des couches de classification → adaptation aux classes champignons.

**Q5 : "Quelles métriques avez-vous utilisées pour évaluer votre modèle de vision ?"**
- **Réponse** : F1-Score, Précision, Rappel avec sklearn.metrics, IoU pour les bounding boxes, courbes de loss dans TensorBoard.

**Q5bis : "Expliquez-moi l'architecture SSD MobileNet V2 que vous utilisez."**
- **Réponse** : SSD = Single Shot Detector utilise un CNN (MobileNet V2) comme backbone pour extraire les features, puis ajoute des couches de détection multi-échelles pour prédire les bounding boxes et classes en une seule passe. MobileNet V2 utilise des blocs résiduels inversés avec convolutions séparables pour être léger et rapide.

**Q5ter : "Pourquoi avoir choisi 320x320 comme taille d'image pour SSD ?"**
- **Réponse** : C'est la taille standard du checkpoint pré-entraîné SSD MobileNet V2 sur COCO. 320x320 offre un bon compromis entre précision et vitesse d'inférence (~150ms). Plus petit (224x224) serait plus rapide mais moins précis, plus grand (640x640) plus précis mais plus lent.

**Q5quater : "Comment avez-vous géré la conversion CSV vers TFRecord ?"**
- **Réponse** : J'ai créé une fonction `create_tf_example` qui lit chaque ligne du CSV d'annotations, charge l'image correspondante, normalise les coordonnées des bounding boxes (0-1), et crée un exemple TFRecord avec toutes les métadonnées nécessaires pour l'entraînement SSD.

**Q5quinquies : "Qu'est-ce que TensorBoard vous a apporté dans ce projet ?"**
- **Réponse** : TensorBoard m'a permis de monitorer l'entraînement en temps réel : courbes de loss, métriques de validation, visualisation des images avec bounding boxes, et debugging des problèmes de convergence. Essential pour ajuster les hyperparamètres.

### 🐱 Machine Learning (CatBoost)

**Q6 : "Pourquoi CatBoost plutôt que Random Forest ou XGBoost ?"**
- **Réponse** : CatBoost gère naturellement les variables catégorielles sans encoding, plus robuste aux outliers, moins de preprocessing requis.

**Q7 : "Comment avez-vous validé votre modèle CatBoost ?"**
- **Réponse** : Validation croisée K-fold, split temporel si données temporelles, métriques de stabilité, feature importance.

**Q8 : "Quelles sont les features les plus importantes dans votre modèle ?"**
- **Réponse** : Dans mon modèle CatBoost, les features les plus importantes sont le type de champignon, le substrat, et le jour d'inoculation. J'ai utilisé l'analyse feature_importance de CatBoost pour identifier ces variables clés.

**Q8bis : "Pourquoi 1200 iterations pour CatBoost ?"**
- **Réponse** : J'ai testé différentes valeurs et utilisé l'early stopping avec 20 rounds. 1200 iterations avec learning_rate=0.1 et depth=3 donnait le meilleur compromis performance/temps sur GPU. L'early stopping évite l'overfitting.

**Q8ter : "Comment avez-vous géré les variables catégorielles dans CatBoost ?"**
- **Réponse** : CatBoost gère nativement les variables catégorielles sans preprocessing. J'ai défini cat_features=["champignon", "substrat", "Jour_inoculation"] et CatBoost applique automatiquement son algorithme de target encoding optimisé.

**Q8quater : "Quelle différence entre le format .cbm et .joblib pour sauvegarder CatBoost ?"**
- **Réponse** : .cbm est le format natif CatBoost (plus compact, chargement plus rapide), .joblib est plus universel pour l'intégration avec d'autres outils Python. J'ai sauvegardé les deux pour plus de flexibilité dans l'API.

**Q9 : "85% d'accuracy, comment être sûr que ce n'est pas du data leakage ?"**
- **Réponse** : Split strict train/validation/test, pas de preprocessing avant split, validation croisée, vérification des corrélations.

## 🏗️ Questions Architecture & Technique

### 🔧 Architecture Système

**Q10 : "Pourquoi avoir choisi une architecture hybride DL + ML ?"**
- **Réponse** : Exploite les forces de chaque approche : DL pour les images, ML pour les données tabulaires. Modularité et possibilité d'évolution indépendante.

**Q11 : "Comment vos deux modèles communiquent-ils ?"**
- **Réponse** : API REST avec FastAPI, chaque modèle est un service indépendant, orchestration via l'API centrale.

**Q12 : "Que se passe-t-il si un des modèles tombe en panne ?"**
- **Réponse** : Système de fallback, mode dégradé, gestion d'erreurs, monitoring des services.

**Q13 : "Comment gérez-vous les montées en charge ?"**
- **Réponse** : Architecture modulaire, possibilité de scale horizontalement, cache des prédictions, optimisation des modèles.

### 🚀 Déploiement & Production

**Q14 : "Votre système est-il prêt pour la production ?"**
- **Réponse** : Prototype avancé avec architecture production-ready, gestion d'erreurs, logging, versioning. Nécessite monitoring et tests de charge.

**Q15 : "Comment gérez-vous les nouvelles versions de modèles ?"**
- **Réponse** : Système de versioning automatique, rollback possible, tests A/B, déploiement graduel.

**Q16 : "Quelles sont les vulnérabilités de sécurité potentielles ?"**
- **Réponse** : Validation des inputs, authentification API, chiffrement des données, limite de taille des uploads.

## 💡 Questions Méthodologiques

### 📊 Données & Preprocessing

**Q17 : "Comment avez-vous constitué votre dataset ?"**
- **Réponse** : Partenariat avec CENTER FOR AGRIC-BUSINESS INNOVATIONS, photos réelles terrain, annotation manuelle, équilibrage des classes.

**Q18 : "Quelles techniques de data augmentation avez-vous utilisées ?"**
- **Réponse** : Rotation, flip, zoom, changement de luminosité, bruit gaussien, préservation des bounding boxes.

**Q19 : "Comment gérez-vous les données déséquilibrées ?"**
- **Réponse** : Échantillonnage équilibré, weights dans la loss function, métriques adaptées (F1-Score vs Accuracy).

### 🔍 Validation & Testing

**Q20 : "Comment avez-vous splité vos données ?"**
- **Réponse** : Split stratifié 70/15/15 train/validation/test, attention aux données temporelles si applicable.

**Q21 : "Avez-vous testé d'autres architectures ?"**
- **Réponse** : J'ai d'abord exploré différentes options : YOLO (plus rapide mais moins précis), R-CNN (plus précis mais plus lent), EfficientNet (bon compromis mais plus complexe à déployer). J'ai choisi SSD MobileNet V2 pour l'équilibre performance/vitesse/taille.

**Q21bis : "Comment avez-vous configuré votre pipeline d'entraînement ?"**
- **Réponse** : J'ai utilisé 30k steps (~40 epochs), batch_size=16 (optimisé pour RTX 4080), learning_rate=0.02 avec cosine schedule, et warmup_steps=1500. Configuration dans un fichier pipeline.config pour TensorFlow Object Detection API.

**Q21ter : "Pourquoi avoir choisi un batch_size de 16 ?"**
- **Réponse** : C'est optimisé pour ma GPU RTX 4080. J'ai testé différentes valeurs : 8 (sous-utilise la GPU), 32 (out of memory), 16 donne le meilleur compromis vitesse/stabilité d'entraînement avec images 320x320.

**Q21quater : "Comment avez-vous géré le Transfer Learning ?"**
- **Réponse** : J'ai utilisé un checkpoint pré-entraîné sur COCO, gelé les couches backbone, et fine-tuné seulement les couches de classification sur mes 2 classes (Healthy/Contaminated). Cela accélère l'entraînement et améliore la généralisation.

**Q22 : "Comment savez-vous que votre modèle généralise bien ?"**
- **Réponse** : Test sur données jamais vues, validation croisée, métriques sur différents sous-ensembles.

## 🎯 Questions Business & Impact

### 💼 Valeur Ajoutée

**Q23 : "Quelle est la valeur business de votre solution ?"**
- **Réponse** : Automatisation détection, réduction erreurs humaines, scalabilité, traçabilité, aide à la décision.

**Q24 : "Qui sont vos utilisateurs cibles ?"**
- **Réponse** : Agriculteurs, centres de recherche agricole, coopératives, inspecteurs qualité.

**Q25 : "Quel est le ROI estimé de votre solution ?"**
- **Réponse** : [À adapter] Réduction temps inspection, diminution pertes, amélioration qualité produits.

### 🔮 Évolutions & Limitations

**Q26 : "Quelles sont les limites actuelles de votre système ?"**
- **Réponse** : Précision perfectible, dataset limité, environnement contrôlé, pas de temps réel strict.

**Q27 : "Comment voyez-vous l'évolution future du projet ?"**
- **Réponse** : Plus de données, modèles plus complexes, déploiement cloud, mobile app, IoT intégration.

**Q28 : "Que feriez-vous différemment si c'était à refaire ?"**
- **Réponse** : Plus de données dès le début, tests A/B, architecture microservices, monitoring avancé.

## 🎓 Questions Pédagogiques

### 📚 Apprentissage & Compétences

**Q29 : "Qu'avez-vous appris de plus important dans ce projet ?"**
- **Réponse** : Importance de l'architecture, choix techniques justifiés, cycle complet ML/DL, contraintes production.

**Q30 : "Quelles ont été vos principales difficultés ?"**
- **Réponse** : Données limitées, optimisation performances, déploiement, debugging.

**Q31 : "Comment vous êtes-vous formé sur ces technologies ?"**
- **Réponse** : Formation Alyra, documentation officielle, projets pratiques, communauté, youtube, intelligence artificiel.

### 🔬 Méthodes & Outils

**Q32 : "Quels outils avez-vous utilisés pour le développement ?"**
- **Réponse** : Jupyter notebooks, TensorBoard, Git, VS Code, .

**Q33 : "Comment avez-vous géré le versioning de votre code ?"**
- **Réponse** : Git avec branches features, commits atomiques, documentation, tests avant merge. 

**Q34 : "Avez-vous fait des tests unitaires ?"**
- **Réponse** : Tests sur les fonctions critiques, validation des inputs/outputs, tests d'intégration API.

## 🤔 Questions Pièges Potentielles

### ⚠️ Questions Critiques

**Q35 : "Votre F1-Score de 60% n'est-il pas trop faible ?"**
- **Réponse** : Pour un prototype avec données limitées, c'est acceptable. Amélioration possible avec plus de données et fine-tuning.

**Q36 : "Pourquoi pas utiliser une solution existante comme Google Vision API ?"**
- **Réponse** : Spécificité du domaine agricole, contrôle total, coût à long terme, apprentissage technique.

**Q37 : "N'auriez-vous pas pu faire plus simple avec juste un modèle ?"**
- **Réponse** : Approche hybride justifiée par la nature différente des données (images vs tabulaires), meilleure modularité.

**Q38 : "Comment être sûr que votre modèle ne discrimine pas ?"**
- **Réponse** : Validation sur différents sous-groupes, métriques de fairness, dataset équilibré.

**Q39 : "Avec seulement 2 classes, était-ce vraiment nécessaire d'utiliser SSD ?"**
- **Réponse** : SSD ne fait pas que classifier, il fait de la détection d'objets avec localisation. Même avec 2 classes, il faut détecter ET localiser les champignons dans l'image avec des bounding boxes.

**Q40 : "Pourquoi TensorFlow Object Detection API au lieu de PyTorch ?"**
- **Réponse** : TensorFlow OD API offre des checkpoints pré-entraînés compatibles, pipeline.config standardisé, et TensorBoard intégré. Plus mature pour la détection d'objets en production.

**Q41 : "Votre dataset est-il assez grand pour éviter l'overfitting ?"**
- **Réponse** : J'ai utilisé transfer learning, data augmentation, early stopping, et validation croisée. Le modèle généralise bien sur le test set grâce à ces techniques de régularisation.

**Q42 : "Comment justifier le choix des hyperparamètres CatBoost ?"**
- **Réponse** : depth=3 évite l'overfitting, iterations=1200 avec early stopping trouve le bon équilibre, learning_rate=0.1 sur GPU donne une convergence stable. J'ai testé différentes combinaisons.

---

## 💡 Conseils pour Répondre

### ✅ Bonnes Pratiques
- **Soyez honnête** sur les limitations
- **Justifiez vos choix** techniques
- **Donnez des exemples concrets**
- **Montrez votre compréhension** des enjeux
- **Préparez des schémas** si nécessaire

### ❌ À Éviter
- Réponses vagues ou évasives
- "Je ne sais pas" sans proposition
- Sur-vendre les performances
- Ignorer les limitations
- Réponses trop techniques sans contexte

---

## 🎯 Questions à Poser au Jury

**En fin de soutenance, vous pouvez poser :**
- "Auriez-vous fait des choix différents ?"
- "Que pensez-vous du potentiel de cette approche ?"
- "Quels domaines me conseillez-vous d'approfondir ?"

---

*Préparé pour la soutenance Alyra - 2025*
*Bonne chance ! 🍀*
