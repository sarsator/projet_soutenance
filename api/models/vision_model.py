import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Union

# Logger pour suivre ce qui se passe
logger = logging.getLogger(__name__)

class VisionModel:
    """
    Mon wrapper pour le modèle de vision.
    
    Supporte maintenant deux types de modèles :
    - EfficientNetB0 (ancien format Keras) pour la classification
    - SSD MobileNet V2 (nouveau format SavedModel) pour la détection d'objets
    
    Le modèle SSD détecte et classifie les champignons (sains/contaminés)
    avec des performances exceptionnelles !
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialise le modèle de vision.
        
        Args:
            model_path: Chemin vers le modèle sauvegardé (optionnel)
        """
        self.model = None
        self.model_type = None  # 'keras' ou 'savedmodel'
        self.metadata = None
        
        if model_path:
            # Chemin spécifique fourni
            self.model_path = model_path
        else:
            # Logique automatique : essayer le versioning d'abord (SavedModel)
            current_dir = Path(__file__).parent
            
            # Priorité 1: Système de versioning avec SavedModel
            versions_path = current_dir / "dl_model" / "versions"
            if versions_path.exists():
                # Trouver la version la plus récente
                latest_version = self._find_latest_version(versions_path)
                if latest_version:
                    self.model_path = latest_version
                    logger.info(f"Modele vision via versioning: {latest_version.name}")
                else:
                    self.model_path = self._fallback_path(current_dir)
            else:
                self.model_path = self._fallback_path(current_dir)
        
        # Configuration par défaut (sera mise à jour selon le type de modèle)
        self.class_names = ["contamine", "sain"]
        self.input_size = (640, 640)
        
    def _find_latest_version(self, versions_path: Path) -> Path:
        """Trouve la version la plus récente dans le dossier versions"""
        
        # Priorité 1: Utiliser le symlink 'current' s'il existe
        current_symlink = versions_path / "current"
        if current_symlink.exists() and current_symlink.is_symlink():
            target_path = current_symlink.resolve()
            if target_path.exists():
                logger.info(f"Utilisation du symlink current: {target_path.name}")
                return target_path / "saved_model"
        
        # Fallback: Recherche manuelle de la version la plus récente
        import re
        from datetime import datetime
        
        versions = []
        for folder in versions_path.iterdir():
            if folder.is_dir() and folder.name != "current":
                # Format: v1.3_20250716_132518
                match = re.match(r'v(\d+)\.(\d+)_(\d{8}_\d{6})', folder.name)
                if match:
                    major, minor, timestamp = match.groups()
                    try:
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        versions.append((int(major), int(minor), dt, folder))
                    except ValueError:
                        continue
        
        if versions:
            # Trier par version puis par timestamp
            versions.sort(key=lambda x: (x[0], x[1], x[2]))
            latest_folder = versions[-1][3]
            return latest_folder / "saved_model"
        
        return None
    
    def _fallback_path(self, current_dir: Path) -> Path:
        """Chemin de fallback vers les anciens modeles"""
        paths_to_try = [
            current_dir / "dl_model" / "current",
            current_dir / "dl_model" / "final_model.keras"
        ]
        
        for path in paths_to_try:
            if path.exists():
                logger.info(f"Fallback vers modele: {path.name}")
                return path
        
        # Si rien n'existe, retourner le dernier pour debugging
        return paths_to_try[-1] 
        
    def charger_modele(self) -> bool:
        """
        Charge le modèle de vision depuis le fichier.
        Supporte maintenant les formats Keras et SavedModel.
        
        Returns:
            bool: True si ça marche, False sinon
        """
        try:
            model_path = Path(self.model_path)
            
            # Gestion des liens symboliques
            if model_path.is_symlink():
                model_path = model_path.resolve()
                logger.info(f"Lien symbolique resolu vers: {model_path}")
            
            # Detecter le type de modele et charger
            if self._is_savedmodel_format(model_path):
                return self._load_savedmodel(model_path)
            else:
                return self._load_keras_model(model_path)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle de vision: {e}")
            return False
    
    def _is_savedmodel_format(self, model_path: Path) -> bool:
        """Détermine si le chemin contient un SavedModel"""
        # Vérifier si c'est un dossier de version avec saved_model
        if model_path.is_dir():
            saved_model_dir = model_path / "saved_model"
            if saved_model_dir.exists() and (saved_model_dir / "saved_model.pb").exists():
                return True
            
            # Vérifier si c'est directement un dossier SavedModel
            if (model_path / "saved_model.pb").exists():
                return True
        
        return False
    
    def _load_savedmodel(self, model_path: Path) -> bool:
        """Charge un modèle au format SavedModel (SSD)"""
        try:
            # Charger les métadonnées si disponibles
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadonnees chargees: {self.metadata.get('architecture', 'Unknown')}")
            
            # Déterminer le chemin du SavedModel
            saved_model_dir = model_path / "saved_model"
            if not saved_model_dir.exists():
                saved_model_dir = model_path
            
            logger.info(f"Chargement du SavedModel depuis: {saved_model_dir}")
            logger.info(f"Taille: {self._get_dir_size(saved_model_dir):.1f} MB")
            
            # Charger le SavedModel
            self.model = tf.saved_model.load(str(saved_model_dir))
            self.model_type = 'savedmodel'
            
            # Configuration pour SSD MobileNet V2
            self.input_size = (320, 320)  # SSD utilise 320x320
            self.class_names = ["background", "healthy", "contaminated"]  # Classes SSD
            
            # Vérifier les signatures
            signatures = list(self.model.signatures.keys())
            logger.info(f"Signatures disponibles: {signatures}")
            
            if 'serving_default' in signatures:
                serving_fn = self.model.signatures['serving_default']
                logger.info("Signature 'serving_default' trouvee")
                
                # Log des inputs/outputs
                for name, spec in serving_fn.structured_input_signature[1].items():
                    logger.info(f"  Input {name}: {spec.shape} ({spec.dtype})")
                
                for name, spec in serving_fn.structured_outputs.items():
                    logger.info(f"  Output {name}: {spec.shape}")
            
            logger.info("SavedModel charge avec succes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du SavedModel: {e}")
            return False
    
    def _load_keras_model(self, model_path: Path) -> bool:
        """Charge un modèle au format Keras (ancien EfficientNet)"""
        try:
            if model_path.is_file():
                model_file = model_path
            elif model_path.is_dir():
                # Chercher dans l'ordre de préférence
                keras_files = list(model_path.glob("*.keras"))
                h5_files = list(model_path.glob("*.h5"))
                
                if keras_files:
                    model_file = keras_files[0]
                elif h5_files:
                    model_file = h5_files[0]
                else:
                    raise FileNotFoundError(f"Aucun fichier Keras trouvé dans: {model_path}")
            else:
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
            
            logger.info(f"Chargement du modèle Keras depuis: {model_file}")
            logger.info(f"Taille du fichier: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
            
            self.model = tf.keras.models.load_model(str(model_file))
            self.model_type = 'keras'
            
            # Configuration pour EfficientNet
            self.input_size = (640, 640)
            self.class_names = ["contamine", "sain"]
            
            # Vérifier la structure du modèle chargé
            logger.info(f"Modèle chargé: {self.model.name if hasattr(self.model, 'name') else 'Sans nom'}")
            logger.info(f"Nombre de sorties: {len(self.model.outputs)}")
            for i, output in enumerate(self.model.outputs):
                logger.info(f"  Sortie {i}: {output.name} - Shape: {output.shape}")
            
            logger.info("Modele Keras charge avec succes")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modele Keras: {e}")
            return False
    
    def _get_dir_size(self, dir_path: Path) -> float:
        """Calcule la taille d'un dossier en MB"""
        total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
        return total_size / 1024 / 1024
    
    def preprocess_image(self, image_path: str) -> Union[np.ndarray, tf.Tensor]:
        """
        Préprocesse une image pour la prédiction selon le type de modèle
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image préprocessée (format dépend du type de modèle)
        """
        try:
            # Charger l'image
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(self.input_size)
            
            if self.model_type == 'savedmodel':
                # Pour SSD: format uint8, pas de normalisation
                img_array = np.array(img, dtype=np.uint8)
                # SSD attend (1, height, width, 3)
                img_tensor = tf.convert_to_tensor(img_array)
                img_tensor = tf.expand_dims(img_tensor, 0)
                return img_tensor
                
            else:
                # Pour Keras/EfficientNet: normalisation float32
                img_array = np.array(img, dtype=np.float32)
                img_array = img_array / 255.0  # Normalisation [0, 1]
                # Ajouter la dimension batch
                img_array = np.expand_dims(img_array, axis=0)
                return img_array
            
        except Exception as e:
            logger.error(f"Erreur lors du préprocessing de l'image: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Effectue une prédiction sur une image selon le type de modèle
        
        Args:
            image_path: Chemin vers l'image à analyser
            
        Returns:
            Dict contenant la prédiction et la confiance
        """
        if self.model is None:
            if not self.charger_modele():
                raise RuntimeError("Impossible de charger le modèle de vision")
        
        try:
            # Préprocesser l'image
            preprocessed_img = self.preprocess_image(image_path)
            
            if self.model_type == 'savedmodel':
                return self._predict_savedmodel(preprocessed_img)
            else:
                return self._predict_keras(preprocessed_img)
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction vision: {e}")
            raise
    
    def _analyze_contamination_score(self, score: float, class_type: str) -> tuple:
        """Analyse un score de contamination et retourne prediction et confidence"""
        if class_type == "contaminated":
            if score >= 0.30:
                return "contamine", score * 1.3
            elif score >= 0.20:
                return "contamine", score * 1.5
            elif score >= 0.15:
                return "incertain", score * 1.2
            else:
                return "sain", 0.6
        else:  # healthy
            if score >= 0.7:
                return "sain", score * 1.1
            elif score < 0.15:
                return "sain", score * 4.0
            elif score < 0.3:
                return "sain", score * 3.0
            else:
                return "sain", score * 2.0

    def _analyze_mixed_detection(self, max_contaminated_score: float, max_healthy_score: float) -> dict:
        """Analyse le cas mixte contamination + sain"""
        if max_contaminated_score >= 0.25:
            return {
                "prediction": "contamine",
                "confidence": min(max_contaminated_score * 1.5, 0.90),
                "contamination_probability": min(max_contaminated_score * 1.3, 0.95)
            }
        elif max_contaminated_score >= 0.15:
            logger.warning(f"Contamination possible detectee: score={max_contaminated_score:.3f}")
            return {
                "prediction": "incertain", 
                "confidence": max_contaminated_score * 1.2,
                "contamination_probability": max_contaminated_score * 1.5
            }
        else:
            if max_healthy_score > max_contaminated_score:
                confidence = max_healthy_score * (2.5 if max_healthy_score < 0.2 else 1.5)
                return {
                    "prediction": "sain",
                    "confidence": min(confidence, 0.75),
                    "contamination_probability": max_contaminated_score
                }
            else:
                return {
                    "prediction": "incertain",
                    "confidence": 0.3,
                    "contamination_probability": 0.4
                }

    def _predict_savedmodel(self, img_tensor: tf.Tensor) -> Dict[str, Any]:
        """Prediction avec le modele SSD SavedModel"""
        try:
            # Obtenir la fonction d'inference
            infer = self.model.signatures['serving_default']
            
            # Effectuer l'inference
            predictions = infer(input_tensor=img_tensor)
            
            # Extraire les resultats
            detection_boxes = predictions['detection_boxes'].numpy()[0]
            detection_classes = predictions['detection_classes'].numpy()[0]
            detection_scores = predictions['detection_scores'].numpy()[0]
            num_detections = int(predictions['num_detections'].numpy()[0])
            
            # Analyser les detections valides
            valid_detections = []
            contaminated_count = 0
            healthy_count = 0
            max_contaminated_score = 0.0
            max_healthy_score = 0.0
            
            # Log pour debug
            logger.info("=== ANALYSE DETAILLEE DES DETECTIONS ===")
            for i in range(min(10, len(detection_scores))):
                score = float(detection_scores[i])
                class_id = int(detection_classes[i])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                logger.info(f"  Detection {i}: classe={class_name} (id={class_id}), score={score:.4f}")
            
            for i in range(min(num_detections, len(detection_scores))):
                score = float(detection_scores[i])
                if score > 0.08:
                    class_id = int(detection_classes[i])
                    box = detection_boxes[i].tolist()
                    
                    valid_detections.append({
                        "class_id": class_id,
                        "class_name": self.class_names[class_id] if class_id < len(self.class_names) else "unknown",
                        "score": score,
                        "box": box
                    })
                    
                    # Compter par type
                    if class_id == 2:  # contaminated
                        contaminated_count += 1
                        max_contaminated_score = max(max_contaminated_score, score)
                    elif class_id == 1:  # healthy
                        healthy_count += 1
                        max_healthy_score = max(max_healthy_score, score)
            
            # Determiner la prediction finale
            if contaminated_count > 0 and healthy_count > 0:
                # Cas mixte
                contaminated_ratio = contaminated_count / (contaminated_count + healthy_count)
                logger.info(f"Analyse mixte: {contaminated_count} contamine(s), {healthy_count} sain(s)")
                logger.info(f"Ratio contamination: {contaminated_ratio:.3f}")
                logger.info(f"Meilleur score contamine: {max_contaminated_score:.3f}")
                logger.info(f"Meilleur score sain: {max_healthy_score:.3f}")
                
                result = self._analyze_mixed_detection(max_contaminated_score, max_healthy_score)
                prediction = result["prediction"]
                confidence = result["confidence"]
                contamination_probability = result["contamination_probability"]
                        
            elif contaminated_count > 0:
                # Seulement contamination
                logger.info(f"Contamination pure detectee: score max {max_contaminated_score:.3f}")
                prediction, confidence = self._analyze_contamination_score(max_contaminated_score, "contaminated")
                confidence = min(confidence, 0.85)
                contamination_probability = min(max_contaminated_score * 1.8, 0.90)
                
            elif healthy_count > 0:
                # Seulement sain - verification supplementaire
                if max_healthy_score < 0.7:
                    # Chercher contamination faible
                    low_contaminated_scores = [
                        float(detection_scores[i]) for i in range(min(num_detections, len(detection_scores)))
                        if int(detection_classes[i]) == 2 and float(detection_scores[i]) > 0.1
                    ]
                    
                    if low_contaminated_scores:
                        max_low_contaminated = max(low_contaminated_scores)
                        logger.info(f"Detections contamination faibles trouvees: max={max_low_contaminated:.3f}")
                        
                        if healthy_count >= 3 and max_healthy_score < 0.65 and max_low_contaminated > 0.15:
                            prediction = "incertain"
                            confidence = max_healthy_score * 0.6
                            contamination_probability = 0.4
                            logger.warning("Detection ambigue: objets classes 'sains' mais scores moderes avec traces de contamination")
                        else:
                            prediction = "sain"
                            confidence = max_healthy_score * 0.85
                            contamination_probability = 1.0 - max_healthy_score
                    else:
                        prediction, confidence = self._analyze_contamination_score(max_healthy_score, "healthy")
                        confidence = min(confidence, 0.75)
                        contamination_probability = 1.0 - max_healthy_score
                else:
                    prediction, confidence = self._analyze_contamination_score(max_healthy_score, "healthy")
                    confidence = min(confidence, 0.85)
                    contamination_probability = 1.0 - max_healthy_score
                
            else:
                # Aucune detection valide - analyser les faibles
                weak_detections = []
                very_weak_detections = []
                
                logger.info("=== ANALYSE DES DETECTIONS FAIBLES ===")
                for i in range(min(num_detections, len(detection_scores))):
                    score = float(detection_scores[i])
                    class_id = int(detection_classes[i])
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                    
                    if score > 0.15:
                        weak_detections.append((class_id, score, class_name))
                        logger.info(f"  Detection faible: {class_name} score={score:.4f}")
                    elif score > 0.05:
                        very_weak_detections.append((class_id, score, class_name))
                
                contaminated_weak = [d for d in weak_detections if d[0] == 2]
                healthy_weak = [d for d in weak_detections if d[0] == 1]
                
                if contaminated_weak:
                    best_contaminated = max(contaminated_weak, key=lambda x: x[1])
                    prediction = "incertain"
                    confidence = best_contaminated[1] * 0.5
                    contamination_probability = best_contaminated[1] * 1.2
                    logger.info(f"Contamination faible detectee: score={best_contaminated[1]:.4f}")
                elif healthy_weak:
                    best_healthy = max(healthy_weak, key=lambda x: x[1])
                    prediction = "sain"
                    confidence = best_healthy[1] * 0.7
                    contamination_probability = 1.0 - best_healthy[1]
                elif very_weak_detections and any(d[0] == 2 for d in very_weak_detections):
                    logger.warning("Traces tres faibles de contamination detectees")
                    prediction = "incertain"
                    confidence = 0.25
                    contamination_probability = 0.6
                else:
                    prediction = "incertain"
                    confidence = 0.2
                    contamination_probability = 0.5
            
            logger.info(f"SSD Prediction: {prediction} (confiance: {confidence:.3f})")
            logger.info(f"Detections: {contaminated_count} contamine(s), {healthy_count} sain(s)")
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "model_type": "ssd_object_detection",
                "contamination_probability": float(contamination_probability),
                "detection_summary": {
                    "total_detections": len(valid_detections),
                    "contaminated_count": contaminated_count,
                    "healthy_count": healthy_count,
                    "max_contaminated_score": float(max_contaminated_score),
                    "max_healthy_score": float(max_healthy_score)
                },
                "detections": valid_detections,
                "model_info": {
                    "architecture": self.metadata.get('architecture', 'SSD_MobileNet_V2') if self.metadata else 'SSD_MobileNet_V2',
                    "version": self.metadata.get('version', 'unknown') if self.metadata else 'unknown',
                    "input_size": self.input_size
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prediction SSD: {e}")
            raise
    
    def _predict_keras(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Prédiction avec le modèle Keras (EfficientNet)"""
        try:
            # Prédiction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Le modèle retourne deux sorties: [multi_sac, contaminated_present]
            if isinstance(predictions, list) and len(predictions) == 2:
                multi_sac_pred, contamination_pred = predictions
                
                # Extraction de la prédiction de contamination
                contamination_prob = float(contamination_pred[0][0])
                predicted_class = "contamine" if contamination_prob > 0.5 else "sain"
                confidence = contamination_prob if contamination_prob > 0.5 else 1 - contamination_prob
                
                # Extraction de la prédiction multi_sac
                multi_sac_value = float(multi_sac_pred[0][0])
                
                logger.info(f"Keras Prédiction: {predicted_class} (confiance: {confidence:.3f}), multi_sac: {multi_sac_value:.2f}")
                
                return {
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "model_type": "keras_classification",
                    "contamination_probability": contamination_prob,
                    "multi_sac_value": multi_sac_value,
                    "raw_outputs": {
                        "multi_sac": multi_sac_pred[0].tolist(),
                        "contaminated_present": contamination_pred[0].tolist()
                    }
                }
            else:
                # Fallback pour un seul output (ancien comportement)
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_class_idx])
                    predicted_class = self.class_names[predicted_class_idx]
                else:
                    confidence = float(predictions[0][0])
                    predicted_class = "contamine" if confidence > 0.5 else "sain"
                    confidence = confidence if confidence > 0.5 else 1 - confidence
                
                logger.info(f"Keras Prédiction (fallback): {predicted_class} (confiance: {confidence:.3f})")
                
                return {
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "model_type": "keras_classification",
                    "raw_output": predictions[0].tolist()
                }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction Keras: {e}")
            raise
    
    def est_charge(self) -> bool:
        """Vérifie si le modèle est chargé"""
        return self.model is not None