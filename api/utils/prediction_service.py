import logging
from typing import Dict, Any, Optional
from pathlib import Path

from api.models.catboost_model import CatBoostModel
from api.models.vision_model import VisionModel
from api.models.model_version_manager import ModelVersionManager

logger = logging.getLogger(__name__)

class PredictionService:
    
    
    def __init__(self, catboost_model_path: str = None, vision_model_path: str = None):
        """
        Initialise le service de prédiction
        
        Args:
            catboost_model_path: Chemin vers le modèle CatBoost
            vision_model_path: Chemin vers le modèle de vision
        """
        self.catboost_model = CatBoostModel(catboost_model_path)
        self.vision_model = VisionModel(vision_model_path)
        self._models_loaded = False
        
        # Initialiser le gestionnaire de versions pour récupérer les infos
        try:
            models_dir = Path(__file__).parent.parent / "models"
            self.version_manager = ModelVersionManager(models_dir)
        except Exception as e:
            logger.warning(f"Impossible d'initialiser le gestionnaire de versions: {e}")
            self.version_manager = None
    
    def get_model_versions(self) -> Dict[str, str]:
        """
        Récupère les versions des modèles actuellement chargés
        
        Returns:
            Dict avec les versions des modèles ML et DL
        """
        versions = {
            "catboost": "v1.0",  # Version CatBoost par défaut
            "vision": "Non disponible"
        }
        
        try:
            # Récupérer la version du modèle Vision directement depuis le modèle chargé
            if self.vision_model and self.vision_model.is_loaded():
                if hasattr(self.vision_model, 'metadata') and self.vision_model.metadata:
                    # Si nous avons des métadonnées (SavedModel), utiliser la version
                    version = self.vision_model.metadata.get('version', 'unknown')
                    versions["vision"] = f"v{version}"
                elif hasattr(self.vision_model, 'model_path'):
                    # Extraire la version depuis le chemin du modèle
                    model_path = str(self.vision_model.model_path)
                    if 'v1.3_' in model_path:
                        versions["vision"] = "v1.3"
                    elif 'v1.2_' in model_path:
                        versions["vision"] = "v1.2"
                    else:
                        versions["vision"] = "v1.0"
                else:
                    versions["vision"] = "v1.0"
                    
            # Fallback avec le gestionnaire de versions si disponible
            if versions["vision"] == "Non disponible" and self.version_manager:
                try:
                    dl_version = self.version_manager.get_latest_version("dl")
                    if dl_version:
                        versions["vision"] = f"v{dl_version['version']}"
                except Exception as e:
                    logger.warning(f"Erreur lors de la récupération des versions via version_manager: {e}")
                    
        except Exception as e:
            logger.warning(f"Erreur lors de la récupération des versions: {e}")
        
        logger.info(f"Versions des modèles récupérées: {versions}")
        return versions
    
    def load_models(self) -> bool:
        """
        Charge les deux modèles
        
        Returns:
            bool: True si tous les modèles sont chargés avec succès
        """
        try:
            logger.info("Chargement des modèles...")
            
            catboost_success = self.catboost_model.load_model()
            vision_success = self.vision_model.load_model()
            
            self._models_loaded = catboost_success and vision_success
            
            if self._models_loaded:
                logger.info("Tous les modèles sont chargés avec succès")
            else:
                logger.warning("Échec du chargement de certains modèles")
                
            return self._models_loaded
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            return False
    
    def predict(self, 
                race_champignon: str,
                type_substrat: str,
                jours_inoculation: int,
                hygrometrie: float,
                co2_ppm: float,
                image_path: str = None) -> Dict[str, Any]:
        """
        Effectue une prédiction orchestrée
        
        Args:
            race_champignon: Race du champignon
            type_substrat: Type de substrat
            jours_inoculation: Nombre de jours depuis l'inoculation
            hygrometrie: Taux d'hygrométrie
            co2_ppm: Taux de CO2 en PPM
            image_path: Chemin vers l'image (optionnel)
            
        Returns:
            Dict contenant les résultats de prédiction
        """
        logger.info("=== DÉBUT DE PRÉDICTION SERVICE ===")
        logger.info(f"Paramètres: race_champignon={race_champignon}, type_substrat={type_substrat}")
        logger.info(f"Valeurs numériques: jours_inoculation={jours_inoculation}, hygrometrie={hygrometrie}, co2_ppm={co2_ppm}")
        logger.info(f"Image fournie: {image_path is not None}")
        
        if not self._models_loaded:
            logger.info("Modèles non chargés, tentative de chargement...")
            if not self.load_models():
                logger.error("❌ Impossible de charger les modèles")
                raise RuntimeError("Impossible de charger les modèles")
            logger.info("✅ Modèles chargés avec succès")
        
        try:
            # Étape 1: Prédiction CatBoost
            logger.info("=== ÉTAPE 1: Prédiction CatBoost ===")
            
            # Préparation des données pour CatBoost
            input_data = {
                "race_champignon": race_champignon,
                "type_substrat": type_substrat,
                "jours_inoculation": jours_inoculation,
                "hygrometrie": hygrometrie,
                "co2_ppm": co2_ppm
            }
            logger.info(f"Données préparées pour CatBoost: {input_data}")
            
            catboost_result = self.catboost_model.predict(input_data)
            logger.info(f"✅ Résultat CatBoost: {catboost_result}")
            
            # Structure de réponse de base
            response = {
                "catboost_prediction": catboost_result,
                "vision_prediction": None,
                "final_decision": None,
                "confidence_score": catboost_result["confidence"],  # CORRIGÉ: Utiliser confidence (float) au lieu de probability (list)
                "confidence_source": "catboost",  # Nouvelle information sur la source de confiance
                "models_used": ["catboost"],
                "analysis_steps": ["catboost_analysis"],
                "multi_sac_count": None  # Nouvelle information sur le nombre de sacs
            }
            
            # Étape 2: Décision d'utiliser la vision
            catboost_prediction = catboost_result["prediction"]
            catboost_probability = catboost_result["probability"]
            catboost_confidence = catboost_result["confidence"]  # Utiliser la confiance pour les logs
            use_vision = False
            
            if catboost_result["risk_level"] == "high":  # Risque élevé (probabilité > 0.49)
                use_vision = True
                logger.info(f"CatBoost détecte un risque élevé (confiance={catboost_confidence:.3f}), passage au modèle de vision")
            else:
                logger.info(f"CatBoost détecte un risque faible (confiance={catboost_confidence:.3f}), pas de passage au modèle de vision")
            
            # Étape 3: Prédiction Vision si nécessaire
            if use_vision and image_path:
                if not Path(image_path).exists():
                    logger.warning(f"Image non trouvée: {image_path}")
                    response["final_decision"] = catboost_prediction
                    response["warning"] = "Image non trouvée, utilisation du résultat CatBoost uniquement"
                else:
                    try:
                        logger.info("Étape 2: Prédiction Vision")
                        vision_result = self.vision_model.predict(image_path)
                        response["vision_prediction"] = vision_result
                        response["models_used"].append("vision")
                        response["analysis_steps"].append("vision_analysis")
                        
                        # Ajouter les informations de Vision
                        response["multi_sac_count"] = vision_result.get("multi_sac_value", None)
                        response["confidence_source"] = "vision"  # La confiance vient maintenant de Vision
                        
                        # Combinaison des résultats
                        response["final_decision"] = self._combine_predictions(
                            catboost_result, vision_result
                        )
                        response["confidence_score"] = vision_result.get("confidence", 0.5)  # Utiliser directement la confiance de Vision
                        
                    except Exception as e:
                        logger.error(f"Erreur lors de la prédiction vision: {e}")
                        response["final_decision"] = catboost_prediction
                        response["error"] = f"Erreur vision: {str(e)}"
            else:
                # Utilisation du résultat CatBoost uniquement
                response["final_decision"] = catboost_prediction
                if not image_path:
                    response["note"] = "Aucune image fournie, utilisation du modèle CatBoost uniquement"
            
            # Ajouter les versions des modèles
            response["model_versions"] = self.get_model_versions()
            
            logger.info(f"✅ Prédiction finale: {response['final_decision']}")
            logger.info("=== FIN DE PRÉDICTION SERVICE ===")
            return response
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ ERREUR CRITIQUE dans prediction_service:")
            logger.error(f"Type d'erreur: {type(e).__name__}")
            logger.error(f"Message: {str(e)}")
            logger.error(f"Traceback complet:\n{error_details}")
            logger.error("=== FIN DE PRÉDICTION SERVICE (ERREUR) ===")
            raise
    
    def _combine_predictions(self, catboost_result: Dict, vision_result: Dict) -> str:
        """
        Combine les prédictions des deux modèles
        PRIORITÉ AU MODÈLE VISION car il analyse l'image réelle
        
        Args:
            catboost_result: Résultat CatBoost
            vision_result: Résultat Vision
            
        Returns:
            str: Prédiction finale
        """
        catboost_pred = catboost_result["prediction"]
        catboost_risk = catboost_result["risk_level"]
        vision_pred = vision_result["prediction"]
        vision_confidence = vision_result.get("confidence", 0.5)
        
        logger.info(f"Combinaison des prédictions:")
        logger.info(f"  - CatBoost: {catboost_pred} (risque: {catboost_risk})")
        logger.info(f"  - Vision: {vision_pred} (confiance: {vision_confidence:.3f})")
        
        # NOUVEAU: Gestion du cas "incertain" de Vision
        if vision_pred == "incertain":
            logger.info("Vision incertain -> utilisation de CatBoost avec prudence")
            # Si CatBoost est très confiant, on peut s'y fier
            catboost_prob = max(catboost_result["probability"])
            if catboost_risk == "high" and catboost_prob > 0.9:
                return "contamine"
            else:
                return "sain"  # En cas de doute, privilégier "sain"
        
        # NOUVEAU: Gestion des scores modérés de Vision (0.3-0.6)
        if 0.3 <= vision_confidence <= 0.6:
            logger.info(f"Vision avec confiance modérée ({vision_confidence:.3f})")
            # Dans ce cas, on fait plus confiance à la cohérence entre les deux modèles
            if catboost_risk == "high" and vision_pred == "contamine":
                logger.info("Accord CatBoost (risque élevé) + Vision (contaminé) -> contaminé")
                return "contamine"
            elif catboost_risk == "low" and vision_pred == "sain":
                logger.info("Accord CatBoost (risque faible) + Vision (sain) -> sain")
                return "sain"
            else:
                # Désaccord avec confiance modérée -> être conservateur
                logger.info("Désaccord avec confiance modérée -> sain par précaution")
                return "sain"
        
        # Si Vision est confiant (>60%), on suit sa décision
        elif vision_confidence > 0.60:
            logger.info(f"Vision confiant ({vision_confidence:.3f}), décision: {vision_pred}")
            return vision_pred
        
        # Si Vision est moyennement confiant (>40%), on considère les deux modèles
        elif vision_confidence > 0.40:
            if catboost_risk == "high" and vision_pred == "contamine":
                # Les deux modèles s'accordent sur la contamination
                logger.info("Accord entre CatBoost (risque élevé) et Vision (contaminé)")
                return "contamine"
            elif catboost_risk == "low" and vision_pred == "sain":
                # Les deux modèles s'accordent sur la propreté
                logger.info("Accord entre CatBoost (risque faible) et Vision (sain)")
                return "sain"
            else:
                # Désaccord : on privilégie Vision car il voit l'image réelle
                logger.info(f"Désaccord - CatBoost: {catboost_risk}, Vision: {vision_pred}. Priorité à Vision.")
                return vision_pred
        
        # Si Vision n'est pas confiant (<40%), on utilise CatBoost en backup AVEC PRUDENCE
        else:
            logger.warning(f"Vision peu confiant ({vision_confidence:.3f})")
            # NOUVEAU: Si CatBoost dit "risque élevé" mais Vision n'est pas sûr, 
            # on reste prudent et on dit "sain" sauf si vraiment critique
            if catboost_risk == "high":
                # Vérifier la probabilité exacte de CatBoost
                catboost_prob = max(catboost_result["probability"])
                if catboost_prob > 0.9:  # CatBoost très sûr
                    logger.info("CatBoost très confiant sur risque élevé, décision: contaminé")
                    return "contamine"
                else:
                    logger.info("CatBoost moyennement confiant, Vision incertain -> sain par précaution")
                    return "sain"
            else:
                logger.info("CatBoost indique risque faible, décision: sain")
                return "sain"
    
    def _calculate_combined_confidence(self, catboost_result: Dict, vision_result: Dict) -> float:
        """
        Calcule la confiance combinée en privilégiant Vision
        
        Args:
            catboost_result: Résultat CatBoost
            vision_result: Résultat Vision
            
        Returns:
            float: Score de confiance combiné
        """
        catboost_prob = catboost_result["probability"]
        catboost_risk = catboost_result["risk_level"]
        vision_conf = vision_result["confidence"]
        vision_pred = vision_result["prediction"]
        
        # Calcul basé sur la nouvelle logique de priorité à Vision
        
        # Si Vision est très confiant, la confiance finale est principalement basée sur Vision
        if vision_conf > 0.90:
            return vision_conf * 0.9 + catboost_prob * 0.1
        
        # Si Vision est confiant, on donne plus de poids à Vision
        elif vision_conf > 0.70:
            return vision_conf * 0.75 + catboost_prob * 0.25
        
        # Si Vision est moyennement confiant
        elif vision_conf > 0.50:
            # Si les deux modèles s'accordent, confiance élevée
            if (catboost_risk == "high" and vision_pred == "contamine") or \
               (catboost_risk == "low" and vision_pred == "sain"):
                # Accord -> confiance combinée élevée
                return (vision_conf + catboost_prob) * 0.6
            else:
                # Désaccord -> on suit Vision mais avec confiance réduite
                return vision_conf * 0.8
        
        # Si Vision n'est pas confiant, on utilise principalement CatBoost
        else:
            return catboost_prob * 0.7 + vision_conf * 0.3
    
    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état des modèles
        
        Returns:
            Dict avec l'état de chaque modèle
        """
        return {
            "catboost_loaded": self.catboost_model.is_loaded(),
            "vision_loaded": self.vision_model.is_loaded(),
            "all_models_ready": self._models_loaded
        }