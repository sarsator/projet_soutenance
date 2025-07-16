#!/usr/bin/env python3
"""
Modèle CatBoost pour la prédiction sur données tabulaires
"""
#je vais commenter correctement

import logging 
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class CatBoostModel:
    """
    Classe pour charger et utiliser le modèle CatBoost entraîné.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise le modèle CatBoost.
        
        Args:
            model_path: Chemin vers le fichier du modèle (.joblib)
        """
        self.model_path = model_path
        self.model = None
        self._loaded = False
        
        # Charger automatiquement si un chemin est fourni
        if model_path:
            self.load_model()
    
    def load_model(self) -> bool:
        """
        Charge le modèle CatBoost depuis le fichier.
        
        Returns:
            bool: True si le chargement réussit, False sinon
        """
        try:
            if not self.model_path:
                # Utiliser le chemin par défaut
                current_dir = Path(__file__).parent
                self.model_path = current_dir / "ml_model" / "model_catboost_best.joblib"
            
            model_path = Path(self.model_path)
            
            if not model_path.exists():
                logger.error(f"Fichier modèle CatBoost non trouvé : {model_path}")
                return False
            
            logger.info(f"Chargement du modèle CatBoost : {model_path}")
            self.model = joblib.load(model_path)
            self._loaded = True
            
            logger.info("✅ Modèle CatBoost chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle CatBoost : {e}")
            self.model = None
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """
        Vérifie si le modèle est chargé.
        
        Returns:
            bool: True si le modèle est chargé
        """
        return self._loaded and self.model is not None
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Effectue une prédiction avec le modèle CatBoost.
        
        Args:
            input_data: Dictionnaire avec les features d'entrée
                       - race_champignon: str
                       - type_substrat: str
                       - jours_inoculation: int
                       - hygrometrie: float
                       - co2_ppm: float
        
        Returns:
            Dict contenant la prédiction et la confiance
        """
        logger.info("=== DÉBUT PRÉDICTION CATBOOST ===")
        logger.info(f"Données d'entrée reçues: {input_data}")
        
        if not self.is_loaded():
            logger.error("❌ Modèle CatBoost non chargé")
            return {
                "prediction": 0,
                "probability": 0.0,
                "confidence": 0.0,
                "risk_level": "low",
                "error": "Modèle non chargé"
            }
        
        try:
            # Préparer les données d'entrée
            logger.info("Préparation des données d'entrée...")
            df_input = self._prepare_input_data(input_data)
            logger.info(f"DataFrame préparé: {df_input}")
            logger.info(f"Colonnes du DataFrame: {df_input.columns.tolist()}")
            logger.info(f"Types des données: {df_input.dtypes.to_dict()}")
            
            # Prédiction
            logger.info("Exécution de la prédiction...")
            prediction = self.model.predict(df_input)[0]
            logger.info(f"Prédiction brute: {prediction}")
            
            probabilities = self.model.predict_proba(df_input)[0]
            logger.info(f"Probabilités brutes: {probabilities}")
            
            # Calculer la confiance (probabilité de la classe prédite)
            confidence = float(max(probabilities))
            logger.info(f"Confiance calculée: {confidence}")
            
            # Déterminer le niveau de risque basé sur la prédiction
            # Version très permissive : si la probabilité de la classe 1 ("à analyser") > 0.10, c'est un risque élevé
            # Même si la prédiction finale est 0, on regarde quand même la probabilité de contamination
            probability_class_1 = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            risk_level = "high" if probability_class_1 > 0.10 else "low"
            logger.info(f"Niveau de risque: {risk_level} (seuil: 0.10, prob_classe_1: {probability_class_1:.3f})")
            
            result = {
                "prediction": int(prediction),
                "probability": probabilities.tolist(),
                "confidence": confidence,
                "risk_level": risk_level,
                "prediction_label": "à analyser" if prediction == 1 else "inutilisable"
            }
            
            logger.info(f"✅ Prédiction CatBoost : {result['prediction_label']} (confiance: {confidence:.3f})")
            logger.info("=== FIN PRÉDICTION CATBOOST ===")
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ ERREUR dans prédiction CatBoost:")
            logger.error(f"Type d'erreur: {type(e).__name__}")
            logger.error(f"Message: {str(e)}")
            logger.error(f"Traceback complet:\n{error_details}")
            logger.error("=== FIN PRÉDICTION CATBOOST (ERREUR) ===")
            return {
                "prediction": 0,
                "probability": [1.0, 0.0],
                "confidence": 0.0,
                "risk_level": "low",
                "error": str(e)
            }
    
    def _prepare_input_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prépare les données d'entrée pour la prédiction.
        
        Args:
            input_data: Données d'entrée brutes
            
        Returns:
            DataFrame formaté pour le modèle
        """
        # Mapping des noms de colonnes selon l'entraînement
        data = {
            "champignon": input_data.get("race_champignon", ""),
            "substrat": input_data.get("type_substrat", ""),
            "Jour_inoculation": input_data.get("jours_inoculation", 0),
            "hygrometrie": input_data.get("hygrometrie", 0.0),  # Minuscule pour correspondre au modèle
            "co2": input_data.get("co2_ppm", 0.0)
        }
        
        return pd.DataFrame([data])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle.
        
        Returns:
            Dict avec les informations du modèle
        """
        if not self.is_loaded():
            return {"loaded": False, "error": "Modèle non chargé"}
        
        try:
            info = {
                "loaded": True,
                "model_type": str(type(self.model).__name__),
                "model_path": str(self.model_path),
                "features": ["champignon", "substrat", "Jour_inoculation", "hygrometrie", "co2"]  # Noms exacts du modèle
            }
            
            # Ajouter des infos spécifiques CatBoost si disponibles
            if hasattr(self.model, 'feature_names_'):
                info["feature_names"] = self.model.feature_names_.tolist()
            
            if hasattr(self.model, 'classes_'):
                info["classes"] = self.model.classes_.tolist()
                
            return info
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos modèle : {e}")
            return {"loaded": True, "error": str(e)}
