"""
Configuration de l'API Gaia Vision
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration de l'application"""
    
    # API Configuration
    API_KEY = os.getenv("API_KEY", "gaia_vision_api_key")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    #variables d'environnement
    valeur_min_catboost = 0.12  # Seuil pour CatBoost, peut être modifié via .env
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_DIR = BASE_DIR / "api" / "images_a_traiter"
    PROCESSED_DIR = BASE_DIR / "api" / "images_traitees"
    
    # Model Paths (utilisation du système de versioning)
    MODELS_BASE_DIR = BASE_DIR / "api" / "models"
    CATBOOST_MODEL_PATH = MODELS_BASE_DIR / "ml_model" / "current"
    VISION_MODEL_PATH = MODELS_BASE_DIR / "dl_model" / "current"
    
    # Compatibility paths (pour fallback si current n'existe pas)
    CATBOOST_MODEL_FALLBACK = MODELS_BASE_DIR / "ml_model" / "model_catboost_best.joblib"
    VISION_MODEL_FALLBACK = MODELS_BASE_DIR / "dl_model" / "final_model.keras"
    
    # Model Configuration
    VISION_INPUT_SIZE = (640, 640)
    VISION_CLASS_NAMES = ["contamine", "sain"]
    
    # Prediction Logic
    CATBOOST_CONFIDENCE_THRESHOLD = 0.8  # Seuil pour déclencher la vision
    VISION_CONFIDENCE_THRESHOLD = 0.5    # Seuil pour la classification vision
    
    # File Management
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    @classmethod
    def create_directories(cls):
        """Crée les dossiers nécessaires"""
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Créer la structure pour le versioning des modèles
        cls.MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)
        (cls.MODELS_BASE_DIR / "ml_model" / "versions").mkdir(parents=True, exist_ok=True)
        (cls.MODELS_BASE_DIR / "dl_model" / "versions").mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Valide la configuration"""
        errors = []
        
        if not cls.API_KEY or cls.API_KEY == "gaia-vision-test-key-2025":
            errors.append("API_KEY non définie ou utilise la valeur par défaut")
        
        # Vérifier les modèles avec fallback
        catboost_exists = cls.CATBOOST_MODEL_PATH.exists() or cls.CATBOOST_MODEL_FALLBACK.exists()
        if not catboost_exists:
            errors.append(f"Modèle CatBoost non trouvé: {cls.CATBOOST_MODEL_PATH} ou {cls.CATBOOST_MODEL_FALLBACK}")
        
        vision_exists = cls.VISION_MODEL_PATH.exists() or cls.VISION_MODEL_FALLBACK.exists()
        if not vision_exists:
            errors.append(f"Modèle Vision non trouvé: {cls.VISION_MODEL_PATH} ou {cls.VISION_MODEL_FALLBACK}")
        
        return errors

# Instance globale
config = Config()
