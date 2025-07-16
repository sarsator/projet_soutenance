from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv
from uuid import uuid4
from pathlib import Path
import sys

# Import des modules custom (j'ai organisé le code en modules)
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from api.utils.prediction_service import PredictionService
from api.config import config

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging (j'aime bien savoir ce qui se passe)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
API_KEY = config.API_KEY
UPLOAD_DIR = config.UPLOAD_DIR
config.create_directories()  # Création des dossiers si ils existent pas

# Création de l'app FastAPI avec metadata
app = FastAPI(
    title="🍄 Gaia Vision API", 
    version="1.0.0",
    description="API d'analyse de contamination de champignons - Projet de soutenance Alyra"
)

# Configuration CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
from uuid import uuid4
from pathlib import Path
import sys

# Import des modules custom (j'ai organisé le code en modules)
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from api.utils.prediction_service import PredictionService
from api.config import config

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging (j'aime bien savoir ce qui se passe)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
API_KEY = config.API_KEY
UPLOAD_DIR = config.UPLOAD_DIR
config.create_directories()  # Création des dossiers si ils existent pas

# Création de l'app FastAPI avec metadata
app = FastAPI(
    title="� Gaia Vision API", 
    version="1.0.0",
    description="API d'analyse de contamination de champignons - Projet de soutenance Alyra"
)

# Initialisation du service de prédiction (le cœur du système)
logger.info("🌱 Initialisation du service de prédiction...")
prediction_service = PredictionService(
    catboost_model_path=str(config.CATBOOST_MODEL_PATH),
    vision_model_path=str(config.VISION_MODEL_PATH)
)

def check_api_key(auth: str):
    """
    Vérification de la clé API (sécurité basique mais suffisante pour le projet).
    
    Args:
        auth: Header d'autorisation
        
    Raises:
        HTTPException: Si la clé est invalide
    """
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=403, 
            detail="🚫 Header Authorization manquant ou invalide"
        )
    
    key = auth.split("Bearer ")[-1]
    if key != API_KEY:
        raise HTTPException(
            status_code=403, 
            detail="🔑 Clé API invalide"
        )

@app.get("/status")
def status():
    """Check de base pour voir si l'API répond"""
    return {
        "status": "🌿 API opérationnelle", 
        "version": "1.0.0",
        "project": "Gaia Vision - Soutenance Alyra"
    }

@app.get("/health")
def health():
    """Vérification complète de l'état du système"""
    try:
        health_status = prediction_service.health_check()
        return {
            "status": "healthy" if health_status["all_models_ready"] else "partial",
            "models": health_status
        }
    except Exception as e:
        logger.error(f"Erreur lors du health check: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/predict-image")
async def predict_image(
    authorization: str = Header(None),
    race_champignon: str = Form(..., description="Race du champignon"),
    type_substrat: str = Form(..., description="Type de substrat"),
    jours_inoculation: int = Form(..., description="Nombre de jours depuis l'inoculation"),
    hygrometrie: float = Form(..., description="Taux d'hygrométrie (%)"),
    co2_ppm: float = Form(..., description="Taux de CO2 en PPM"),
    commentaire: str = Form("", description="Commentaire optionnel"),
    image: UploadFile = File(..., description="Image à analyser")
):
    """
    Prédiction orchestrée utilisant CatBoost + Vision
    """
    logger.info("=== DÉBUT DE REQUÊTE PREDICT-IMAGE ===")
    logger.info(f"Paramètres reçus: race_champignon={race_champignon}, type_substrat={type_substrat}, jours_inoculation={jours_inoculation}, hygrometrie={hygrometrie}, co2_ppm={co2_ppm}")
    logger.info(f"Image reçue: {image.filename}, taille: {image.size if hasattr(image, 'size') else 'inconnue'}")
    
    try:
        logger.info("Vérification de la clé API...")
        check_api_key(authorization)
        logger.info("✅ Clé API validée")
        
    except Exception as e:
        logger.error(f"❌ Erreur de validation API Key: {e}")
        raise HTTPException(status_code=401, detail=f"Erreur d'autorisation: {str(e)}")
    
    try:
        logger.info("Début de sauvegarde de l'image...")
        # Sauvegarde de l'image
        file_id = str(uuid4())
        file_ext = Path(image.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        logger.info(f"Chemin de sauvegarde: {file_path}")
        
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        logger.info(f"✅ Image sauvegardée: {file_path} ({len(content)} bytes)")
        
        # Prédiction orchestrée
        logger.info("Début de la prédiction orchestrée...")
        result = prediction_service.predict(
            race_champignon=race_champignon,
            type_substrat=type_substrat,
            jours_inoculation=jours_inoculation,
            hygrometrie=hygrometrie,
            co2_ppm=co2_ppm,
            image_path=str(file_path)
        )
        logger.info(f"✅ Prédiction terminée: {result.get('final_decision', 'N/A')}")
        
        # Enrichissement de la réponse
        logger.info("Construction de la réponse...")
        response = {
            "prediction": result["final_decision"],
            "confidence": result["confidence_score"],
            "confidence_source": result["confidence_source"],  # Source de la confiance
            "multi_sac_count": result["multi_sac_count"],  # Nombre de sacs
            "analysis_method": "Intelligence Artificielle",  # Texte pour l'interface
            "details": {
                "catboost_prediction": result["catboost_prediction"],
                "vision_prediction": result["vision_prediction"],
                "models_used": result["models_used"],
                "analysis_steps": result["analysis_steps"],
                "model_versions": result.get("model_versions", {})  # Ajouter les versions
            },
            "input_parameters": {
                "race_champignon": race_champignon,
                "type_substrat": type_substrat,
                "jours_inoculation": jours_inoculation,
                "hygrometrie": hygrometrie,
                "co2_ppm": co2_ppm,
                "commentaire": commentaire,
                "image_file": file_path.name
            }
        }
        
        # Ajout d'éventuels warnings ou erreurs
        if "warning" in result:
            response["warning"] = result["warning"]
        if "error" in result:
            response["error"] = result["error"]
        
        logger.info(f"✅ Réponse construite avec succès")
        logger.info(f"Décision finale: {result['final_decision']}")
        logger.info("=== FIN DE REQUÊTE PREDICT-IMAGE ===")
        return JSONResponse(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ ERREUR CRITIQUE dans predict-image:")
        logger.error(f"Type d'erreur: {type(e).__name__}")
        logger.error(f"Message: {str(e)}")
        logger.error(f"Traceback complet:\n{error_details}")
        
        # Nettoyer le fichier en cas d'erreur
        if 'file_path' in locals() and file_path.exists():
            logger.info(f"Nettoyage du fichier temporaire: {file_path}")
            file_path.unlink()
        
        logger.error("=== FIN DE REQUÊTE PREDICT-IMAGE (ERREUR) ===")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/predict-parameters-only")
async def predict_parameters_only(
    authorization: str = Header(None),
    race_champignon: str = Form(...),
    type_substrat: str = Form(...),
    jours_inoculation: int = Form(...),
    hygrometrie: float = Form(...),
    co2_ppm: float = Form(...),
    commentaire: str = Form(""),
):
    """
    Prédiction basée uniquement sur les paramètres (CatBoost seul)
    """
    check_api_key(authorization)
    
    try:
        # Prédiction sans image
        result = prediction_service.predict(
            race_champignon=race_champignon,
            type_substrat=type_substrat,
            jours_inoculation=jours_inoculation,
            hygrometrie=hygrometrie,
            co2_ppm=co2_ppm,
            image_path=None
        )
        
        response = {
            "prediction": result["final_decision"],
            "confidence": result["confidence_score"],
            "details": result["catboost_prediction"],
            "input_parameters": {
                "race_champignon": race_champignon,
                "type_substrat": type_substrat,
                "jours_inoculation": jours_inoculation,
                "hygrometrie": hygrometrie,
                "co2_ppm": co2_ppm,
                "commentaire": commentaire
            }
        }
        
        logger.info(f"Prédiction paramètres seuls: {result['final_decision']}")
        return JSONResponse(response)
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction paramètres: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'API"""
    logger.info("Démarrage de l'API Gaia Vision...")
    
    # Validation de la configuration
    config_errors = config.validate_config()
    if config_errors:
        logger.warning("Problèmes de configuration détectés:")
        for error in config_errors:
            logger.warning(f"  - {error}")
    
    # Précharger les modèles
    try:
        if prediction_service.load_models():
            logger.info("Modèles préchargés avec succès")
        else:
            logger.warning("Certains modèles n'ont pas pu être chargés")
    except Exception as e:
        logger.error(f"Erreur lors du préchargement des modèles: {e}")

@app.get("/")
def root():
    """Documentation de base de l'API"""
    return {
        "name": "Gaia Vision API",
        "version": "1.0.0",
        "description": "API d'analyse de contamination de champignons utilisant CatBoost et Vision",
        "endpoints": {
            "/status": "Statut de l'API",
            "/health": "État des modèles",
            "/predict-image": "Prédiction avec image (CatBoost + Vision)",
            "/predict-parameters-only": "Prédiction sans image (CatBoost seul)",
            "/heatmap": "Génération de heatmap de contamination",
            "/heatmap-overlay": "Génération d'overlay de contamination",
            "/docs": "Documentation Swagger"
        },
        "models": {
            "catboost": "Analyse des paramètres de culture",
            "vision": "Analyse visuelle d'image"
        }
    }

# ===== ENDPOINTS HEATMAP DE CONTAMINATION =====

# Support OPTIONS pour CORS
@app.options("/heatmap")
async def heatmap_options():
    """Support CORS OPTIONS pour l'endpoint heatmap"""
    return {"message": "OK"}

@app.options("/heatmap-overlay")
async def heatmap_overlay_options():
    """Support CORS OPTIONS pour l'endpoint heatmap-overlay"""
    return {"message": "OK"}

@app.post("/heatmap")
async def generate_heatmap(
    authorization: str = Header(None),
    file: UploadFile = File(...)
):
    """
    Génère une heatmap de contamination pour une image uploadée
    
    Returns:
        Image PNG avec heatmap overlay des zones de contamination
    """
    check_api_key(authorization)
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        import tempfile
        from io import BytesIO
        from PIL import Image
        sys.path.append(str(current_dir / "utils"))
        from heatmap_generator import ContaminationHeatmapGenerator
        from models.vision_model import VisionModel
        
        # Sauvegarder temporairement le fichier uploadé
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        logger.info(f"🔥 Génération heatmap pour: {file.filename}")
        
        # Charger le modèle et faire la prédiction
        model = VisionModel()
        if not model.load_model():
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle de vision")
        
        # Obtenir les détections
        result = model.predict(temp_image_path)
        
        # Vérifier s'il y a des contaminations
        contaminated_detections = [d for d in result.get('detections', []) if d.get('class_name') == 'contaminated']
        
        if not contaminated_detections:
            logger.info("⚠️ Aucune contamination détectée, retour image originale")
            # Retourner l'image originale si pas de contamination
            from fastapi.responses import Response
            return Response(content=content, media_type="image/jpeg")
        
        # Générer la heatmap
        generator = ContaminationHeatmapGenerator()
        heatmap_img = generator.create_contamination_heatmap(temp_image_path, result['detections'])
        
        # Convertir en bytes pour la réponse
        pil_img = Image.fromarray(heatmap_img)
        img_buffer = BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        logger.info(f"✅ Heatmap générée avec {len(contaminated_detections)} zone(s) de contamination")
        
        # Nettoyer le fichier temporaire
        os.unlink(temp_image_path)
        
        from fastapi.responses import Response
        return Response(content=img_buffer.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"❌ Erreur génération heatmap: {e}")
        # Nettoyer en cas d'erreur
        if 'temp_image_path' in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la heatmap: {str(e)}")


@app.post("/heatmap-overlay")  
async def generate_heatmap_overlay(
    authorization: str = Header(None),
    file: UploadFile = File(...)
):
    """
    Génère un overlay style PIL avec rectangles de contamination
    
    Returns:
        Image PNG avec overlay rectangulaire des zones de contamination
    """
    check_api_key(authorization)
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        import tempfile
        from io import BytesIO
        from PIL import Image
        sys.path.append(str(current_dir / "utils"))
        from heatmap_generator import ContaminationHeatmapGenerator
        from models.vision_model import VisionModel
        
        # Sauvegarder temporairement le fichier uploadé
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        logger.info(f"🎯 Génération overlay pour: {file.filename}")
        
        # Charger le modèle et faire la prédiction
        model = VisionModel()
        if not model.load_model():
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle de vision")
        
        # Obtenir les détections
        result = model.predict(temp_image_path)
        
        # Vérifier s'il y a des contaminations
        contaminated_detections = [d for d in result.get('detections', []) if d.get('class_name') == 'contaminated']
        
        if not contaminated_detections:
            logger.info("⚠️ Aucune contamination détectée, retour image originale")
            from fastapi.responses import Response
            return Response(content=content, media_type="image/jpeg")
        
        # Générer l'overlay
        generator = ContaminationHeatmapGenerator()
        overlay_img = generator.create_contamination_overlay_pil(temp_image_path, result['detections'])
        
        # Convertir en bytes pour la réponse
        img_buffer = BytesIO()
        overlay_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        logger.info(f"✅ Overlay généré avec {len(contaminated_detections)} zone(s) de contamination")
        
        # Nettoyer le fichier temporaire
        os.unlink(temp_image_path)
        
        from fastapi.responses import Response
        return Response(content=img_buffer.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"❌ Erreur génération overlay: {e}")
        # Nettoyer en cas d'erreur
        if 'temp_image_path' in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de l'overlay: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

