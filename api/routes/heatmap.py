from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response
import tempfile
import os
import sys
import logging
from io import BytesIO
from PIL import Image
from vision_model import VisionModel
from heatmap_generator import ContaminationHeatmapGenerator


# Ajouter le chemin des modèles
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))


logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/heatmap")
async def generate_heatmap(file: UploadFile = File(...)):
       
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        # Sauvegarder temporairement le fichier uploadé
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        logger.info(f"🔥 Génération heatmap pour: {file.filename}")
        
        # Charger le modèle et faire la prédiction
        model = VisionModel()
        if not model.charger_modele():
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle de vision")
        
        # Obtenir les détections
        result = model.predict(temp_image_path)
        
        # Vérifier s'il y a des contaminations
        contaminated_detections = [d for d in result.get('detections', []) if d.get('class_name') == 'contaminated']
        
        if not contaminated_detections:
            logger.info("⚠️ Aucune contamination détectée, retour image originale")
            # Retourner l'image originale si pas de contamination
            return Response(content=content, media_type="image/jpeg")
        
        # Générer la heatmap
        generator = ContaminationHeatmapGenerator()
        heatmap_img = generator.create_contamination_heatmap(temp_image_path, result['detections'])
        
        # Convertir en bytes pour la réponse
        pil_img = Image.fromarray(heatmap_img)
        img_buffer = BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        logger.info(f" Heatmap générée avec {len(contaminated_detections)} zone(s) de contamination")
        
        # Nettoyer le fichier temporaire
        os.unlink(temp_image_path)
        
        return Response(content=img_buffer.getvalue(), media_type="image/png")
        :smile: 
    except Exception as e:
        logger.error(f"❌ Erreur génération heatmap: {e}")
        # Nettoyer en cas d'erreur
        if 'temp_image_path' in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la heatmap: {str(e)}")


@router.post("/heatmap-overlay") 
async def generate_heatmap_overlay(file: UploadFile = File(...)):
       
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail= "❌ Le fichier doit être une image")
    
    try:
        # Sauvegarder temporairement le fichier uploadé
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        logger.info(f" Génération overlay pour: {file.filename}")
        
        # Charger le modèle et faire la prédiction
        model = VisionModel()
        if not model.charger_modele():
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle de vision")
        
        # Obtenir les détections
        result = model.predict(temp_image_path)
        
        # Vérifier s'il y a des contaminations
        contaminated_detections = [d for d in result.get('detections', []) if d.get('class_name') == 'contaminated']
        
        if not contaminated_detections:
            logger.info("⚠️ Aucune contamination détectée, retour image originale")
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
        
        return Response(content=img_buffer.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"❌ Erreur génération overlay: {e}")
        if 'temp_image_path' in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de l'overlay: {str(e)}")


@router.get("/health-heatmap")
async def health_check_heatmap():
   
    try:
        generator = ContaminationHeatmapGenerator()
        model = VisionModel()
        
        return {
            "status": "healthy",
            "heatmap_generator": "available",
            "vision_model": "available"
        }
    except Exception as e:
        logger.error(f"❌ Health check heatmap échoué: {e}")
        raise HTTPException(status_code=500, detail=f"Service heatmap indisponible: {str(e)}")
