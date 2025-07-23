#!/usr/bin/env python3
"""
Script de debug pour l'API avec logs complets
"""
import sys
import os
from pathlib import Path
import logging

# Configuration du logging verbose
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/gaia_vision_api.log')
    ]
)

# Ajouter le dossier parent au path pour les imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    import uvicorn
    from api.config import config
    
    print("🔍 DÉMARRAGE API EN MODE DEBUG...")
    print(f"📍 Host: {config.HOST}:{config.PORT}")
    print(f"🔑 API Key: {config.API_KEY}")
    print(f"📁 Upload Dir: {config.UPLOAD_DIR}")
    print(f"🤖 CatBoost Path: {config.CATBOOST_MODEL_PATH}")
    print(f"👁️ Vision Path: {config.VISION_MODEL_PATH}")
    print("📝 Logs sauvegardés dans: /tmp/gaia_vision_api.log")

    
    # Mode développement avec logs complets
    uvicorn.run(
        "api.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,  # Pas de reload pour éviter la perte de logs
        log_level="debug",
        access_log=True
    )
