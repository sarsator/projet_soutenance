#!/usr/bin/env python3
"""
Script de lancement de l'API Gaia Vision
"""
import sys
import os
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

if __name__ == "__main__":
    import uvicorn
    from api.config import config
    
    print("🚀 Démarrage de l'API Gaia Vision...")
    print(f"📍 Host: {config.HOST}:{config.PORT}")
    print(f"🔧 Debug: {config.DEBUG}")
    
    if config.DEBUG:
        # Mode développement avec reload
        uvicorn.run(
            "api.main:app",
            host=config.HOST,
            port=config.PORT,
            reload=True,
            log_level="info"
        )
    else:
        # Mode production
        from api.main import app
        uvicorn.run(
            app,
            host=config.HOST,
            port=config.PORT,
            log_level="info"
        )
