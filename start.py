#!/usr/bin/env python3
"""
LANCEMENT ULTRA-RAPIDE GAIA VISION
Usage: python start.py
"""

import subprocess
import time
import webbrowser
import os

def quick_start():
    print("🚀 DÉMARRAGE RAPIDE GAIA VISION")
    print("=" * 40)
    
    # Nettoyer
    print("🧹 Nettoyage...")
    os.system("pkill -f 'python.*api/run_api.py' 2>/dev/null")
    os.system("pkill -f 'python.*frontend/app.py' 2>/dev/null")
    os.system("lsof -ti:8000,5000 | xargs -r kill -9 2>/dev/null")
    
    # Lancer en parallèle
    print("🚀 Lancement des services...")
    subprocess.Popen(["python", "api/run_api.py"], cwd="/home/sarsator/projets/gaia_vision", 
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.Popen(["python", "frontend/app.py"], cwd="/home/sarsator/projets/gaia_vision",
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Attendre et ouvrir
    print("⏳ Attente (30 secondes)...")
    time.sleep(30)
    
    print("🌐 Ouverture du navigateur...")
    webbrowser.open("http://localhost:5000")
    
    print("✅ Prêt! Interface: http://localhost:5000")
    print("⚠️  Appuyez sur Ctrl+C puis lancez:")
    print("    pkill -f 'python.*api'; pkill -f 'python.*frontend'")

if __name__ == "__main__":
    quick_start()
