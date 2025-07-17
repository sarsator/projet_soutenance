#!/usr/bin/env python3
"""
ARRÊT GAIA VISION
Usage: python stop.py
Arrête tous les services (API + Frontend + TensorBoard)
"""

import os
import subprocess
import time

def stop_all_services():
    print("🛑 ARRÊT DE TOUS LES SERVICES GAIA VISION")
    print("=" * 45)
    
    # Commandes d'arrêt
    stop_commands = [
        ("🔧 Arrêt API...", "pkill -f 'python.*api/main.py' 2>/dev/null"),
        ("🔧 Arrêt API (run_api)...", "pkill -f 'python.*api/run_api.py' 2>/dev/null"),
        ("🔧 Arrêt API (uvicorn)...", "pkill -f 'uvicorn.*api.main' 2>/dev/null"),
        ("🎨 Arrêt Frontend...", "pkill -f 'python.*frontend/app.py' 2>/dev/null"),
        ("📊 Arrêt TensorBoard...", "pkill -f 'tensorboard' 2>/dev/null"),
        ("🧹 Libération des ports...", "lsof -ti:8000,5000,6006 | xargs -r kill -9 2>/dev/null")
    ]
    
    for description, command in stop_commands:
        print(f"   {description}")
        os.system(command)
        time.sleep(0.5)
    
    # Vérification finale
    print("\n🔍 Vérification finale...")
    time.sleep(2)
    
    # Vérifier les ports
    ports_to_check = [5000, 8000, 6006]
    for port in ports_to_check:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print(f"   ⚠️  Port {port} encore utilisé")
        else:
            print(f"   ✅ Port {port} libéré")
    
    print("\n✅ ARRÊT TERMINÉ")
    print("   Tous les services Gaia Vision ont été arrêtés")

if __name__ == "__main__":
    stop_all_services()
