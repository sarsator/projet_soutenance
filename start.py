import subprocess
import time
import webbrowser
import os
from pathlib import Path

def quick_start():
    print("DÉMARRAGE COMPLET DE GAIA VISION")
    
    # Dossier de travail
    project_dir = Path("/home/sarsator/projets/gaia_vision")
    venv_python = project_dir / ".venv" / "bin" / "python"
    tensorboard_script = project_dir / "tensorboard.sh"
    
    # Vérifications
    print("Vérifications...")
    if not venv_python.exists():
        print(f"❌ Environnement virtuel non trouvé: {venv_python}")
        print("Créez-le avec: python -m venv .venv")
        return
    
    if not tensorboard_script.exists():
        print(f"❌ Script TensorBoard non trouvé: {tensorboard_script}")
        print("Le script tensorboard.sh doit être présent")
        return
    
    print(f"✅ Environnement virtuel: {venv_python}")
    print(f"✅ Script TensorBoard: {tensorboard_script}")
    
    # Nettoyer les processus existants
    print("\nNettoyage des processus existants...")
    cleanup_commands = [
        "pkill -f 'python.*api/run_api.py' 2>/dev/null",
        "pkill -f 'python.*frontend/app.py' 2>/dev/null", 
        "pkill -f 'python.*api/main.py' 2>/dev/null",
        "pkill -f 'uvicorn.*api.main' 2>/dev/null",
        "pkill -f 'tensorboard' 2>/dev/null",
        "lsof -ti:8000,5000,6006 | xargs -r kill -9 2>/dev/null"
    ]
    
    for cmd in cleanup_commands:
        os.system(cmd)
    
    print("✅ Nettoyage terminé")
    
    # Lancer les services en parallèle
    print("\nLancement des services...")
    
    # 1. API avec .venv
    print("Lancement de l'API...")
    api_process = subprocess.Popen(
        [str(venv_python), "api/main.py"], 
        cwd=str(project_dir),
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    
    # 2. Frontend avec .venv  
    print("Lancement du Frontend...")
    frontend_process = subprocess.Popen(
        [str(venv_python), "frontend/app.py"], 
        cwd=str(project_dir),
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    
    # 3. TensorBoard (optionnel)
    print("Lancement de TensorBoard...")
    tensorboard_process = None
    if tensorboard_script.exists():
        print(f"✅ Script TensorBoard trouvé: {tensorboard_script}")
        try:
            tensorboard_process = subprocess.Popen(
                ["bash", str(tensorboard_script)], 
                cwd=str(project_dir),
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            print(f"✅ TensorBoard lancé (PID: {tensorboard_process.pid})")
        except Exception as e:
            print(f"❌ Erreur TensorBoard: {e}")
    else:
        print(f"❌ Script TensorBoard non trouvé: {tensorboard_script}")
    
    # Attendre que les services démarrent
    print("\nAttente du démarrage des services...")
    for i in range(15):
        print(f"   {i+1}/15 secondes...", end="\r")
        time.sleep(1)
    
    print("\n")
    
    # Vérifier les processus
    print("🔍 Vérification des services...")
    services_status = {
        "API": api_process.poll() is None,
        "Frontend": frontend_process.poll() is None, 
        "TensorBoard": tensorboard_process.poll() is None if tensorboard_process else False
    }
    
    for service, running in services_status.items():
        status = "✅ ACTIF" if running else "❌ ERREUR"
        print(f"   {service}: {status}")
        
        # Afficher les erreurs TensorBoard si présentes
        if service == "TensorBoard" and not running and tensorboard_process:
            try:
                stdout, stderr = tensorboard_process.communicate(timeout=1)
                if stderr:
                    print(f"⚠️  Erreur TensorBoard: {stderr.decode()[:200]}...")
            except:
                pass
    
    # Informations d'accès
    print("\nSERVICES DISPONIBLES:")

    print("Frontend:    http://localhost:5000")
    print("API:         http://localhost:8000")
    print("TensorBoard: http://localhost:6006")
    print("API Docs:    http://localhost:8000/docs")
    
    print("\nCONSEILS:")
    print("• Ouvrez http://localhost:5000 pour l'interface principale")
    print("• Utilisez http://localhost:6006 pour voir l'entraînement")
    print("• Consultez http://localhost:8000/docs pour l'API")
    
    print("\nPOUR ARRÊTER TOUS LES SERVICES:")
    print("Appuyez sur Ctrl+C puis lancez:")
    print("pkill -f 'python.*api'; pkill -f 'python.*frontend'; pkill -f 'tensorboard'")

if __name__ == "__main__":
    quick_start()
