#!/usr/bin/env python3
"""
Script de lancement TensorBoard pour visualiser l'entraînement SSD MobileNet V2
"""
import os
import subprocess
import sys
import time
from pathlib import Path
import webbrowser
from datetime import datetime

def launch_tensorboard():
    """Lance TensorBoard avec les bons paramètres pour votre entraînement"""
    
    print("LANCEMENT TENSORBOARD - ENTRAÎNEMENT SSD MOBILENET V2")
    
    # Chemins vers les logs d'entraînement
    base_dir = Path("/home/sarsator/projets/gaia_vision")
    
    # Priorité aux logs d'entraînement (si ils existent)
    possible_logdirs = [
        base_dir / "training" / "logs",
        base_dir / "training" / "models" / "dl_model" / "outputs" / "ssd_mnv2_simple",
        base_dir / "training" / "notebook" / "logs",
        base_dir / "api" / "models" / "dl_model" / "outputs" / "ssd_mnv2_simple"
    ]
    
    print("RECHERCHE DES LOGS D'ENTRAÎNEMENT...")

    
    existing_logdirs = []
    for logdir in possible_logdirs:
        if logdir.exists():
            # Vérifier s'il y a des fichiers d'événements TensorBoard
            event_files = list(logdir.glob("**/events.out.tfevents.*"))
            if event_files:
                existing_logdirs.append(logdir)
                print(f"✅ Logs trouvés: {logdir}")
                print(f"Fichiers d'événements: {len(event_files)}")
            else:
                print(f"Dossier existe mais pas de logs: {logdir}")
        else:
            print(f"❌ Dossier inexistant: {logdir}")
    
    if not existing_logdirs:
        print("\n❌ AUCUN LOG D'ENTRAÎNEMENT TROUVÉ !")
        print("Vérifiez que l'entraînement a bien été lancé et qu'il y a des fichiers events.out.tfevents.*")
        return False
    
    # Utiliser le premier dossier de logs trouvé
    logdir = existing_logdirs[0]
    print(f"\nUTILISATION DU DOSSIER: {logdir}")
    
    # Paramètres TensorBoard
    port = 6006
    host = "0.0.0.0"  # Accessible depuis l'extérieur
    
    print(f"\nLANCEMENT TENSORBOARD...")
    print(f"   Logdir: {logdir}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   🔗 URL: http://localhost:{port}")
    
    # Construire la commande
    cmd = [
        "tensorboard",
        f"--logdir={logdir}",
        f"--port={port}",
        f"--host={host}",
        "--reload_interval=1",  # Rechargement automatique
        "--load_fast=true",  # Chargement rapide
    ]
    
    print(f"\n📋 COMMANDE:")
    print(f"   {' '.join(cmd)}")
    
    try:
        print(f"\nDÉMARRAGE EN COURS...")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Lancer TensorBoard
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(base_dir)
        )
        
        # Attendre un peu et vérifier les premières lignes
        print(f"\nSORTIE TENSORBOARD:")

        
        for i in range(10):  # Lire les premières lignes
            line = process.stdout.readline()
            if line:
                print(f"   {line.strip()}")
                if "http://" in line:
                    # Extraire l'URL
                    if f":{port}" in line:
                        print(f"\n✅ TENSORBOARD LANCÉ AVEC SUCCÈS !")
                        break
            time.sleep(0.1)
        
        # Informations finales
        print(f"\nTENSORBOARD EN COURS D'EXÉCUTION")
        print(f"   🔗 URL locale: http://localhost:{port}")
        print(f"   Logs: {logdir}")
        
        print(f"\nMÉTRIQUES À SURVEILLER:")
        print(f"   • Loss/total_loss - Perte totale")
        print(f"   • Loss/classification_loss - Perte classification")
        print(f"   • Loss/localization_loss - Perte localisation")
        print(f"   • DetectionBoxes_Precision/mAP - Précision moyenne")
        print(f"   • DetectionBoxes_Recall/AR@100 - Rappel moyen")
        print(f"   • learning_rate - Taux d'apprentissage")
        
        print(f"\nONGLETS UTILES:")
        print(f"   • SCALARS - Courbes de perte et métriques")
        print(f"   • IMAGES - Visualisations des détections")
        print(f"   • GRAPHS - Architecture du modèle")
        print(f"   • HISTOGRAMS - Distribution des poids")
        
        print(f"\nCONSEILS:")
        print(f"   • Utilisez Ctrl+C pour arrêter TensorBoard")
  
        
        # Attendre l'arrêt
        print(f"\nAPPUYEZ SUR CTRL+C POUR ARRÊTER...")
        try:
            process.wait()
        except KeyboardInterrupt:
            print(f"\n🛑 ARRÊT DE TENSORBOARD...")
            process.terminate()
            process.wait()
            print(f"✅ TensorBoard arrêté proprement")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ TENSORBOARD NON INSTALLÉ !")
        print(f"   Installez-le avec: pip install tensorboard")
        return False
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DU LANCEMENT:")
        print(f"   {e}")
        return False

def check_tensorboard_installed():
    """Vérifie si TensorBoard est installé"""
    try:
        result = subprocess.run(
            ["tensorboard", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ TensorBoard installé: {version}")
            return True
        else:
            print(f"❌ TensorBoard non fonctionnel")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"❌ TensorBoard non installé")
        return False

def main():
    """Fonction principale"""
    print(f"SCRIPT DE LANCEMENT TENSORBOARD")
    print(f"Projet: Gaia Vision - SSD MobileNet V2")
    print(f"Auteur: Davy abderrahman")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Vérifier l'installation
    if not check_tensorboard_installed():
        print(f"\nINSTALLATION:")
        print(f"pip install tensorboard")
        print(f"# ou")
        print(f"conda install tensorboard")
        return
    
    # Lancer TensorBoard
    success = launch_tensorboard()
    
    if success:
        print(f"\nMISSION ACCOMPLIE !")
    else:
        print(f"\n❌ ÉCHEC DU LANCEMENT")

if __name__ == "__main__":
    main()
