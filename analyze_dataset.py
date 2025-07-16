#!/usr/bin/env python3
"""
Script d'analyse du dataset Gaia Vision pour diagnostiquer les problèmes de performance
"""
import os
import pandas as pd
from collections import Counter
import glob

def analyze_dataset():
    print("=== ANALYSE COMPLÈTE DU DATASET GAIA VISION ===\n")
    
    # Chemins
    base_path = "/home/sarsator/projets/gaia_vision/training/data/DL_data"
    csv_path = os.path.join(base_path, "splits/train.csv")
    photos_path = os.path.join(base_path, "photos")
    
    # 1. Analyse du CSV
    print("1. ANALYSE DU FICHIER CSV:")
    try:
        df = pd.read_csv(csv_path)
        print(f"   - Nombre total d'échantillons: {len(df)}")
        
        # Distribution des labels
        label_counts = df['label'].value_counts().sort_index()
        print(f"   - Label 0 (sain): {label_counts[0] if 0 in label_counts else 0}")
        print(f"   - Label 1 (contaminé): {label_counts[1] if 1 in label_counts else 0}")
        
        if 0 in label_counts and 1 in label_counts:
            ratio = label_counts[1] / label_counts[0]
            print(f"   - Ratio contaminé/sain: {ratio:.2f}:1")
            if ratio > 2 or ratio < 0.5:
                print("   ⚠️  PROBLÈME: Déséquilibre important des classes!")
        
    except Exception as e:
        print(f"   ❌ Erreur lors du chargement du CSV: {e}")
        return
    
    # 2. Analyse des noms de fichiers
    print("\n2. ANALYSE DES NOMS DE FICHIERS:")
    filename_prefixes = [f.split('_')[0] for f in df['filename']]
    prefix_counts = Counter(filename_prefixes)
    
    print("   Top 10 des préfixes:")
    for prefix, count in prefix_counts.most_common(10):
        print(f"     {prefix}: {count}")
    
    # Identifier les types problématiques
    problematic_prefixes = ['VID', 'Capture']
    for prefix in problematic_prefixes:
        if prefix in prefix_counts:
            print(f"   ⚠️  PROBLÈME: {prefix_counts[prefix]} images de type '{prefix}' détectées")
    
    # 3. Vérification de l'existence des fichiers
    print("\n3. VÉRIFICATION DE L'EXISTENCE DES FICHIERS:")
    missing_files = []
    existing_files = []
    
    for filename in df['filename']:
        full_path = os.path.join(photos_path, filename)
        if os.path.exists(full_path):
            existing_files.append(filename)
        else:
            missing_files.append(filename)
    
    print(f"   - Fichiers existants: {len(existing_files)}/{len(df)}")
    print(f"   - Fichiers manquants: {len(missing_files)}")
    
    if missing_files:
        print("   ❌ PROBLÈME: Fichiers manquants détectés!")
        print(f"   Premiers fichiers manquants: {missing_files[:5]}")
    
    # 4. Analyse du répertoire photos
    print("\n4. ANALYSE DU RÉPERTOIRE PHOTOS:")
    if os.path.exists(photos_path):
        all_photos = glob.glob(os.path.join(photos_path, "*"))
        photo_files = [f for f in all_photos if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   - Nombre total de fichiers photo: {len(photo_files)}")
        print(f"   - Fichiers dans CSV: {len(df)}")
        print(f"   - Fichiers non utilisés: {len(photo_files) - len(existing_files)}")
        
        if len(photo_files) - len(existing_files) > len(df) * 0.1:
            print("   ⚠️  PROBLÈME: Beaucoup de fichiers photos non utilisés!")
        
        # Vérifier les captures d'écran
        screenshots = [f for f in photo_files if 'capture' in os.path.basename(f).lower()]
        if screenshots:
            print(f"   ⚠️  PROBLÈME: {len(screenshots)} captures d'écran détectées dans le répertoire")
    
    # 5. Analyse de la qualité des images (format et taille)
    print("\n5. ANALYSE DE LA QUALITÉ DES IMAGES:")
    sample_files = df['filename'].head(10).tolist()
    resolutions = []
    
    for filename in sample_files:
        full_path = os.path.join(photos_path, filename)
        if os.path.exists(full_path):
            try:
                # Utiliser la commande file pour obtenir les informations
                import subprocess
                result = subprocess.run(['file', full_path], capture_output=True, text=True)
                if 'x' in result.stdout:
                    # Extraire la résolution
                    parts = result.stdout.split(',')
                    for part in parts:
                        if 'x' in part and any(char.isdigit() for char in part):
                            resolution = part.strip()
                            resolutions.append(resolution)
                            break
            except:
                pass
    
    if resolutions:
        print("   Résolutions échantillon:")
        for i, res in enumerate(resolutions[:5]):
            print(f"     {sample_files[i]}: {res}")
        
        # Vérifier les petites résolutions
        small_res_count = sum(1 for res in resolutions if '640x640' in res or '224x224' in res)
        if small_res_count > len(resolutions) * 0.5:
            print("   ⚠️  PROBLÈME: Beaucoup d'images avec petite résolution détectées!")
    
    # 6. Recommandations
    print("\n6. RECOMMANDATIONS POUR AMÉLIORER LES PERFORMANCES:")
    
    recommendations = []
    
    if len(missing_files) > 0:
        recommendations.append("- Nettoyer le CSV pour supprimer les fichiers manquants")
    
    if 'VID' in prefix_counts:
        recommendations.append("- Examiner la qualité des extraits vidéo (frames) - peuvent être flous")
    
    if 'Capture' in prefix_counts or screenshots:
        recommendations.append("- Supprimer les captures d'écran qui ne sont pas de vraies photos")
    
    if 0 in label_counts and 1 in label_counts:
        ratio = label_counts[1] / label_counts[0]
        if ratio > 2 or ratio < 0.5:
            recommendations.append("- Équilibrer le dataset ou utiliser des poids de classe")
    
    if len(photo_files) - len(existing_files) > len(df) * 0.1:
        recommendations.append("- Utiliser plus d'images disponibles pour augmenter la taille du dataset")
    
    recommendations.append("- Vérifier manuellement la qualité des labels")
    recommendations.append("- Augmenter l'augmentation de données pour améliorer la généralisation")
    recommendations.append("- Utiliser un learning rate adaptatif")
    recommendations.append("- Implémenter une validation croisée")
    
    if recommendations:
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("   ✅ Le dataset semble correct au niveau technique")
    
    print(f"\n=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    analyze_dataset()
