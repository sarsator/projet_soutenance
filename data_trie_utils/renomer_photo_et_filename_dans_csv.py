#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour nettoyer les noms de fichiers dans le CSV annotations.csv
et renommer les photos correspondantes pour éliminer les espaces et caractères problématiques.

Auteur: GitHub Copilot
Date: 2025-07-15
"""

import os
import csv
import re
from pathlib import Path
import shutil

def clean_filename(filename):
    """
    Nettoie un nom de fichier en remplaçant les caractères problématiques
    
    Args:
        filename (str): Nom de fichier original
        
    Returns:
        str: Nom de fichier nettoyé
    """
    if not filename:
        return filename
    
    # Séparer le nom et l'extension
    name, ext = os.path.splitext(filename)
    
    # Remplacer les espaces par des underscores
    name = name.replace(' ', '_')
    
    # Supprimer ou remplacer les caractères spéciaux problématiques
    # Garder seulement lettres, chiffres, underscores, tirets
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    
    # Supprimer les underscores multiples consécutifs
    name = re.sub(r'_+', '_', name)
    
    # Supprimer les underscores en début et fin
    name = name.strip('_')
    
    # Reconstituer le nom avec extension
    cleaned_filename = f"{name}{ext.lower()}"
    
    return cleaned_filename

def rename_photos_and_update_csv():
    """
    Fonction principale pour nettoyer les noms de fichiers et renommer les photos
    """
    # Chemins
    csv_path = Path("../training/data/DL_data/etiquettes/annotations.csv")
    photos_dir = Path("../training/data/DL_data/photos")
    
    print("DÉBUT DU NETTOYAGE DES NOMS DE FICHIERS")
    print("=" * 60)
    
    # Vérification des chemins
    if not csv_path.exists():
        print(f"❌ ERREUR: Fichier CSV non trouvé: {csv_path}")
        return False
    
    if not photos_dir.exists():
        print(f"❌ ERREUR: Dossier photos non trouvé: {photos_dir}")
        return False
    
    print(f"✅ Fichier CSV trouvé: {csv_path}")
    print(f"✅ Dossier photos trouvé: {photos_dir}")
    
    # Chargement du CSV
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = list(reader)
            fieldnames = reader.fieldnames
        print(f"CSV chargé: {len(data)} lignes")
    except Exception as e:
        print(f"❌ ERREUR lors du chargement du CSV: {e}")
        return False
    
    # Vérification des colonnes nécessaires
    required_columns = ['filename']
    missing_columns = [col for col in required_columns if col not in fieldnames]
    if missing_columns:
        print(f"❌ ERREUR: Colonnes manquantes dans le CSV: {missing_columns}")
        return False
    
    # Vérification si nom_image existe
    has_nom_image = 'nom_image' in fieldnames
    if has_nom_image:
        print("✅ Colonne 'nom_image' détectée")
    else:
        print("Colonne 'nom_image' non présente")
    
    print(f"Colonnes disponibles: {list(fieldnames)}")
    
    # Compteurs pour le rapport
    files_processed = 0
    files_renamed = 0
    files_not_found = 0
    csv_updated = 0
    
    print(f"\nTRAITEMENT DES FICHIERS...")
    
    # Traitement de chaque ligne
    for idx, row in enumerate(data):
        original_filename = str(row['filename'])
        cleaned_filename = clean_filename(original_filename)
        
        files_processed += 1
        
        # Si le nom est déjà propre, on passe
        if original_filename == cleaned_filename:
            continue
        
        print(f"\nLigne {idx + 1}: {original_filename} → {cleaned_filename}")
        
        # Chemins des fichiers
        original_photo_path = photos_dir / original_filename
        cleaned_photo_path = photos_dir / cleaned_filename
        
        # Vérifier si la photo originale existe
        if not original_photo_path.exists():
            print(f"Photo non trouvée: {original_photo_path}")
            files_not_found += 1
            continue
        
        # Vérifier si le nouveau nom existe déjà
        if cleaned_photo_path.exists() and cleaned_photo_path != original_photo_path:
            print(f"Le fichier {cleaned_filename} existe déjà, ajout d'un suffixe")
            # Ajouter un suffixe numérique
            name, ext = os.path.splitext(cleaned_filename)
            counter = 1
            while cleaned_photo_path.exists():
                cleaned_filename = f"{name}_{counter}{ext}"
                cleaned_photo_path = photos_dir / cleaned_filename
                counter += 1
            print(f"Nouveau nom: {cleaned_filename}")
        
        try:
            # Renommer la photo
            shutil.move(str(original_photo_path), str(cleaned_photo_path))
            print(f"✅ Photo renommée: {original_filename} → {cleaned_filename}")
            files_renamed += 1
            
            # Mettre à jour le CSV dans les données
            data[idx]['filename'] = cleaned_filename
            
            # Mettre à jour nom_image si elle existe et est identique à filename
            if has_nom_image and str(row['nom_image']) == original_filename:
                data[idx]['nom_image'] = cleaned_filename
                print(f"✅ nom_image aussi mis à jour")
            
            csv_updated += 1
            
        except Exception as e:
            print(f"❌ ERREUR lors du renommage de {original_filename}: {e}")
            continue
    
    # Sauvegarde du CSV mis à jour
    if csv_updated > 0:
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            print(f"\nCSV mis à jour et sauvegardé: {csv_path}")
        except Exception as e:
            print(f"❌ ERREUR lors de la sauvegarde du CSV: {e}")
            return False
    else:
        print(f"\nAucune modification nécessaire dans le CSV")
    
    # Rapport final
    print(f"\nRAPPORT FINAL:")

    print(f"Fichiers traités: {files_processed}")
    print(f"Fichiers renommés: {files_renamed}")
    print(f"Lignes CSV mises à jour: {csv_updated}")
    print(f"Photos non trouvées: {files_not_found}")
    
    if files_renamed > 0:
        print(f"\n✅ SUCCÈS: {files_renamed} fichiers nettoyés avec succès!")
    else:
        print(f"\nPARFAIT: Tous les noms de fichiers sont déjà propres!")
    
    return True

def main():
    """Point d'entrée principal"""
    print("🧹 SCRIPT DE NETTOYAGE DES NOMS DE FICHIERS")

    print("Ce script va:")
    print("1. Lire le fichier annotations.csv")
    print("2. Nettoyer les noms de fichiers (espaces → underscores, caractères spéciaux)")
    print("3. Renommer les photos correspondantes")
    print("4. Mettre à jour le CSV directement (pas de sauvegarde)")

    
    # Demander confirmation
    response = input("\n🤔 Voulez-vous continuer? (o/n): ").lower().strip()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("❌ Opération annulée par l'utilisateur")
        return
    
    # Exécuter le nettoyage
    success = rename_photos_and_update_csv()
    
    if success:
        print(f"TERMINÉ AVEC SUCCÈS!")
    else:
        print(f"\nÉCHEC DU TRAITEMENT")

if __name__ == "__main__":
    main()
