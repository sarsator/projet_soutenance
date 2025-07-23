#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de nettoyage du CSV d'annotations
========================================

Ce script parcourt le CSV d'annotations et supprime les entrées dont les fichiers
images correspondants n'existent plus dans le dossier photos.

Fonctionnalités :
- Vérifie l'existence des fichiers images référencés dans le CSV
- Déplace les entrées sans fichier vers un CSV de sauvegarde
- Met à jour le CSV principal avec seulement les entrées valides
- Gestion robuste des erreurs

Utilisation :
    python enlever_doublons_csv.py

Auteur : Générateur d'IA
Date : Juillet 2025
"""

import os
import pandas as pd
from typing import Set, Tuple


def obtenir_fichiers_images(dossier_photos: str) -> Set[str]:
    """
    Obtient la liste de tous les fichiers JPG présents dans le dossier.
    
    Args:
        dossier_photos: Chemin vers le dossier contenant les photos
        
    Returns:
        Set[str]: Ensemble des noms de fichiers JPG présents
    """
    fichiers_presents = set()
    
    try:
        if not os.path.exists(dossier_photos):
            print(f"Erreur: Le dossier {dossier_photos} n'existe pas.")
            return fichiers_presents
        
        # Lister tous les fichiers .jpg
        for fichier in os.listdir(dossier_photos):
            if fichier.lower().endswith('.jpg'):
                fichiers_presents.add(fichier)
        
        print(f"Fichiers JPG trouvés dans le dossier: {len(fichiers_presents)}")
        
    except Exception as e:
        print(f"Erreur lors de la lecture du dossier {dossier_photos}: {e}")
    
    return fichiers_presents


def nettoyer_annotations_csv(csv_path: str, dossier_photos: str) -> Tuple[int, int]:
    """
    Nettoie le CSV d'annotations en supprimant les entrées sans fichier correspondant.
    
    Args:
        csv_path: Chemin vers le fichier CSV d'annotations
        dossier_photos: Chemin vers le dossier des photos
        
    Returns:
        Tuple[int, int]: (nombre de lignes conservées, nombre de lignes supprimées)
    """
    # Vérifier que le CSV existe
    if not os.path.exists(csv_path):
        print(f"Erreur: Le fichier CSV {csv_path} n'existe pas.")
        return 0, 0
    
    # Obtenir la liste des fichiers présents
    fichiers_presents = obtenir_fichiers_images(dossier_photos)
    
    if not fichiers_presents:
        print("Aucun fichier JPG trouvé. Arrêt du traitement.")
        return 0, 0
    
    try:
        # Charger le CSV
        print(f"Chargement du CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Lignes dans le CSV original: {len(df)}")
        
        # Vérifier que la colonne 'filename' existe
        if 'filename' not in df.columns:
            print("Erreur: La colonne 'filename' n'existe pas dans le CSV.")
            return 0, 0
        
        # Séparer les lignes valides des lignes invalides
        lignes_valides = []
        lignes_invalides = []
        
        for index, row in df.iterrows():
            filename = row['filename']
            
            # Gérer les valeurs NaN ou vides
            if pd.isna(filename) or filename == '':
                print(f"Ligne {index + 1}: filename vide ou NaN")
                lignes_invalides.append(row)
                continue
            
            # Vérifier si le fichier existe
            if filename in fichiers_presents:
                lignes_valides.append(row)
            else:
                print(f"Fichier manquant: {filename}")
                lignes_invalides.append(row)
        
        # Créer les DataFrames résultants
        df_valide = pd.DataFrame(lignes_valides)
        df_invalide = pd.DataFrame(lignes_invalides)
        
        nb_conservees = len(lignes_valides)
        nb_supprimees = len(lignes_invalides)
        
        print(f"Lignes conservées: {nb_conservees}")
        print(f"Lignes à supprimer: {nb_supprimees}")
        
        # Sauvegarder les lignes supprimées dans un fichier de sauvegarde
        if nb_supprimees > 0:
            dossier_csv = os.path.dirname(csv_path)
            fichier_doublons = os.path.join(dossier_csv, "photos_en_doublons.csv")
            
            try:
                df_invalide.to_csv(fichier_doublons, index=False)
                print(f"Lignes supprimées sauvegardées dans: {fichier_doublons}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde des doublons: {e}")
        
        # Sauvegarder le CSV nettoyé (remplace l'original)
        if nb_conservees > 0:
            try:
                # Créer une sauvegarde de l'original avant modification
                backup_path = csv_path + ".backup"
                df.to_csv(backup_path, index=False)
                print(f"Sauvegarde de l'original créée: {backup_path}")
                
                # Sauvegarder la version nettoyée
                df_valide.to_csv(csv_path, index=False)
                print(f"CSV nettoyé sauvegardé: {csv_path}")
                
            except Exception as e:
                print(f"Erreur lors de la sauvegarde du CSV nettoyé: {e}")
                return 0, 0
        else:
            print("Attention: Aucune ligne valide trouvée. Le CSV original n'est pas modifié.")
        
        return nb_conservees, nb_supprimees
        
    except Exception as e:
        print(f"Erreur lors du traitement du CSV: {e}")
        return 0, 0


def main():
    """Fonction principale du script."""
    print("=== Nettoyage du CSV d'annotations ===")
    
    # Chemins des fichiers
    dossier_photos = "/home/sarsator/projets/gaia_vision/training/data/DL_data/photos"
    csv_annotations = "/home/sarsator/projets/gaia_vision/training/data/DL_data/etiquettes/annotations.csv"
    
    print(f"Dossier photos: {dossier_photos}")
    print(f"CSV annotations: {csv_annotations}")
    print()
    
    # Nettoyer le CSV
    nb_conservees, nb_supprimees = nettoyer_annotations_csv(csv_annotations, dossier_photos)
    
    # Résumé final
    print("\n=== Résumé ===")
    print(f"✅ Lignes conservées: {nb_conservees}")
    print(f"❌  Lignes supprimées: {nb_supprimees}")
    
    if nb_supprimees > 0:
        print(f"📁 Lignes supprimées sauvegardées dans: photos_en_doublons.csv")
        print(f"💾 Sauvegarde originale créée: annotations.csv.backup")
    
    if nb_conservees == 0 and nb_supprimees == 0:
        print("❌ Aucune opération effectuée.")
    elif nb_supprimees == 0:
        print(" Aucune ligne à supprimer trouvée. Le CSV est déjà propre!")
    else:
        print("✅ Nettoyage terminé avec succès.")


if __name__ == "__main__":
    main()