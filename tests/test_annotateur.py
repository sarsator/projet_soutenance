#!/usr/bin/env python3
"""Test de l'annotateur sans interface graphique"""

import os
import glob
import pandas as pd

def test_annotateur():
    # Chemins
    photos_dir = "/home/sarsator/projets/gaia_vision/training/data/DL_data/photos"
    annotations_csv = "/home/sarsator/projets/gaia_vision/training/data/DL_data/etiquettes/annotations.csv"
    
    # Charger les images
    patterns = [
        os.path.join(photos_dir, "*.jpg"),
        os.path.join(photos_dir, "*.jpeg"), 
        os.path.join(photos_dir, "*.png")
    ]
    
    images_trouvees = []
    for pattern in patterns:
        images_trouvees.extend(glob.glob(pattern))
    
    images_disponibles = [os.path.basename(img) for img in images_trouvees]
    print(f"Images disponibles: {len(images_disponibles)}")
    
    # Charger le CSV
    if not os.path.exists(annotations_csv):
        print(f"Erreur: {annotations_csv} introuvable")
        return
        
    df_annotations = pd.read_csv(annotations_csv)
    print(f"Lignes dans CSV: {len(df_annotations)}")
    
    # Filtrer les images à annoter
    images_a_traiter = []
    count_a_annoter = 0
    count_existant = 0
    
    for _, row in df_annotations.iterrows():
        nom_image = row.get('filename', row.get('nom_image', ''))
        statut = row.get('statut', '')
        
        if statut == 'à annoter':
            count_a_annoter += 1
            print(f"Image à annoter: {nom_image}")
            
            if nom_image in images_disponibles:
                count_existant += 1
                images_a_traiter.append(nom_image)
                print(f"  ✓ Existe physiquement")
            else:
                print(f"  ✗ N'existe pas physiquement")
    
    print(f"\nRésumé:")
    print(f"Images avec statut 'à annoter': {count_a_annoter}")
    print(f"Images à annoter qui existent: {count_existant}")
    print(f"Images à traiter: {len(images_a_traiter)}")
    
    if images_a_traiter:
        print(f"Première image à traiter: {images_a_traiter[0]}")

if __name__ == "__main__":
    test_annotateur()
