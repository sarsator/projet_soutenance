import csv
import os
from pathlib import Path
import sys

def verify_photos_existence():
    """
    Vérifie l'existence des photos référencées dans train.csv
    """
    # Chemins relatifs depuis data_trie_utils
    csv_path = Path("../training/data/DL_data/etiquettes/annotations.csv")
    photos_dir = Path("../training/data/DL_data/photos")
    
    print("🔍 VÉRIFICATION DE L'EXISTENCE DES PHOTOS")

    
    # Vérification de l'existence des chemins
    if not csv_path.exists():
        print(f"❌ ERREUR: Fichier CSV non trouvé: {csv_path}")
        print(f"Chemin absolu: {csv_path.resolve()}")
        return False
    
    if not photos_dir.exists():
        print(f"❌ ERREUR: Dossier photos non trouvé: {photos_dir}")
        print(f"Chemin absolu: {photos_dir.resolve()}")
        return False
    
    print(f"✅ Fichier CSV trouvé: {csv_path}")
    print(f"✅ Dossier photos trouvé: {photos_dir}")
    print(f"Chemin absolu photos: {photos_dir.resolve()}")
    
    # Lecture du fichier CSV
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = list(reader)
            fieldnames = reader.fieldnames
        
        print(f"CSV lu: {len(data)} lignes")
        print(f"Colonnes: {list(fieldnames)}")
        
    except Exception as e:
        print(f"❌ ERREUR lors de la lecture du CSV: {e}")
        return False
    
    # Vérification de la colonne filename
    if 'filename' not in fieldnames:
        print(f"❌ ERREUR: Colonne 'filename' non trouvée dans le CSV")
        return False
    
    # Compteurs
    total_files = len(data)
    existing_files = 0
    missing_files = 0
    missing_list = []
    
    print(f"\n🔍 Vérification de l'existence de {total_files} photos...")
    
    # Vérification de chaque fichier
    for i, row in enumerate(data, 1):
        filename = row['filename']
        photo_path = photos_dir / filename
        
        if photo_path.exists():
            existing_files += 1
            if i <= 5:  # Afficher les 5 premiers pour vérification
                print(f"✅ {i:3d}/{total_files}: {filename}")
        else:
            missing_files += 1
            missing_list.append(filename)
            if missing_files <= 10:  # Afficher les 10 premiers manquants
                print(f"❌ {i:3d}/{total_files}: {filename} - MANQUANT")
    
    # Résumé
    print(f"\nRÉSUMÉ DE LA VÉRIFICATION:")

    print(f"Total de fichiers dans le CSV: {total_files}")
    print(f"Photos existantes: {existing_files}")
    print(f"Photos manquantes: {missing_files}")
    print(f"Taux de succès: {(existing_files/total_files)*100:.1f}%")
    
    if missing_files > 0:
        print(f"\nPHOTOS MANQUANTES:")
        if missing_files <= 20:
            # Afficher toutes si peu nombreuses
            for filename in missing_list:
                print(f"   - {filename}")
        else:
            # Afficher les 15 premières et 5 dernières
            print("Premières manquantes:")
            for filename in missing_list[:15]:
                print(f"   - {filename}")
            print(f"   ... ({missing_files - 20} autres) ...")
            print("Dernières manquantes:")
            for filename in missing_list[-5:]:
                print(f"   - {filename}")
        
        # Sauvegarde de la liste des fichiers manquants
        missing_csv_path = Path("../training/data/DL_data/splits/train_missing_photos.csv")
        try:
            with open(missing_csv_path, 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['filename'])
                for filename in missing_list:
                    writer.writerow([filename])
            print(f"\nListe des photos manquantes sauvegardée: {missing_csv_path}")
        except Exception as e:
            print(f"Impossible de sauvegarder la liste: {e}")
    
    else:
        print(f"\nPARFAIT: Toutes les photos existent!")
    
    # Vérification inverse: photos dans le dossier mais pas dans le CSV
    print(f"\nVérification inverse: photos orphelines...")
    try:
        # Extensions d'images courantes
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # Fichiers dans le dossier photos
        photos_in_dir = set()
        for ext in image_extensions:
            photos_in_dir.update(f.name for f in photos_dir.glob(f"*{ext}"))
            photos_in_dir.update(f.name for f in photos_dir.glob(f"*{ext.upper()}"))
        
        # Fichiers dans le CSV
        photos_in_csv = set(row['filename'] for row in data)
        
        # Photos orphelines (dans dossier mais pas dans CSV)
        orphan_photos = photos_in_dir - photos_in_csv
        
        print(f"Photos dans le dossier: {len(photos_in_dir)}")
        print(f"Photos dans le CSV: {len(photos_in_csv)}")
        print(f"Photos orphelines: {len(orphan_photos)}")
        
        if orphan_photos:
            print(f"\nQuelques photos orphelines (dans dossier mais pas dans CSV):")
            for i, photo in enumerate(sorted(orphan_photos)[:10]):
                print(f"   - {photo}")
            if len(orphan_photos) > 10:
                print(f"   ... et {len(orphan_photos) - 10} autres")
    
    except Exception as e:
        print(f"Erreur lors de la vérification inverse: {e}")
    
    # Statut final
    success = missing_files == 0
    if success:
        print(f"\nVALIDATION RÉUSSIE: Toutes les photos du train.csv existent!")
    else:
        print(f"\nVALIDATION ÉCHOUÉE: {missing_files} photos manquantes sur {total_files}")
        
    return success

def main():
    """Point d'entrée principal"""
    print("VÉRIFICATEUR D'EXISTENCE DES PHOTOS")

    print("Ce script vérifie que toutes les photos référencées dans")
    print("training/data/DL_data/splits/train.csv existent bien dans")
    print("training/data/DL_data/photos/")

    
    success = verify_photos_existence()
    
    # Code de sortie
    exit_code = 0 if success else 1
    
    if success:
        print(f"\nSUCCÈS: Toutes les vérifications sont passées!")
    else:
        print(f"\nÉCHEC: Des photos sont manquantes!")
        
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
