#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour nettoyer les doublons dans le fichier annotations.csv
"""

import pandas as pd
import os

def nettoyer_doublons_csv():
    """
    Nettoie les doublons dans le CSV en gardant la version avec le statut le plus récent.
    Priorité : ANNOTE > PASSE > à annoter > vide
    """
    annotations_csv = "/home/sarsator/projets/gaia_vision/training/data/DL_data/etiquettes/annotations.csv"
    
    if not os.path.exists(annotations_csv):
        print(f"Fichier non trouvé : {annotations_csv}")
        return
        
    try:
        df = pd.read_csv(annotations_csv)
        print(f"Nombre de lignes avant nettoyage : {len(df)}")
        
        # Définir la priorité des statuts
        def priorite_statut(statut):
            if pd.isna(statut) or statut == '':
                return 0
            elif statut == 'à annoter':
                return 1
            elif statut == 'PASSE':
                return 2
            elif statut == 'ANNOTE':
                return 3
            else:
                return 0
        
        # Ajouter une colonne de priorité
        df['priorite'] = df['statut'].apply(priorite_statut)
        
        # Grouper par nom d'image et garder celui avec la priorité la plus élevée
        df_nettoye = df.loc[df.groupby('filename')['priorite'].idxmax()]
        
        # Supprimer la colonne temporaire
        df_nettoye = df_nettoye.drop('priorite', axis=1)
        
        # Vérifier s'il y avait des doublons
        nombre_avant = len(df)
        nombre_apres = len(df_nettoye)
        
        if nombre_avant > nombre_apres:
            # Sauvegarder le CSV nettoyé
            df_nettoye.to_csv(annotations_csv, index=False)
            print(f"CSV nettoyé : {nombre_avant - nombre_apres} doublons supprimés ({nombre_avant} → {nombre_apres} lignes)")
        else:
            print("Aucun doublon trouvé dans le CSV")
            
    except Exception as e:
        print(f"Erreur lors du nettoyage des doublons : {e}")

if __name__ == "__main__":
    print("Nettoyage des doublons dans annotations.csv...")
    nettoyer_doublons_csv()
    print("Terminé !")
