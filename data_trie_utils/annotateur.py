#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'annotation d'images pour le projet Gaia Vision
=======================================================

Ce script permet d'annoter des images JPG avec une interface graphique simple.

Fonctionnalités :
- Charge les images depuis training/data/dl_data/photos/
- Gère un CSV d'annotations avec sauvegarde automatique
- Interface graphique avec aperçu image et champs d'annotation
- Reprend automatiquement là où on s'est arrêté
- Compatible WSL/Linux et VS Code

Utilisation :
    python annotateur.py

Dépendances :
    - tkinter (inclus avec Python)
    - PIL (Pillow)
    - pandas
    - os, glob

Auteur : Générateur d'IA
Date : Juillet 2025
"""

import os
import glob
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json


class AnnotateurImages:
    """
    Classe principale pour l'annotation d'images avec interface graphique.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Annotateur d'Images - Gaia Vision")
        self.root.geometry("1200x800")
        
        # Chemins des fichiers
        self.photos_dir = "/home/sarsator/projets/gaia_vision/training/data/DL_data/photos"
        self.annotations_csv = "/home/sarsator/projets/gaia_vision/training/data/DL_data/etiquettes/annotations.csv"
        
        # Variables pour l'état actuel
        self.images_a_traiter = []
        self.images_traitees = []  # Historique des images traitées pour le retour
        self.current_index = 0
        self.current_image = None
        self.current_image_tk = None
        
        # Variables d'annotation
        self.setup_annotation_variables()
        
        # Initialisation
        self.charger_images()
        self.nettoyer_doublons_csv()  # Nettoyer les doublons avant de charger
        self.charger_images_a_annoter()
        self.setup_interface()
        self.afficher_image_courante()
        
    def setup_annotation_variables(self):
        """Initialise les variables tkinter pour les annotations."""
        self.etat_champignon = tk.StringVar(value="healthy")
        self.elastique = tk.BooleanVar()
        self.multi_sac = tk.IntVar()
        self.vue_sac = tk.StringVar(value="dessus")
        self.couleur_ou_objet_ou_personne_visible = tk.BooleanVar()
        self.etiquette_sur_sac = tk.BooleanVar()
        self.artefact_sur_sac = tk.BooleanVar()
        self.arriere_plan_flou = tk.BooleanVar()
        
    def charger_images(self):
        """Charge la liste des images à traiter (JPG, JPEG, PNG)."""
        patterns = [
            os.path.join(self.photos_dir, "*.jpg"),
            os.path.join(self.photos_dir, "*.jpeg"), 
            os.path.join(self.photos_dir, "*.png")
        ]
        
        images_trouvees = []
        for pattern in patterns:
            images_trouvees.extend(glob.glob(pattern))
        
        # Garde seulement les noms de fichiers
        self.images_disponibles = [os.path.basename(img) for img in images_trouvees]
        
        if not self.images_disponibles:
            messagebox.showerror("Erreur", f"Aucune image JPG trouvée dans {self.photos_dir}")
            self.root.quit()
            return
            
        print(f"Images trouvées : {len(self.images_disponibles)}")
        
    def nettoyer_doublons_csv(self):
        """
        Nettoie les doublons dans le CSV en gardant la version avec le statut le plus récent.
        Priorité : ANNOTE > PASSE > à annoter > vide
        """
        if not os.path.exists(self.annotations_csv):
            return
            
        try:
            df = pd.read_csv(self.annotations_csv)
            
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
                df_nettoye.to_csv(self.annotations_csv, index=False)
                print(f"CSV nettoyé : {nombre_avant - nombre_apres} doublons supprimés ({nombre_avant} → {nombre_apres} lignes)")
            else:
                print("Aucun doublon trouvé dans le CSV")
                
        except Exception as e:
            print(f"Erreur lors du nettoyage des doublons : {e}")
        
    def charger_images_a_annoter(self):
        """
        Charge les images à annoter depuis le fichier annotations.csv.
        Ne garde que celles avec statut vide ou "à annoter" et qui ont déjà Contaminated/Healthy remplis.
        """
        self.images_a_traiter = []
        self.annotations_existantes = {}
        
        if not os.path.exists(self.annotations_csv):
            messagebox.showerror("Erreur", f"Fichier annotations.csv introuvable : {self.annotations_csv}")
            self.root.quit()
            return
            
        try:
            df_annotations = pd.read_csv(self.annotations_csv)
            
            # Filtrer les images qui doivent être annotées
            for _, row in df_annotations.iterrows():
                nom_image = row.get('filename', row.get('nom_image', ''))
                statut = row.get('statut', '')
                contaminated = row.get('Contaminated', '')
                healthy = row.get('Healthy', '')
                
                # Vérifier si l'image doit être annotée (statut "à annoter" uniquement)
                doit_etre_annotee = (statut == 'à annoter')
                
                # Vérifier si l'image existe physiquement
                existe_physiquement = nom_image in self.images_disponibles
                
                if doit_etre_annotee and existe_physiquement:
                    self.images_a_traiter.append(nom_image)
                    self.annotations_existantes[nom_image] = row.to_dict()
                    
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les annotations : {e}")
            self.root.quit()
            return
        
        if not self.images_a_traiter:
            messagebox.showinfo("Terminé", "Aucune image à annoter trouvée !")
            self.root.quit()
            return
            
        print(f"Images à annoter trouvées : {len(self.images_a_traiter)}")
        
    def charger_annotations_existantes(self):
        """
        Ancienne fonction - remplacée par charger_images_a_annoter()
        """
        pass
        
    def setup_interface(self):
        """Met en place l'interface graphique."""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame pour l'image (côté gauche)
        image_frame = ttk.LabelFrame(main_frame, text="Image", padding=10)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack()
        
        # Info sur l'image courante
        self.info_label = ttk.Label(image_frame, text="", font=("Arial", 10))
        self.info_label.pack(pady=5)
        
        # Frame pour les annotations (côté droit)
        annotation_frame = ttk.LabelFrame(main_frame, text="Annotations", padding=10)
        annotation_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Champs d'annotation
        row = 0
        
        # Contaminated/Healthy (important en premier)
        ttk.Label(annotation_frame, text="État du champignon :", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1
        
        contaminated_frame = ttk.Frame(annotation_frame)
        contaminated_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Utiliser une variable StringVar pour les boutons radio exclusifs
        self.etat_champignon = tk.StringVar(value="healthy")
        ttk.Radiobutton(contaminated_frame, text="Contaminated", variable=self.etat_champignon, 
                       value="contaminated").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(contaminated_frame, text="Healthy", variable=self.etat_champignon, 
                       value="healthy").pack(side=tk.LEFT)
        row += 1
        
        # Séparateur
        ttk.Separator(annotation_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
           
        # Multi sac
        ttk.Label(annotation_frame, text="Nombre de sacs :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(annotation_frame, from_=1, to=20, textvariable=self.multi_sac, width=10).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        

        # Vue sac
        ttk.Label(annotation_frame, text="Vue du sac :").grid(row=row, column=0, sticky=tk.W, pady=2)
        vue_frame = ttk.Frame(annotation_frame)
        vue_frame.grid(row=row, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(vue_frame, text="Dessus", variable=self.vue_sac, value="dessus").pack(anchor=tk.W)
        ttk.Radiobutton(vue_frame, text="Côté", variable=self.vue_sac, value="cote").pack(anchor=tk.W)
        ttk.Radiobutton(vue_frame, text="3/4", variable=self.vue_sac, value="3/4").pack(anchor=tk.W)
        row += 1
                
         # Elastique
        ttk.Label(annotation_frame, text="Élastique :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(annotation_frame, variable=self.elastique).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Couleur/objet/personne visible
        ttk.Label(annotation_frame, text="Couleur/objet/personne visible :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(annotation_frame, variable=self.couleur_ou_objet_ou_personne_visible).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Étiquette sur sac
        ttk.Label(annotation_frame, text="Étiquette sur sac :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(annotation_frame, variable=self.etiquette_sur_sac).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Artefact sur sac
        ttk.Label(annotation_frame, text="Artefact sur sac :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(annotation_frame, variable=self.artefact_sur_sac).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Arrière-plan flou
        ttk.Label(annotation_frame, text="Arrière-plan flou :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(annotation_frame, variable=self.arriere_plan_flou).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Séparateur
        ttk.Separator(annotation_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
        
        # Boutons d'action
        boutons_frame = ttk.Frame(annotation_frame)
        boutons_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        ttk.Button(boutons_frame, text="Sauvegarder", command=self.sauvegarder_annotation).pack(fill=tk.X, pady=2)
        ttk.Button(boutons_frame, text="Passer", command=self.passer_image).pack(fill=tk.X, pady=2)
        ttk.Button(boutons_frame, text="Retour", command=self.retour_image_precedente).pack(fill=tk.X, pady=2)
        
        # Raccourcis clavier
        self.root.bind('<Return>', lambda e: self.sauvegarder_annotation())
        self.root.bind('<space>', lambda e: self.passer_image())
        self.root.bind('<BackSpace>', lambda e: self.retour_image_precedente())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        # Info raccourcis
        ttk.Label(annotation_frame, text="\nRaccourcis :\nEntrée : Sauvegarder\nEspace : Passer\nBackspace : Retour\nEchap : Quitter", 
                 font=("Arial", 8), foreground="gray").grid(row=row+1, column=0, columnspan=2, pady=5)
        
    def afficher_image_courante(self):
        """Affiche l'image courante dans l'interface."""
        if not self.images_a_traiter:
            return
            
        if self.current_index >= len(self.images_a_traiter):
            messagebox.showinfo("Terminé", "Toutes les images ont été traitées !")
            self.root.quit()
            return
            
        nom_image = self.images_a_traiter[self.current_index]
        chemin_image = os.path.join(self.photos_dir, nom_image)
        
        try:
            # Charger et redimensionner l'image
            image = Image.open(chemin_image)
            
            # Redimensionner pour l'affichage (max 600x600)
            image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            
            self.current_image_tk = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.current_image_tk)
            
            # Mettre à jour les infos
            self.info_label.configure(text=f"Image {self.current_index + 1}/{len(self.images_a_traiter)}: {nom_image}")
            
            # Charger les annotations existantes ou reset
            self.charger_ou_reset_annotations(nom_image)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image {nom_image}: {e}")
            self.passer_image()
            
    def charger_ou_reset_annotations(self, nom_image):
        """Charge les annotations existantes pour une image ou remet à zéro."""
        if nom_image in self.annotations_existantes:
            # Charger les valeurs existantes
            annotation = self.annotations_existantes[nom_image]
            
            # Contaminated/Healthy
            contaminated_val = annotation.get('Contaminated', 0.0)
            if pd.notna(contaminated_val) and float(contaminated_val) == 1.0:
                self.etat_champignon.set("contaminated")
            else:
                self.etat_champignon.set("healthy")
            
            # Autres champs avec gestion des valeurs NaN
            # Elastique
            elastique_val = annotation.get('elastique', False)
            self.elastique.set(bool(elastique_val) if pd.notna(elastique_val) and elastique_val != '' else False)
            
            # Multi sac - gestion spéciale pour les NaN
            multi_sac_val = annotation.get('multi_sac', 1)
            if pd.notna(multi_sac_val) and multi_sac_val != '':
                try:
                    self.multi_sac.set(int(float(multi_sac_val)))
                except (ValueError, TypeError):
                    self.multi_sac.set(1)
            else:
                self.multi_sac.set(1)
            
            # Vue sac
            vue_sac_val = annotation.get('vue_sac', 'dessus')
            self.vue_sac.set(str(vue_sac_val) if pd.notna(vue_sac_val) and vue_sac_val != '' else 'dessus')
            
            # Autres champs booléens
            couleur_val = annotation.get('couleur_ou_objet_ou_personne_visible', False)
            self.couleur_ou_objet_ou_personne_visible.set(bool(couleur_val) if pd.notna(couleur_val) and couleur_val != '' else True)
            
            etiquette_val = annotation.get('etiquette_sur_sac', False)
            self.etiquette_sur_sac.set(bool(etiquette_val) if pd.notna(etiquette_val) and etiquette_val != '' else False)
            
            artefact_val = annotation.get('artefact_sur_sac', False)
            self.artefact_sur_sac.set(bool(artefact_val) if pd.notna(artefact_val) and artefact_val != '' else False)
            
            arriere_plan_val = annotation.get('arriere_plan_flou', False)
            self.arriere_plan_flou.set(bool(arriere_plan_val) if pd.notna(arriere_plan_val) and arriere_plan_val != '' else False)
            
            print(f"Annotations chargées pour {nom_image}: Contaminated={contaminated_val}, État={self.etat_champignon.get()}")
        else:
            # Pas d'annotation existante, valeurs par défaut
            self.reset_annotations()
    
    def reset_annotations(self):
        """Remet à zéro tous les champs d'annotation."""
        self.etat_champignon.set("healthy")  # Défaut Healthy
        self.multi_sac.set(1)
        self.vue_sac.set("dessus")
        self.elastique.set(False)
        self.couleur_ou_objet_ou_personne_visible.set(True)
        self.etiquette_sur_sac.set(False)
        self.artefact_sur_sac.set(False)
        self.arriere_plan_flou.set(False)
        
    def sauvegarder_annotation(self):
        """Sauvegarde l'annotation courante et passe à l'image suivante."""
        if not self.images_a_traiter:
            return
            
        nom_image = self.images_a_traiter[self.current_index]
        
        # Créer l'annotation
        annotation = {
            "filename": nom_image,
            "Contaminated": 1.0 if self.etat_champignon.get() == "contaminated" else 0.0,
            "Healthy": 1.0 if self.etat_champignon.get() == "healthy" else 0.0,
            "nom_image": nom_image,
            "elastique": self.elastique.get(),
            "multi_sac": self.multi_sac.get(),
            "vue_sac": self.vue_sac.get(),
            "couleur_ou_objet_ou_personne_visible": self.couleur_ou_objet_ou_personne_visible.get(),
            "etiquette_sur_sac": self.etiquette_sur_sac.get(),
            "artefact_sur_sac": self.artefact_sur_sac.get(),
            "arriere_plan_flou": self.arriere_plan_flou.get(),
            "statut": "ANNOTE"  # Marqueur pour identifier les images annotées
        }
        
        # Sauvegarder dans le CSV d'annotations
        self.sauvegarder_dans_csv(annotation)
        
        # Retirer l'image de la liste locale
        self.retirer_image_de_liste(nom_image)
        
        # Passer à l'image suivante
        self.image_suivante()
        
    def passer_image(self):
        """Passe à l'image suivante sans sauvegarder d'annotation."""
        if not self.images_a_traiter:
            return
            
        nom_image = self.images_a_traiter[self.current_index]
        
        # Créer une entrée "passée" dans le CSV pour éviter que l'image revienne
        annotation_passee = {
            "filename": nom_image,
            "Contaminated": "",  # Vide pour indiquer "passé"
            "Healthy": "",       # Vide pour indiquer "passé"
            "nom_image": nom_image,
            "elastique": "",
            "multi_sac": "",
            "vue_sac": "",
            "couleur_ou_objet_ou_personne_visible": "",
            "etiquette_sur_sac": "",
            "artefact_sur_sac": "",
            "arriere_plan_flou": "",
            "statut": "PASSE"  # Marqueur pour identifier les images passées
        }
        
        # Sauvegarder l'entrée "passée" dans le CSV
        self.sauvegarder_dans_csv(annotation_passee)
        
        # Retirer l'image de la liste locale
        self.retirer_image_de_liste(nom_image)
        
        # Passer à l'image suivante
        self.image_suivante()
        
    def sauvegarder_dans_csv(self, annotation):
        """Sauvegarde une annotation dans le CSV en mettant à jour la ligne existante."""
        try:
            if not os.path.exists(self.annotations_csv):
                messagebox.showerror("Erreur", f"Fichier annotations.csv introuvable : {self.annotations_csv}")
                return
                
            # Charger le CSV existant
            df = pd.read_csv(self.annotations_csv)
            
            # Trouver l'index de la ligne à mettre à jour
            nom_image = annotation['nom_image']
            mask = (df['filename'] == nom_image) | (df['nom_image'] == nom_image)
            indices = df.index[mask].tolist()
            
            if indices:
                # Mettre à jour la ligne existante
                index = indices[0]
                for key, value in annotation.items():
                    if key in df.columns:
                        df.at[index, key] = value
                print(f"Annotation mise à jour pour {nom_image}")
            else:
                # Ajouter une nouvelle ligne si pas trouvée (cas de sécurité)
                nouvelle_ligne = pd.DataFrame([annotation])
                df = pd.concat([df, nouvelle_ligne], ignore_index=True)
                print(f"Nouvelle annotation ajoutée pour {nom_image}")
            
            # Sauvegarder
            df.to_csv(self.annotations_csv, index=False)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder l'annotation : {e}")
            
    def retirer_image_de_liste(self, nom_image):
        """Retire une image de la liste locale des images à traiter."""
        try:
            if nom_image in self.images_a_traiter:
                # Ajouter à l'historique avant de retirer
                self.images_traitees.append(nom_image)
                self.images_a_traiter.remove(nom_image)
                print(f"Image {nom_image} retirée de la liste locale et ajoutée à l'historique")
        except Exception as e:
            print(f"Erreur lors du retrait de l'image de la liste : {e}")
            
    def retour_image_precedente(self):
        """Revient à l'image précédente traitée."""
        if not self.images_traitees:
            messagebox.showinfo("Info", "Aucune image précédente disponible !")
            return
            
        # Récupérer la dernière image traitée
        nom_image_precedente = self.images_traitees.pop()
        
        # Remettre son statut à "à annoter" dans le CSV pour qu'elle puisse être re-annotée
        self.remettre_statut_a_annoter(nom_image_precedente)
        
        # La remettre au début de la liste des images à traiter
        self.images_a_traiter.insert(0, nom_image_precedente)
        
        # Remettre l'index à 0 pour afficher cette image
        self.current_index = 0
        
        print(f"Retour à l'image précédente : {nom_image_precedente}")
        
        # Afficher l'image
        self.afficher_image_courante()
        
    def remettre_statut_a_annoter(self, nom_image):
        """Remet le statut d'une image à 'à annoter' pour permettre la re-annotation."""
        try:
            if not os.path.exists(self.annotations_csv):
                return
                
            # Charger le CSV
            df = pd.read_csv(self.annotations_csv)
            
            # Trouver l'image et remettre son statut à "à annoter"
            mask = (df['filename'] == nom_image) | (df['nom_image'] == nom_image)
            indices = df.index[mask].tolist()
            
            if indices:
                index = indices[0]
                df.at[index, 'statut'] = 'à annoter'
                # Sauvegarder
                df.to_csv(self.annotations_csv, index=False)
                print(f"Statut remis à 'à annoter' pour {nom_image}")
                
        except Exception as e:
            print(f"Erreur lors de la remise du statut : {e}")
            
    def image_suivante(self):
        """Passe à l'image suivante."""
        # Pas besoin d'incrémenter l'index car on supprime l'image courante de la liste
        self.afficher_image_courante()
        
    def run(self):
        """Lance l'application."""
        self.root.mainloop()


def main():
    """
    Fonction principale du script.
    """
    print("Démarrage de l'annotateur d'images...")
    
    # Vérifier les dépendances
    try:
        import PIL
        print("✓ Pillow disponible")
    except ImportError:
        print("✗ Pillow manquant. Installez-le avec: pip install Pillow")
        return
        
    try:
        import pandas
        print("✓ Pandas disponible")
    except ImportError:
        print("✗ Pandas manquant. Installez-le avec: pip install pandas")
        return
    
    # Lancer l'application
    app = AnnotateurImages()
    app.run()


if __name__ == "__main__":
    main()