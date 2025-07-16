#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de détection et suppression de doublons visuels d'images
===============================================================

Ce script parcourt un dossier d'images JPG et détecte les doublons visuels
même s'ils ont subi des transformations (rotation, flip, crop, bruit).

Fonctionnalités :
- Détection de doublons avec hachage perceptuel (pHash)
- Interface graphique pour sélectionner quelle image garder
- Déplacement automatique des doublons vers un sous-dossier
- Gestion robuste des erreurs et fichiers corrompus
- Compatible WSL/Linux et VS Code

Dépendances :
    - tkinter (inclus avec Python)
    - PIL (Pillow)
    - imagehash
    - os, shutil, glob

Utilisation :
    python antidoublon.py

Auteur : Générateur d'IA
Date : Juillet 2025
"""

import os
import glob
import shutil
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import imagehash


class DetecteurDoublons:
    """
    Classe principale pour la détection et gestion des doublons d'images.
    """
    
    def __init__(self):
        # Chemins
        self.photos_dir = "/home/sarsator/projets/gaia_vision/training/data/DL_data/photos"
        self.doublons_dir = os.path.join(self.photos_dir, "doublons")
        
        # Variables pour l'état
        self.images_files = []
        self.groupes_doublons = []
        self.groupe_actuel_index = 0
        self.images_selectionnees = set()  # Changé pour supporter plusieurs sélections
        
        # Interface graphique
        self.root = None
        self.image_labels = []
        self.image_buttons = []
        self.image_checkboxes = []  # Nouveau : cases à cocher
        self.checkbox_vars = []     # Variables des cases à cocher
        self.current_images = []
        self.scrollable_frame = None  # Pour le défilement
        
        # Seuil de similarité pour considérer deux images comme doublons
        # Plus la valeur est faible, plus les images doivent être similaires
        self.seuil_similarite = 8
        
    def detecter_doublons(self) -> List[List[str]]:
        """
        Détecte les groupes de doublons visuels.
        
        Returns:
            List[List[str]]: Liste des groupes de doublons (chaque groupe = liste de chemins)
        """
        print("Chargement des images...")
        
        # Charger toutes les images JPG
        pattern = os.path.join(self.photos_dir, "*.jpg")
        self.images_files = glob.glob(pattern)
        
        if not self.images_files:
            print("Aucune image JPG trouvée dans le dossier.")
            return []
        
        print(f"Analyse de {len(self.images_files)} images...")
        
        # Calculer les hash perceptuels pour chaque image
        hashes = {}
        images_valides = []
        
        for i, image_path in enumerate(self.images_files):
            try:
                # Afficher le progrès
                if i % 100 == 0:
                    print(f"Traitement: {i+1}/{len(self.images_files)}")
                
                # Ouvrir l'image et calculer plusieurs hash pour robustesse
                with Image.open(image_path) as img:
                    # Convertir en RGB si nécessaire
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Calculer différents types de hash perceptuels
                    phash = imagehash.phash(img, hash_size=16)
                    dhash = imagehash.dhash(img, hash_size=16)
                    whash = imagehash.whash(img, hash_size=16)
                    
                    # Tester aussi les rotations pour plus de robustesse
                    phash_90 = imagehash.phash(img.rotate(90), hash_size=16)
                    phash_180 = imagehash.phash(img.rotate(180), hash_size=16)
                    phash_270 = imagehash.phash(img.rotate(270), hash_size=16)
                    
                    # Tester les flips
                    phash_flip_h = imagehash.phash(img.transpose(Image.FLIP_LEFT_RIGHT), hash_size=16)
                    phash_flip_v = imagehash.phash(img.transpose(Image.FLIP_TOP_BOTTOM), hash_size=16)
                    
                    hashes[image_path] = {
                        'phash': phash,
                        'dhash': dhash,
                        'whash': whash,
                        'rotations': [phash_90, phash_180, phash_270],
                        'flips': [phash_flip_h, phash_flip_v]
                    }
                    images_valides.append(image_path)
                    
            except Exception as e:
                print(f"Erreur lors du traitement de {os.path.basename(image_path)}: {e}")
                continue
        
        print(f"Images valides analysées: {len(images_valides)}")
        
        # Grouper les images similaires
        groupes = self._grouper_images_similaires(hashes, images_valides)
        
        # Filtrer pour ne garder que les groupes avec plus d'une image
        groupes_doublons = [groupe for groupe in groupes if len(groupe) > 1]
        
        print(f"Groupes de doublons détectés: {len(groupes_doublons)}")
        
        return groupes_doublons
    
    def _grouper_images_similaires(self, hashes: Dict, images_valides: List[str]) -> List[List[str]]:
        """
        Groupe les images similaires ensemble.
        
        Args:
            hashes: Dictionnaire des hash pour chaque image
            images_valides: Liste des chemins d'images valides
            
        Returns:
            List[List[str]]: Groupes d'images similaires
        """
        groupes = []
        images_traitees = set()
        
        for i, image1 in enumerate(images_valides):
            if image1 in images_traitees:
                continue
                
            groupe_actuel = [image1]
            images_traitees.add(image1)
            
            # Comparer avec toutes les autres images
            for j, image2 in enumerate(images_valides[i+1:], i+1):
                if image2 in images_traitees:
                    continue
                
                if self._images_similaires(hashes[image1], hashes[image2]):
                    groupe_actuel.append(image2)
                    images_traitees.add(image2)
            
            groupes.append(groupe_actuel)
        
        return groupes
    
    def _images_similaires(self, hash1: Dict, hash2: Dict) -> bool:
        """
        Détermine si deux images sont similaires basé sur leurs hash.
        
        Args:
            hash1, hash2: Dictionnaires contenant les différents hash
            
        Returns:
            bool: True si les images sont considérées comme similaires
        """
        # Tester la similarité avec les hash principaux
        if (hash1['phash'] - hash2['phash']) <= self.seuil_similarite:
            return True
        if (hash1['dhash'] - hash2['dhash']) <= self.seuil_similarite:
            return True
        if (hash1['whash'] - hash2['whash']) <= self.seuil_similarite:
            return True
        
        # Tester avec les rotations de la première image
        for rotation_hash in hash1['rotations']:
            if (rotation_hash - hash2['phash']) <= self.seuil_similarite:
                return True
        
        # Tester avec les flips de la première image
        for flip_hash in hash1['flips']:
            if (flip_hash - hash2['phash']) <= self.seuil_similarite:
                return True
        
        # Tester l'inverse (rotations de la seconde image)
        for rotation_hash in hash2['rotations']:
            if (hash1['phash'] - rotation_hash) <= self.seuil_similarite:
                return True
        
        for flip_hash in hash2['flips']:
            if (hash1['phash'] - flip_hash) <= self.seuil_similarite:
                return True
        
        return False
    
    def creer_interface(self):
        """Crée l'interface graphique Tkinter."""
        try:
            self.root = tk.Tk()
            self.root.title("Détecteur de Doublons - Gaia Vision")
            self.root.geometry("1400x900")
            self.root.configure(bg='white')
            
            # Tester la création d'une image simple pour vérifier X11
            try:
                test_img = Image.new('RGB', (10, 10), color='red')
                test_photo = ImageTk.PhotoImage(test_img)
                del test_photo  # Nettoyer le test
            except Exception as e:
                print(f"Erreur lors du test X11: {e}")
                raise
            
        except Exception as e:
            print(f"Erreur lors de la création de l'interface: {e}")
            print("Vérifiez que vous avez accès à l'affichage X11 (export DISPLAY=:0)")
            raise
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Titre et informations
        titre_frame = ttk.Frame(main_frame)
        titre_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.titre_label = ttk.Label(titre_frame, text="Sélection des doublons", 
                                    font=("Arial", 16, "bold"))
        self.titre_label.pack()
        
        self.info_label = ttk.Label(titre_frame, text="", font=("Arial", 10))
        self.info_label.pack(pady=5)
        
        # Instructions
        instructions = ttk.Label(titre_frame, 
                                text="🔍 COCHEZ les images à GARDER ou CLIQUEZ sur UNE image pour ne garder qu'elle.\n"
                                     "📦 Les images non sélectionnées seront déplacées vers le dossier 'doublons'.",
                                font=("Arial", 10), foreground="blue")
        instructions.pack(pady=5)
        
        # Frame avec scrollbar pour les images
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Canvas et scrollbar
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas et scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind de la molette de souris pour le défilement
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))  # Linux
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))   # Linux
        
        # Frame pour les boutons
        boutons_frame = ttk.Frame(main_frame)
        boutons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(boutons_frame, text="✅ Confirmer sélection", 
                  command=self.confirmer_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(boutons_frame, text="Passer ce groupe", 
                  command=self.passer_groupe).pack(side=tk.LEFT, padx=5)
        ttk.Button(boutons_frame, text="Tout sélectionner", 
                  command=self.tout_selectionner).pack(side=tk.LEFT, padx=5)
        ttk.Button(boutons_frame, text="Tout désélectionner", 
                  command=self.tout_deselectionner).pack(side=tk.LEFT, padx=5)
        ttk.Button(boutons_frame, text="Quitter", 
                  command=self.quitter).pack(side=tk.RIGHT, padx=5)
        
        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.pack(pady=10)
        
    def afficher_groupe_doublons(self, groupe: List[str]):
        """
        Affiche un groupe de doublons dans l'interface avec défilement.
        
        Args:
            groupe: Liste des chemins des images du groupe
        """
        # Nettoyer l'affichage précédent
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.current_images = groupe
        self.images_selectionnees = set()  # Réinitialiser les sélections
        self.image_buttons = []
        self.image_checkboxes = []
        self.checkbox_vars = []
        
        # Mettre à jour les informations
        self.info_label.configure(text=f"Groupe {self.groupe_actuel_index + 1}/{len(self.groupes_doublons)} - {len(groupe)} images similaires")
        
        # Mettre à jour la barre de progression
        progress = (self.groupe_actuel_index / len(self.groupes_doublons)) * 100
        self.progress_var.set(progress)
        
        # Afficher TOUTES les images (plus de limitation)
        print(f"Affichage de {len(groupe)} images avec défilement")
        
        # Calculer la disposition des images (4 colonnes pour optimiser l'espace)
        nb_images = len(groupe)
        nb_colonnes = 4  # 4 colonnes pour avoir plus d'images visibles
        nb_lignes = (nb_images + nb_colonnes - 1) // nb_colonnes
        
        images_reussies = 0
        
        for i, image_path in enumerate(groupe):
            ligne = i // nb_colonnes
            colonne = i % nb_colonnes
            
            try:
                # Créer un frame pour chaque image
                image_frame = ttk.Frame(self.scrollable_frame)
                image_frame.grid(row=ligne, column=colonne, padx=5, pady=5, sticky="nsew")
                
                # Variable pour la case à cocher
                var = tk.BooleanVar()
                self.checkbox_vars.append(var)
                
                # Case à cocher en haut
                checkbox = ttk.Checkbutton(image_frame, text="Garder", variable=var,
                                         command=lambda idx=i: self.on_checkbox_change(idx))
                checkbox.pack(pady=2)
                self.image_checkboxes.append(checkbox)
                
                # Pré-tester l'image avant de l'afficher
                with Image.open(image_path) as img_test:
                    if img_test.width == 0 or img_test.height == 0:
                        raise ValueError(f"Image avec dimensions invalides: {img_test.width}x{img_test.height}")
                
                # Charger et redimensionner l'image (taille plus petite pour plus d'images)
                with Image.open(image_path) as img:
                    if img.width <= 0 or img.height <= 0:
                        raise ValueError(f"Image avec dimensions invalides: {img.width}x{img.height}")
                    
                    # Taille plus petite pour avoir plus d'images visibles
                    max_size = 180
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    if img.width < 20 or img.height < 20:
                        img = img.resize((50, 50), Image.Resampling.LANCZOS)
                    
                    print(f"Affichage image: {os.path.basename(image_path)} ({img.width}x{img.height})")
                    
                    # Créer l'image Tkinter
                    try:
                        photo = ImageTk.PhotoImage(img)
                        
                        # Bouton image cliquable (clic = sélection unique)
                        btn = tk.Button(image_frame, image=photo, 
                                      command=lambda idx=i: self.selectionner_image_unique(idx),
                                      relief=tk.RAISED, borderwidth=2,
                                      cursor="hand2")
                        btn.image = photo  # Garder une référence
                        btn.pack(pady=2)
                        
                    except Exception as e:
                        print(f"⚠️ Erreur X11 pour l'image {os.path.basename(image_path)}: {e}")
                        # Bouton de remplacement
                        btn = tk.Button(image_frame, 
                                      text=f"Image\n{os.path.basename(image_path)[:15]}...\n(Erreur d'affichage)",
                                      command=lambda idx=i: self.selectionner_image_unique(idx),
                                      relief=tk.RAISED, borderwidth=2,
                                      width=12, height=6,
                                      cursor="hand2")
                        btn.pack(pady=2)
                    
                    # Label avec le nom du fichier (plus court)
                    nom_fichier = os.path.basename(image_path)
                    if len(nom_fichier) > 20:
                        nom_fichier = nom_fichier[:17] + "..."
                    label = ttk.Label(image_frame, text=nom_fichier, 
                                    font=("Arial", 7), wraplength=150)
                    label.pack(pady=2)
                    
                    self.image_buttons.append(btn)
                    images_reussies += 1
                        
            except Exception as e:
                print(f"Erreur lors de l'affichage de {os.path.basename(image_path)}: {e}")
                # Placeholder en cas d'erreur
                error_frame = ttk.Frame(self.scrollable_frame)
                error_frame.grid(row=ligne, column=colonne, padx=5, pady=5)
                
                # Variable pour la case à cocher même en cas d'erreur
                var = tk.BooleanVar()
                self.checkbox_vars.append(var)
                
                checkbox = ttk.Checkbutton(error_frame, text="Garder", variable=var,
                                         command=lambda idx=i: self.on_checkbox_change(idx))
                checkbox.pack()
                self.image_checkboxes.append(checkbox)
                
                error_label = ttk.Label(error_frame, text=f"Erreur:\n{os.path.basename(image_path)[:15]}...", 
                                      font=("Arial", 8), foreground="red")
                error_label.pack()
                
                btn_placeholder = tk.Button(error_frame, text="IMAGE\nCORROMPUE", 
                                           command=lambda idx=i: self.selectionner_image_unique(idx),
                                           bg='red', fg='white', width=12, height=4)
                btn_placeholder.pack()
                self.image_buttons.append(btn_placeholder)
        
        print(f"Images affichées avec succès: {images_reussies}/{len(groupe)}")
        
        # Configurer la grille pour être responsive
        for i in range(nb_colonnes):
            self.scrollable_frame.columnconfigure(i, weight=1)
        
        # Mettre à jour la région de défilement
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_checkbox_change(self, index: int):
        """
        Gère le changement d'état d'une case à cocher.
        
        Args:
            index: Index de l'image dans la liste courante
        """
        if index < len(self.checkbox_vars):
            if self.checkbox_vars[index].get():
                self.images_selectionnees.add(index)
                # Mettre en surbrillance le bouton
                if index < len(self.image_buttons):
                    self.image_buttons[index].configure(bg='lightgreen', relief=tk.SUNKEN)
            else:
                self.images_selectionnees.discard(index)
                # Enlever la surbrillance
                if index < len(self.image_buttons):
                    self.image_buttons[index].configure(bg='lightgray', relief=tk.RAISED)
    
    def selectionner_image_unique(self, index: int):
        """
        Sélectionne uniquement cette image (désélectionne toutes les autres).
        
        Args:
            index: Index de l'image dans la liste courante
        """
        # Décocher toutes les cases
        for i, var in enumerate(self.checkbox_vars):
            var.set(False)
            if i < len(self.image_buttons):
                self.image_buttons[i].configure(bg='lightgray', relief=tk.RAISED)
        
        # Sélectionner uniquement cette image
        if index < len(self.checkbox_vars):
            self.checkbox_vars[index].set(True)
            self.images_selectionnees = {index}
            if index < len(self.image_buttons):
                self.image_buttons[index].configure(bg='lightblue', relief=tk.SUNKEN)
    
    def tout_selectionner(self):
        """
        Sélectionne toutes les images du groupe.
        """
        for i, var in enumerate(self.checkbox_vars):
            var.set(True)
            if i < len(self.image_buttons):
                self.image_buttons[i].configure(bg='lightgreen', relief=tk.SUNKEN)
        self.images_selectionnees = set(range(len(self.current_images)))
    
    def tout_deselectionner(self):
        """
        Désélectionne toutes les images du groupe.
        """
        for i, var in enumerate(self.checkbox_vars):
            var.set(False)
            if i < len(self.image_buttons):
                self.image_buttons[i].configure(bg='lightgray', relief=tk.RAISED)
        self.images_selectionnees.clear()
    
    def confirmer_selection(self):
        """
        Confirme la sélection des images à garder et déplace les non-sélectionnées.
        """
        if not self.images_selectionnees:
            messagebox.showwarning("Attention", "Aucune image sélectionnée !")
            return
        
        # Images à garder (indices sélectionnés)
        images_a_garder = [self.current_images[i] for i in self.images_selectionnees]
        # Images à déplacer (indices non sélectionnés)
        images_a_deplacer = [self.current_images[i] for i in range(len(self.current_images)) 
                           if i not in self.images_selectionnees]
        
        message = f"Garder {len(images_a_garder)} image(s) et déplacer {len(images_a_deplacer)} image(s) ?"
        
        if messagebox.askyesno("Confirmer", message):
            # Déplacer les images non sélectionnées
            for image_path in images_a_deplacer:
                self.deplacer_image(image_path)
            
            # Passer au groupe suivant
            self.passer_au_suivant()
    
    def passer_groupe(self):
        """Passe au groupe suivant sans déplacer d'images."""
        self.groupe_suivant()
    
    def groupe_suivant(self):
        """Affiche le groupe de doublons suivant."""
        self.groupe_actuel_index += 1
        
        if self.groupe_actuel_index >= len(self.groupes_doublons):
            # Terminé
            messagebox.showinfo("Terminé", 
                              "✅ Tous les groupes de doublons ont été traités!\n\n"
                              "Aucun doublon restant détecté.", 
                              parent=self.root)
            self.root.quit()
        else:
            # Afficher le groupe suivant
            groupe = self.groupes_doublons[self.groupe_actuel_index]
            self.afficher_groupe_doublons(groupe)
    
    def quitter(self):
        """Ferme l'application."""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter ?", parent=self.root):
            self.root.quit()
    
    def executer(self):
        """Méthode principale pour exécuter le processus de détection de doublons."""
        print("=== Détecteur de Doublons Visuels ===")
        print(f"Dossier analysé: {self.photos_dir}")
        
        # Vérifier que le dossier existe
        if not os.path.exists(self.photos_dir):
            print(f"Erreur: Le dossier {self.photos_dir} n'existe pas.")
            return
        
        # Détecter les doublons
        self.groupes_doublons = self.detecter_doublons()
        
        if not self.groupes_doublons:
            print("✅ Aucun doublon détecté!")
            messagebox.showinfo("Résultat", "✅ Aucun doublon détecté!\n\nToutes les images sont uniques.")
            return
        
        print(f"Groupes de doublons trouvés: {len(self.groupes_doublons)}")
        
        # Créer et lancer l'interface graphique
        self.creer_interface()
        
        # Afficher le premier groupe
        self.groupe_actuel_index = 0
        groupe = self.groupes_doublons[self.groupe_actuel_index]
        self.afficher_groupe_doublons(groupe)
        
        # Lancer la boucle principale
        self.root.mainloop()
    
    def deplacer_image(self, image_path: str):
        """
        Déplace une image vers le dossier doublons.
        
        Args:
            image_path: Chemin de l'image à déplacer
        """
        try:
            nom_fichier = os.path.basename(image_path)
            destination = os.path.join(self.doublons_dir, nom_fichier)
            
            # Gérer les noms de fichiers en conflit
            compteur = 1
            while os.path.exists(destination):
                nom_base, extension = os.path.splitext(nom_fichier)
                nouveau_nom = f"{nom_base}_{compteur}{extension}"
                destination = os.path.join(self.doublons_dir, nouveau_nom)
                compteur += 1
            
            shutil.move(image_path, destination)
            print(f"✓ Image déplacée: {nom_fichier} -> doublons/")
            
        except Exception as e:
            print(f"❌ Erreur lors du déplacement de {os.path.basename(image_path)}: {e}")
    
    def passer_au_suivant(self):
        """
        Passe au groupe suivant de doublons.
        """
        self.groupe_actuel_index += 1
        
        if self.groupe_actuel_index < len(self.groupes_doublons):
            self.afficher_groupe_doublons(self.groupes_doublons[self.groupe_actuel_index])
        else:
            messagebox.showinfo("Terminé", "Tous les groupes de doublons ont été traités !")
            self.root.quit()
    
    def quitter(self):
        """Ferme l'application."""
        if messagebox.askyesno("Quitter", "Voulez-vous vraiment quitter ?"):
            self.root.quit()


def main():
    """Fonction principale du script."""
    print("Vérification des dépendances...")
    
    # Vérifier l'environnement X11 pour l'affichage graphique
    if 'DISPLAY' not in os.environ:
        print("⚠️  Variable DISPLAY non définie. Tentative de configuration automatique...")
        os.environ['DISPLAY'] = ':0'
    
    print(f"DISPLAY configuré sur: {os.environ.get('DISPLAY', 'Non défini')}")
    
    # Vérifier les dépendances
    try:
        import PIL
        print("✓ Pillow disponible")
    except ImportError:
        print("✗ Pillow manquant. Installez-le avec: pip install Pillow")
        return
    
    try:
        import imagehash
        print("✓ imagehash disponible")
    except ImportError:
        print("✗ imagehash manquant. Installez-le avec: pip install imagehash")
        return
    
    # Tester l'accès à X11
    try:
        test_root = tk.Tk()
        test_root.withdraw()  # Cacher la fenêtre de test
        test_root.destroy()
        print("✓ Accès X11 disponible")
    except Exception as e:
        print(f"✗ Erreur d'accès X11: {e}")
        print("Solutions possibles:")
        print("1. Exécutez: export DISPLAY=:0")
        print("2. Vérifiez que vous êtes dans un environnement avec interface graphique")
        print("3. Si vous utilisez WSL, installez un serveur X11 comme VcXsrv")
        return
    
    # Lancer le détecteur
    detecteur = DetecteurDoublons()
    detecteur.executer()


if __name__ == "__main__":
    main()