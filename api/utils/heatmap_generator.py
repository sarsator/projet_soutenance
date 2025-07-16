#!/usr/bin/env python3
"""
Générateur de heatmap pour visualiser les zones de contamination détectées
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
from typing import List, Dict, Tuple, Any

class ContaminationHeatmapGenerator:
    """
    Générateur de heatmap pour visualiser les zones de contamination
    """
    
    def __init__(self):
        # Palette de couleurs pour la heatmap (du vert au rouge)
        self.colormap = cv2.COLORMAP_JET
        
    def create_contamination_heatmap(self, 
                                   image_path: str, 
                                   detections: List[Dict], 
                                   output_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Crée une heatmap des zones de contamination détectées
        
        Args:
            image_path: Chemin vers l'image originale
            detections: Liste des détections avec bounding boxes et scores
            output_size: Taille de sortie (largeur, hauteur), None pour garder l'original
            
        Returns:
            Image numpy array avec heatmap overlay
        """
        # Charger l'image originale
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        h, w = original_img.shape[:2]
        
        # Redimensionner si nécessaire
        if output_size:
            original_img = cv2.resize(original_img, output_size)
            scale_x = output_size[0] / w
            scale_y = output_size[1] / h
            h, w = output_size[1], output_size[0]
        else:
            scale_x = scale_y = 1.0
        
        # Créer la heatmap de base (toute noire)
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Filtrer seulement les détections de contamination
        contaminated_detections = [d for d in detections if d.get('class_name') == 'contaminated']
        
        if not contaminated_detections:
            # Pas de contamination, retourner l'image originale
            return original_img
        
        print(f"🔥 Génération heatmap pour {len(contaminated_detections)} zone(s) contaminée(s)")
        
        for detection in contaminated_detections:
            score = detection['score']
            box = detection['box']  # [ymin, xmin, ymax, xmax] normalisé
            
            # Convertir les coordonnées normalisées en pixels
            ymin = int(box[0] * h * scale_y)
            xmin = int(box[1] * w * scale_x) 
            ymax = int(box[2] * h * scale_y)
            xmax = int(box[3] * w * scale_x)
            
            # S'assurer que les coordonnées sont dans les limites
            ymin = max(0, min(h-1, ymin))
            ymax = max(0, min(h-1, ymax))
            xmin = max(0, min(w-1, xmin))
            xmax = max(0, min(w-1, xmax))
            
            # Créer un masque gaussien pour cette détection
            mask = self._create_gaussian_mask(h, w, (ymin, xmin, ymax, xmax), score)
            
            # Ajouter à la heatmap globale
            heatmap = np.maximum(heatmap, mask)
        
        # Normaliser la heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Appliquer la colormap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), self.colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Mélanger avec l'image originale
        # Alpha blending: zones sans contamination restent normales
        alpha = heatmap[..., np.newaxis] * 0.6  # Intensité de l'overlay
        blended = original_img.astype(np.float32) * (1 - alpha) + heatmap_colored.astype(np.float32) * alpha
        
        return blended.astype(np.uint8)
    
    def _create_gaussian_mask(self, h: int, w: int, bbox: Tuple[int, int, int, int], intensity: float) -> np.ndarray:
        """
        Crée un masque gaussien pour une bounding box
        
        Args:
            h, w: Dimensions de l'image
            bbox: (ymin, xmin, ymax, xmax)
            intensity: Intensité basée sur le score de détection
            
        Returns:
            Masque gaussien 2D
        """
        ymin, xmin, ymax, xmax = bbox
        
        # Créer une grille de coordonnées
        y, x = np.ogrid[:h, :w]
        
        # Centre de la bounding box
        center_y = (ymin + ymax) / 2
        center_x = (xmin + xmax) / 2
        
        # Taille de la région (pour déterminer l'écart-type du gaussien)
        box_height = ymax - ymin
        box_width = xmax - xmin
        
        # Écart-type basé sur la taille de la box (plus large = plus diffus)
        sigma_y = box_height / 3.0
        sigma_x = box_width / 3.0
        
        # Gaussienne 2D
        gaussian = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))
        
        # Appliquer l'intensité basée sur le score
        gaussian *= intensity * 3.0  # Amplifier pour une meilleure visibilité
        
        # Limiter les valeurs à la région de la bounding box étendue
        margin = 1.2  # Élargir un peu au-delà de la box
        extended_ymin = max(0, int(center_y - box_height * margin / 2))
        extended_ymax = min(h, int(center_y + box_height * margin / 2))
        extended_xmin = max(0, int(center_x - box_width * margin / 2))
        extended_xmax = min(w, int(center_x + box_width * margin / 2))
        
        # Masquer en dehors de la région étendue
        mask = np.zeros_like(gaussian)
        mask[extended_ymin:extended_ymax, extended_xmin:extended_xmax] = gaussian[extended_ymin:extended_ymax, extended_xmin:extended_xmax]
        
        return mask
    
    def create_contamination_overlay_pil(self, 
                                       image_path: str, 
                                       detections: List[Dict],
                                       alpha: float = 0.5) -> Image.Image:
        """
        Version PIL pour créer un overlay de contamination plus artistique
        
        Args:
            image_path: Chemin vers l'image
            detections: Détections de contamination
            alpha: Transparence de l'overlay
            
        Returns:
            Image PIL avec overlay
        """
        # Charger l'image originale
        original_img = Image.open(image_path).convert('RGB')
        w, h = original_img.size
        
        # Créer un calque pour l'overlay
        overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Filtrer les contaminations
        contaminated_detections = [d for d in detections if d.get('class_name') == 'contaminated']
        
        if not contaminated_detections:
            return original_img
        
        for detection in contaminated_detections:
            score = detection['score']
            box = detection['box']  # [ymin, xmin, ymax, xmax] normalisé
            
            # Convertir en coordonnées pixel
            ymin = int(box[0] * h)
            xmin = int(box[1] * w)
            ymax = int(box[2] * h)
            xmax = int(box[3] * w)
            
            # Couleur basée sur l'intensité (rouge vif pour forte contamination)
            intensity = min(1.0, score * 3.0)  # Amplifier pour visibilité
            red = int(255 * intensity)
            alpha_val = int(180 * intensity)  # Transparence variable
            
            # Dessiner un rectangle rempli semi-transparent
            draw.rectangle([xmin, ymin, xmax, ymax], 
                         fill=(red, 50, 50, alpha_val))
            
            # Ajouter un contour plus visible
            draw.rectangle([xmin, ymin, xmax, ymax], 
                         outline=(255, 0, 0, 255), width=3)
        
        # Mélanger avec l'image originale
        result = Image.alpha_composite(original_img.convert('RGBA'), overlay)
        return result.convert('RGB')
    
    def save_heatmap_image(self, heatmap_img: np.ndarray, output_path: str):
        """Sauvegarde la heatmap"""
        if heatmap_img.dtype != np.uint8:
            heatmap_img = (heatmap_img * 255).astype(np.uint8)
        
        cv2.imwrite(output_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        print(f"💾 Heatmap sauvegardée: {output_path}")
    
    def get_heatmap_base64(self, heatmap_img: np.ndarray) -> str:
        """Convertit la heatmap en base64 pour affichage web"""
        if heatmap_img.dtype != np.uint8:
            heatmap_img = (heatmap_img * 255).astype(np.uint8)
        
        # Convertir en PIL Image
        pil_img = Image.fromarray(heatmap_img)
        
        # Encoder en base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"


def test_heatmap_generator():
    """Test de la génération de heatmap"""
    generator = ContaminationHeatmapGenerator()
    
    # Exemple de détections de test
    test_detections = [
        {
            'class_name': 'contaminated',
            'score': 0.85,
            'box': [0.1, 0.2, 0.4, 0.6]  # [ymin, xmin, ymax, xmax]
        },
        {
            'class_name': 'healthy',
            'score': 0.75,
            'box': [0.5, 0.1, 0.8, 0.4]
        },
        {
            'class_name': 'contaminated',
            'score': 0.65,
            'box': [0.6, 0.7, 0.9, 0.95]
        }
    ]
    
    image_path = "api/images_a_traiter/4d17b151-36ae-4e30-b01c-8998087d4497.jpeg"
    
    try:
        # Test génération heatmap
        heatmap = generator.create_contamination_heatmap(image_path, test_detections)
        print(f"✅ Heatmap générée: {heatmap.shape}")
        
        # Test overlay PIL
        overlay = generator.create_contamination_overlay_pil(image_path, test_detections)
        print(f"✅ Overlay PIL généré: {overlay.size}")
        
        # Test base64
        base64_str = generator.get_heatmap_base64(heatmap)
        print(f"✅ Base64 généré: {len(base64_str)} caractères")
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_heatmap_generator()
