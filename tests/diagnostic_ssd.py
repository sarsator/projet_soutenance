#!/usr/bin/env python3
"""
Script de diagnostic du modèle SSD pour comprendre pourquoi il ne détecte rien
"""
import sys
sys.path.append('/home/sarsator/projets/gaia_vision')

import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import json

def test_ssd_raw():
    """Test direct du modèle SSD pour voir ses sorties brutes"""
    
    # Charger le modèle
    model_path = Path("/home/sarsator/projets/gaia_vision/api/models/dl_model/versions/v1.3_20250716_132518/saved_model")
    
    print("🔍 DIAGNOSTIC DU MODÈLE SSD")

    print(f"Chemin du modèle: {model_path}")
    
    try:
        # Charger le SavedModel
        model = tf.saved_model.load(str(model_path))
        print("✅ Modèle chargé")
        
        # Charger une image de test
        image_path = "/home/sarsator/projets/gaia_vision/api/images_a_traiter/01640b85-29ca-4cb6-b8dc-26264ee26008.jpg"
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((320, 320))
        
        # Préprocesser
        img_array = np.array(img, dtype=np.uint8)
        img_tensor = tf.convert_to_tensor(img_array)
        img_tensor = tf.expand_dims(img_tensor, 0)
        
        print(f"Image préprocessée: shape={img_tensor.shape}, dtype={img_tensor.dtype}")
        
        # Prédiction
        infer = model.signatures['serving_default']
        predictions = infer(input_tensor=img_tensor)
        
        # Analyser toutes les sorties
        detection_boxes = predictions['detection_boxes'].numpy()[0]
        detection_classes = predictions['detection_classes'].numpy()[0]
        detection_scores = predictions['detection_scores'].numpy()[0]
        num_detections = int(predictions['num_detections'].numpy()[0])
        
        print(f"\n📊 RÉSULTATS BRUTS:")
        print(f"Nombre de détections: {num_detections}")
        print(f"Scores max: {np.max(detection_scores):.4f}")
        print(f"Scores min: {np.min(detection_scores):.4f}")
        print(f"Scores moyens: {np.mean(detection_scores):.4f}")
        
        # Analyser par seuils
        seuils = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        print(f"\n🎯 ANALYSE PAR SEUILS:")
        for seuil in seuils:
            detections_valides = sum(1 for score in detection_scores if score > seuil)
            print(f"Seuil {seuil}: {detections_valides} détections")
        
        # Détailler les 10 meilleures détections
        print(f"\n🏆 TOP 10 DÉTECTIONS:")
        indices_sorted = np.argsort(detection_scores)[::-1][:10]
        
        for i, idx in enumerate(indices_sorted):
            score = detection_scores[idx]
            class_id = int(detection_classes[idx])
            box = detection_boxes[idx]
            
            class_names = ["background", "healthy", "contaminated"]
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            print(f"  {i+1:2d}. Score: {score:.4f} | Classe: {class_name} ({class_id}) | Box: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
        
        # Test avec différentes classes
        print(f"\n🔢 RÉPARTITION PAR CLASSES:")
        for class_id in [0, 1, 2]:
            class_detections = sum(1 for cls in detection_classes if int(cls) == class_id)
            class_scores = [detection_scores[i] for i, cls in enumerate(detection_classes) if int(cls) == class_id]
            max_score = max(class_scores) if class_scores else 0
            
            class_names = ["background", "healthy", "contaminated"]
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            print(f"  Classe {class_id} ({class_name}): {class_detections} détections, score max: {max_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ssd_raw()
