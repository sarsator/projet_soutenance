#!/usr/bin/env python3
"""
Test de génération de heatmap avec de vraies détections de contamination
"""

import sys
import os
import logging

# Ajouter le dossier api au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from models.vision_model import VisionModel
from utils.heatmap_generator import ContaminationHeatmapGenerator

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_heatmap_with_real_detections():
    """Test avec de vraies détections de contamination"""
    
    # Chemin vers l'image avec contamination
    image_path = "api/images_a_traiter/4d17b151-36ae-4e30-b01c-8998087d4497.jpeg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    print(f"🔍 Test heatmap avec l'image: {image_path}")
    print("=" * 60)
    
    try:
        # 1. Obtenir les détections réelles avec le modèle
        model = VisionModel()
        if not model.load_model():
            print("❌ Impossible de charger le modèle")
            return
        
        print("✅ Modèle chargé, obtention des détections...")
        result = model.predict(image_path)
        
        print(f"🎯 Détections obtenues: {len(result.get('detections', []))}")
        
        # Afficher les détections de contamination
        contaminated_detections = [d for d in result.get('detections', []) if d.get('class_name') == 'contaminated']
        print(f"☠️ Contaminations détectées: {len(contaminated_detections)}")
        
        for i, det in enumerate(contaminated_detections):
            print(f"  {i+1}. Score: {det['score']:.3f}, Box: {[f'{x:.3f}' for x in det['box']]}")
        
        if not contaminated_detections:
            print("⚠️ Aucune contamination détectée, impossible de générer une heatmap")
            return
        
        # 2. Générer la heatmap
        print("\n🔥 Génération de la heatmap...")
        generator = ContaminationHeatmapGenerator()
        
        # Test avec OpenCV heatmap
        heatmap_cv = generator.create_contamination_heatmap(image_path, result['detections'])
        generator.save_heatmap_image(heatmap_cv, "heatmap_opencv.png")
        print("✅ Heatmap OpenCV sauvegardée: heatmap_opencv.png")
        
        # Test avec PIL overlay
        heatmap_pil = generator.create_contamination_overlay_pil(image_path, result['detections'])
        heatmap_pil.save("heatmap_pil.png")
        print("✅ Overlay PIL sauvegardé: heatmap_pil.png")
        
        # Test base64 pour le web
        base64_str = generator.get_heatmap_base64(heatmap_cv)
        print(f"✅ Base64 généré: {len(base64_str)} caractères")
        
        # Créer un fichier HTML de test pour visualiser
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Heatmap Contamination</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .image-container {{ margin: 20px 0; }}
        .image-container img {{ max-width: 100%; height: auto; border: 2px solid #ccc; }}
        .image-container h3 {{ margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>🔥 Test Heatmap de Contamination</h1>
    
    <div class="image-container">
        <h3>📊 Résultats de détection:</h3>
        <ul>
            <li>Total détections: {result.get('detection_summary', {}).get('total_detections', 0)}</li>
            <li>Contaminations: {result.get('detection_summary', {}).get('contaminated_count', 0)}</li>
            <li>Objets sains: {result.get('detection_summary', {}).get('healthy_count', 0)}</li>
            <li>Prédiction finale: <strong>{result.get('prediction', 'N/A')}</strong></li>
            <li>Confiance: <strong>{result.get('confidence', 0)*100:.1f}%</strong></li>
        </ul>
    </div>
    
    <div class="image-container">
        <h3>🔥 Heatmap OpenCV (zones de contamination en rouge chaud):</h3>
        <img src="{base64_str}" alt="Heatmap OpenCV">
    </div>
    
    <div class="image-container">
        <h3>🎯 Overlay PIL (rectangles de contamination):</h3>
        <img src="heatmap_pil.png" alt="Overlay PIL">
    </div>
    
    <div class="image-container">
        <h3>📋 Détections de contamination détaillées:</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Index</th>
                <th>Classe</th>
                <th>Score</th>
                <th>Position (ymin, xmin, ymax, xmax)</th>
            </tr>
        """
        
        for i, det in enumerate(contaminated_detections):
            html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td>{det['class_name']}</td>
                <td>{det['score']:.3f}</td>
                <td>{', '.join([f'{x:.3f}' for x in det['box']])}</td>
            </tr>
            """
        
        html_content += """
        </table>
    </div>
</body>
</html>
        """
        
        with open("test_heatmap.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("✅ Fichier de test HTML créé: test_heatmap.html")
        print("\n🌐 Ouvrez test_heatmap.html dans un navigateur pour voir les résultats !")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_heatmap_with_real_detections()
