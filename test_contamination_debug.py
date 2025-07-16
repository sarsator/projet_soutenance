#!/usr/bin/env python3
"""
Script de test pour analyser le problème de détection de contamination
"""

import sys
import os
import logging

# Ajouter le dossier api au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from models.vision_model import VisionModel

# Configuration du logging pour voir tous les détails
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_contamination_image():
    """Test avec l'image problématique"""
    
    # Chemin vers l'image avec contamination
    image_path = "api/images_a_traiter/4d17b151-36ae-4e30-b01c-8998087d4497.jpeg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    print(f"🔍 Test avec l'image: {image_path}")
    print("=" * 60)
    
    # Initialiser le modèle
    try:
        model = VisionModel()
        print("✅ VisionModel initialisé")
        
        # Charger le modèle
        if not model.load_model():
            print("❌ Impossible de charger le modèle")
            return
        
        print(f"✅ Modèle chargé: {model.model_type}")
        print(f"📐 Taille d'entrée: {model.input_size}")
        print(f"🏷️ Classes: {model.class_names}")
        
        # Effectuer la prédiction
        print("\n🚀 Lancement de la prédiction...")
        result = model.predict(image_path)
        
        # Afficher les résultats détaillés
        print("\n📊 RÉSULTATS DÉTAILLÉS:")
        print("=" * 60)
        print(f"🏆 Prédiction finale: {result['prediction']}")
        print(f"📈 Confiance: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
        print(f"☠️ Probabilité contamination: {result.get('contamination_probability', 'N/A'):.3f}")
        
        if 'detection_summary' in result:
            summary = result['detection_summary']
            print(f"\n🔍 RÉSUMÉ DES DÉTECTIONS:")
            print(f"  📊 Total détections: {summary['total_detections']}")
            print(f"  ☠️ Objets contaminés: {summary['contaminated_count']}")
            print(f"  ✅ Objets sains: {summary['healthy_count']}")
            print(f"  📈 Meilleur score contaminé: {summary['max_contaminated_score']:.3f}")
            print(f"  📈 Meilleur score sain: {summary['max_healthy_score']:.3f}")
            
        if 'detections' in result:
            print(f"\n🎯 DÉTECTIONS INDIVIDUELLES ({len(result['detections'])}):")
            for i, detection in enumerate(result['detections']):
                print(f"  {i+1}. {detection['class_name']} - Score: {detection['score']:.3f}")
                print(f"     Box: {[f'{x:.3f}' for x in detection['box']]}")
        
        # Diagnostic du problème
        print("\n🩺 DIAGNOSTIC:")
        print("=" * 60)
        
        if result['prediction'] == 'sain' and result.get('contamination_probability', 0) < 0.5:
            if result.get('detection_summary', {}).get('healthy_count', 0) > 0:
                max_healthy = result.get('detection_summary', {}).get('max_healthy_score', 0)
                if max_healthy < 0.7:
                    print("⚠️ PROBLÈME DÉTECTÉ: Objets classés 'sains' mais avec des scores modérés")
                    print(f"   Le meilleur score 'sain' est de {max_healthy:.3f} (< 0.7)")
                    print("   Cela pourrait indiquer une contamination non détectée")
                else:
                    print("✅ Classification correcte: objets vraiment sains avec bons scores")
            else:
                print("⚠️ Aucun objet détecté - difficile de juger la qualité")
        elif result['prediction'] == 'contamine':
            print("✅ Contamination détectée correctement")
        else:
            print("❓ Prédiction incertaine - nécessite investigation")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_contamination_image()
