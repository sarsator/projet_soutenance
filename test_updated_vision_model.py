#!/usr/bin/env python3
"""
Script de test pour valider le nouveau VisionModel qui supporte les SavedModel
"""

import sys
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ajouter le dossier API au path
sys.path.append('/home/sarsator/projets/gaia_vision/api')

from models.vision_model import VisionModel

def test_model_loading():
    """Teste le chargement automatique du modèle"""
    print("🧪 Test 1: Chargement automatique du modèle")
    print("=" * 50)
    
    # Initialiser le modèle (devrait automatiquement trouver la version la plus récente)
    model = VisionModel()
    
    print(f"📁 Chemin du modèle détecté: {model.model_path}")
    print(f"📏 Taille d'entrée configurée: {model.input_size}")
    print(f"🏷️  Classes configurées: {model.class_names}")
    
    # Tenter de charger le modèle
    success = model.load_model()
    
    if success:
        print(f"✅ Modèle chargé avec succès!")
        print(f"🔧 Type de modèle: {model.model_type}")
        
        if model.metadata:
            print(f"📋 Architecture: {model.metadata.get('architecture', 'Unknown')}")
            print(f"📊 Performance F1: {model.metadata.get('performance_metrics', {}).get('f1_score', 'Unknown')}")
        
        return True
    else:
        print("❌ Échec du chargement du modèle")
        return False

def test_model_prediction():
    """Teste une prédiction sur une image"""
    print("\n🧪 Test 2: Prédiction sur une image")
    print("=" * 50)
    
    # Chercher une image de test
    test_images_paths = [
        "/home/sarsator/projets/gaia_vision/api/images_a_traiter",
        "/home/sarsator/projets/gaia_vision/images_a_traiter"
    ]
    
    test_image = None
    for path in test_images_paths:
        test_dir = Path(path)
        if test_dir.exists():
            # Prendre la première image trouvée
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images = list(test_dir.glob(ext))
                if images:
                    test_image = images[0]
                    break
            if test_image:
                break
    
    if not test_image:
        print("⚠️  Aucune image de test trouvée")
        return False
    
    print(f"🖼️  Image de test: {test_image.name}")
    
    # Initialiser et charger le modèle
    model = VisionModel()
    if not model.load_model():
        print("❌ Impossible de charger le modèle pour le test")
        return False
    
    try:
        # Effectuer la prédiction
        print("🔍 Prédiction en cours...")
        result = model.predict(str(test_image))
        
        print(f"📊 Résultat de la prédiction:")
        print(f"   └── Prédiction: {result['prediction']}")
        print(f"   └── Confiance: {result['confidence']:.3f}")
        print(f"   └── Type de modèle: {result['model_type']}")
        
        if 'detection_summary' in result:
            # Pour SSD
            summary = result['detection_summary']
            print(f"   └── Détections totales: {summary['total_detections']}")
            print(f"   └── Contaminés: {summary['contaminated_count']}")
            print(f"   └── Sains: {summary['healthy_count']}")
        
        if 'contamination_probability' in result:
            print(f"   └── Probabilité contamination: {result['contamination_probability']:.3f}")
        
        print("✅ Prédiction réussie!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🚀 TEST DU VISIONMODEL MIS À JOUR")
    print("=" * 60)
    
    # Test 1: Chargement du modèle
    loading_success = test_model_loading()
    
    if loading_success:
        # Test 2: Prédiction
        prediction_success = test_model_prediction()
        
        if prediction_success:
            print("\n🎉 TOUS LES TESTS RÉUSSIS!")
            print("Le VisionModel est prêt à utiliser le nouveau SavedModel SSD!")
        else:
            print("\n⚠️  Chargement OK mais prédiction échouée")
    else:
        print("\n❌ ÉCHEC DU CHARGEMENT")
        print("Vérifiez que le modèle a bien été déployé")

if __name__ == "__main__":
    main()
