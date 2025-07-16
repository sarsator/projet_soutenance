#!/usr/bin/env python3
"""
Test du nouveau modèle SSD MobileNet V2 déployé
"""
import requests
import json
import time
import sys
from pathlib import Path

def test_api_health():
    """Test du statut de l'API"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("✅ API en ligne")
            print(f"   - Modèles chargés: {health['models']}")
            return True
        else:
            print(f"❌ API indisponible (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion API: {e}")
        return False

def test_predict_with_ssd(image_path):
    """Test de prédiction avec le nouveau modèle SSD"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'race_champignon': 'pleurote',
                'type_substrat': 'paille',
                'jours_inoculation': 10,
                'hygrometrie': 85.0,
                'co2_ppm': 800.0,
                'commentaire': 'Test automatique du nouveau modèle SSD'
            }
            headers = {
                'Authorization': 'Bearer gaia_vision_api_key'
            }
            
            print(f"🔄 Test de prédiction avec {image_path.name}...")
            response = requests.post('http://localhost:8000/predict-image', 
                                   files=files, 
                                   data=data, 
                                   headers=headers,
                                   timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Prédiction réussie:")
                print(f"   - Résultat: {result['prediction']}")
                print(f"   - Confiance: {result['confidence']:.1f}%")
                print(f"   - Source: {result.get('confidence_source', 'N/A')}")
                print(f"   - Nb sacs détectés: {result.get('multi_sac_count', 'N/A')}")
                
                if 'details' in result:
                    models_used = result['details'].get('models_used', [])
                    print(f"   - Modèles utilisés: {', '.join(models_used)}")
                    
                    # Vérification que le nouveau modèle SSD est bien utilisé
                    if 'vision' in models_used:
                        vision_pred = result['details'].get('vision_prediction', {})
                        if 'model_type' in vision_pred and 'SSD' in vision_pred['model_type']:
                            print("✅ Nouveau modèle SSD MobileNet V2 confirmé")
                        else:
                            print("⚠️  Modèle vision utilisé mais type non confirmé")
                
                return True
            else:
                print(f"❌ Erreur de prédiction (status: {response.status_code})")
                print(f"   Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def main():
    """Test principal"""
    print("🧪 TEST DU NOUVEAU MODÈLE SSD MOBILENET V2")
    print("=" * 50)
    
    # 1. Test de santé de l'API
    print("\n1. Test de santé de l'API...")
    if not test_api_health():
        print("\n❌ L'API n'est pas disponible. Assurez-vous qu'elle soit démarrée.")
        sys.exit(1)
    
    # 2. Test avec plusieurs images
    print("\n2. Test de prédictions...")
    images_dir = Path("/home/sarsator/projets/gaia_vision/api/images_a_traiter")
    
    if not images_dir.exists():
        print(f"❌ Dossier d'images non trouvé: {images_dir}")
        sys.exit(1)
    
    # Prendre quelques images pour les tests
    image_files = list(images_dir.glob("*.jpg"))[:3]  # Premières 3 images
    
    if not image_files:
        print("❌ Aucune image trouvée pour les tests")
        sys.exit(1)
    
    success_count = 0
    for img_path in image_files:
        if test_predict_with_ssd(img_path):
            success_count += 1
        time.sleep(1)  # Petite pause entre les tests
    
    # 3. Résumé
    print(f"\n📊 RÉSUMÉ DES TESTS")
    print("=" * 30)
    print(f"Tests réussis: {success_count}/{len(image_files)}")
    
    if success_count == len(image_files):
        print("🎉 Tous les tests sont passés ! Le nouveau modèle SSD est opérationnel.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les logs.")
    
    return success_count == len(image_files)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
