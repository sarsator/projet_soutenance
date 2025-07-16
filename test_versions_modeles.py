#!/usr/bin/env python3
"""
Test rapide des versions de modèles affichées dans l'API
"""
import requests
import json
import sys

def test_model_versions():
    """Test simple pour vérifier les versions affichées"""
    try:
        print("🧪 Test des versions de modèles")
        print("=" * 40)
        
        # Préparer les données de test
        files = {'image': open('/home/sarsator/projets/gaia_vision/api/images_a_traiter/01640b85-29ca-4cb6-b8dc-26264ee26008.jpg', 'rb')}
        data = {
            'race_champignon': 'pleurote',
            'type_substrat': 'paille',
            'jours_inoculation': 10,
            'hygrometrie': 85.0,
            'co2_ppm': 800.0,
            'commentaire': 'Test versions automatique'
        }
        headers = {'Authorization': 'Bearer gaia-vision-test-key-2025'}
        
        print("🔄 Envoi de la requête...")
        response = requests.post('http://localhost:8000/predict-image', 
                               files=files, data=data, headers=headers, timeout=20)
        
        files['image'].close()  # Fermer le fichier
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prédiction réussie!")
            print(f"   Résultat: {result['prediction']}")
            print(f"   Confiance: {result['confidence']:.1f}%")
            
            # Afficher les versions
            print("\n📊 VERSIONS DES MODÈLES:")
            if 'details' in result and 'model_versions' in result['details']:
                versions = result['details']['model_versions']
                print(f"   CatBoost: {versions.get('catboost', 'N/A')}")
                print(f"   Vision: {versions.get('vision', 'N/A')}")
                
                # Vérifier si c'est v1.3
                vision_version = versions.get('vision', '')
                if 'v1.3' in vision_version:
                    print("✅ Nouveau modèle SSD v1.3 détecté!")
                else:
                    print(f"⚠️  Version vision attendue: v1.3, obtenue: {vision_version}")
            else:
                print("❌ Pas d'informations de version dans la réponse")
            
            # Détails du modèle vision
            if 'details' in result and 'vision_prediction' in result['details']:
                vision = result['details']['vision_prediction']
                print("\n🔍 DÉTAILS MODÈLE VISION:")
                print(f"   Type: {vision.get('model_type', 'N/A')}")
                
                if 'model_info' in vision:
                    info = vision['model_info']
                    print(f"   Architecture: {info.get('architecture', 'N/A')}")
                    print(f"   Version détaillée: {info.get('version', 'N/A')}")
                    
                    # Vérifier l'architecture
                    arch = info.get('architecture', '')
                    if 'SSD_MobileNet_V2' in arch:
                        print("✅ Architecture SSD MobileNet V2 confirmée!")
                    else:
                        print(f"⚠️  Architecture inattendue: {arch}")
            
            return True
            
        else:
            print(f"❌ Erreur API: {response.status_code}")
            print(f"Réponse: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = test_model_versions()
    sys.exit(0 if success else 1)
