#!/usr/bin/env python3
"""
Script de test pour valider les améliorations du modèle SSD
"""
import requests
import json
import sys
import time
from pathlib import Path

def test_prediction(image_name, expected_result=None, description=""):
    """Test une prédiction et affiche les détails"""
    try:
        print(f"\n🧪 Test: {description}")
        print(f"Image: {image_name}")
        print("-" * 50)
        
        with open(f'/home/sarsator/projets/gaia_vision/api/images_a_traiter/{image_name}', 'rb') as f:
            files = {'image': f}
            data = {
                'race_champignon': 'pleurote',
                'type_substrat': 'paille',
                'jours_inoculation': 10,
                'hygrometrie': 85.0,
                'co2_ppm': 800.0,
                'commentaire': f'Test automatique - {description}'
            }
            headers = {'Authorization': 'Bearer gaia-vision-test-key-2025'}
            
            response = requests.post('http://localhost:8000/predict-image', 
                                   files=files, data=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Résumé de la prédiction
                print(f"✅ Prédiction finale: {result['prediction']}")
                print(f"   Confiance globale: {result['confidence']:.3f}")
                print(f"   Source confiance: {result['confidence_source']}")
                
                # Détails Vision
                if 'details' in result and 'vision_prediction' in result['details']:
                    vision = result['details']['vision_prediction']
                    print(f"\n👁️  Vision SSD:")
                    print(f"   Prédiction: {vision['prediction']}")
                    print(f"   Confiance: {vision['confidence']:.3f}")
                    
                    summary = vision['detection_summary']
                    print(f"   Détections: {summary['total_detections']} total")
                    print(f"   Contaminés: {summary['contaminated_count']} (max score: {summary['max_contaminated_score']:.3f})")
                    print(f"   Sains: {summary['healthy_count']} (max score: {summary['max_healthy_score']:.3f})")
                
                # Détails CatBoost
                if 'details' in result and 'catboost_prediction' in result['details']:
                    catboost = result['details']['catboost_prediction']
                    print(f"\n🎯 CatBoost:")
                    print(f"   Prédiction: {catboost['prediction_label']}")
                    print(f"   Risque: {catboost['risk_level']}")
                    print(f"   Confiance: {catboost['confidence']:.3f}")
                
                # Validation si résultat attendu fourni
                if expected_result:
                    if result['prediction'] == expected_result:
                        print(f"✅ CORRECT - Attendu: {expected_result}, Obtenu: {result['prediction']}")
                        return True
                    else:
                        print(f"❌ INCORRECT - Attendu: {expected_result}, Obtenu: {result['prediction']}")
                        return False
                else:
                    return True
                    
            else:
                print(f"❌ Erreur API: {response.status_code}")
                print(f"Réponse: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Tests principaux"""
    print("🔬 TESTS DES AMÉLIORATIONS DU MODÈLE SSD")
    print("=" * 60)
    
    # Vérifier que l'API est disponible
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code != 200:
            print("❌ API non disponible")
            return False
    except:
        print("❌ Impossible de contacter l'API")
        return False
    
    print("✅ API disponible")
    
    # Liste des images disponibles
    images_dir = Path("/home/sarsator/projets/gaia_vision/api/images_a_traiter")
    images = list(images_dir.glob("*.jpg"))[:5]  # Prendre 5 images pour les tests
    
    if not images:
        print("❌ Aucune image trouvée pour les tests")
        return False
    
    print(f"📁 {len(images)} images trouvées pour les tests")
    
    success_count = 0
    total_tests = len(images)
    
    for i, img_path in enumerate(images):
        success = test_prediction(
            img_path.name, 
            description=f"Image {i+1}/{total_tests} - Détection générale"
        )
        if success:
            success_count += 1
        
        time.sleep(2)  # Pause entre les tests
    
    # Résumé final
    print(f"\n📊 RÉSUMÉ DES TESTS")
    print("=" * 40)
    print(f"Tests réussis: {success_count}/{total_tests}")
    print(f"Taux de succès: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 Tous les tests sont passés!")
    elif success_count >= total_tests * 0.8:
        print("👍 La plupart des tests sont passés")
    else:
        print("⚠️  Plusieurs tests ont échoué")
    
    print("\n📝 Les améliorations apportées:")
    print("- Seuil de détection abaissé à 0.3")
    print("- Logique de cas mixtes (contaminé + sain)")
    print("- Gestion de l'incertitude avec scores modérés")
    print("- Principe de précaution en cas de doute")
    
    return success_count >= total_tests * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
