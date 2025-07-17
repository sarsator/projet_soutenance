#!/usr/bin/env python3
"""
Script principal pour le système de versioning des modèles
Permet de déployer, lister, rollback avec logging complet
"""
import sys
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Ajouter le dossier racine au path
sys.path.append('/home/sarsator/projets/gaia_vision')

from api.models.model_version_manager import ModelVersionManager

class ModelVersioningCLI:
    """Interface en ligne de commande pour le système de versioning"""
    
    def __init__(self):
        """Initialise le CLI avec logging"""
        self.setup_logging()
        self.models_dir = Path("/home/sarsator/projets/gaia_vision/api/models")
        self.manager = ModelVersionManager(self.models_dir)
        
        self.logger.info("🚀 Système de versioning initialisé")
        self.logger.info(f"📁 Dossier modèles: {self.models_dir}")
    
    def setup_logging(self):
        """Configure le système de logging"""
        # Créer le dossier de logs
        log_dir = Path("/home/sarsator/projets/gaia_vision/logs")
        log_dir.mkdir(exist_ok=True)
        
        # Nom du fichier de log avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"versioning_{timestamp}.log"
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("ModelVersioning")
        self.logger.info(f"📝 Logging configuré - Fichier: {log_file}")
    
    def deployer_modele(self, model_type: str, source_path: str = None, metadata: dict = None):
        """Déploie un nouveau modèle"""
        self.logger.info(f"🚀 DÉPLOIEMENT MODÈLE {model_type.upper()}")
        self.logger.info("=" * 50)
        
        try:
            # Déterminer le chemin source automatiquement si non fourni
            if not source_path:
                if model_type == "dl":
                    # Utiliser la dernière version SSD
                    current_path = self.manager.obtenir_chemin_modele_actuel("dl")
                    if current_path and current_path.exists():
                        if current_path.name == "saved_model":
                            source_path = str(current_path)
                        else:
                            # Chercher le dossier SavedModel dans le même dossier
                            parent_dir = current_path.parent
                            savedmodel_path = parent_dir / "saved_model"
                            if savedmodel_path.exists():
                                source_path = str(savedmodel_path)
                            else:
                                source_path = str(current_path)
                    else:
                        self.logger.error("❌ Aucun modèle SSD trouvé")
                        return False
                
                elif model_type == "ml":
                    # Utiliser la dernière version CatBoost
                    current_path = self.manager.obtenir_chemin_modele_actuel("ml")
                    if current_path and current_path.exists():
                        source_path = str(current_path)
                    else:
                        self.logger.error("❌ Aucun modèle CatBoost trouvé")
                        return False
            
            self.logger.info(f"📁 Source: {source_path}")
            
            # Métadonnées par défaut
            if not metadata:
                metadata = self._get_default_metadata(model_type)
            
            # Afficher la prochaine version calculée
            version_info = self.manager._generate_version_info(model_type, metadata)
            self.logger.info(f"🔢 Prochaine version: v{version_info['version']}")
            
            # Déploiement
            result = self.manager.deployer_modele(source_path, model_type, metadata)
            
            if result["success"]:
                version_info = result["version_info"]
                self.logger.info(f"✅ DÉPLOIEMENT RÉUSSI!")
                self.logger.info(f"   Version: v{version_info['version']}")
                self.logger.info(f"   Taille: {version_info['model_size_mb']} MB")
                self.logger.info(f"   Format: {version_info['model_format']}")
                self.logger.info(f"   Chemin: {result['current_path']}")
                self.logger.info(f"   ID: {result['version_id']}")
                return True
            else:
                self.logger.error(f"❌ ÉCHEC DU DÉPLOIEMENT: {result['error']}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ ERREUR CRITIQUE: {e}")
            return False
    
    def lister_versions(self, model_type: str = None):
        """Liste les versions disponibles"""
        self.logger.info("📚 LISTE DES VERSIONS")
        self.logger.info("=" * 30)
        
        model_types = [model_type] if model_type else ["dl", "ml"]
        
        for mtype in model_types:
            self.logger.info(f"\n🎯 MODÈLES {mtype.upper()}:")
            self.logger.info("-" * 20)
            
            versions = self.manager.lister_versions(mtype)
            
            if not versions:
                self.logger.info("   Aucune version trouvée")
                continue
            
            for i, version in enumerate(versions, 1):
                status = "🟢 ACTUEL" if self._is_current_version(mtype, version['version']) else "⚪"
                
                self.logger.info(f"   {i}. {status} v{version['version']}")
                self.logger.info(f"      📅 Déployé: {version.get('deployed_at', 'Date inconnue')}")
                self.logger.info(f"      📊 Taille: {version.get('model_size_mb', 0)} MB")
                self.logger.info(f"      🔗 Format: {version.get('model_format', 'Inconnu')}")
                
                if 'architecture' in version:
                    self.logger.info(f"      🏗️  Architecture: {version['architecture']}")
                if 'algorithm' in version:
                    self.logger.info(f"      ⚙️  Algorithme: {version['algorithm']}")
                if 'deployment_reason' in version:
                    self.logger.info(f"      📝 Raison: {version['deployment_reason']}")
                
                self.logger.info("")
    
    def rollback(self, model_type: str, version_id: str):
        """Effectue un rollback vers une version spécifique"""
        self.logger.info(f"🔄 ROLLBACK MODÈLE {model_type.upper()}")
        self.logger.info("=" * 40)
        
        # Version actuelle
        current_version = self.manager.obtenir_version_actuelle(f"{model_type}_model")
        self.logger.info(f"📍 Version actuelle: v{current_version}")
        self.logger.info(f"🎯 Version cible: {version_id}")
        
        result = self.manager.rollback_to_version(model_type, version_id)
        
        if result["success"]:
            self.logger.info("✅ ROLLBACK RÉUSSI!")
            self.logger.info(f"   Message: {result['message']}")
            self.logger.info(f"   Nouveau chemin: {result['current_path']}")
            
            # Vérifier la nouvelle version
            new_version = self.manager.get_current_version(f"{model_type}_model")
            self.logger.info(f"   Version confirmée: v{new_version}")
            return True
        else:
            self.logger.error(f"❌ ÉCHEC DU ROLLBACK: {result['error']}")
            return False
    
    def status(self):
        """Affiche le statut complet du système"""
        self.logger.info("📊 STATUT DU SYSTÈME DE VERSIONING")
        self.logger.info("=" * 50)
        
        # Statut général
        self.logger.info("\n🎯 VERSIONS ACTUELLES:")
        ssd_version = self.manager.get_current_version("dl_model")
        catboost_version = self.manager.get_current_version("ml_model")
        
        self.logger.info(f"   SSD MobileNet V2: v{ssd_version}")
        self.logger.info(f"   CatBoost: v{catboost_version}")
        
        # Chemins actuels
        self.logger.info("\n📁 CHEMINS ACTUELS:")
        ssd_path = self.manager.get_current_model_path("dl")
        catboost_path = self.manager.get_current_model_path("ml")
        
        self.logger.info(f"   SSD: {ssd_path}")
        self.logger.info(f"   CatBoost: {catboost_path}")
        
        # Validation
        self.logger.info("\n✅ VALIDATION:")
        ssd_valid = ssd_path.exists() if ssd_path else False
        catboost_valid = catboost_path.exists() if catboost_path else False
        
        self.logger.info(f"   SSD valide: {'✅' if ssd_valid else '❌'}")
        self.logger.info(f"   CatBoost valide: {'✅' if catboost_valid else '❌'}")
        
        # Statistiques
        ssd_versions = self.manager.list_versions("dl")
        ml_versions = self.manager.list_versions("ml")
        
        self.logger.info("\n📈 STATISTIQUES:")
        self.logger.info(f"   Total versions SSD: {len(ssd_versions)}")
        self.logger.info(f"   Total versions CatBoost: {len(ml_versions)}")
        
        # Espace disque
        total_ssd_size = sum(v.get('model_size_mb', 0) for v in ssd_versions)
        total_ml_size = sum(v.get('model_size_mb', 0) for v in ml_versions)
        
        self.logger.info(f"   Espace SSD: {total_ssd_size:.2f} MB")
        self.logger.info(f"   Espace CatBoost: {total_ml_size:.2f} MB")
        self.logger.info(f"   Espace total: {total_ssd_size + total_ml_size:.2f} MB")
    
    def cleanup(self, model_type: str, keep_count: int = 3):
        """Nettoie les anciennes versions"""
        self.logger.info(f"🧹 NETTOYAGE MODÈLE {model_type.upper()}")
        self.logger.info("=" * 40)
        
        versions_before = self.manager.list_versions(model_type)
        self.logger.info(f"📊 Versions avant nettoyage: {len(versions_before)}")
        
        if len(versions_before) <= keep_count:
            self.logger.info(f"✅ Pas besoin de nettoyage (≤ {keep_count} versions)")
            return
        
        self.logger.info(f"🗑️  Suppression des versions anciennes (garde {keep_count})")
        
        try:
            self.manager.cleanup_old_versions(model_type, keep_count)
            
            versions_after = self.manager.list_versions(model_type)
            deleted_count = len(versions_before) - len(versions_after)
            
            self.logger.info(f"✅ NETTOYAGE TERMINÉ!")
            self.logger.info(f"   Versions supprimées: {deleted_count}")
            self.logger.info(f"   Versions restantes: {len(versions_after)}")
            
        except Exception as e:
            self.logger.error(f"❌ ERREUR DE NETTOYAGE: {e}")
    
    def _get_default_metadata(self, model_type: str) -> dict:
        """Génère des métadonnées par défaut"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if model_type == "dl":
            return {
                "architecture": "SSD MobileNet V2",
                "input_size": "320x320",
                "classes": ["background", "healthy", "contaminated"],
                "detection_threshold": 0.12,
                "framework": "TensorFlow Object Detection API",
                "deployment_reason": f"Déploiement automatique via CLI - {timestamp}",
                "cli_deployment": True
            }
        else:
            return {
                "algorithm": "CatBoost Classifier",
                "features": ["race_champignon", "type_substrat", "jours_inoculation", "hygrometrie", "co2_ppm"],
                "framework": "CatBoost",
                "deployment_reason": f"Déploiement automatique via CLI - {timestamp}",
                "cli_deployment": True
            }
    
    def _is_current_version(self, model_type: str, version: str) -> bool:
        """Vérifie si une version est actuellement déployée"""
        current = self.manager.get_current_version(f"{model_type}_model")
        return current == version

def main():
    """Fonction principale avec interface CLI"""
    parser = argparse.ArgumentParser(
        description="Système de versioning des modèles Gaia Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s deploy dl                    # Déploie une nouvelle version SSD
  %(prog)s deploy ml                    # Déploie une nouvelle version CatBoost
  %(prog)s list                         # Liste toutes les versions
  %(prog)s list dl                      # Liste les versions SSD uniquement
  %(prog)s status                       # Affiche le statut du système
  %(prog)s rollback dl v1.3_20250716_132518  # Rollback SSD vers version
  %(prog)s cleanup dl --keep 3          # Garde seulement les 3 dernières versions SSD
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande deploy
    deploy_parser = subparsers.add_parser('deploy', help='Déploie une nouvelle version')
    deploy_parser.add_argument('model_type', choices=['dl', 'ml'], help='Type de modèle')
    deploy_parser.add_argument('--source', help='Chemin source (optionnel)')
    
    # Commande list
    list_parser = subparsers.add_parser('list', help='Liste les versions')
    list_parser.add_argument('model_type', nargs='?', choices=['dl', 'ml'], help='Type de modèle (optionnel)')
    
    # Commande status
    subparsers.add_parser('status', help='Affiche le statut du système')
    
    # Commande rollback
    rollback_parser = subparsers.add_parser('rollback', help='Rollback vers une version')
    rollback_parser.add_argument('model_type', choices=['dl', 'ml'], help='Type de modèle')
    rollback_parser.add_argument('version_id', help='ID de la version (ex: v1.3_20250716_132518)')
    
    # Commande cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Nettoie les anciennes versions')
    cleanup_parser.add_argument('model_type', choices=['dl', 'ml'], help='Type de modèle')
    cleanup_parser.add_argument('--keep', type=int, default=3, help='Nombre de versions à garder')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialiser le CLI
    cli = ModelVersioningCLI()
    
    try:
        # Exécuter la commande
        if args.command == 'deploy':
            success = cli.deploy_model(args.model_type, args.source)
            sys.exit(0 if success else 1)
            
        elif args.command == 'list':
            cli.list_versions(args.model_type)
            
        elif args.command == 'status':
            cli.status()
            
        elif args.command == 'rollback':
            success = cli.rollback(args.model_type, args.version_id)
            sys.exit(0 if success else 1)
            
        elif args.command == 'cleanup':
            cli.cleanup(args.model_type, args.keep)
            
    except KeyboardInterrupt:
        cli.logger.info("\n🛑 Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        cli.logger.error(f"❌ ERREUR FATALE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
