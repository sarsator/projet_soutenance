"""
Gestionnaire de versions pour les modèles ML/DL
Gère le versioning, les métadonnées et les déploiements
"""
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class ModelVersionManager:
    """Gestionnaire de versions pour les modèles"""
    
    def __init__(self, base_models_dir: Union[str, Path]):
        """
        Initialise le gestionnaire de versions
        
        Args:
            base_models_dir: Dossier racine des modèles (api/models/)
        """
        self.base_dir = Path(base_models_dir)
        self.ml_model_dir = self.base_dir / "ml_model"
        self.dl_model_dir = self.base_dir / "dl_model"
        
        # Créer la structure de dossiers
        self._ensure_directory_structure()
    
    def _ensure_directory_structure(self):
        """Crée la structure de dossiers nécessaire"""
        for model_dir in [self.ml_model_dir, self.dl_model_dir]:
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "versions").mkdir(exist_ok=True)
    
    def _generate_version_info(self, model_type: str, metadata: Dict = None) -> Dict:
        """
        Génère les informations de version
        
        Args:
            model_type: Type de modèle ('ml' ou 'dl')
            metadata: Métadonnées additionnelles
            
        Returns:
            Dictionnaire avec les infos de version
        """
        timestamp = datetime.now()
        
        # Obtenir la prochaine version
        current_version = self.get_latest_version(model_type)
        if current_version:
            major, minor = current_version['version'].split('.')
            next_version = f"{major}.{int(minor) + 1}"
        else:
            next_version = "1.0"
        
        version_info = {
            "version": next_version,
            "timestamp": timestamp.isoformat(),
            "model_type": model_type,
            "deployed_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "deployment_id": timestamp.strftime("%Y%m%d_%H%M%S")
        }
        
        # Ajouter les métadonnées si fournies
        if metadata:
            version_info.update(metadata)
        
        return version_info
    
    def deploy_model(self, 
                    source_path: Union[str, Path], 
                    model_type: str,
                    metadata: Dict = None) -> Dict:
        """
        Déploie un nouveau modèle avec versioning
        
        Args:
            source_path: Chemin vers le modèle source
            model_type: Type de modèle ('ml' ou 'dl')
            metadata: Métadonnées du modèle (accuracy, loss, etc.)
            
        Returns:
            Informations sur le déploiement
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Modèle source non trouvé : {source_path}")
            
            # Générer les infos de version
            version_info = self._generate_version_info(model_type, metadata)
            version_id = f"v{version_info['version']}_{version_info['deployment_id']}"
            
            # Définir les chemins
            if model_type == "ml":
                model_dir = self.ml_model_dir
                filename = "model_catboost_best.joblib"
            elif model_type == "dl":
                model_dir = self.dl_model_dir
                filename = "final_model.keras"
            else:
                raise ValueError(f"Type de modèle invalide : {model_type}")
            
            # Créer le dossier de version
            version_dir = model_dir / "versions" / version_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copier le modèle vers le dossier de version
            dest_model_path = version_dir / filename
            shutil.copy2(source_path, dest_model_path)
            
            # Ajouter les infos sur le fichier
            version_info["model_file"] = filename
            version_info["model_size_bytes"] = dest_model_path.stat().st_size
            version_info["model_size_mb"] = round(version_info["model_size_bytes"] / (1024 * 1024), 2)
            version_info["source_path"] = str(source_path)
            version_info["deployed_path"] = str(dest_model_path)
            
            # Sauvegarder les métadonnées
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(version_info, f, indent=2, ensure_ascii=False)
            
            # Créer/mettre à jour le lien symbolique vers 'current'
            current_link = model_dir / "current"
            if current_link.exists() or current_link.is_symlink():
                current_link.unlink()
            
            # Créer un nouveau lien symbolique
            current_link.symlink_to(dest_model_path.relative_to(model_dir))
            
            # Mettre à jour l'historique des déploiements
            self._update_deployment_history(model_type, version_info)
            
            logger.info(f"Modèle {model_type} déployé avec succès : version {version_info['version']}")
            
            return {
                "success": True,
                "version_info": version_info,
                "version_id": version_id,
                "current_path": str(current_link)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du déploiement du modèle {model_type} : {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_deployment_history(self, model_type: str, version_info: Dict):
        """Met à jour l'historique des déploiements"""
        if model_type == "ml":
            history_file = self.ml_model_dir / "deployment_history.json"
        else:
            history_file = self.dl_model_dir / "deployment_history.json"
        
        # Charger l'historique existant
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception:
                history = []
        
        # Ajouter la nouvelle version
        history.append(version_info)
        
        # Garder seulement les 10 dernières versions dans l'historique
        history = history[-10:]
        
        # Sauvegarder l'historique mis à jour
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def get_deployment_history(self, model_type: str) -> List[Dict]:
        """Récupère l'historique des déploiements"""
        if model_type == "ml":
            history_file = self.ml_model_dir / "deployment_history.json"
        else:
            history_file = self.dl_model_dir / "deployment_history.json"
        
        if not history_file.exists():
            return []
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def get_latest_version(self, model_type: str) -> Optional[Dict]:
        """Récupère les infos de la dernière version"""
        history = self.get_deployment_history(model_type)
        return history[-1] if history else None
    
    def list_versions(self, model_type: str) -> List[Dict]:
        """Liste toutes les versions disponibles"""
        if model_type == "ml":
            versions_dir = self.ml_model_dir / "versions"
        else:
            versions_dir = self.dl_model_dir / "versions"
        
        if not versions_dir.exists():
            return []
        
        versions = []
        for version_dir in sorted(versions_dir.iterdir()):
            if version_dir.is_dir():
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            versions.append(metadata)
                    except Exception:
                        continue
        
        return versions
    
    def rollback_to_version(self, model_type: str, version_id: str) -> Dict:
        """
        Effectue un rollback vers une version spécifique
        
        Args:
            model_type: Type de modèle ('ml' ou 'dl')
            version_id: ID de la version (ex: v1.0_20250711_142530)
            
        Returns:
            Résultat du rollback
        """
        try:
            if model_type == "ml":
                model_dir = self.ml_model_dir
                filename = "model_catboost_best.joblib"
            else:
                model_dir = self.dl_model_dir
                filename = "final_model.keras"
            
            version_dir = model_dir / "versions" / version_id
            if not version_dir.exists():
                raise ValueError(f"Version {version_id} non trouvée")
            
            model_file = version_dir / filename
            if not model_file.exists():
                raise ValueError(f"Fichier modèle non trouvé dans la version {version_id}")
            
            # Mettre à jour le lien symbolique 'current'
            current_link = model_dir / "current"
            if current_link.exists() or current_link.is_symlink():
                current_link.unlink()
            
            current_link.symlink_to(model_file.relative_to(model_dir))
            
            logger.info(f"Rollback effectué vers la version {version_id} pour le modèle {model_type}")
            
            return {
                "success": True,
                "message": f"Rollback vers {version_id} effectué avec succès",
                "current_path": str(current_link)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du rollback : {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_current_model_path(self, model_type: str) -> Optional[Path]:
        """Récupère le chemin du modèle actuellement déployé"""
        if model_type == "ml":
            current_link = self.ml_model_dir / "current"
        else:
            current_link = self.dl_model_dir / "current"
        
        if current_link.exists() and current_link.is_symlink():
            return current_link.resolve()
        
        return None
    
    def get_current_version(self, model_type: str) -> Optional[str]:
        """
        Récupère la version actuellement déployée
        
        Args:
            model_type: Type de modèle ('ml_model' ou 'dl_model')
            
        Returns:
            Version actuelle ou None si aucune version déployée
        """
        if model_type == "ml_model":
            current_link = self.ml_model_dir / "current"
        elif model_type == "dl_model":
            current_link = self.dl_model_dir / "current"
        else:
            return None
        
        if current_link.exists() and current_link.is_symlink():
            try:
                # Le lien pointe vers le fichier du modèle dans versions/vX.Y_timestamp/
                target = current_link.resolve()
                version_dir = target.parent
                
                # Extraire la version du nom du dossier
                version_dir_name = version_dir.name
                if version_dir_name.startswith('v') and '_' in version_dir_name:
                    version = version_dir_name.split('_')[0][1:]  # Enlever le 'v' et prendre la partie avant '_'
                    return version
                    
            except Exception as e:
                logger.error(f"Erreur lors de la récupération de la version actuelle : {e}")
        
        return None

    def cleanup_old_versions(self, model_type: str, keep_count: int = 5):
        """
        Nettoie les anciennes versions (garde seulement les N plus récentes)
        
        Args:
            model_type: Type de modèle
            keep_count: Nombre de versions à garder
        """
        versions = self.list_versions(model_type)
        if len(versions) <= keep_count:
            return
        
        # Trier par timestamp et garder seulement les plus récentes
        versions.sort(key=lambda x: x['timestamp'])
        versions_to_delete = versions[:-keep_count]
        
        if model_type == "ml":
            versions_dir = self.ml_model_dir / "versions"
        else:
            versions_dir = self.dl_model_dir / "versions"
        
        for version_info in versions_to_delete:
            version_id = f"v{version_info['version']}_{version_info['deployment_id']}"
            version_dir = versions_dir / version_id
            
            if version_dir.exists():
                try:
                    shutil.rmtree(version_dir)
                    logger.info(f"Version {version_id} supprimée")
                except Exception as e:
                    logger.error(f"Erreur lors de la suppression de {version_id} : {e}")
