#!/bin/bash
#
# Script wrapper simple pour le système de versioning Gaia Vision
# Usage facile pour les opérations courantes
#

SCRIPT_DIR="/home/sarsator/projets/gaia_vision"
VERSIONING_SCRIPT="$SCRIPT_DIR/model_versioning.py"

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}GAIA VISION - SYSTÈME DE VERSIONING${NC}"
echo -e "${BLUE}=====================================${NC}"

# Fonction d'aide
show_help() {
    echo -e "${YELLOW}Commandes disponibles:${NC}"
    echo "  ./versioning.sh status           - Afficher le statut du système"
    echo "  ./versioning.sh list             - Lister toutes les versions"
    echo "  ./versioning.sh deploy ssd       - Déployer nouvelle version SSD"
    echo "  ./versioning.sh deploy catboost  - Déployer nouvelle version CatBoost"
    echo "  ./versioning.sh deploy both      - Déployer les deux modèles"
    echo "  ./versioning.sh logs             - Voir les derniers logs"
    echo "  ./versioning.sh cleanup          - Nettoyer les anciennes versions"
    echo ""
    echo -e "${YELLOW}Exemples:${NC}"
    echo "  ./versioning.sh deploy both      # Déploie SSD v1.6 et CatBoost v1.4"
    echo "  ./versioning.sh status           # État actuel: SSD v1.5, CatBoost v1.3"
}

# Fonction pour afficher les logs
show_logs() {
    echo -e "${BLUE}DERNIERS LOGS:${NC}"
    echo "=============="
    LATEST_LOG=$(ls -t $SCRIPT_DIR/logs/versioning_*.log | head -1)
    if [ -f "$LATEST_LOG" ]; then
        echo -e "${GREEN}Fichier: $(basename $LATEST_LOG)${NC}"
        echo ""
        tail -20 "$LATEST_LOG"
    else
        echo -e "${RED}Aucun log trouvé${NC}"
    fi
}

# Fonction de nettoyage
cleanup_versions() {
    echo -e "${YELLOW}NETTOYAGE DES ANCIENNES VERSIONS${NC}"
    echo "================================="
    
    echo -e "${BLUE}Nettoyage SSD (garde 3 versions)...${NC}"
    cd "$SCRIPT_DIR" && python "$VERSIONING_SCRIPT" cleanup dl --keep 3
    
    echo -e "${BLUE}Nettoyage CatBoost (garde 3 versions)...${NC}"
    cd "$SCRIPT_DIR" && python "$VERSIONING_SCRIPT" cleanup ml --keep 3
    
    echo -e "${GREEN}Nettoyage terminé${NC}"
}

# Fonction de déploiement
deploy_model() {
    local model_type=$1
    
    case $model_type in
        "ssd"|"dl")
            echo -e "${BLUE}Déploiement SSD MobileNet V2...${NC}"
            cd "$SCRIPT_DIR" && python "$VERSIONING_SCRIPT" deploy dl
            ;;
        "catboost"|"ml")
            echo -e "${BLUE}Déploiement CatBoost...${NC}"
            cd "$SCRIPT_DIR" && python "$VERSIONING_SCRIPT" deploy ml
            ;;
        "both"|"all")
            echo -e "${BLUE}Déploiement des deux modèles...${NC}"
            echo ""
            echo -e "${YELLOW}1/2 - Déploiement SSD...${NC}"
            cd "$SCRIPT_DIR" && python "$VERSIONING_SCRIPT" deploy dl
            echo ""
            echo -e "${YELLOW}2/2 - Déploiement CatBoost...${NC}"
            cd "$SCRIPT_DIR" && python "$VERSIONING_SCRIPT" deploy ml
            echo ""
            echo -e "${GREEN}Déploiement complet terminé${NC}"
            ;;
        *)
            echo -e "${RED}Type de modèle invalide: $model_type${NC}"
            echo "Types supportés: ssd, catboost, both"
            exit 1
            ;;
    esac
}

# Traitement des arguments
case $1 in
    "status")
        cd "$SCRIPT_DIR" && python "$VERSIONING_SCRIPT" status
        ;;
    "list")
        cd "$SCRIPT_DIR" && python "$VERSIONING_SCRIPT" list
        ;;
    "deploy")
        if [ -z "$2" ]; then
            echo -e "${RED}Spécifiez le type de modèle à déployer${NC}"
            show_help
            exit 1
        fi
        deploy_model "$2"
        ;;
    "logs")
        show_logs
        ;;
    "cleanup")
        cleanup_versions
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo -e "${RED}Commande inconnue: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
