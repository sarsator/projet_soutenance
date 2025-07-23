
echo "LANCEMENT TENSORBOARD - ENTRAÎNEMENT SSD MOBILENET V2"


# Aller dans le dossier du projet
cd /home/sarsator/projets/gaia_vision

# Dossiers possibles pour les logs
LOGDIRS=(
    training/models/dl_model/outputs/ssd_mnv2_320/

)

# Chercher le premier dossier de logs valide
SELECTED_LOGDIR=""
for logdir in "${LOGDIRS[@]}"; do
    if [ -d "$logdir" ]; then
        # Vérifier s'il y a des fichiers d'événements
        if find "$logdir" -name "events.out.tfevents.*" -type f | head -1 | grep -q .; then
            SELECTED_LOGDIR="$logdir"
            echo "✅ Logs trouvés dans: $logdir"
            break
        fi
    fi
done

if [ -z "$SELECTED_LOGDIR" ]; then
    echo "AUCUN LOG D'ENTRAÎNEMENT TROUVÉ !"
    echo "Vérifiez que l'entraînement a bien été lancé"
    exit 1
fi

# Paramètres TensorBoard
PORT=6006
HOST="0.0.0.0"

echo ""
echo "LANCEMENT TENSORBOARD..."
echo "Logdir: $SELECTED_LOGDIR"
echo "Host: $HOST"
echo "Port: $PORT"
echo "URL: http://localhost:$PORT"
echo ""

# Vérifier si TensorBoard est installé
if ! command -v tensorboard &> /dev/null; then
    echo "TensorBoard n'est pas installé !"
    echo "Installez-le avec: pip install tensorboard"
    exit 1
fi

echo "COMMANDE: tensorboard --logdir=$SELECTED_LOGDIR --port=$PORT --host=$HOST"

echo "CONSEILS:"
echo "Ouvrez http://localhost:$PORT dans votre navigateur"
echo "Utilisez Ctrl+C pour arrêter TensorBoard"
echo "Surveillez les onglets SCALARS et IMAGES"
echo "APPUYEZ SUR CTRL+C POUR ARRÊTER..."


# Lancer TensorBoard avec les paramètres optimaux
tensorboard \
    --logdir="$SELECTED_LOGDIR" \
    --port=$PORT \
    --host=$HOST \
    # --reload_interval=1 \
    # --load_fast=true


echo "TensorBoard arrêté proprement"
