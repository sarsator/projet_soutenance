#!/bin/bash
set -e

echo "Démarrage de l'entraînement SSD MobileNet V2 320x320..."
echo "TensorBoard: http://localhost:6006"

export PYTHONPATH="/home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research:/home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research/slim:$PYTHONPATH"
cd "/home/sarsator/projets/gaia_vision/training/notebook"

echo "Vérification des fichiers..."
if [ ! -f "../models/dl_model/outputs/ssd_mnv2_320/pipeline_working.config" ]; then
    echo "❌ Fichier de configuration manquant"
    exit 1
fi

echo "✅ Fichiers vérifiés"
echo "Lancement de l'entraînement avec patch tf-slim..."

/home/sarsator/projets/gaia_vision/.venv/bin/python "train_with_patch.py" \
    --model_dir="../models/dl_model/outputs/ssd_mnv2_320" \
    --pipeline_config_path="../models/dl_model/outputs/ssd_mnv2_320/pipeline_working.config" \
    --alsologtostderr \
    --checkpoint_dir=../models/dl_model/outputs/ssd_mnv2_320 \
    --run_once=True

echo "✅ Entraînement terminé!"