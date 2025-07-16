#!/bin/bash
set -e

echo "🚀 Démarrage de l'entraînement SSD MobileNet V2 Simple (100% Compatible)..."
echo "📊 TensorBoard: http://localhost:6006"

export PYTHONPATH="/home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research:/home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research/slim:$PYTHONPATH"
cd "/home/sarsator/projets/gaia_vision/training/notebook"

echo "🔍 Vérification des fichiers..."
if [ ! -f "../models/dl_model/outputs/ssd_mnv2_simple/pipeline.config" ]; then
    echo "❌ Fichier de configuration manquant"
    exit 1
fi

echo "✅ Fichiers vérifiés"
echo "🏃 Lancement de l'entraînement avec modèle ultra-compatible..."

/home/sarsator/projets/gaia_vision/.venv/bin/python "tensorflow_models/research/object_detection/model_main_tf2.py" \
    --model_dir="../models/dl_model/outputs/ssd_mnv2_simple" \
    --pipeline_config_path="../models/dl_model/outputs/ssd_mnv2_simple/pipeline.config" \
    --num_train_steps=30000 \
    --sample_1_of_n_eval_examples=1 \
    --alsologtostderr

echo "✅ Entraînement terminé!"
