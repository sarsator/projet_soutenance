#!/bin/bash
# Script automatique d'entrainement SSD MobileNet V2

echo "Lancement entrainement SSD MobileNet V2..."

cd /home/sarsator/projets/gaia_vision/training/notebook

export PYTHONPATH=/home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research:/home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research/slim:$PYTHONPATH

python /home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research/object_detection/model_main_tf2.py \
  --model_dir=../models/dl_model/outputs/ssd_mnv2_320 \
  --pipeline_config_path=../models/dl_model/outputs/ssd_mnv2_320/pipeline_working.config \
  --num_train_steps=80000 \
  --checkpoint_every_n=4000 \
  --alsologtostderr

echo "Entrainement termine!"
echo "Resultats dans: ../models/dl_model/outputs/ssd_mnv2_320"
echo "TensorBoard: http://localhost:6006"
