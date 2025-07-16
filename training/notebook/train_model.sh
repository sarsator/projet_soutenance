#!/bin/bash
export PYTHONPATH=/home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research:/home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research/slim:/home/sarsator/.local/lib/python3.12/site-packages:$PYTHONPATH
cd /home/sarsator/projets/gaia_vision/training/notebook
python3 /home/sarsator/projets/gaia_vision/training/notebook/tensorflow_models/research/object_detection/model_main_tf2.py --model_dir=../models/dl_model/outputs/ssd_mnv3_512 --pipeline_config_path=ssd_mobilenet_v3_small.config --num_train_steps=30000 --sample_1_of_n_eval_examples=1 --alsologtostderr
