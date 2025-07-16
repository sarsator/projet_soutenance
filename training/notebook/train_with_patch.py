#!/usr/bin/env python3
"""
Script d'entraînement avec patch tf-slim pour TensorFlow 2.15
"""

import os
import sys

# Ajouter les chemins nécessaires
current_dir = os.path.dirname(os.path.abspath(__file__))
research_path = os.path.join(current_dir, "tensorflow_models", "research")
slim_path = os.path.join(current_dir, "tensorflow_models", "research", "slim")

sys.path.insert(0, research_path)
sys.path.insert(0, slim_path)

# ═══════════════════════════════════════════════════════════
# 🔧 PATCH TF-SLIM POUR TENSORFLOW 2.15
# ═══════════════════════════════════════════════════════════

print("🔧 Application du patch tf-slim...")

try:
    from tensorflow.python.ops import control_flow_ops
    import tensorflow as tf
    
    # Patch pour control_flow_ops.case
    if not hasattr(control_flow_ops, 'case'):
        def case_wrapper(pred_fn_pairs, default=None, exclusive=False, name='case'):
            """Wrapper pour remplacer control_flow_ops.case"""
            return tf.case(pred_fn_pairs, default=default, exclusive=exclusive, name=name)
        
        control_flow_ops.case = case_wrapper
        print("✅ Patch control_flow_ops.case appliqué")
    
    # Patch pour control_flow_ops.cond 
    if not hasattr(control_flow_ops, 'cond'):
        def cond_wrapper(pred, true_fn=None, false_fn=None, name=None):
            """Wrapper pour remplacer control_flow_ops.cond"""
            return tf.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
        
        control_flow_ops.cond = cond_wrapper
        print("✅ Patch control_flow_ops.cond appliqué")
        
    # Patch pour control_flow_ops.while_loop si nécessaire
    if not hasattr(control_flow_ops, 'while_loop'):
        def while_loop_wrapper(cond, body, loop_vars, shape_invariants=None, 
                              parallel_iterations=10, back_prop=True, 
                              swap_memory=False, name=None, maximum_iterations=None,
                              return_same_structure=False):
            """Wrapper pour remplacer control_flow_ops.while_loop"""
            return tf.while_loop(
                cond=cond, body=body, loop_vars=loop_vars,
                shape_invariants=shape_invariants, 
                parallel_iterations=parallel_iterations,
                back_prop=back_prop, swap_memory=swap_memory, 
                name=name, maximum_iterations=maximum_iterations,
                return_same_structure=return_same_structure
            )
        
        control_flow_ops.while_loop = while_loop_wrapper
        print("✅ Patch control_flow_ops.while_loop appliqué")
        
    print("✅ Patch tf-slim complet")
        
except Exception as e:
    print(f"⚠️  Erreur de patch: {e}")

# ═══════════════════════════════════════════════════════════
# 🚀 LANCEMENT DE L'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════

print("🚀 Lancement de l'entraînement SSD MobileNet V2...")

# Lancer le script original avec le patch appliqué
if __name__ == '__main__':
    # Exécuter le script original model_main_tf2.py
    script_path = os.path.join(research_path, "object_detection", "model_main_tf2.py")
    
    # Remplacer sys.argv par les arguments appropriés
    original_argv = sys.argv[:]
    sys.argv = [
        script_path,
        '--model_dir=../models/dl_model/outputs/ssd_mnv2_320',
        '--pipeline_config_path=../models/dl_model/outputs/ssd_mnv2_320/pipeline.config',
        '--num_train_steps=30000',
        '--sample_1_of_n_eval_examples=1',
        '--alsologtostderr'
    ]
    
    print(f"📄 Exécution: {script_path}")
    print(f"� Arguments: {sys.argv[1:]}")
    
    # Importer et exécuter le module
    exec(open(script_path).read())
