import os
import sys
import datetime
import subprocess

# Ajouter les chemins nécessaires
current_dir = os.path.dirname(os.path.abspath(__file__))
research_path = os.path.join(current_dir, "tensorflow_models", "research")
slim_path = os.path.join(current_dir, "tensorflow_models", "research", "slim")

sys.path.insert(0, research_path)
sys.path.insert(0, slim_path)


# PATCH TF-SLIM POUR TENSORFLOW 2.15


print("Application du patch tf-slim...")

try:
    from tensorflow.python.ops import control_flow_ops
    import tensorflow as tf
    
    # Patch pour control_flow_ops.case
    if not hasattr(control_flow_ops, 'case'):
        def case_wrapper(pred_fn_pairs, default=None, exclusive=False, name='case'):
            """Wrapper pour remplacer control_flow_ops.case"""
            return tf.case(pred_fn_pairs, default=default, exclusive=exclusive, name=name)
        
        control_flow_ops.case = case_wrapper
        print("Patch control_flow_ops.case appliqué")
    
    # Patch pour control_flow_ops.cond 
    if not hasattr(control_flow_ops, 'cond'):
        def cond_wrapper(pred, true_fn=None, false_fn=None, name=None):
            """Wrapper pour remplacer control_flow_ops.cond"""
            return tf.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
        
        control_flow_ops.cond = cond_wrapper
        print("Patch control_flow_ops.cond appliqué")
        
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
        print("Patch control_flow_ops.while_loop appliqué")

    print("Tous les patches tf-slim appliqués avec succès")

except Exception as e:
    print(f"Erreur lors de l'application des patches: {e}")

if __name__ == "__main__":
    print("\nDémarrage de l'entraînement avec patches appliqués...")
    
    # Créer le fichier de log avec timestamp
    
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_log_{timestamp}.txt"
    
    # Exécuter le script original model_main_tf2.py
    script_path = os.path.join(research_path, "object_detection", "model_main_tf2.py")
    
    print(f"Arguments reçus: {sys.argv}")
    print(f"Exécution: {script_path}")
    print(f"Arguments transmis: {sys.argv[1:]}")
    print(f"Log de sortie: {log_file}")
    
    # Préparer la commande avec les arguments originaux
    cmd = [sys.executable, script_path] + sys.argv[1:]
    
    # Configurer l'environnement
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{research_path}:{slim_path}:{env.get('PYTHONPATH', '')}"
    
    print(f"Commande: {' '.join(cmd)}")
    print(f"Démarrage de l'entraînement... (logs dans {log_file})")
    
    # Exécuter avec redirection vers fichier ET console
    try:
        with open(log_file, 'w') as f:
            # Écrire l'en-tête du log
            f.write(f"=== ENTRAINEMENT SSD MOBILENET V2 ===\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write(f"Commande: {' '.join(cmd)}\n")
            f.write(f"PYTHONPATH: {env['PYTHONPATH']}\n")

            f.flush()
            
            # Lancer le processus avec redirection
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
                bufsize=1
            )
            
            # Lire et afficher en temps réel + sauvegarder
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())  # Afficher sur console
                f.write(line)         # Sauvegarder dans fichier
                f.flush()             # Forcer l'écriture
            
            process.wait()
            
            # Écrire le statut final
            f.write(f"\n" + "=" * 50 + "\n")
            f.write(f"Fin de l'entraînement: {datetime.datetime.now()}\n")
            f.write(f"Code de retour: {process.returncode}\n")
            
        print(f"\nEntraînement terminé. Code de retour: {process.returncode}")
        print(f"Log complet sauvé dans: {log_file}")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
        with open(log_file, 'a') as f:
            f.write(f"\nERREUR: {e}\n")
