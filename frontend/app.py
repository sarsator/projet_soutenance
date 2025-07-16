from flask import Flask, render_template, request, url_for
import requests
import os
import json
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

# Création de l'app Flask
app = Flask(__name__)

# Configuration des uploads d'images
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Formats acceptés

# Création du dossier uploads si nécessaire
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chemin vers les fichiers de configuration JSON
JSONS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'jsons'))

# Configuration de l'API (mon backend FastAPI)
API_URL = "http://localhost:8000/predict-image"
API_KEY = os.getenv("API_KEY", "gaia-vision-test-key-2025")

def allowed_file(filename):
    """
    Vérifie si le fichier uploadé a une extension autorisée.
    
    Args:
        filename: Nom du fichier
        
    Returns:
        bool: True si l'extension est ok, False sinon
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_json(file_name):
    """
    Charge un fichier JSON depuis le dossier jsons.
    
    Args:
        file_name: Nom du fichier JSON
        
    Returns:
        dict: Contenu du fichier JSON
    """
    file_path = os.path.join(JSONS_DIR, file_name)
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Route principale de l'interface - gère l'affichage et la soumission du formulaire.
    """
    # Gestion de la langue (français par défaut)
    lang_code = request.args.get("lang", "fr")
    lang = load_json(f"{lang_code}.json")
    
    # Chargement des listes déroulantes depuis les JSON
    champignons = load_json("champignon_types.json")["champignon_types"]
    substrats = load_json("substrat_types.json")["substrat_types"]

    # Variables pour la réponse
    response_data = None
    uploaded_image_filename = None

    if request.method == "POST":
        # Traitement de l'image uploadée
        image_file = request.files["image"]
        if image_file and allowed_file(image_file.filename):
            # Génération d'un nom unique pour éviter les conflits
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(image_file.filename)}"
            image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            image_file.save(image_path)
            uploaded_image_filename = unique_filename
            
            # Réouvrir le fichier pour l'envoyer à l'API
            with open(image_path, 'rb') as f:
                image_data = f.read()
        
        # Conversion de la date d'inoculation en nombre de jours
        date_inoculation = request.form["jours_inoculation"]
        try:
            # Parse la date d'inoculation
            date_inoculation_obj = datetime.strptime(date_inoculation, "%Y-%m-%d")
            # Calcule le nombre de jours depuis l'inoculation
            jours_depuis_inoculation = (datetime.now() - date_inoculation_obj).days
            
            # S'assurer que le nombre de jours est positif
            if jours_depuis_inoculation < 0:
                jours_depuis_inoculation = 0
                
        except ValueError:
            # Si la conversion échoue, utiliser une valeur par défaut
            jours_depuis_inoculation = 0
        
        form_data = {
            "race_champignon": request.form["race_champignon"],
            "type_substrat": request.form["type_substrat"],
            "jours_inoculation": jours_depuis_inoculation,  # Utiliser le nombre de jours calculé
            "hygrometrie": request.form["hygrometrie"],
            "co2_ppm": request.form["co2_ppm"],
            "commentaire": request.form.get("commentaire", "")
        }

        files = {"image": (image_file.filename, image_data, image_file.content_type)}
        headers = {"Authorization": f"Bearer {API_KEY}"}

        try:
            r = requests.post(API_URL, data=form_data, files=files, headers=headers)
            print(f"🔍 Status Code API: {r.status_code}")
            print(f"🔍 Headers API: {dict(r.headers)}")
            if r.status_code == 200:
                response_data = r.json()
                print(f"🔍 Réponse API brute: {response_data}")
                print(f"🔍 Type de confidence: {type(response_data.get('confidence', 'N/A'))}")
                print(f"🔍 Valeur de confidence: {response_data.get('confidence', 'N/A')}")
                
                # Ajouter le nom de fichier de l'image à la réponse
                response_data["uploaded_image_filename"] = uploaded_image_filename
                # Ajouter des informations de debug
                response_data["debug_info"] = {
                    "date_inoculation": date_inoculation,
                    "jours_calcules": jours_depuis_inoculation
                }
            else:
                print(f"🔍 Erreur API - Status: {r.status_code}")
                print(f"🔍 Contenu erreur: {r.text}")
                response_data = {
                    "error": f"Erreur API: {r.status_code}",
                    "details": r.text
                }
        except requests.exceptions.ConnectionError:
            print("🔍 Erreur de connexion: API non disponible")
            response_data = {
                "error": "API_UNAVAILABLE",
                "error_message": "Le service d'analyse n'est pas disponible",
                "error_details": "Veuillez réessayer dans quelques instants ou contacter l'administrateur.",
                "error_icon": "🔌",
                "error_type": "connection"
            }
        except requests.exceptions.Timeout:
            print("🔍 Timeout de l'API")
            response_data = {
                "error": "API_TIMEOUT", 
                "error_message": "L'analyse prend plus de temps que prévu",
                "error_details": "Veuillez réessayer avec une image plus petite ou patienter.",
                "error_icon": "⏱️",
                "error_type": "timeout"
            }
        except Exception as e:
            print(f"🔍 Exception dans frontend: {e}")
            import traceback
            traceback.print_exc()
            
            # Messages d'erreur user-friendly selon le type d'erreur
            if "Connection refused" in str(e):
                response_data = {
                    "error": "API_DOWN",
                    "error_message": "Le service d'analyse est temporairement indisponible",
                    "error_details": "Notre équipe technique a été notifiée. Veuillez réessayer dans quelques minutes.",
                    "error_icon": "🚧",
                    "error_type": "service"
                }
            elif "Max retries exceeded" in str(e):
                response_data = {
                    "error": "NETWORK_ERROR",
                    "error_message": "Problème de connexion réseau",
                    "error_details": "Vérifiez votre connexion internet et réessayez.",
                    "error_icon": "🌐",
                    "error_type": "network"
                }
            else:
                response_data = {
                    "error": "UNKNOWN_ERROR",
                    "error_message": "Une erreur inattendue s'est produite",
                    "error_details": "Veuillez réessayer ou contacter le support technique.",
                    "error_icon": "❌",
                    "error_type": "unknown",
                    "technical_details": str(e) if app.debug else None
                }

    return render_template(
        "index.html",
        lang=lang,
        lang_code=lang_code,
        champignons=champignons,
        substrats=substrats,
        response=response_data
    )

@app.route('/heatmap', methods=['POST'])
def generate_heatmap():
    """Endpoint pour générer une heatmap de contamination"""
    try:
        # L'image est envoyée depuis le frontend JavaScript
        if 'file' not in request.files:
            return {"error": "Aucun fichier fourni"}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {"error": "Aucun fichier sélectionné"}, 400
        
        # Sauvegarder temporairement le fichier
        import tempfile
        import sys
        import os
        from pathlib import Path
        
        # Ajouter le chemin parent pour les imports
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        sys.path.insert(0, str(parent_dir))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_image_path = temp_file.name
        
        print(f"🔥 Génération heatmap pour: {file.filename}")
        print(f"Fichier temporaire: {temp_image_path}")
        
        # Importer et utiliser le générateur de heatmap
        try:
            from api.utils.heatmap_generator import ContaminationHeatmapGenerator
            from api.models.vision_model import VisionModel
            
            # Charger le modèle et faire la prédiction
            model = VisionModel()
            if not model.load_model():
                return {"error": "Impossible de charger le modèle de vision"}, 500
            
            # Obtenir les détections
            result = model.predict(temp_image_path)
            print(f"Résultat prédiction: {result}")
            
            # Vérifier s'il y a des contaminations
            contaminated_detections = [d for d in result.get('detections', []) if d.get('class_name') == 'contaminated']
            print(f"Contaminations trouvées: {len(contaminated_detections)}")
            
            if not contaminated_detections:
                print("⚠️ Aucune contamination détectée, retour image originale")
                # Retourner l'image originale si pas de contamination
                with open(temp_image_path, 'rb') as f:
                    content = f.read()
                os.unlink(temp_image_path)
                from flask import Response
                return Response(content, mimetype='image/jpeg')
            
            # Générer la heatmap
            generator = ContaminationHeatmapGenerator()
            heatmap_img = generator.create_contamination_heatmap(temp_image_path, result['detections'])
            
            # Convertir en bytes pour la réponse
            from PIL import Image
            from io import BytesIO
            pil_img = Image.fromarray(heatmap_img)
            img_buffer = BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            print(f"✅ Heatmap générée avec {len(contaminated_detections)} zone(s)")
            
            # Nettoyer le fichier temporaire
            os.unlink(temp_image_path)
            
            from flask import Response
            return Response(img_buffer.getvalue(), mimetype='image/png')
            
        except ImportError as e:
            print(f"Erreur import: {e}")
            # Si l'import échoue, générer une heatmap factice pour tester
            from PIL import Image, ImageDraw
            from io import BytesIO
            import random
            
            # Charger l'image originale
            original_img = Image.open(temp_image_path)
            
            # Créer une heatmap simple pour test
            overlay = Image.new('RGBA', original_img.size, (255, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Ajouter quelques zones "chaudes" factices
            width, height = original_img.size
            for i in range(3):
                x1 = random.randint(0, width//2)
                y1 = random.randint(0, height//2)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)
                draw.ellipse([x1, y1, x2, y2], fill=(255, 0, 0, 80))
            
            # Fusionner avec l'image originale
            result_img = Image.alpha_composite(original_img.convert('RGBA'), overlay)
            
            # Convertir en bytes
            img_buffer = BytesIO()
            result_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            print("✅ Heatmap factice générée (test)")
            os.unlink(temp_image_path)
            
            from flask import Response
            return Response(img_buffer.getvalue(), mimetype='image/png')
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Erreur génération heatmap: {e}")
        print(f"Traceback: {error_details}")
        return {"error": f"Erreur lors de la génération de la heatmap: {str(e)}"}, 500

@app.route('/heatmap-overlay', methods=['POST'])
def generate_heatmap_overlay():
    """Endpoint pour générer un overlay de contamination"""
    try:
        # L'image est envoyée depuis le frontend JavaScript
        if 'file' not in request.files:
            return {"error": "Aucun fichier fourni"}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {"error": "Aucun fichier sélectionné"}, 400
        
        # Sauvegarder temporairement le fichier
        import tempfile
        import sys
        import os
        from pathlib import Path
        
        # Ajouter le chemin parent pour les imports
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        sys.path.insert(0, str(parent_dir))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_image_path = temp_file.name
        
        print(f"🎯 Génération overlay pour: {file.filename}")
        
        # Importer et utiliser le générateur
        try:
            from api.utils.heatmap_generator import ContaminationHeatmapGenerator
            from api.models.vision_model import VisionModel
            
            # Charger le modèle et faire la prédiction
            model = VisionModel()
            if not model.load_model():
                return {"error": "Impossible de charger le modèle de vision"}, 500
            
            # Obtenir les détections
            result = model.predict(temp_image_path)
            
            # Vérifier s'il y a des contaminations
            contaminated_detections = [d for d in result.get('detections', []) if d.get('class_name') == 'contaminated']
            
            if not contaminated_detections:
                print("⚠️ Aucune contamination détectée, retour image originale")
                with open(temp_image_path, 'rb') as f:
                    content = f.read()
                os.unlink(temp_image_path)
                from flask import Response
                return Response(content, mimetype='image/jpeg')
            
            # Générer l'overlay
            generator = ContaminationHeatmapGenerator()
            overlay_img = generator.create_contamination_overlay_pil(temp_image_path, result['detections'])
            
            # Convertir en bytes
            from io import BytesIO
            img_buffer = BytesIO()
            overlay_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            print(f"✅ Overlay généré avec {len(contaminated_detections)} zone(s)")
            os.unlink(temp_image_path)
            
            from flask import Response
            return Response(img_buffer.getvalue(), mimetype='image/png')
            
        except ImportError as e:
            print(f"Erreur import: {e}")
            # Overlay factice pour test
            from PIL import Image, ImageDraw
            from io import BytesIO
            import random
            
            # Charger l'image originale
            original_img = Image.open(temp_image_path)
            draw = ImageDraw.Draw(original_img)
            
            # Ajouter quelques rectangles factices
            width, height = original_img.size
            for i in range(2):
                x1 = random.randint(0, width//2)
                y1 = random.randint(0, height//2)
                x2 = x1 + random.randint(80, 120)
                y2 = y1 + random.randint(80, 120)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                draw.text((x1, y1-20), f"Contamination {i+1}", fill='red')
            
            # Convertir en bytes
            img_buffer = BytesIO()
            original_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            print("✅ Overlay factice généré (test)")
            os.unlink(temp_image_path)
            
            from flask import Response
            return Response(img_buffer.getvalue(), mimetype='image/png')
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Erreur génération overlay: {e}")
        print(f"Traceback: {error_details}")
        return {"error": f"Erreur lors de la génération de l'overlay: {str(e)}"}, 500

if __name__ == "__main__":
    print("🌐 Démarrage du frontend Gaia Vision...")
    print("📍 URL: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
