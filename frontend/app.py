from flask import Flask, render_template, request
import requests
import os
import json

app = Flask(__name__)

# Chemins relatifs aux JSON de configuration
JSONS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'jsons'))

# API externe
API_URL = "http://localhost:8000/predict-image"
API_KEY = os.getenv("API_KEY", "123456")

def load_json(file_name):
    file_path = os.path.join(JSONS_DIR, file_name)
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    lang_code = request.args.get("lang", "fr")
    lang = load_json(f"{lang_code}.json")
    champignons = load_json("champignon_types.json")["champignon_types"]
    substrats = load_json("substrat_types.json")["substrat_types"]

    response_data = None

    if request.method == "POST":
        form_data = {
            "race_champignon": request.form["race_champignon"],
            "type_substrat": request.form["type_substrat"],
            "jours_inoculation": request.form["jours_inoculation"],
            "hygrometrie": request.form["hygrometrie"],
            "co2_ppm": request.form["co2_ppm"],
            "commentaire": request.form.get("commentaire", "")
        }
        image_file = request.files["image"]

        files = {"image": (image_file.filename, image_file.stream, image_file.content_type)}
        headers = {"Authorization": f"Bearer {API_KEY}"}

        try:
            r = requests.post(API_URL, data=form_data, files=files, headers=headers)
            response_data = r.json()
        except Exception as e:
            response_data = {"error": str(e)}

    return render_template(
        "index.html",
        lang=lang,
        lang_code=lang_code,
        champignons=champignons,
        substrats=substrats,
        response=response_data
    )
