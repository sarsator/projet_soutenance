from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from uuid import uuid4
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("API_KEY")
UPLOAD_DIR = Path("images_a_traiter")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

def check_api_key(auth: str):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Missing or invalid Authorization header")
    key = auth.split("Bearer ")[-1]
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.get("/status")
def status():
    return {"status": "API OK", "version": "1.0"}

@app.post("/predict-image")
async def predict_image(
    authorization: str = Header(None),
    race_champignon: str = Form(...),
    type_substrat: str = Form(...),
    jours_inoculation: int = Form(...),
    hygrometrie: float = Form(...),
    co2_ppm: float = Form(...),
    commentaire: str = Form(""),
    image: UploadFile = File(...)
):
    check_api_key(authorization)

    file_id = str(uuid4())
    file_ext = Path(image.filename).suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    with open(file_path, "wb") as f:
        f.write(await image.read())

    # Simulation de prédiction pour test
    return JSONResponse({
        "prediction": "sain",
        "confidence": 0.72,
        "input": {
            "race": race_champignon,
            "substrat": type_substrat,
            "jours": jours_inoculation,
            "hygro": hygrometrie,
            "co2": co2_ppm,
            "commentaire": commentaire,
            "image_file": file_path.name
        }
    })

