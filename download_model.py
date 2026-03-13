import os
import gdown  #type: ignore
from dotenv import load_dotenv #type: ignore
load_dotenv()

MODEL_PATH = os.getenv("CHECKPOINT_PATH", "models/best_model.pt")
MODEL_URL  = os.getenv("MODEL_URL") 

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde Drive")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    print("Modelo listo.")
else:
    print("Modelo ya existe, saltando descarga.")
