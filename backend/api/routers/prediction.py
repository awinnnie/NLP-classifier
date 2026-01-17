from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "Models" / "tuned" / "svm.pkl"

model = joblib.load(MODEL_PATH)
router = APIRouter()

class InputText(BaseModel):
    text: str

@router.post("/predict")
def predict(data: InputText):
    pred = model.predict([data.text])[0]  # wrap in list
    return {"prediction": pred} 