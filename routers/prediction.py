from fastapi import APIRouter
from pydantic import BaseModel
import joblib

model = joblib.load("Models/tuned/svm.pkl")
router = APIRouter()

class InputText(BaseModel):
    text: str

@router.post("/predict")
def predict(data: InputText):
    pred = model.predict([data.text])[0]  # wrap in list
    return {"prediction": pred} 


