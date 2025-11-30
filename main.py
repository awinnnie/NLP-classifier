from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib

model = joblib.load("Models/tuned/svm.pkl")
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    pred = model.predict([data.text])[0]  # wrap in list
    return {"prediction": int(pred)} 



# app = FastAPI() 
# @app.get("/") 
# async def root(): 
#     return {"message": "hellou"}

# @app.post("/items/")
# async def create_item(item: Item):
#     return item

# @app.put("/items/{item_id}")
# async def create_item(item_id: int, item: Item):
#     return {"item_id": item_id, **item.dict()}

# @app.delete("/heroes/{hero_id}")
# def delete_hero(hero_id: int):
#     with Session(engine) as session:
#         hero = session.get(Hero, hero_id)
#         if not hero:
#             raise HTTPException(status_code=404, detail="Hero not found")
#         session.delete(hero)
#         session.commit()
#         return {"ok": True}