from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import pandas as pd
import json

model = joblib.load("Models/tuned/svm.pkl")
app = FastAPI()

data = []
with open("Data\\News_Category_Dataset_v3.json", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)
df = df[['headline', 'category']].reset_index(drop=True)

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    pred = model.predict([data.text])[0]  # wrap in list
    return {"prediction": pred} 

# 1 get count of category

class CategoryRequest(BaseModel):
    category: str

@app.post("/count_category")
def count_category(req: CategoryRequest):
    cat = req.category.strip().upper()
    count = df[df["category"] == cat].shape[0]
    return count

# 2 ten headlines from category

@app.post("/ten_from_category")
def top_10_category(req: CategoryRequest):
    cat = req.category.strip().upper()
    subset = df[df["category"] == cat].reset_index()
    top10 = []
    for _, row in subset.head(10).iterrows():
        top10.append({"id": int(row["index"]), "headline": row["headline"]})
    return {"category": cat, "headlines": top10}

# add headlines
@app.post("/add_headline")
def add_headline(req: InputText):
    global df
    pred = model.predict([req.text])[0]

    new_row = {
        "category": pred,
        "headline": req.text
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_json("Data/News_Category_Dataset_v3.json", orient="records", lines=True)
    
    return {
        "message": "Headline added",
        "id": len(df) - 1,
        "headline": req.text,
        "predicted_category": pred
    }
    
# update headline (by id and new headline)
class UpdateHeadlineRequest(BaseModel):
    id: int
    text: str
    
@app.put("/update_headline")
def update_headline(req: UpdateHeadlineRequest):
    global df
    if req.id < 0 or req.id >= len(df):
        return {"error": "Invalid ID"}

    pred = model.predict([req.text])[0]

    df.at[req.id, "headline"] = req.text
    df.at[req.id, "category"] = pred

    df.to_json("Data/News_Category_Dataset_v3.json", orient="records", lines=True)

    return {
        "message": "Headline updated",
        "id": req.id,
        "new_headline": req.text,
        "new_category": pred
    }

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
