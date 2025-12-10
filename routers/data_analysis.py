from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import joblib

model = joblib.load("Models/tuned/svm.pkl")
router = APIRouter()

data = []
with open("Data\\News_Category_Dataset_v3.json", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)
df = df[['headline', 'category']].reset_index(drop=True)

# 1 get count of category

@router.get("/count/{category}")
def count_category(category: str):
    cat = category.strip().upper()
    count = df[df["category"] == cat].shape[0]
    return count

# 2 ten headlines from category

@router.get("/top 10/{category}")
def top_10_category(category: str):
    cat = category.strip().upper()
    subset = df[df["category"] == cat].reset_index()
    top10 = []
    for _, row in subset.head(10).iterrows():
        top10.append({"id": int(row["index"]), "headline": row["headline"]})
    return {"category": cat, "headlines": top10}

# 3 add headlines
class InputText(BaseModel):
    text: str
    
@router.post("/add_headline")
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
    
# 4 update headline (by id and new headline)
class UpdateHeadlineRequest(BaseModel):
    id: int
    text: str

@router.put("/update_headline")
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
    
# 4 delete headline by id  
class DeleteHeadlineRequest(BaseModel):
    id: int

@router.delete("/delete_headline")
def delete_headline(req: DeleteHeadlineRequest):
    global df
    if req.id < 0 or req.id >= len(df):
        return {"error": "Invalid ID"}
    
    df = df.drop(req.id).reset_index(drop=True)
    df.to_json("Data/News_Category_Dataset_v3.json", orient="records", lines=True)
    
    return {
        "message": "Headline deleted",
        "id": req.id
    }