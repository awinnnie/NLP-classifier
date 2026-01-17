from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
import joblib
from pathlib import Path

from db import get_db
from models import NewsRow

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "Models" / "tuned" / "svm.pkl"

model = joblib.load(MODEL_PATH)
router = APIRouter()

# 1 get count of category

@router.get("/count/{category}")
def count_category(category: str, db: Session = Depends(get_db)):
    cat = category.strip().upper()

    count = (
        db.query(func.count(NewsRow.id))
        .filter(NewsRow.category == cat)
        .scalar()
    )

    return count


# 2 ten headlines from category

@router.get("/top10/{category}")
def top_10_category(category: str, db: Session = Depends(get_db)):
    cat = category.strip().upper()

    rows = (
        db.query(NewsRow)
        .filter(NewsRow.category == cat)
        .order_by(NewsRow.id)
        .limit(10)
        .all()
    )

    return {
        "category": cat,
        "headlines": [
            {"id": row.id, "headline": row.headline}
            for row in rows
        ],
    }


# 3 add headlines
class HeadlineRequest(BaseModel):
    text: str


@router.post("/add_headline")
def add_headline(payload: HeadlineRequest, db: Session = Depends(get_db)):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Headline cannot be empty")

    pred = model.predict([text])[0]

    row = NewsRow(
        headline=text,
        category=pred
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "message": "Headline added",
        "id": row.id,
        "headline": row.headline,
        "predicted_category": row.category,
    }

    
# 4 update headline (by id and new headline)
class UpdateHeadlineRequest(BaseModel):
    id: int
    text: str


@router.put("/update_headline")
def update_headline(req: UpdateHeadlineRequest, db: Session = Depends(get_db)):
    row = db.get(NewsRow, req.id)

    if not row:
        raise HTTPException(status_code=404, detail="Headline not found")

    pred = model.predict([req.text])[0]

    row.headline = req.text
    row.category = pred

    db.commit()

    return {
        "message": "Headline updated",
        "id": row.id,
        "new_headline": row.headline,
        "new_category": row.category,
    }

    
# 5 delete headline by id  

class DeleteRequest(BaseModel):
    id: int


@router.delete("/delete_headline")
def delete_headline(req: DeleteRequest, db: Session = Depends(get_db)):
    row = db.get(NewsRow, req.id)

    if not row:
        raise HTTPException(status_code=404, detail="Headline not found")

    db.delete(row)
    db.commit()

    return {
        "message": "Headline deleted",
        "id": req.id,
    }

