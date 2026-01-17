import json
from pathlib import Path
from sqlalchemy import select, func

from db import engine, SessionLocal, Base
from models import NewsRow

PROJECT_ROOT = Path(__file__).resolve().parent[1]
JSON_PATH = PROJECT_ROOT / "Data" / "News_Category_Dataset_v3.json"

if not JSON_PATH.exists():
    raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")


def seed_from_jsonl():
    Base.metadata.create_all(bind=engine)

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")
    rows = []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                rows.append(
                    NewsRow(
                        headline=obj["headline"],
                        category=obj["category"]
                    )
                )

    with SessionLocal() as db:
        existing = db.execute(
            select(func.count(NewsRow.id))
        ).scalar_one()

        if existing > 0:
            print(f"DB already has {existing} rows. Skipping seed.")
            return

        db.add_all(rows)
        db.commit()

    print(f"Seeded {len(rows)} news rows into database.")

if __name__ == "__main__":
    seed_from_jsonl()