from fastapi import FastAPI
from api.routers.prediction import router as prediction_router
from api.routers.data_analysis import router as data_router

app = FastAPI()

app.include_router(prediction_router, tags=["Prediction"])
app.include_router(data_router, tags=["Data Analysis"])


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