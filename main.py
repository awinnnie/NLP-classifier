from fastapi import FastAPI
from routers.prediction import router as prediction_router
from routers.data_analysis import router as data_router

app = FastAPI()

app.include_router(prediction_router, tags=["Prediction"])
app.include_router(data_router, tags=["Data Analysis"])
