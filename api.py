from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    instrumentator.expose(app, endpoint="/metrics")
    yield

app = FastAPI(lifespan=lifespan)
instrumentator.instrument(app)

class Features(BaseModel):
    data: list

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get("/")
async def healthcheck():
    return {"message": "radio check"}

@app.post("/predict")
async def predict(features: Features):
    data = np.array(features.data).reshape(1, -1)
    pred = model.predict(data)[0]
    return {"pred": float(pred)}
