from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("training/model.pkl")
label_map = joblib.load("training/label_map.pkl")


class NewsRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: NewsRequest):
    prediction = model.predict([req.text])[0]

    probabilities = model.predict_proba([req.text])[0]
    confidence = float(max(probabilities))

    return {
        "category": label_map[int(prediction)],
        "confidence": confidence
    }
