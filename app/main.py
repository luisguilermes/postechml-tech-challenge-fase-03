from __future__ import annotations

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class PlayerForPredict(BaseModel):
    age: int
    overall: int
    potential: int
    wage_eur: float
    club_name: str | None = None
    club_position: str | None = None
    nationality_name: str | None = None

app = FastAPI()
MODEL = joblib.load("../ml/models/fc26_value_model.joblib")

@app.post("/predict")
def predict(p: PlayerForPredict):
    df = pd.DataFrame([{
      "age": p.age, "overall": p.overall, "potential": p.potential,
      "wage_eur": p.wage_eur, "club_name": p.club_name, "club_position": p.club_position, "nationality_name": p.nationality_name
    }])
    pred = MODEL.predict(df)[0]
    return {"predicted_value_eur": float(pred)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)