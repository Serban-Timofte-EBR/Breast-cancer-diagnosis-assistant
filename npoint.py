from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import uvicorn
from pydantic import BaseModel
import os
from src.data_loader import preprocess_data

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/best_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class CancerFeatures(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_cancer(data: CancerFeatures):
    try:
        input_data = pd.DataFrame([data.features]) 
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        result = "Malignant" if prediction[0] == 1 else "Benign"
        
        return {
            "prediction": result,
            "probability": {
                "Benign": round(probability[0][0], 4),
                "Malignant": round(probability[0][1], 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
