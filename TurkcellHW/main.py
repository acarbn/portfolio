from fastapi import FastAPI, HTTPException,Response
from pydantic import BaseModel, Field
from model.utils import load_model, predict_outcome
import pandas as pd
import numpy as np

app = FastAPI(title="Job Outcome Prediction Using SVC")


# Ana sayfa (root) endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Veri modelleri
class PredictRequest(BaseModel):
    experience_year: float = Field(..., ge=0, le=40, description="Programming experience must be between 0 and 40 years")
    tech_score: float = Field(..., le=100, description="Tech score must be 100 or less")

@app.get("/data")
def get_products():
    # Read relevant data
    getdf = pd.read_json('data/labeled_data.json')
    data = getdf.to_dict(orient="records")
    return data

@app.post("/predict")
def predict(req: PredictRequest):
    model = load_model("model/linearSVCmodel.joblib")
    experience_year = req.experience_year
    tech_score = req.tech_score
    if experience_year >40:
        raise HTTPException(status_code=400, detail="Years of programming experience must be 40 or less.")
    elif tech_score > 100:
        raise HTTPException(status_code=400, detail="Tech score must be 100 or less.")
    
    try:
    
    # Prepare the example DataFrame
        example = pd.DataFrame({'experience_year': [experience_year], 'tech_score': [tech_score]})
        example_np = example.to_numpy()
        prediction = predict_outcome(example_np)
        prediction_value = int(prediction[0])

        if prediction_value == 0:
            result = "Will get the job."
        elif prediction_value == 1:
            result = "Will NOT get the job."
        else:
            result = "Unknown" 
    
      # Ensure this is a simple int
        return result
    except Exception as e:
        return "There is an error."

