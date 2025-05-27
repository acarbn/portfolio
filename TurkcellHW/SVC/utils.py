import matplotlib.pyplot as plt
import numpy as np
from joblib import load

def load_model(path: str):
    return load(path)

def predict_outcome(example):
    model=load('model/linearSVCmodel.joblib')
    scaler = load('model/scaler.pkl')
    example_scaler=scaler.transform(example)
    outcome=model.predict(example_scaler)
    return outcome

    
