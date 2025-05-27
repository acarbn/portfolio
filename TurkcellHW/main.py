
"""
PAİR ÜYELERİ
İZEMNUR BUDAK
BURÇİN ACAR
HANDE NUR UYGUN
NİSASU BOZKURT
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import tensorflow as tf
from utils import feature_eng, generate_return_labels, explain_order

app = FastAPI(title="Deep Learning Based Order Return Prediction",description="Given an order it will predict if the customer will give it back")

@app.get("/")
async def root():
    return {"message": "Welcome to Our API!"}


@app.get("/orders")
async def get_orders():
    df = pd.read_csv('deep_learning/HW/HW2/pair1/combined_originaldata.csv')
    return df.to_dict(orient="records")

class Applicant(BaseModel):
    order_id: int
    product_id: List[int]
    unit_price: List[float]
    quantity: List[int]
    discount: List[float]

#@app.post("/return_risk",tags=['Prediction'])
@app.api_route("/return_risk", methods=["GET", "POST"], tags=['Prediction'])
async def predict_return(applicant:Applicant):
        dt_model=tf.keras.models.load_model('deep_learning/HW/HW2/pair1/fullmodel.keras')
        input_data = pd.DataFrame({
            'order_id': [applicant.order_id],
            'product_id': [applicant.product_id],
            'unit_price': [applicant.unit_price],
            'quantity': [applicant.quantity],
            'discount': [applicant.discount]})
        # Explode the lists into separate rows
        exploded_df = input_data.explode(['product_id', 'unit_price', 'quantity', 'discount'])
        # Reset index if needed
        exploded_df = exploded_df.reset_index(drop=True)
        print(input_data)
        print(exploded_df)
        x_scaler=joblib.load('deep_learning/HW/HW2/pair1/scaler.joblib')
        exploded_df["min_disc"] = exploded_df.groupby("order_id")['discount'].transform('min')
        exploded_df=feature_eng(exploded_df)
        exploded_df=generate_return_labels(exploded_df)
        inputX=exploded_df.drop(columns=['order_id','return_label'])
        inputX_scaled = x_scaler.transform(inputX)
        orderY=dt_model.predict(inputX_scaled)
        prediction = (orderY > 0.5).astype(int)  # Convert to 0 or 1
        result = "High risk of return" if prediction==1 else "Low risk of return"
        X= pd.read_csv('deep_learning/HW/HW2/pair1/Xdata.csv')
        X_scaled=x_scaler.transform(X)
        Y= pd.read_csv('deep_learning/HW/HW2/pair1/Ydata.csv')

        explain=explain_order(dt_model, inputX_scaled, inputX.columns,X,X_scaled,Y,x_scaler)
        print(explain)

        return {
        "prediction":result,
        "details":{
            "order_id":applicant.order_id,
            "product_id":applicant.product_id,
            "unit_price":applicant.unit_price,
            "quantity":applicant.quantity,
            "discount":applicant.discount

        },
        "explain":explain
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 
