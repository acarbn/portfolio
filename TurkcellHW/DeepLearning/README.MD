# Order Return Prediction API

A FastAPI-based service that predicts the likelihood of an order being returned by a customer using a deep learning model.

## Features
- Predicts return risk for customer orders
- Provides explanations for predictions
- RESTful API endpoints for easy integration
- Built with FastAPI for high performance

## Requirements
- Python 3.7+
- Required packages (see `requirements.txt`):
  - fastapi
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - joblib
  - uvicorn

## Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the API server:
   ```bash
   python main.py
   ```

2. The API will be available at `http://127.0.0.1:8000`

## API Endpoints

### GET /
- Welcome message
- **Response**: 
  ```json
  {"message": "Welcome to Our API!"}
  ```

### GET /orders
- Retrieves all order data
- **Response**: List of all orders in the dataset

### POST /return_risk
- Predicts return risk for a given order
- **Request Body**:
  ```json
  {
    "order_id": 12345,
    "product_id": [1, 2, 3],
    "unit_price": [10.99, 5.99, 7.50],
    "quantity": [2, 1, 3],
    "discount": [0.1, 0.0, 0.15]
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "High risk of return",
    "details": {
      "order_id": 12345,
      "product_id": [1, 2, 3],
      "unit_price": [10.99, 5.99, 7.50],
      "quantity": [2, 1, 3],
      "discount": [0.1, 0.0, 0.15]
    },
    "explain": "Explanation of prediction..."
  }
  ```

## Model Details
- Uses a TensorFlow deep learning model (`fullmodel.keras`)
- Features are preprocessed using a saved scaler (`scaler.joblib`)
- Includes feature engineering and label generation utilities

## Files
- `main.py`: FastAPI application and endpoints
- `utils.py`: Helper functions for feature engineering and explanations
- `fullmodel.keras`: Trained TensorFlow model
- `scaler.joblib`: Feature scaler
- `combined_originaldata.csv`: Sample order data
- `Xdata.csv`, `Ydata.csv`: Training data

## Extra
- model.py create, compile, train, test, validate and tune the deep learning model. This code creates fullmodel.keras, scaler.joblob and csv files.

