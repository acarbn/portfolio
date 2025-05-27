# Job Application Outcome Prediction Model
- Burçin Acar
- 16.04.25

This project includes a SVC model that predicts whether job applicants will be accepted (0) or rejected (1) using Faker generated data for years of programming experience and technical score, along with a FastAPI-based web service that serves this model. 

## Project Description

This system predicts the outcome of a job application in the future by analyzing Faker-generated data. The model is trained using an SVM classifier and leverages the following features:

- Years of programming experience (experience_year)
- Technical exam score (tech_score)

## Requirements

To run this project, you'll need the following components:

- Python 3.8 or higher
- The following Python packages:
  - fastapi
  - uvicorn
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - matplotlib
  - Faker 
  - pydantic

## Installation

1. Unzip the singleHW1 folder.

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Training the Model

1. Run the script to generate data, train and save the model:

```bash
python SVCmodel.py
```

The training process:
- Creates applicant data using Faker
- Standardizes the data
- Trains and saves the SVC model with optimized parameters
- Evaluates model performance and creates visualizations

## Running the API Service

To make the model accessible via API:

```bash
uvicorn main:app --reload
```
The service will run at http://localhost:8000 by default.

## API Reference

Once the API is running, you can access the swagger API documentation at: http://localhost:8000/docs

### Main Endpoints

- `GET /`: Check API status
- `GET /data`: List all applicant data
- `POST /predict`: Makes a prediction for a new applicant

### Example Prediction Request

```json
{
  "experience_year": 40,
  "tech_score": 100
}
```
## Docker image
The following linux-based Docker image can be used to reach the API:
  Docker Container : https://hub.docker.com/r/brcnacar/jobsvc3 

## API URL with Render:
The API is deployed on Render and can be reached with the following URL:
  https://burcinacar-job-outcome.onrender.com/docs 

## Model Performance

The best SVC model with linear kernel ('C': 100) has shown the following performance :

- Best cross-validation accuracy for ['linear'] kernel: 0.9437
- Classification Report for ['linear'] kernel:
              precision    recall  f1-score   support

           0       0.82      0.82      0.82        11
           1       0.93      0.93      0.93        29

    accuracy                           0.90        40
   macro avg       0.87      0.87      0.87        40
weighted avg       0.90      0.90      0.90        40

- Confusion Matrix:
  [[ 9  2]
  [ 2 27]]
- Test accuracy: 0.9000

Models with other kernels resulted scores and figures given under SVCmodelResults folder.
