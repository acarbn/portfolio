from faker import Faker
import numpy as np
from sklearn.svm import SVC     # CLASSIFIER
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from joblib import dump
from utilsmodel import plot_decision_boundary, predict_outcome
import json

path="singleHW/api/"


output_file = open("singleHW/model_evaluation_results.txt", "a")  # open a file in append mode

fake=Faker()
Faker.seed(42)  # or any integer


experiences_years = []
tech_scores=[]
labels=[]
n_samples=200

for _ in range(n_samples):
    experience_year=fake.pyfloat(min_value=0, max_value=10)                          # 0–10 years
    tech_score=fake.pyfloat(min_value=0, max_value=100)
    label=0 if experience_year>2 and tech_score>60 else 1 # 
    experiences_years.append(experience_year)
    tech_scores.append(tech_score)
    labels.append(label)

X=np.column_stack((experiences_years,tech_scores))
y=np.array(labels)

# Convert to native Python types
labeled_data = []
for i in range(len(X)):
    entry = {
        "experience_year": float(X[i][0]),
        "tech_score": float(X[i][1]),
        "label": int(y[i])
    }
    labeled_data.append(entry)

# Save to JSON
with open(path+"data/labeled_data.json", "w") as json_file:
    json.dump(labeled_data, json_file, indent=4)




scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
dump(scaler, 'singleHW/api/model/scaler.pkl')

X_train, X_test, y_train, y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)
svc = SVC()

param_grid1 = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]},
    {'kernel': ['poly'], 'C': [0.1, 1, 10], 'gamma': ['scale'], 'degree': [2, 3, 4]},
    {'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
]

for gridx in param_grid1:
    grid_search = GridSearchCV(svc, gridx, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)


    # Replace your model with the best one found
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    y_pred = best_model.predict(X_test)

    if gridx['kernel'][0]=="linear":
        best_model_linear=best_model

    output_file.write(f"Best parameters for {gridx['kernel']} kernel: {grid_search.best_params_}\n")
    output_file.write(f"Best cross-validation accuracy for {gridx['kernel']} kernel: {grid_search.best_score_:.4f}\n")

    # Classification and confusion matrix
    output_file.write(f"Classification Report for {gridx['kernel']} kernel:\n{classification_report(y_test, y_pred)}\n")
    output_file.write(f"Confusion Matrix for {gridx['kernel']} kernel:\n{confusion_matrix(y_test, y_pred)}\n")
    output_file.write(f"Test accuracy with best model for {gridx['kernel']} kernel: {accuracy:.4f}\n\n")

    kernel=gridx['kernel'][0]
## Non-linear
    plot_decision_boundary(kernel,best_model, X_scaled, y)

output_file.close()
# Fit the best model with the entire dataset
model=best_model_linear.fit(X_scaled, y)
dump(model, 'singleHW/api/model/linearSVCmodel.joblib')  # or 'model.pkl'
# You can now use this model to make predictions on new data or evaluate it

example=np.array([5.14467032,94.1687]).reshape(1,-1)
example1=np.array([ 5.11201029, 55.25639745]).reshape(1,-1)
example2=np.array( [ 2.22204911, 78.48      ]).reshape(1,-1)
example3=np.array([ 6.9048,     58.39599   ]).reshape(1,-1)
example=np.array([15,40]).reshape(1,-1)

print(example)
print(predict_outcome(example))
print(example2)
print(predict_outcome(example2))
