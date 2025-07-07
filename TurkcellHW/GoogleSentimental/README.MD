# GoEmotions Multi-Label Emotion Classification with GloVe Embeddings and BiLSTM

This repository contains code for training and evaluating a multi-label deep learning model to classify text into 28 emotion categories from the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions). The model uses pre-trained GloVe embeddings and a Bidirectional LSTM architecture implemented with TensorFlow/Keras.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Data Preprocessing](#data-preprocessing)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Usage](#usage)  
- [Results](#results)  
- [Dependencies](#dependencies)  
- [License](#license)  

---

## Project Overview

This project tackles multi-label emotion classification on text inputs, predicting which emotions apply to a given sentence. The model uses:

- The GoEmotions dataset containing Reddit comments labeled with 28 emotion categories.
- Pre-trained GloVe 100-dimensional word embeddings.
- A Bidirectional LSTM network with L2 regularization and dropout.
- Optimal threshold tuning per emotion label to improve classification metrics.

---

## Dataset

- The original GoEmotions dataset is split across three CSV files.  
- The scripts download and merge these files automatically (if not already present).  
- Labels are one-hot encoded 28-dimensional vectors representing emotions such as admiration, amusement, anger, etc.

---

## Data Preprocessing

- Text is cleaned by fixing encoding issues (`ftfy`), lowercasing, removing digits, and stripping punctuation.  
- Text is tokenized using `keras.preprocessing.text.Tokenizer` with an out-of-vocabulary token.  
- Sequences are padded to the max sequence length in the training set.  
- The dataset is split into training and testing sets (80% train, 20% test).

---

## Model Architecture

- Embedding layer initialized with GloVe 100-dimensional vectors (trainable).  
- Bidirectional LSTM layer with 64 units and L2 kernel regularization.  
- Dropout layer (rate=0.5) for regularization.  
- Output layer: Dense with 28 sigmoid units for multi-label prediction.  
- Loss function: Binary crossentropy.  
- Optimizer: Adam.  
- Metrics: Precision and Recall.

---

## Training

- Early stopping on validation loss with patience of 5 epochs.  
- Maximum 100 epochs (usually stopped earlier by early stopping).  
- Batch training on padded sequences.

---

## Evaluation

- Predictions output probabilities per label.  
- Optimal thresholds per label found by maximizing F1 score.  
- Metrics reported: Classification report, Label Ranking Average Precision (LRAP), Macro ROC-AUC, Hamming Loss, Subset Accuracy, Coverage Error, Label Ranking Loss.  
- ROC curves plotted for each label.  
- Thresholds visualization per label.

---

## Usage

1. Clone the repository.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the main script to download data, preprocess, train, and save the model:

   ```bash
   python train_model.py

4. Run the evaluation script to load the model and tokenizer, run predictions, and visualize results:

   ```bash
   python evaluate_model.py

## Results
- Achieved reasonable precision and recall across multiple emotion categories.
- Optimizing thresholds per label improved classification F1 scores.
- Visualizations of ROC curves and optimal thresholds per emotion help interpret model performance.
- All relevant results can be found under Results folder. 

## Dependencies
- Python 3.7+
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- ftfy
- tqdm
- requests

## License
This project is licensed under the MIT License.

## Author 
Burcin Acar

## Date
2025-07-07
