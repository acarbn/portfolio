import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, classification_report, label_ranking_average_precision_score, roc_auc_score

# --- Load tokenizer ---
with open('google_sentimentHW/tokenizer_glove.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- Load model ---
model = load_model('google_sentimentHW/goSent_model_glove.h5')

# --- Load data ---
df = pd.read_csv("google_sentimentHW/data/goemotions_merged.csv", encoding='latin-1')
df = df.drop(df.columns[1:9], axis=1)

# Preprocess text
import re, string
from ftfy import fix_text

df['text'] = df['text'].apply(fix_text)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['clean_text'] = df['text'].apply(clean_text)
df.drop(['text'], axis=1, inplace=True)

# Split data
from sklearn.model_selection import train_test_split
X = df['clean_text']
y = df.drop(columns=['clean_text'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad
X_test_seq = tokenizer.texts_to_sequences(X_test)
max_length = max([len(seq) for seq in tokenizer.texts_to_sequences(X_train)])
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# --- Predict probabilities ---
y_pred_proba = model.predict(X_test_pad)
target_names = y_test.columns.tolist()

# --- Find optimal thresholds per label ---
def find_best_thresholds(y_true, y_pred_proba, metric=f1_score, average='binary'):
    thresholds = []
    for i in range(y_true.shape[1]):
        best_thresh = 0.5
        best_score = 0
        for t in np.arange(0.05, 0.95, 0.01):
            y_pred_bin = (y_pred_proba[:, i] >= t).astype(int)
            score = metric(y_true.iloc[:, i], y_pred_bin, average=average, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = t
        thresholds.append(best_thresh)
    return np.array(thresholds)

optimal_thresholds = find_best_thresholds(y_test, y_pred_proba)

# --- Apply optimal thresholds ---
y_pred_optimized = (y_pred_proba >= optimal_thresholds).astype(int)

# --- Evaluation ---
print("\n🧾 Classification Report with Optimized Thresholds:\n")
print(classification_report(y_test, y_pred_optimized, target_names=target_names))

print("📈 LRAP:", label_ranking_average_precision_score(y_test, y_pred_proba))
print("📉 Macro ROC-AUC:", roc_auc_score(y_test, y_pred_proba, average='macro'))

# --- Visualize thresholds ---
plt.figure(figsize=(12, 6))
plt.bar(range(len(optimal_thresholds)), optimal_thresholds)
plt.xticks(range(len(optimal_thresholds)), target_names, rotation=90)
plt.ylabel('Optimal Threshold')
plt.title('Per-label Optimal Thresholds')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- Ensure y_test is in array format ---
y_true = y_test.values
n_classes = y_true.shape[1]

# --- Plotting ROC Curves for all labels ---
plt.figure(figsize=(16, 12))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{target_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves per Emotion Label', fontsize=16)
plt.legend(loc='lower right', fontsize=10, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import hamming_loss
hl = hamming_loss(y_test, y_pred_optimized)
print("Hamming Loss:", hl)

from sklearn.metrics import accuracy_score
subset_acc = accuracy_score(y_test, y_pred_optimized)
print("Subset Accuracy:", subset_acc)

from sklearn.metrics import coverage_error, label_ranking_loss

print("Coverage Error:", coverage_error(y_test, y_pred_proba))
print("Label Ranking Loss:", label_ranking_loss(y_test, y_pred_proba))
