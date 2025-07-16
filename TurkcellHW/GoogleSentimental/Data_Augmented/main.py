### LOADING INPUT DATA ###############################################################################
import os
import pandas as pd
import requests
from io import StringIO
from tabulate import tabulate
from text_processor import process_text,AugProcessor

save_dir = "google_sentimentHW_dataaug/data/"
merged_filename = "goemotions_merged.csv"
merged_path = os.path.join(save_dir, merged_filename)

if os.path.exists(merged_path):
    print(f"Merged file already exists at {merged_path}, skipping download.")
    print_df = pd.read_csv(merged_path, encoding='latin-1')
else:
    print("Merged file not found. Downloading and merging...")
    os.makedirs(save_dir, exist_ok=True)
    urls = [
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"
    ]
    dfs = []
    for url in urls:
        print(f"Fetching {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), encoding='latin-1')
            dfs.append(df)
            print(f"Loaded {url}")
        else:
            print(f"Failed to download {url}")
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        print_df = merged_df
        merged_df.to_csv(merged_path, index=False)
        print(f"Merged dataset saved to {merged_path}")
    else:
        print("No files downloaded. Merged file not created.")

df = print_df.drop(print_df.columns[1:9], axis=1)

### PREPROCESSING #####################################################################################
import re
import string
from ftfy import fix_text

df['text'] = df['text'].apply(fix_text)

df['clean_text'] = df['text'].apply(process_text)
df.drop(['text'], axis=1, inplace=True)
aug_processor = AugProcessor()
print(df['clean_text'].iloc[0])
print(aug_processor.augment_text(df['clean_text'].iloc[0]))

df['aug_text']= df['clean_text'].apply(aug_processor.augment_text)
#print(df.head(10))
df_aug=df.explode('aug_text',ignore_index=True)
#print(df_aug.head(10))
df_aug=df_aug.drop(columns=['clean_text'])
df_aug=df_aug.rename(columns={'aug_text':'clean_text'})
df=df.drop(columns=['aug_text'])

print(df_aug.head(10))
print(df.head(10))
df_merged = pd.concat([df, df_aug], ignore_index=True)
df_merged = df_merged.dropna(subset=['clean_text'])


### SPLIT DATA #########################################################################################
from sklearn.model_selection import train_test_split

X = df_merged['clean_text']
y = df_merged.drop(columns=['clean_text'])
print(df.shape)
print(df_aug.shape)
print(df_merged.shape)
df_merged.to_csv("google_sentimentHW_dataaug/data/augm.csv",index=False)

print(df_merged.head(10))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test)
X_test.to_csv("google_sentimentHW_dataaug/data/Xtest.csv", index=False)
y_test.to_csv("google_sentimentHW_dataaug/data/ytest.csv", index=False)
X_train.to_csv("google_sentimentHW_dataaug/data/Xtrain.csv", index=False)

print("STOPPPPPPPP")
### TOKENIZE ###########################################################################################
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

### PADDING ############################################################################################
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

seq_lengths = [len(seq) for seq in X_train_seq]
max_length = np.max(seq_lengths)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

### DOWNLOAD AND EXTRACT GLOVE #########################################################################
import zipfile
from tqdm import tqdm

glove_dir = "google_sentimentHW_dataaug/"
glove_zip_path = os.path.join(glove_dir, "glove.6B.zip")
glove_path = os.path.join(glove_dir, "glove.6B.100d.txt")

if not os.path.exists(glove_path):
    print("Downloading GloVe embeddings...")
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    os.makedirs(glove_dir, exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    with open(glove_zip_path, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading GloVe", unit='KB'):
            if chunk:
                f.write(chunk)

    print("Extracting GloVe embeddings...")
    with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
        zip_ref.extractall(glove_dir)

else:
    print("GloVe embeddings already downloaded.")

### LOAD GLOVE EMBEDDINGS #############################################################################
embedding_index = {}
embedding_dim = 100

print("Loading GloVe embeddings into memory...")
with open(glove_path, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeffs

print(f"Loaded {len(embedding_index)} word vectors from GloVe.")

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in tokenizer.word_index.items():
    if idx < vocab_size:
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector

### MODEL BUILDING #####################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall

model = Sequential()
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_length,
    trainable=True,
    mask_zero=True
))
model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))))
model.add(Dropout(0.5))
model.add(Dense(28, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[Precision(name='precision'), Recall(name='recall')]
)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=100,
    validation_data=(X_test_pad, y_test),
    callbacks=[early_stopping]
)

### SAVE MODEL AND TOKENIZER ##########################################################################
model.save("google_sentimentHW_dataaug/goSent_model_glove.h5")

import pickle
with open('google_sentimentHW_dataaug//tokenizer_glove.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

### EVALUATE ###########################################################################################    
from sklearn.metrics import classification_report

y_pred = model.predict(X_test_pad)
y_pred_binary = (y_pred > 0.3).astype(int)
target_names1 = y_test.columns.tolist()
print(classification_report(y_test, y_pred_binary, target_names=target_names1))

from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score

print("LRAP:", label_ranking_average_precision_score(y_test, y_pred))
print("Macro ROC-AUC:", roc_auc_score(y_test, y_pred, average='macro'))
