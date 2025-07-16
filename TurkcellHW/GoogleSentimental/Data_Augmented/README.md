# GoEmotions Sentiment Classification with Data Augmentation and GloVe Embeddings

This project demonstrates a complete machine learning pipeline for multi-label emotion classification using the [GoEmotions dataset](https://github.com/google-research/goemotions). The workflow includes data preprocessing, text cleaning, data augmentation, word embedding with GloVe, and training a Bidirectional LSTM neural network using TensorFlow/Keras.

---

## 🔍 Dataset

The GoEmotions dataset consists of over 58k Reddit comments labeled with 27 emotion categories + neutral. This script fetches and merges the 3 provided CSVs into one unified dataset.

---

## 🧼 Text Preprocessing

- Lowercasing
- Removing URLs, numbers, special characters, and user mentions
- Removing English stopwords (using `nltk`)
- Optional: Fixing Unicode text issues with `ftfy`

---

## 🔁 Data Augmentation

Implemented with `nlpaug`:
- **Synonym Replacement** (WordNet-based)
- **Random Word Swap**
- **Random Word Deletion**

Each clean text sample is expanded to multiple augmented versions, increasing data diversity for model robustness.

---

## 🔤 Tokenization & Padding

- Tokenized using Keras' `Tokenizer`
- Vocabulary size calculated based on training data
- Sequences are padded to maximum length for batch consistency

---

## 💬 Word Embeddings

- Downloads [GloVe 6B (100d)](https://nlp.stanford.edu/projects/glove/) vectors
- Embedding matrix is initialized using these pretrained vectors
- Used as the input layer in the Keras model

---

## 🧠 Model Architecture

```text
- Embedding (GloVe) → BiLSTM → Dropout → Dense (sigmoid)
- Multi-label classification (28 outputs, one per label)
- Loss: binary_crossentropy 
- Optimizer: adam
- Metrics: Precision and Recall
- Includes early stopping on validation loss
```

---

## 📊 Evaluation
```text
Classification Report (Precision, Recall, F1-score per class)
LRAP (Label Ranking Average Precision)
Macro-averaged ROC-AUC
```

---

## ✅ Outputs
```text
- goSent_model_glove.h5 — Trained Keras model
- tokenizer_glove.pkl — Fitted tokenizer for future inference
```

## 📦 Dependencies
```bash
pip install pandas requests ftfy nltk nlpaug scikit-learn tqdm tensorflow
```
Note: Ensure to download nltk stopwords when running for the first time:

```python
import nltk
nltk.download('stopwords')
```

## 🚀 Run the Project
```bash
python main.py
```

## 🧑‍💻 Author
Burçin Acar


