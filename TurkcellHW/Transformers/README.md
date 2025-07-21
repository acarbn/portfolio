# News Summarization with Fine-Tuned BART

This project demonstrates how to use a fine-tuned BART model from Hugging Face Transformers to generate summaries from raw news text. The summarization pipeline includes preprocessing with custom text cleaning utilities and optional data augmentation.

## Contents

* `predict.py`: Loads the model and tokenizer, processes a list of news articles, generates summaries, and writes the full articles with their summaries to an output text file.
* `text_processor.py`: Defines the text preprocessing pipeline, including cleaning, stopword removal, and optional data augmentation using `nlpaug`.
* `main.py`: Training script for the BART model (uploaded separately).

---

## 📁 Directory Structure

```
news_summary/
├── saved_model/  # Fine-tuned model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── ...
├── predict.py
├── text_processor.py
├── main.py  # (uploaded)
└── summaries_output.txt  # Output file with summaries
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install torch transformers nltk nlpaug
```

### 2. Download NLTK Resources

```python
import nltk
nltk.download('stopwords')
```

### 3. Prepare Model and Tokenizer

Ensure the trained model is placed in `news_summary/saved_model/`.

### 4. Run Prediction

```bash
python predict.py
```

The script will:

* Load and preprocess sample articles.
* Generate summaries using beam search.
* Write the original and summarized text to `summaries_output.txt`.

---

## 🧹 Text Preprocessing

`text_processor.py` performs:

* Lowercasing
* Removal of URLs, numbers, punctuation, mentions
* Stopword filtering
* (Optional) Data augmentation using:

  * Synonym replacement
  * Word swapping
  * Word deletion

You can extend the `AugProcessor` class to include contextual insertions using BERT or other models.

---

## 🧠 Model Details

* Model: `facebook/bart-base` (fine-tuned)
* Training Epochs: 6
* Max Input Length: 512 tokens
* Max Output Length: 150 tokens
* Beam Search: `num_beams=4`, `length_penalty=2.0`, `early_stopping=True`
* Prototype model was trained on 5000 instances, validated and tested on two set of 100 instances
---

## ✍️ Example Output

```
--- Full Article 1 ---
(gen x is having...)

--- Summary 1 ---
Gen X faces societal pressure as they age, leading some to turn to extreme weight loss measures.
```

---

## 📌 Notes

* Ensure your GPU is enabled for faster generation.
* Modify the `texts` list in `predict.py` to test your own inputs.
* This repo is a minimal inference/demo pipeline; for fine-tuning code, refer to `main.py`.

---

## 🛠️ TODO

* Add CLI arguments for batch summarization.
* Integrate web UI (e.g., Streamlit) for interactive summarization.
* Enhance augmentation strategies.

---

## 📜 License

MIT License

---

## 👤 Author

Burcin Acar
