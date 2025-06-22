# 📰 News Headlines NLP Analysis

This project demonstrates a complete NLP pipeline applied to a real-world dataset of news headlines. It includes text preprocessing, vectorization (Bag-of-Words and TF-IDF), sentiment analysis using two different methods (TextBlob and VADER), and topic modeling using LDA and NMF.

---

## 📦 Dataset

The dataset is sourced from the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) on Kaggle. It contains over 200,000 headlines from the HuffPost website, categorized by topics such as Politics, Entertainment, Health, etc.

---

## 🧼 Preprocessing Steps

* **Tokenization**: Headlines are tokenized using NLTK.
* **Lowercasing**: All tokens are converted to lowercase.
* **Cleaning**: Removes punctuation and non-alphabetic tokens.
* **Stopword Removal**: English stopwords are filtered out.
* **Lemmatization**: Words are reduced to their base form.
* **POS Tagging**: Each token is tagged with its part of speech.
* **POS-Appended Tokens**: Tokens are combined with their POS tags (e.g., `woman_NN`).

---

## 🔢 Vectorization

Two vectorization techniques are used:

### 1. CountVectorizer (Bag-of-Words)

* Converts text into token frequency counts.
* Does not distinguish importance of words.
* Example output:

```
million_cd: 1
american_jj: 1
roll_nn: 1
sleeve_nn: 1
omicrontargeted_vbn: 1
covid_jj: 1
booster_nn: 1
```

### 2. TfidfVectorizer (TF-IDF)

* Calculates the importance of words based on their frequency and rarity.
* Highlights unique and meaningful words in context.
* Example output:

```
million_cd: 0.2448
american_jj: 0.2200
roll_nn: 0.3208
sleeve_nn: 0.4562
omicrontargeted_vbn: 0.5000
covid_jj: 0.4103
booster_nn: 0.4027
```

---

## 😄 Sentiment Analysis

Two approaches are compared:

### 1. TextBlob

* Lexicon-based.
* Computes average polarity of words.
* Best suited for long text (not ideal for headlines).

### 2. VADER

* Lexicon + rule-based.
* Tailored for short texts, headlines, and social media.
* Better performance on detecting nuanced sentiment in news.

### 🎊 Sentiment Stats:

| Method   | Neutral | Positive | Negative |
| -------- | ------- | -------- | -------- |
| TextBlob | 109,583 | 68,210   | 31,734   |
| VADER    | 81,661  | 69,169   | 58,697   |

---

## 🧠 Topic Modeling

Two unsupervised methods are used to uncover latent topics in the headlines:

### 1. LDA (Latent Dirichlet Allocation)

* Probabilistic model using word-topic distributions.
* Assigns prior probabilities.

### 2. NMF (Non-negative Matrix Factorization)

* Decomposes document-term matrix into parts.
* Uses weights for topic-word relevance.

### 📍 Example Topics:

#### **LDA Topics**

* Topic 1: 🎥 Lifestyle & Media
* Topic 2: 🗳️ Politics & US Affairs
* Topic 3: 👨‍👩‍👧 Family & Women’s Topics
* Topic 4: 🍏 Health, Wellness & Research
* Topic 5: 🌍 Global News

#### **NMF Topics**

* Topic 1: 👗 Lifestyle, Fashion & Home
* Topic 2: 🗳️ US Politics & Campaigns
* Topic 3: 🎉 New Year & Events
* Topic 4: 👨‍👧 Everyday Life & Parenting
* Topic 5: 😂 Humor, Social Media & Identity

---

## 📁 Output Example

```
TF-IDF - Word weights:
omicrontargeted_vbn: 0.5000
sleeve_nn: 0.4562
covid_jj: 0.4103
booster_nn: 0.4027
roll_nn: 0.3208
```

---

## 💾 Code Structure

* `data_loading`: Downloads and loads the Kaggle dataset.
* `preprocessing`: Applies tokenization, lemmatization, cleaning, and POS tagging.
* `vectorization`: Creates Count and TF-IDF representations.
* `sentiment_analysis`: Applies TextBlob and VADER.
* `topic_modeling`: Extracts topics using LDA and NMF.
* `outputs`: Prints top words and example headlines for each topic.
* `sentiment.csv`: Export of sentiment scores per headline.

---

## 📄 Requirements

* Python 3.8+
* pandas
* nltk
* scikit-learn
* textblob
* vaderSentiment
* kagglehub

Install via:

```bash
pip install -r requirements.txt
```

---

## 🧐 Author

**Burçin Acar**
PhD in Protein Dynamics, working on NLP and machine learning applications in scientific domains.

---

## 📩 Feedback

Feel free to open issues or contribute with pull requests! ⭐

