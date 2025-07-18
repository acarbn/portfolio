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
* Example output for “Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters”:

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
* Example output “Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters”:

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
* Example: “Climate Poses Many Threats To U.S. Financial System & Natural Gas May Be Major Risk” : 0.165625, positive

### 2. VADER

* Lexicon + rule-based.
* Tailored for short texts, headlines, and social media.
* Better performance on detecting nuanced sentiment in news.
* Example: “Climate Poses Many Threats To U.S. Financial System & Natural Gas May Be Major Risk”: -0.4005, negative

### 🎊 Sentiment Stats:

| Method   | Neutral | Positive | Negative |
| -------- | ------- | -------- | -------- |
| TextBlob | 109,583 | 68,210   | 31,734   |
| VADER    | 81,661  | 69,169   | 58,697   |

* Conclusion: VADER method distinguished news with negative sentiments better. 
---

## 🧠 Topic Modeling

Two unsupervised methods are used to uncover latent topics in the headlines:

### 1. LDA (Latent Dirichlet Allocation)

* Probabilistic model using word-topic distributions.
* Assigns prior probabilities.

#### **Example Topics**

Topic 1: 🎥 Lifestyle & Media: photo, video, new, way, food, best, day, life, time, get

* Reporter Gets Adorable Surprise From Her Boyfriend While Live On TV
* Fiona Threatens To Become Category 4 Storm Headed To Bermuda
* Golden Globes Returning To NBC In January After Year Off-Air

Topic 2: 🗳️ Politics & US Affairs: trump, donald, clinton, hillary, court, say


* How A New Documentary Captures The Complexity Of Being A Child Of Immigrants
* Man Sets Himself On Fire In Apparent Protest Of Funeral For Japan's Abe
* 'Reboot' Is A Clever And Not Too Navel-Gazey Look Inside TV Reboots

Topic 3: 👨‍👩‍👧 Family & Women’s Topics: photo, woman, week, mom, parent, life

* 23 Of The Funniest Tweets About Cats And Dogs This Week (Sept. 17-23)
* The Funniest Tweets From Parents This Week (Sept. 17-23)
* Woman Who Called Cops On Black Bird-Watcher Loses Lawsuit Against Ex-Employer

Topic 4: 🍏 Health, Wellness & Research: photo, way, video, woman, study, healthy, make


* Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters
* American Airlines Flyer Charged, Banned For Life After Punching Flight Attendant On Video
* Russian Cosmonaut Valery Polyakov Who Broke Record With 437-Day Stay In Space Dies At 80

Topic 5: 🌍 Global News: trump, year, world, news, woman


* Biden At UN To Call Russian War An Affront To Body's Charter
* World Cup Captains Want To Wear Rainbow Armbands In Qatar
* Virginia Thomas Agrees To Interview With Jan. 6 Panel

### 2. NMF (Non-negative Matrix Factorization)

* Decomposes document-term matrix into parts.
* Uses weights for topic-word relevance.

#### **Example Topics**

Topic 1: 👗 Lifestyle, Fashion & Home


* 'Reboot' Is A Clever And Not Too Navel-Gazey Look Inside TV Reboots
* German Customs Officials Follow Trail Of Slime And Find Bags Of Giant Snails
* Watch Harry Styles Pay Tribute To Queen Elizabeth II At NYC Concert

Topic 2: 🗳️ US Politics & Campaigns


* Biden At UN To Call Russian War An Affront To Body's Charter
* Fiona Threatens To Become Category 4 Storm Headed To Bermuda
* Twitch Bans Gambling Sites After Streamer Scams Folks Out Of $200,000

Topic 3: 🎉 New Year & Events


* How A New Documentary Captures The Complexity Of Being A Child Of Immigrants
* Golden Globes Returning To NBC In January After Year Off-Air
* ‘Beautiful And Sad At The Same Time’: Ukrainian Cultural Festival Takes On A Deeper Meaning This Year

Topic 4: 👨‍👧 Everyday Life & Parenting


* Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters
* American Airlines Flyer Charged, Banned For Life After Punching Flight Attendant On Video
* Cleaner Was Dead In Belk Bathroom For 4 Days Before Body Found: Police

Topic 5: 😂 Humor, Social Media & Identity


* 23 Of The Funniest Tweets About Cats And Dogs This Week (Sept. 17-23)
* The Funniest Tweets From Parents This Week (Sept. 17-23)
* Woman Who Called Cops On Black Bird-Watcher Loses Lawsuit Against Ex-Employer

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

---

## 🧐 Author

**Burçin Acar**
PhD in Protein Dynamics Modeling, working on NLP and machine learning applications in scientific domains.

---

## 📩 Feedback

Feel free to open issues or contribute with pull requests! ⭐

