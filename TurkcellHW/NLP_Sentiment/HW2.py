import os
import kagglehub
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation, NMF

####   Loading Kaggle Dataset   ####

# Define a local path for where the dataset will live
dataset_dir = os.path.expanduser("~/.cache/kagglehub/datasets/rmisra/news-category-dataset/versions/3")

# Check if path exists — if not, download it
if not os.path.exists(dataset_dir):
    print("Dataset not found. Downloading...")
    path = kagglehub.dataset_download("rmisra/news-category-dataset")
else:
    print("Dataset already exists.")
    path = dataset_dir

# Now load the JSON file
json_file = os.path.join(path, "News_Category_Dataset_v3.json")
df = pd.read_json(json_file, lines=True)
df = df[['headline', 'category']]

print(df.head())
print("Number of headlines:", df.shape[0])

#### Downloading pre-processing models
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#### Pre-processing #####
## Tokenization
df['tokens'] = df['headline'].apply(word_tokenize)
print(df.head())

## Lower-case
df['tokens'] = df['tokens'].apply(lambda x: [w.lower() for w in x])
print(df.head())
def clean_tokens(tokens):
    # Remove weird symbols
    return [re.sub(r'[^\w\s]','',w) for w in tokens if re.match(r'^[a-zA-Z\-]+$', w)]
df['tokens'] = df['tokens'].apply(clean_tokens) 

## Stopword Removal
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [w for w in x if w not in stop_words and len(w) > 1])
print(df.head())
## ! using isalpha also removes "bird-watching" due to hyphen -

## Lemmatization
lemmatizer = WordNetLemmatizer()
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
print(df.head())

## POS-Tagging
df['pos_tags'] = df['tokens'].apply(nltk.pos_tag)
print(df.head())

def tokens_with_pos(tokens_pos):
    return [f"{word}_{pos}" for word, pos in tokens_pos]

df['tokens_pos'] = df['pos_tags'].apply(tokens_with_pos)
df['clean_text_pos'] = df['tokens_pos'].apply(lambda x: ' '.join(x))
print(df.head())

## Vectorization: to convert your text data into numerical features
# 1- CountVectorizer() : This method converts your text into a Bag-of-Words (BoW) representation.
#                        It creates a vocabulary of all unique tokens (words or tokens like "woman_NN")
#                        from your text.
#                        Then it counts how many times each token appears in each document (headline).
#                        The output is a sparse matrix of shape (num_documents, num_unique_tokens).
#                        Each row = one document (headline), each column = token count.

cv = CountVectorizer() # Bag-of-Words (BoW) sparse matrix representation
X_bow = cv.fit_transform(df['clean_text_pos'])
features = cv.get_feature_names_out()
#for f in features:
#    print(f)
print(features)
print(X_bow.toarray()[0])

tfidf = TfidfVectorizer()
X_tfidf= tfidf.fit_transform(df['clean_text_pos'])
features2 = tfidf.get_feature_names_out()
print(features2)
print(X_tfidf.toarray()[0])


print("BoW (CountVectorizer) - Word frequencies:")
for idx, count in zip(X_bow[0].indices, X_bow[0].data):
    print(f"{features[idx]}: {count}")

print("\nTF-IDF - Word weights:")
for idx, weight in zip(X_tfidf[0].indices, X_tfidf[0].data):
    print(f"{features2[idx]}: {weight:.4f}")

#### Sentiment
df['polarity'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

analyzer = SentimentIntensityAnalyzer()
df['vader_score'] = df['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['vader_sentiment'] = df['vader_score'].apply(lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral')
pd.set_option('display.max_colwidth', None)

print(df[['headline', 'polarity', 'sentiment', 'vader_score', 'vader_sentiment']].head())
df[['headline', 'polarity', 'sentiment', 'vader_score', 'vader_sentiment']].to_csv('/Users/burcinacar/Desktop/GYK/NLP/sentiment.csv')

# TextBlob sentiment counts
textblob_counts = df['sentiment'].value_counts()

# VADER sentiment counts
vader_counts = df['vader_sentiment'].value_counts()

print("TextBlob Sentiment Counts:")
print(textblob_counts)
print("\nVADER Sentiment Counts:")
print(vader_counts)

#### Topic Modelling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_tfidf)  # or X_tfidf for no POS

nmf = NMF(n_components=5, random_state=42)
nmf.fit(X_tfidf)

def print_topics(model, feature_names, n_top_words=10, method="Topic Model"):
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i].split('_')[0] for i in top_indices]
        print(f"\n📌 {method} Topic #{topic_idx + 1}")
        print("Top words:", ", ".join(top_words))

print_topics(lda, features2, method="LDA")
print_topics(nmf, features2, method="NMF")

df['nmf_topic'] = nmf.transform(X_tfidf).argmax(axis=1)
df['lda_topic'] = lda.transform(X_tfidf).argmax(axis=1)


for i in range(5):
    print(f"\n📌 Example headlines for Topic {i+1}:")
    print("\n".join(df[df['nmf_topic'] == i]['headline'].head(3).tolist()))

for i in range(5):
    print(f"\n📌 Example headlines for Topic {i+1}:")
    print("\n".join(df[df['lda_topic'] == i]['headline'].head(3).tolist()))
