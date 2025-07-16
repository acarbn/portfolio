# For good coding practices, we should use single-responsibility principle  & don't repeat yourself  

import re
import nltk
import random 
import nlpaug.augmenter.word as naw

random.seed(42) # global seed = 42

nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words('english')

def process_text(text):
    text=clean_text(text)
    text=remove_mentions(text)
    text=remove_stopwords(text)
    return text.strip() # remove leading and trailing spaces 

def clean_text(text):
    """General text cleaning function"""
    text=text.lower()
    text=re.sub(r"http\S+","",text)  # remove links - important for outsourced online data
    text=re.sub(r'\d+','',text) # remove numbers 
    text=re.sub(r'[^a-zA-Z\s]','',text) # remove special characters & punctuation  

    return text

def remove_mentions(text):
    """Remove mentions from text"""
    text=re.sub(r'@\S+','',text)
    return text

def remove_stopwords(text):
    """Remove stopwords from text"""
    text=" ".join([word for word in text.split() if word not in stopwords])
    return text 

##############################################################################################
# Data Augmentation - populate input data by using the same data but with different variations 
# 1. Synonym Replacement- This is [really/pretty/...] cool.
# 2. Random Insertion
# 3. Random Deletion
# 4. Random Swap
# EDA
# Back Translation

class AugProcessor:
    def __init__(self):
        self.synonym_aug = naw.SynonymAug(aug_src="wordnet")
        self.swap_aug = naw.RandomWordAug(action="swap")
        self.del_aug = naw.RandomWordAug(action="delete")
        #self.bert_insert = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="insert")
        #self.bert_insert = naw.ContextualWordEmbsAug(model_path='prajjwal1/bert-tiny', action="insert")
        #self.insert_aug=naw.ContextualWordEmbsAug(action="insert")
        #self.insert_aug = naw.RandomWordAug(action="insert")

    def augment_text(self, text):
        # Return multiple augmented strings as a list
        aug_lists = [
        self.synonym_aug.augment(text),
        self.swap_aug.augment(text),
        self.del_aug.augment(text)
        ]
        # flatten into a single list
        flat_aug = [item for sublist in aug_lists for item in (sublist if isinstance(sublist, list) else [sublist])]
        return flat_aug
    
 #self.insert_aug.augment(text)
