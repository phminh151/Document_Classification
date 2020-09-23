# Importing Libraries
from model.model import load_NB_model
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--demo_sentence', help="Query sentence for classification", dest= 'demo', default="Images/bicycle.jpg")
args = parser.parse_args()

# Loading model
model = load_NB_model()
# Vectorize words
from sklearn.feature_extraction.text import TfidfVectorizer
# Create TfidVectorizer
train = pd.read_csv('data\BBC News Train.csv')
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
train_set = tfidf.fit_transform(train.Text)
# Dictionary to translate
category_to_id = {'business':0, 'tech':1, 'politics':2, 'sport':3, 'entertainment':4}
id_to_category = {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}
# demo
class_id = model.predict(tfidf.transform([args.demo]))
print(id_to_category[class_id[0]])