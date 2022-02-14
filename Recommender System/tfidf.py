import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np


interactions_train = pd.read_csv("data/interactions_train.csv")
recipes = pd.read_csv("data/recipes.csv")

interactions_full = interactions_train.groupby(['user_id', 'item_id'])['rating'].sum().apply(lambda x: np.log10(x+1)*2).reset_index()

vectorizer = TfidfVectorizer(analyzer='word',
                    ngram_range=(1, 2),
                    min_df=0.003,
                    max_df=0.5,
                    max_features=5000,
                    stop_words=stopwords.words('english'))

item_ids = recipes['item_id'].tolist()
vectorizer_input = recipes['name'] + " " + recipes['description'] # TODO add other columns to this
tfidf_matrix = vectorizer.fit_transform(vectorizer_input)

np.save('tfidf.npy', tfidf_matrix, allow_pickle=True, fix_imports=True)

print(tfidf_matrix.shape)