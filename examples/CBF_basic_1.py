import numpy as np
import scipy
import pandas as pd
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

recipes = pd.read_csv('data/recipes.csv')
interactions_train = pd.read_csv('data/interactions_train.csv')
interactions_test = pd.read_csv('data/interactions_test.csv')

n = 5
users_interactions_count = interactions_train.groupby(['user_id', 'item_id']).size().groupby('user_id').size()
print(f"Number of users: {len(users_interactions_count)}")
users_with_enough_interactions = users_interactions_count[users_interactions_count >= n].reset_index()[['user_id']]
print(f"Number of users with at least {n} interactions: {len(users_with_enough_interactions)}")

print(f"Number of  of interactions: {len(interactions_train)}")
interactions_from_selected_users = interactions_train.merge(users_with_enough_interactions, how = 'right', left_on = 'user_id', right_on = 'user_id')
print(f"Number of interactions from users with at least n interactions:{len(interactions_from_selected_users)}")

interactions_full = interactions_from_selected_users.groupby(['user_id', 'item_id'])['rating'].sum().apply(lambda x: np.log10(x+1)*2).reset_index()

print(f"Number of unique user/item interactions: {len(interactions_full)}")

interactions_train, interactions_test = train_test_split(interactions_full, stratify=interactions_full['user_id'], 
                                                               test_size=0.2, random_state=666)

vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords.words('english'))

item_ids = recipes['item_id'].tolist()
vectorizer_input = recipes['name'] + " " + recipes['description']
tfidf_matrix = vectorizer.fit_transform(vectorizer_input)

def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    try:
        item_profiles_list = [get_item_profile(x) for x in ids]
    except:
        item_profiles_list = [get_item_profile(x) for x in [ids]]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(user_id, interactions_indexed):
    interactions_person = interactions_indexed.loc[user_id]
    user_item_profiles = get_item_profiles(interactions_person['item_id'])
    
    user_item_strengths = np.array(interactions_person['rating']).reshape(-1,1)
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed = interactions_train[interactions_train['item_id'] \
                                                   .isin(recipes['item_id'])].set_index('user_id')
    user_profiles = {}
    for user_id in interactions_indexed.index.unique():
        user_profiles[user_id] = build_users_profile(user_id, interactions_indexed)
    return user_profiles


# interactions_train[interactions_train['item_id'].isin(recipes['item_id'])].set_index('user_id').index.unique()
user_profiles = build_users_profiles()


class ContentBasedRecommender:
    def __init__(self, items=None):
        pass
                
    def _get_similar_items_to_user_profile(self, user_id, topn=1000):
        cosine_similarities = cosine_similarity(user_profiles[user_id], tfidf_matrix)
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations = pd.DataFrame(similar_items_filtered, columns=['item_id', 'recStrength']).head(topn)

        return recommendations
    
content_based_recommender_model = ContentBasedRecommender(recipes)


recommendations = content_based_recommender_model.recommend_items(599450)
recommendations = pd.merge(recommendations, recipes[['name', 'item_id']], how='left', on='item_id')
print(recommendations)