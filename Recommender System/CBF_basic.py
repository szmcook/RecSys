import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import scipy
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

np.seterr(all='raise')

class ContentRecommender:
    def __init__(self, active_user_id):
        self.type = 'Content Based Filter'
        self.recipes, self.interactions = self.load_data()
        self.active_user_id = active_user_id
        self.clean_data()
        
    def load_data(self):
        recipes = pd.read_csv('data/clean_recipes.csv')
        interactions = pd.read_csv('data/clean_interactions.csv')
        return recipes, interactions

    def clean_data(self):
        n = 5
        users_interactions_count = self.interactions.groupby(['user_id', 'item_id']).size().groupby('user_id').size()
        # print(f"Number of users: {len(users_interactions_count)}")
        users_with_enough_interactions = users_interactions_count[users_interactions_count >= n].reset_index()[['user_id']]
        # print(f"Number of users with at least {n} interactions: {len(users_with_enough_interactions)}")
        # print(f"Number of interactions: {len(self.interactions)}")
        interactions_from_selected_users = self.interactions.merge(users_with_enough_interactions, how = 'right', left_on = 'user_id', right_on = 'user_id')
        # print(f"Number of interactions from users with at least n interactions:{len(interactions_from_selected_users)}")

        interactions_full = interactions_from_selected_users.groupby(['user_id', 'item_id'])['rating'].sum().apply(lambda x: np.log10(x+1)*2).reset_index()

        interactions_train, interactions_test = train_test_split(interactions_full, stratify=interactions_full['user_id'], 
                                                                    test_size=0.2, random_state=666)

        vectorizer = TfidfVectorizer(analyzer='word',
                            ngram_range=(1, 2),
                            min_df=0.003,
                            max_df=0.5,
                            max_features=5000,
                            stop_words=stopwords.words('english'))

        self.item_ids = self.recipes['item_id'].tolist()
        vectorizer_input = self.recipes['name'] + " " + self.recipes['description']
        self.tfidf_matrix = vectorizer.fit_transform(vectorizer_input)

        def get_item_profile(item_id):
            idx = self.item_ids.index(item_id)
            item_profile = self.tfidf_matrix[idx:idx+1]
            return item_profile

        def get_item_profiles(ids):
            try:
                item_profiles_list = [get_item_profile(x) for x in ids]
            except:
                item_profiles_list = [get_item_profile(x) for x in [ids]]
            item_profiles = scipy.sparse.vstack(item_profiles_list)
            return item_profiles

        def build_user_profile(user_id, interactions_indexed):
            interactions_person = interactions_indexed.loc[user_id]
            user_item_profiles = get_item_profiles(interactions_person['item_id'])
            
            user_item_strengths = np.array(interactions_person['rating']).reshape(-1,1)
            try:
                user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths) # numpy.matrix (1, 1381)
            except FloatingPointError as e: # there can be a problem if the sum of the user ratings is 0.
                print('Runtime error, setting user_item_strength_weighted_average to 0s')
                user_item_strengths_weighted_avg = np.zeros((1, user_item_strengths.shape[1]))
            user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
            return user_profile_norm

        interactions_indexed = interactions_train[interactions_train['item_id'].isin(self.recipes['item_id'])].set_index('user_id')
        self.active_user_profile = build_user_profile(self.active_user_id, interactions_indexed)



    def _get_similar_items_to_user_profile(self, topn=1000):
        cosine_similarities = cosine_similarity(self.active_user_profile, self.tfidf_matrix)
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, items_to_ignore=[], topn=10):
        similar_items = self._get_similar_items_to_user_profile()
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        recommendations = pd.DataFrame(similar_items_filtered, columns=['item_id', 'recStrength']).head(topn)
        return recommendations

recommender = ContentRecommender(599450)

recommendations = recommender.recommend_items()
recommendations = pd.merge(recommendations, recommender.recipes[['name', 'item_id']], how='left', on='item_id')
print(recommendations)