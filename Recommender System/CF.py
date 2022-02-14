import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader, SVD, SVDpp, CoClustering
from surprise.model_selection import KFold
from collections import defaultdict
from datetime import datetime

class CollaborativeRecommender:
    def __init__(self, active_user_id, load=True):
        """Constructor for CollaborativeRecommender"""
        self.type = 'Collaborative Filter'
        self.recipes, self.interactions_train = self.load_data()
        self.active_user_id = active_user_id
        self.model = SVD(verbose=False) # SVDpp(verbose=True) or NMF() # Doesn't really work or CoClustering(n_cltr_i= 6,verbose = False)
        self.load = load


    def load_data(self):
        """Load the data"""
        interactions_train = pd.read_csv('data/interactions_train.csv')
        recipes = pd.read_csv('data/recipes.csv')
        interactions_train.drop(columns=['review', 'date'], inplace=True)
        reader = Reader(rating_scale=(0, 5))
        interactions_train = Dataset.load_from_df(interactions_train[['user_id', 'item_id', 'rating']], reader)
        return recipes, interactions_train


    def kfold_train(self, kfold=5, k=10, threshold=3.5):
        """Train the model on the interactions_train dataset with the model specified in the constructor"""

        kf = KFold(n_splits=kfold)

        for trainset, testset in kf.split(self.interactions_train):
            self.model.fit(trainset)
            predictions = self.model.test(testset)

        with open(f'models/{type(self.model).__name__}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        return 


    def recommend_items(self, n=5):
        """Recommend a set of n items to the active user"""
        if self.load: self.model = pickle.load(open(f'models/{type(self.model).__name__}.pkl', 'rb'))
        else: self.kfold_train()
 
        iids = self.recipes['item_id'].to_list() # All the recipe ids from the dataset as a list
        test_set = [[self.active_user_id, iid, 1.] for iid in iids] # TODO WORK OUT WHAT THIS 1. IS/SHOULD BE
        predictions = self.model.test(test_set)
        pred_ratings = [pred.est for pred in predictions]

        # Add the predicted ratings to the test_set
        p = pd.concat([pd.DataFrame(test_set, columns=[0, 'item_id', 2]).drop(columns=[0, 2]), pd.DataFrame(pred_ratings, columns=['Predicted Rating'])], axis=1)
        # Add in the data needed for contextual filtering
        p = pd.merge(p, self.recipes[['name', 'item_id', 'minutes', 'nutrition']], how='left', on='item_id')

        def time_filter(row):
            """If it's a weekday or past 8pm and the recipe takes longer than half an hour, adjust the score down by 1% for every 5 minutes over half an hour"""
            if datetime.now().weekday() < 4:
                time = row[3]
                p = row.iloc[1]
                p = p*0.995**(max(0, time-30))
                row.iloc[1] = p
            return row

        def health_filter(row):
            health_weight = 0.8
            nutrition = eval(row[4])
            calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat, _ = nutrition   # PDV is percent daily value
            
            def z(x, mu, sigma):
                return (x - mu) / sigma

            health_coefficient = 1 + (z(calories, 475.390531, 2823.977013) + z(total_fat_PDV, 34.809417, 61.341142) + z(sugar_PDV, 93.393067, 2307.438741) + z(sodium_PDV, 29.804066, 133.302684) - z(protein_PDV, 34.498343, 58.939851) + z(saturated_fat, 43.880586, 87.361438) / 6) # could also multiply the values
            # TODO make sure protein goes the other way
            # print("health_coefficient", health_coefficient)
            # TODO limit the effects of the health filter and revisit the coefficient.
            row.iloc[1] = min(row.iloc[1] * health_coefficient * health_weight, row.iloc[1])
            return row

        p = p.apply(time_filter, axis=1)
        print("We are using the current day of the week to enhance the relevance of your recommendations, this data will not be stored\n")
        p = p.apply(health_filter, axis=1)

        p.sort_values(by='Predicted Rating', ascending=False, inplace=True)
        p = p[:n]

        while p.iloc[-1]['Predicted Rating'] < 4:
            p = p[:-1]
        if p.shape[0] < n:
            print(f"We were unable to find {n} items which we believe you'll like, we have shortened the list of recommendations to ensure a high quality of recommendations\n")

        p = p.drop(columns=['item_id', 'nutrition']).rename(columns={"name":"Recipe Name","minutes":"Minutes to prepare"})
        return p


# recommender = CollaborativeRecommender(599450)
# recommendations = recommender.recommend_items()
# print(recommendations)