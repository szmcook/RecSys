import pandas as pd
import numpy as np
import math

from CF import CollaborativeRecommender
from CBF_basic import ContentRecommender

test_set = pd.read_csv('data/interactions_test.csv')
recipes = pd.read_csv('data/recipes.csv')


def evaluate_RMSE_CF():
    """Returns the average RMSE for the users in the test set on the Collaborative Filter"""
    RMSEs = []

    users = test_set['user_id'].unique()

    n = 300

    for i in range(len(users[:n])):
        print(f"\r{100*round(i/len(users[:n]), 3)}%")
        user_ratings = test_set[test_set['user_id'] == users[i]]
        rated_items = recipes[recipes['item_id'].isin(user_ratings['item_id'])]
        
        # r = pd.merge(user_ratings, rated_items, how='left', on='item_id')
        # print(r.head())

        recommender = CollaborativeRecommender(load=True)

        # PREDICT ONE AT A TIME
        ys = []
        for i in range(rated_items.shape[0]):
            item = rated_items.iloc[i]
            y_hat = recommender.predict_rating(users[i], item['item_id'])
            y = user_ratings[user_ratings['item_id'] == item['item_id']]['rating'].iloc[0]
            ys.append(y_hat - y)
        RMSE = math.sqrt(  sum( [(x**2)/len(ys) for x in ys] )  )
        RMSEs.append(RMSE)
        
        # PREDICT IN BATCHES
        # y_hats = np.array(recommender.predict_several_ratings(users[i], list(rated_items)))
        # print(y_hats)
        # ys = user_ratings['rating']

        # print(ys)

    res = sum(RMSEs)/len(RMSEs)
    return res


rmse_CF = evaluate_RMSE_CF()
print(rmse_CF)