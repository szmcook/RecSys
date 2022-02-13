import pandas as pd
import numpy as np
import random
import pickle
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise import SVD,SVDpp, NMF,SlopeOne,CoClustering
from collections import defaultdict

random.seed(0)

interactions = pd.read_csv('data/clean_interactions.csv')
recipes = pd.read_csv('data/clean_recipes.csv')
interactions.drop(columns=['review', 'date'], inplace=True)

reader = Reader(rating_scale=(0, 5))
interactions = Dataset.load_from_df(interactions[['user_id', 'item_id', 'rating']], reader)

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precisions, recalls


def kfold_train_test(data, model, kfold=5, k=10, threshold=3.5):
    kf = KFold(n_splits=kfold)
    precision_kfold = []
    recall_kfold = []

    for trainset, testset in kf.split(data):
        model.fit(trainset)
        predictions = model.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)

        # Precision and recall can then be averaged over all users
        precision_kfold.append(sum(prec for prec in precisions.values()) / len(precisions))
        recall_kfold.append(sum(rec for rec in recalls.values()) / len(recalls))
    
    with open(f'models/{type(model).__name__}.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(f"5-fold precision@10 is {round(np.mean(precision_kfold), 3)}\n5-fold recall@10 is {round(np.mean(recall_kfold), 3)}")
    return precision_kfold, recall_kfold


# use SVD algorithm.
model = SVD(verbose=False)
# Trying other algorithms
# model = SVDpp(verbose=True)
# model = NMF() # Doesn't really work
# model = CoClustering(n_cltr_i= 6,verbose = False)

load = False
if load:
    model = pickle.load(open(f'models/{type(model).__name__}.pkl', 'rb'))
else:
    precision_kfold, recall_kfold = kfold_train_test(interactions, model)


def get_prediction(uid, iid, model):
    test_set = [[uid,iid,4.]]
    predictions = model.test(test_set)
    pred_ratings = [pred.est for pred in predictions]
    return pred_ratings[0]

# p = get_prediction(599450, 263103, model)

def recommend_items(uid, model, n):
    # I think the idea here is to test on everything that hasn't been rated by the user and then return the top 10

    iids = recipes['item_id'].to_list()[:20] # All the recipe ids from the dataset as a list
    test_set = [[uid,iid,1.] for iid in iids]
    predictions = model.test(test_set)
    pred_ratings = [pred.est for pred in predictions]

    # Add the predicted ratings to the test_set
    p = pd.concat([pd.DataFrame(test_set).drop(columns=[0, 2]), pd.DataFrame(pred_ratings, columns=['prediction'])], axis=1)
    p.sort_values(by='prediction', ascending=False, inplace=True)
    return p[:n]


rs = recommend_items(599450, model, 10)
print(rs.head(10))