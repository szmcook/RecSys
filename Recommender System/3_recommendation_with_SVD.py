import pandas as pd
import numpy as np
from collections import Counter
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import pickle
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate,train_test_split,KFold
from surprise import SVD,SVDpp, NMF,SlopeOne,CoClustering
from surprise import accuracy
from collections import defaultdict

random.seed(0)

"""
Now we import the user recipe interaction data to build the recommender base on user interaction
"""


data = pd.read_csv('data/clean_interactions.csv')
recipes = pd.read_csv('data/clean_recipes.csv')



def make_df_from_count(serie,name):
    counts = dict(Counter(serie))
    return pd.DataFrame.from_dict(counts,orient='index').reset_index().rename(columns={'index':name,0:f'{name}_count'})


recipe_df = make_df_from_count(data.recipe_id,'recipe_id')
recipe_df.head()


len(recipe_df[recipe_df['recipe_id_count'] <2])/len(recipe_df)


len(recipe_df[recipe_df['recipe_id_count'] <10])/len(recipe_df) 


"""
Most of the recipes (39%) has only one review. Majority of them has less than 10 reviews (~90%)..
Shall I remove some items so to avoid a matrix too sparse?

What about user behavior?
"""


user_df = make_df_from_count(data.user_id,'user_id')
user_df.head()


user_df['user_id_count'].hist(bins=1000)
ax = plt.gca()
ax.set_xlim((0,50))


len(user_df)


"""
Most users (94%) do not have more than 10 reviews
"""


len(user_df[user_df.user_id_count <10])/len(user_df)


"""
We observe that the histrogram shows quite a right tail and there are very popular recipes

We would like to remove the recipes that have only 1 reviews
"""


data.recipe_id.unique()


data_merge = data.merge(recipe_df,how='left',left_on='recipe_id',right_on = 'recipe_id')


data_merge.head()


"""
Draw a sample to test if the count function works correctly and we can write a unit test for it
"""


def test_count_works (df):
    sample = random.choice(df.recipe_id)
    #print(sample)
    mask = df.recipe_id == sample
    length = len(df[mask])
    #print(length)
    try: 
        count = list(df[mask]['count'])[0]
        return length == count
    except: 
        return False

test_count_works(data_merge)


column_UI_mtx = ['user_id','recipe_id','rating']


df_UI = pd.DataFrame(data_merge[column_UI_mtx])


def id_transformation(values):
    unique_values = np.unique(values)
    return dict([(x, y) for y, x in enumerate(unique_values)])


user_id_transformed = id_transformation(df_UI.user_id)
recipe_id_transformed = id_transformation(df_UI.recipe_id)


transformed_user_id = pd.DataFrame.from_dict(user_id_transformed,orient = 'index').reset_index().rename(columns={'index':'user_id',0:'new_user_id'})
transformed_recipe_id = pd.DataFrame.from_dict(recipe_id_transformed,orient = 'index').reset_index().rename(columns={'index':'recipe_id',0:'new_recipe_id'})


df_UI_new = df_UI.merge(transformed_user_id, how = 'left' ,left_on = 'user_id',right_on='user_id').merge(transformed_recipe_id, how = 'left' ,left_on = 'recipe_id',right_on='recipe_id')


df_UI_new.head()


shape = (len(transformed_user_id),len(transformed_recipe_id))


UI_mtx = coo_matrix ((df_UI_new.rating,(df_UI_new.new_user_id,df_UI_new.new_recipe_id)),shape = shape)


UI_mtx.shape


final_UI_mtx = df_UI_new.drop(columns=['user_id', 'recipe_id'])
final_UI_mtx.head()


"""
Save all the intermediate data for further use
"""


def save_file_to_pickle(item, file_name, file_type = 'obj'):
    file = open(f'output/{file_name}.{file_type}', 'wb') 
    pickle.dump(item, file)
    file.close()


recipes = recipes.merge(transformed_recipe_id, how = 'left' ,left_on = 'id',right_on='recipe_id')


recipes = recipes.drop(columns=['recipe_id','id'])


recipes.head()


# save_file_to_pickle(final_UI_mtx,"UI_mtx",'pkl')


# save_file_to_pickle(recipes,"recipes",'pkl')


final_UI_mtx.info()


"""
We will use surprise for the model so we need to treat the data as per requirement
"""


# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(final_UI_mtx[['new_user_id', 'new_recipe_id', 'rating']], reader)


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


# We'll use the famous SVD algorithm.
algo = SVD(verbose=True)


def kfold_train_test(data,algo,kfold = 5,k = 10,treshold = 3.5):
    kf = KFold(n_splits=kfold)
    precision_kfold = []
    recall_kfold = []

    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=treshold)

        # Precision and recall can then be averaged over all users
        precision_kfold.append(sum(prec for prec in precisions.values()) / len(precisions))
        recall_kfold.append(sum(rec for rec in recalls.values()) / len(recalls))
        
    return precision_kfold,recall_kfold


# kfold_train_test(data)


kf = KFold(n_splits=5)
precision_kfold = []
recall_kfold = []

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3.5)

    # Precision and recall can then be averaged over all users
    precision_kfold.append(sum(prec for prec in precisions.values()) / len(precisions))
    recall_kfold.append(sum(rec for rec in recalls.values()) / len(recalls))


print("5-fold precision@10 is {:.3f}".format(np.mean(precision_kfold)),"\n","5-fold recall@10 is {:.3f}".format(np.mean(recall_kfold)))


# save_file_to_pickle(algo,"SVD_algo",'pkl')


uid = 226571
iids =[23,56,34,111]
recipes_names = dict([(rep_id,name) for name,rep_id in zip(recipes.name,recipes.new_recipe_id)])


def pretty_text (text):
    ''' This function takes in text and try to put it in a human readable format by putting back \' and making it capitalize
    '''
    text = text.replace(" s ","\'s ")
    text_split = text.split(" ")
    #print(text_split)
    text_split = [t.strip().capitalize() for t in text_split if t != '']
    #print(text_split)
    return " ".join(text_split)


[pretty_text(recipes_names[r]) for r in iids]


# save_file_to_pickle(recipes_names,"recipes_names",'pkl')


def get_n_predictions(iids,algo,n = 10, uid = 226571, item_name = recipes_names):
    
    # create the list to search in
    iid_to_test = [iid for iid in range(231637) if iid not in iids]
    # build data for surprise
    test_set = [[uid,iid,4.] for iid in iid_to_test]
    # predict
    predictions = algo.test(test_set)
    #get prediction
    pred_ratings = [pred.est for pred in predictions]
    # return top_n indexes
    top_n = np.argpartition(pred_ratings,1)[-n:]
    # return list of recipe names
    results = [item_name[k] for k in top_n]
    
    return [pretty_text(r) for r in results]


get_n_predictions(iids,algo)


n=10
# create the list to search in
iid_to_test = [iid for iid in range(231637) if iid not in iids]
# build data for surprise
test_set = [[uid,iid,4.] for iid in iid_to_test]
# predict
predictions = algo.test(test_set)
#get prediction
pred_ratings = [pred.est for pred in predictions]
# return top_n indexes
top_n = np.argpartition(pred_ratings,1)[-n:]
# return list of recipe names
results = [recipes_names[k] for k in top_n]

print( [pretty_text(r) for r in results] )


# We'll use the famous SVD algorithm.
algo2 = SVDpp(verbose=True) # we skip this because it is too time consuming for our data... unless we test on a small set of it


algo2 = NMF()


# we will try slopeone
algo3 = CoClustering(n_cltr_i= 6,verbose = False)


precision_kfold,recall_kfold = kfold_train_test(algo = algo3,data = data)


print("5-fold precision@10 is {:.3f}".format(np.mean(precision_kfold)),"\n","5-fold recall@10 is {:.3f}".format(np.mean(recall_kfold)))


