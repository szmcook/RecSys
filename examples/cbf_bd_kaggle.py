
"""
# BD. Lab 8. Recommender Systems
"""


import numpy as np
import scipy
import pandas as pd
import random
import sklearn


from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


"""
# Loading data
"""


recipes = pd.read_csv('../Recommender System/data/recipes.csv')
interactions_df = pd.read_csv('../Recommender System/data/interactions_train.csv')

users_interactions_count_df = interactions_df.groupby(['user_id', 'item_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))


print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'user_id',
               right_on = 'user_id')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))


interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['user_id', 'item_id'])['rating'].sum() \
                    .apply(lambda x: np.log10(x+1)*2).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, stratify=interactions_full_df['user_id'], 
                                                               test_size=0.2, random_state=666)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

interactions_full_indexed_df = interactions_full_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')

def get_items_interacted(person_id, interactions_df):
    interacted_items = interactions_df.loc[person_id]['item_id']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:

    def get_not_interacted_items_sample(self, person_id, sample_size, seed=666):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(recipes['item_id'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index
        
    def _apk(self, actual, predicted, k=10):
        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    def evaluate_model_for_user(self, model, person_id):
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['item_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['item_id'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['item_id'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        for item_id in person_interacted_items_testset:
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            valid_recs_df = person_recs_df[person_recs_df['item_id'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['item_id'].values
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)
        
        apk_at_5 = self._apk(person_interacted_items_testset, valid_recs, 5)
        apk_at_10 = self._apk(person_interacted_items_testset, valid_recs, 10)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10,
                          'apk@5': apk_at_5,
                          'apk@10': apk_at_10}
        return person_metrics

    def evaluate_model(self, model):
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            try:
                person_metrics = self.evaluate_model_for_user(model, person_id)  
                person_metrics['_person_id'] = person_id
                people_metrics.append(person_metrics)
            except:
                pass
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics)
        print(detailed_results_df.columns)
        detailed_results_df = detailed_results_df.sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          'mapk@5': detailed_results_df['apk@5'].mean(),
                          'mapk@10': detailed_results_df['apk@10'].mean()
                         }    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()


interactions_test_indexed_df.index.unique().values.shape

"""Popularity model"""
item_popularity_df = interactions_full_df.groupby('item_id')['rating'].sum().sort_values(ascending=False).reset_index()


"""
# Content-Based Filtering model
"""


vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords.words('english'))

item_ids = recipes['item_id'].tolist()
vectorizer_input = recipes['name'] + " " + recipes['description']
tfidf_matrix = vectorizer.fit_transform(vectorizer_input)
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix


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

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['item_id'])
    
    user_item_strengths = np.array(interactions_person_df['rating']).reshape(-1,1)
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_train_df[interactions_train_df['item_id'] \
                                                   .isin(recipes['item_id'])].set_index('user_id')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles


interactions_train_df[interactions_train_df['item_id'] \
                                                   .isin(recipes['item_id'])].set_index('user_id').index.unique()


user_profiles = build_users_profiles()

print(user_profiles[599450])


pd.DataFrame(sorted(zip(tfidf_feature_names, 
                        user_profiles[599450].flatten().tolist()), key=lambda x: -x[1])[:20],
             columns=['token', 'relevance'])


class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['item_id', 'recStrength']).head(topn)

        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(recipes)


recommendations = content_based_recommender_model.recommend_items(599450)

recommendations = pd.merge(recommendations, recipes[['name', 'item_id']], how='left', on='item_id')

print(recommendations)





import sys
sys.exit()

print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)


"""
## Collaborative Filtering model
"""


"""
## Matrix Factorization
"""


users_items_pivot_matrix_df = interactions_train_df.pivot(index='user_id', 
                                                          columns='item_id', 
                                                          values='rating').fillna(0)

users_items_pivot_matrix_df.head(10)


users_items_pivot_matrix = users_items_pivot_matrix_df.values
users_items_pivot_matrix[:10]


users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]


users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
users_items_pivot_sparse_matrix


NUMBER_OF_FACTORS_MF = 30

U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)


U.shape


Vt.shape


sigma = np.diag(sigma)
sigma.shape


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
np.round(all_user_predicted_ratings, 3)


np.max(np.round(all_user_predicted_ratings, 3) - np.round(users_items_pivot_matrix, 3))


all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())


cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)


len(cf_preds_df.columns)


from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

nmf_model = NMF(20)

vals = nmf_model.fit_transform(users_items_pivot_sparse_matrix)

nmf_result = np.dot(vals, nmf_model.components_)

nmf_result = (nmf_result - nmf_result.min()) / (nmf_result.max() - nmf_result.min())

cf_preds_df = pd.DataFrame(nmf_result, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler, StandardScaler

nmf_model = LatentDirichletAllocation(20)

vals = nmf_model.fit_transform(users_items_pivot_sparse_matrix)

nmf_result = np.dot(vals, nmf_model.components_)

nmf_result = (nmf_result - nmf_result.min()) / (nmf_result.max() - nmf_result.min())

cf_preds_df = pd.DataFrame(nmf_result, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)


interactions = pd.melt(interactions_train_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0).reset_index(), 
        id_vars=['user_id'])

interactions


X = interactions.merge(recipes, on='item_id', how='left').select_dtypes(include=['float', 'int']).fillna(0).drop(columns=['value'])

y = interactions.merge(recipes, on='item_id', how='left')['value'].fillna(0).values.reshape((-1,1))

from sklearn.linear_model import LinearRegression

lin = LinearRegression()

lin.fit(X, y)

lin.predict(X)


y.reshape(1,-1)


pd.DataFrame({'real': lin.predict(X).reshape(1,-1)[0], 'pred': y.reshape(1,-1)[0]})


interactions['pred'] = lin.predict(X).reshape(1,-1)[0]

cf_preds_df = interactions.pivot(index='user_id', columns='item_id', values='pred').fillna(0)
cf_preds_df


class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        recommendations_df = sorted_user_predictions[~sorted_user_predictions['item_id'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, recipes)


print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)


"""
## Association rules
"""


dataset = list(interactions_df.groupby('user_id', as_index=False).agg({'item_id': list})['item_id'].values)


from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import fpgrowth, association_rules

res = association_rules(fpgrowth(df, min_support=0.005, use_colnames=True, max_len=2), metric="lift", min_threshold=1).sort_values('lift', ascending=False)

res['antecedents'] = [list(i)[0] for i in res['antecedents']]
res['consequents'] = [list(i)[0] for i in res['consequents']]

res


#interactions_df


class AprioriRecommender:
    
    MODEL_NAME = 'Apriori'
    
    def __init__(self, rules_df=None):
        self.rules_df = rules_df
        #self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=10):
        cands = list(interactions_df.loc[interactions_df['user_id'] == person_id, 'item_id'].values)
        
        cands = interactions_df.loc[interactions_df['user_id'] == person_id, ['item_id', 'rating']]
        
        cands = cands.merge(self.rules_df, right_on='antecedents', left_on='item_id', how='left')
        
        cands['lift'] = np.log(cands['lift'] * cands['rating'])
        
        selected_cands = cands.loc[:, ['consequents', 'lift']].sort_values('lift', ascending=False)

        return selected_cands
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        similar_items_filtered = similar_items.loc[~similar_items['consequents'].isin(items_to_ignore),:]
        similar_items_filtered.columns=['item_id', 'recStrength']
        recommendations_df = similar_items_filtered.head(topn)
        return recommendations_df
    
apriori_recommender_model = AprioriRecommender(res)


print('Evaluating Apriori model...')
ap_global_metrics, ap_detailed_results_df = model_evaluator.evaluate_model(apriori_recommender_model)
print('\nGlobal metrics:\n%s' % ap_global_metrics)
ap_detailed_results_df.head(10)


"""
## Hybrid Recommender
"""


class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, ap_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0, ap_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.ap_rec_model = ap_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.ap_ensemble_weight = ap_ensemble_weight
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})        
        ap_recs_df = self.ap_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, 
                                                        topn=100).rename(columns={'recStrength': 'recStrengthAP'})
        
        recs_df = cb_recs_df.merge(cf_recs_df, how = 'outer', left_on = 'item_id', right_on = 'item_id').fillna(0.0) \
                            .merge(ap_recs_df, how = 'outer', left_on = 'item_id', right_on = 'item_id').fillna(0.0)
        
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) + (recs_df['recStrengthCF'] * self.cf_ensemble_weight) + (recs_df['recStrengthAP'] * self.ap_ensemble_weight)
        
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)
        
        return recommendations_df
    
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, apriori_recommender_model, recipes,
                                             cb_ensemble_weight=1.0, cf_ensemble_weight=100.0, ap_ensemble_weight=1.0)


print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df.head(10)


"""
## Comparing the methods
"""


global_metrics_df = pd.DataFrame([cb_global_metrics, pop_global_metrics, cf_global_metrics, hybrid_global_metrics, 
                                  ap_global_metrics]) \
                        .set_index('modelName')
global_metrics_df


ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15,8))
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xydescription=(0, 10), descriptioncoords='offset points')


"""
# Testing
"""


def inspect_interactions(person_id, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(recipes, how = 'left', 
                                                      left_on = 'item_id', 
                                                      right_on = 'item_id') \
                          .sort_values('rating', ascending = False)[['rating', 'item_id', 'name', 'url']]


inspect_interactions(-1479311724257856983, test_set=False).head(20)


