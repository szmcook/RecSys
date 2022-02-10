import pandas as pd


items_df1 = pd.read_csv('data-beer-liquor-wine/wine reviews.csv')
items_df2 = pd.read_csv('data-beer-liquor-wine/447_1.csv').drop(columns=['primaryCategories', 'quantities'])
items_df = pd.concat([items_df1, items_df2])
items_df["descriptions"] = items_df["descriptions"].apply(lambda s: 'Carmex' if 'Carmex' in str(s) else s)
items_df = items_df[items_df['descriptions'] != 'Carmex']

print(items_df.columns)

interactions_df = items_df[['reviews.username', 'id', 'reviews.rating', 'reviews.didPurchase', 'reviews.doRecommend']]
# Replace didPurchase with 1 or 0
# Replace doRecommend with 1 or 0

# Create an eventStrength column that takes values from -1 to 1 based on the rating, purchase and recommend columns
interactions_df['eventStrength'] = interactions_df['reviews.rating'] + interactions_df['reviews.doRecommend']

interactions_train_df = train # from train test split

users_items_pivot_matrix_df = interactions_train_df.pivot(index='reviews.username', 
                                                          columns='id', 
                                                          values='eventStrength').fillna(0)

users_items_pivot_matrix_df.head(10)

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

        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        return recommendations_df
    
# cf_recommender_model = CFRecommender(cf_preds_df, articles_df)